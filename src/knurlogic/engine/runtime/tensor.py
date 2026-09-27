"""Tensor split: one model served by N ranks, every layer's weights split
N ways (docs/SERVER.md, "Cluster: tensor split").

    rank 0:   HTTP -> Scheduler -> TensorExecutor --plan--> ranks 1..N-1
    rank r:   follow(): apply the plan, run the same step, never sample

Written for knurlogic (no exo code). The layer split follows the layout of
mlx-lm's Qwen3_5 `Model.shard` (MIT; engine/runtime/PROVENANCE.md) with one
change that is the point of this module: a VQ codebook is REPLICATED, never
sliced. mlx's default predicates split every parameter of a sharded layer;
applied to a VQSwitchLinear they cut the [K, d] codebook (all-to-sharded) or
its d axis (sharded-to-all). VQSwitchLinear's own guard catches only the
first; the second decodes against half a codebook and emits fluent garbage.

The step protocol is engine/runtime/plan.py. Everything that moves between
ranks goes through `Link.exchange`, on the scheduler's thread.
"""

from __future__ import annotations

import logging
import random
from typing import Callable, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map, tree_map_with_path

from . import plan as P
from .executor import (Admission, Checkpoint, Finished, LocalExecutor)

logger = logging.getLogger(__name__)
GIB = 1 << 30


class Desync(RuntimeError):
    """The ranks no longer agree on what they are doing."""


# ------------------------------------------------------------------ split

def predicate(kind: str) -> Callable:
    """The split axis for a parameter of a layer split `kind`
    ("all-to-sharded": output axis; "sharded-to-all": input axis), or None
    to keep it whole. A path ending in `codebook` is ALWAYS None."""
    if kind not in ("all-to-sharded", "sharded-to-all"):
        raise ValueError(kind)

    def pred(path: str, w):
        if path.endswith("codebook"):
            return None
        if kind == "all-to-sharded":
            return -1 if path.endswith("bias") else max(w.ndim - 2, 0)
        return None if path.endswith("bias") else -1
    return pred


_S2A = predicate("sharded-to-all")


def split_params(params, pred, rank: int, n: int, segments=1):
    """`params` with every array `pred` names split n ways along its axis
    (within each of `segments`: a fused QKV splits each part), keeping
    part `rank`. The single-process core of `shard`, so the arithmetic is
    testable without a ring."""
    def one(path, w):
        if not isinstance(w, mx.array):
            return w
        axis = pred(path, w)
        if axis is None:
            return w
        if isinstance(segments, int):
            parts = mx.split(w, segments, axis=axis)
        else:
            parts = mx.split(w, list(segments), axis=axis)
        return mx.contiguous(mx.concatenate(
            [mx.split(p, n, axis=axis)[rank] for p in parts], axis=axis))
    return tree_map_with_path(one, params)


def _split_inplace(module, pred, rank, n, segments=1) -> None:
    if pred is _S2A and "bias" in module:
        # a split input axis gives each rank a partial sum; a bias added on
        # every rank would be summed N times
        raise ValueError(f"{type(module).__name__} has a bias; a "
                         f"sharded-to-all split of it is not built")
    module.update(split_params(module.parameters(), pred, rank, n, segments))


class Reduce(nn.Module):
    """`inner`'s partial output summed across the ranks -- in float32.

    A split layer's ranks each round their partial sum to the activation
    dtype (bf16) and the sum of two rounded halves is rounded again; the
    whole layer rounds once. Summing the halves in float32 leaves one
    rounding after the sum, the whole layer's count plus the partials'."""

    def __init__(self, inner, group):
        super().__init__()
        self.inner = inner
        self._group = group

    def __call__(self, *a, **kw):
        y = self.inner(*a, **kw)
        return mx.distributed.all_sum(y.astype(mx.float32),
                                      group=self._group).astype(y.dtype)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if "inner" not in self:
                raise
            return getattr(self["inner"], name)


def shard(model, group) -> None:
    """Split a qwen3_5 / qwen3_5_moe model across `group` in place.

    Refuse first (tuning/resolve.tensor_refusals) -- this assumes the
    arithmetic was checked."""
    n, rank = group.size(), group.rank()
    a2s, s2a = predicate("all-to-sharded"), _S2A

    def repeat_kv(layer, h):
        # fewer KV heads than ranks: each rank gets a copy of its head
        if n <= h:
            return

        def rep(p):
            s = p.shape
            p = p.reshape(h, s[0] // h, *s[1:])
            return mx.repeat(p, n // h, axis=0).reshape(-1, *s[1:])
        layer.update(tree_map(rep, layer.parameters()))

    for layer in model.layers:
        if layer.is_linear:
            la = layer.linear_attn
            kd = la.key_dim
            _split_inplace(la.conv1d, lambda p, w: 0, rank, n,
                           segments=[kd, 2 * kd])
            la.conv1d.groups //= n
            _split_inplace(la.in_proj_qkv, a2s, rank, n, segments=[kd, 2 * kd])
            for m in (la.in_proj_z, la.in_proj_b, la.in_proj_a):
                _split_inplace(m, a2s, rank, n)
            la.dt_bias = mx.contiguous(mx.split(la.dt_bias, n)[rank])
            la.A_log = mx.contiguous(mx.split(la.A_log, n)[rank])
            _split_inplace(la.out_proj, s2a, rank, n)
            la.num_k_heads //= n
            la.num_v_heads //= n
            la.key_dim //= n
            la.value_dim //= n
            la.conv_dim //= n
            la.sharding_group = None            # Reduce sums it, in fp32
            layer.linear_attn = Reduce(la, group)
        else:
            at = layer.self_attn
            _split_inplace(at.q_proj, a2s, rank, n)
            repeat_kv(at.k_proj, at.num_key_value_heads)
            repeat_kv(at.v_proj, at.num_key_value_heads)
            _split_inplace(at.k_proj, a2s, rank, n)
            _split_inplace(at.v_proj, a2s, rank, n)
            _split_inplace(at.o_proj, s2a, rank, n)
            at.num_attention_heads //= n
            at.num_key_value_heads = max(1, at.num_key_value_heads // n)
            layer.self_attn = Reduce(at, group)

        mlp = layer.mlp
        if hasattr(mlp, "switch_mlp"):
            se = mlp.shared_expert
            _split_inplace(se.gate_proj, a2s, rank, n)
            _split_inplace(se.up_proj, a2s, rank, n)
            _split_inplace(se.down_proj, s2a, rank, n)
            sw = mlp.switch_mlp
            _split_inplace(sw.gate_proj, a2s, rank, n)
            _split_inplace(sw.up_proj, a2s, rank, n)
            _split_inplace(sw.down_proj, s2a, rank, n)
            mlp.sharding_group = None
        else:
            _split_inplace(mlp.gate_proj, a2s, rank, n)
            _split_inplace(mlp.up_proj, a2s, rank, n)
            _split_inplace(mlp.down_proj, s2a, rank, n)
        layer.mlp = Reduce(mlp, group)
    check_codebooks(model)


def check_codebooks(model) -> None:
    """Every VQ module still holds the codebook it was built with."""
    bad = []
    for name, m in model.named_modules():
        k = getattr(m, "_k_expect", None)
        cb = getattr(m, "codebook", None)
        if k is not None and cb is not None and int(cb.shape[0]) != k:
            bad.append(f"{name}: K={cb.shape[0]}, built with {k}")
    if bad:
        raise RuntimeError("a VQ codebook was split: " + "; ".join(bad[:4]))


# ------------------------------------------------------------------- link

class Link:
    """The control exchange between ranks: one all_gather of a fixed-size
    vector per step, then the plan's bytes (all_sum; only rank 0's are
    non-zero) when rank 0 has any."""

    def __init__(self, group):
        self.group = group
        self.rank = group.rank()
        self.size = group.size()
        self.step = 0

    def barrier(self) -> None:
        mx.eval(mx.distributed.all_sum(mx.array(1), group=self.group,
                                       stream=mx.cpu))

    def exchange(self, over: int, payload: Optional[bytes] = None):
        """-> (control rows, one per rank; the plan bytes or None)."""
        import numpy as np
        n = len(payload) if (self.rank == 0 and payload) else 0
        ctl = mx.array([P.control(over, self.step, n)], dtype=mx.int64)
        rows = mx.distributed.all_gather(ctl, group=self.group,
                                         stream=mx.cpu).tolist()
        steps = {r[P.STEP] for r in rows}
        if len(steps) != 1:
            raise Desync(f"ranks at different steps: {[r[P.STEP] for r in rows]}")
        self.step += 1
        length = rows[0][P.LENGTH]
        if not length:
            return rows, None
        if self.rank == 0:
            buf = mx.array(np.frombuffer(payload, dtype=np.uint8))
        else:
            buf = mx.zeros((length,), dtype=mx.uint8)
        out = mx.distributed.all_sum(buf, group=self.group, stream=mx.cpu)
        return rows, np.array(out).tobytes()


class Journal:
    """Rank 0's ops since the last exchange, in the order they happened."""

    def __init__(self):
        self.ops: List[dict] = []

    def add(self, op: str, **fields) -> None:
        self.ops.append({"op": op, **fields})

    def take(self) -> List[dict]:
        ops, self.ops = self.ops, []
        return ops


class Ring:
    """Rank 0's side of the ring, kept across executors and prompt caches
    (both are rebuilt; the ring is not)."""

    def __init__(self, link: Link):
        self.link = link
        self.journal = Journal()
        #: max over ranks >= 1 of (active - limit) at the last exchange, and
        #: rank 0's own active memory then
        self.peer_over: Optional[int] = None
        self.local_then = 0
        self.mismatches = 0

    @property
    def world(self) -> int:
        return self.link.size

    def exchange(self, over: int, plan: dict) -> None:
        payload = None if P.empty(plan) else P.encode(plan)
        rows, _ = self.link.exchange(over, payload)
        self.peer_over = max(r[P.OVER] for r in rows[1:])
        self.local_then = int(mx.get_active_memory())

    def peers_over_now(self) -> Optional[int]:
        """The peers' over-limit, moved by what rank 0 has freed or taken
        since the exchange (the ranks hold equal shards and apply the same
        ops, so memory moves together)."""
        if self.peer_over is None:
            return None
        return self.peer_over + int(mx.get_active_memory()) - self.local_then

    def stop(self) -> None:
        ops = self.journal.take() + [{"op": "stop"}]
        self.link.exchange(0, P.encode({"ops": ops}))


def assign_seed(sampling: dict) -> dict:
    """Every row gets a seed on a ring: rank 0's draw is then the draw any
    rank would make from the same logits."""
    s = dict(sampling or {})
    if s.get("seed") is None:
        s["seed"] = random.SystemRandom().randrange(1 << 31)
    return s


class JournalPromptCache:
    """The scheduler's PromptCache on rank 0 of a ring: count-based (no byte
    cap) and every change recorded for the other ranks. Byte trims (the
    memory guard) become pops of the least recently used entries."""

    def __init__(self, inner, journal: Journal):
        self.inner = inner
        self.journal = journal

    @property
    def lru(self):
        return self.inner.lru

    def fetch(self, key, tokens):
        # the admit op carries the hit; fetching changes nothing
        return self.inner.fetch(key, tokens)

    def insert(self, key, tokens, cache, kind: str, origin=None) -> None:
        if origin is None:
            raise ValueError("a ring's prompt cache inserts only from an "
                             "event (origin=(event, uid))")
        self.inner.insert(key, tokens, cache, kind)
        event, uid = origin
        self.journal.add("insert", uid=int(uid), event=event, kind=kind)

    def trim_to(self, n_bytes: int) -> None:
        before = len(self.inner.lru)
        self.inner.trim_to(n_bytes)
        popped = before - len(self.inner.lru)
        if popped:
            self.journal.add("pop", n=popped)

    @property
    def nbytes(self) -> int:
        return self.inner.nbytes


class TensorExecutor(LocalExecutor):
    """Rank 0's executor on a ring: the local batch engine, with every
    admission and removal journaled and the plan sent before each step."""

    def __init__(self, generator, ring: Ring, over: Callable[[], int]):
        super().__init__(generator)
        self.ring = ring
        self._over = over

    def insert(self, a: Admission) -> int:
        if a.wire is None:
            raise ValueError("a ring admission carries its wire fields")
        uid = super().insert(a)
        segs = [list(map(int, s)) for s in a.segments]
        prompt = list(map(int, a.prefix)) + [t for s in segs for t in s]
        self.ring.journal.add(
            "admit", uid=int(uid), prompt=prompt, segs=segs,
            hit=len(a.prefix), max_tokens=int(a.max_tokens),
            sampling=dict(a.sampling), penalties=dict(a.wire["penalties"]),
            initial=a.wire["initial"])
        return uid

    def remove(self, uids: List[int]) -> None:
        super().remove(uids)
        if uids:
            self.ring.journal.add("remove", uids=[int(u) for u in uids])

    def step(self):
        b = self.gen._batch
        plan = {"ops": self.ring.journal.take()}
        if len(b):
            plan["tokens"] = [[int(u), int(t)]
                              for u, t in zip(b.uids, b.t1.tolist())]
        self.ring.exchange(self._over(), plan)
        return super().step()

    def close(self) -> None:
        try:
            ops = self.ring.journal.take() + [{"op": "reset"}]
            self.ring.link.exchange(0, P.encode({"ops": ops}))
        finally:
            super().close()


# --------------------------------------------------------------- follower

class Mark:
    """A follower's memory limit: the working set less one step's measured
    transient (at least 5% of it, at least 4 GiB) -- the scheduler's rule."""

    def __init__(self, working_set: int):
        self.ws = int(working_set)
        self.spike = 0

    def limit(self) -> int:
        if not self.ws:
            return 0
        return self.ws - max(4 * GIB, self.ws // 20, int(self.spike * 1.25))

    def over(self) -> int:
        lim = self.limit()
        return int(mx.get_active_memory()) - lim if lim else 0

    def around(self, fn):
        mx.reset_peak_memory()
        before = int(mx.get_active_memory())
        out = fn()
        self.spike = max(self.spike, int(mx.get_peak_memory())
                         - max(before, int(mx.get_active_memory())))
        return out


def follow(model, tokenizer, model_key, link: Link, *, prompt_cache_size: int,
           completion_batch_size: int, prefill_step_size: int,
           working_set: int) -> int:
    """Rank >= 1: apply rank 0's plans and step until told to stop. The
    return value is the number of steps taken."""
    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    from .request import control_machine
    from .scheduler import PromptCache

    stream = mx.default_stream(mx.default_device())
    cache = PromptCache(prompt_cache_size)
    mark = Mark(working_set)
    ex: Optional[LocalExecutor] = None
    last: dict = {}
    steps = mismatches = 0

    def executor() -> LocalExecutor:
        nonlocal ex
        if ex is None:
            ex = LocalExecutor(MTPBatchGenerator(
                model, None, stats={}, vision=None,
                completion_batch_size=completion_batch_size,
                prefill_step_size=prefill_step_size, stream=stream))
        return ex

    while True:
        _, data = link.exchange(mark.over(), None)
        plan = P.decode(data) if data else {"ops": []}
        halt = False
        for op in plan.get("ops", []):
            kind = op["op"]
            if kind == "admit":
                prompt = op["prompt"]
                c, rest = cache.fetch(model_key, prompt)
                if len(prompt) - len(rest) != op["hit"]:
                    raise Desync(f"prompt cache hit {len(prompt) - len(rest)} "
                                 f"here, {op['hit']} on rank 0")
                procs = []
                if op["penalties"]:
                    from mlx_lm.sample_utils import make_logits_processors
                    procs = make_logits_processors(**op["penalties"])
                sm, _ = control_machine(tokenizer, op["initial"])
                uid = executor().insert(Admission(
                    segments=op["segs"], max_tokens=op["max_tokens"],
                    cache=c, prefix=prompt[:op["hit"]],
                    sampling=op["sampling"], processors=procs,
                    state_machine=sm))
                if uid != op["uid"]:
                    raise Desync(f"admitted as {uid}, rank 0 has {op['uid']}")
            elif kind == "remove":
                if ex is not None:
                    ex.remove(op["uids"])
            elif kind == "insert":
                got = last.get((op["event"], op["uid"]))
                if got is None:
                    raise Desync(f"no {op['event']} for row {op['uid']} in "
                                 f"the last step")
                cache.insert(model_key, got[0], got[1], op["kind"])
            elif kind == "pop":
                cache.lru.trim_to(n_sequences=len(cache.lru) - op["n"])
            elif kind == "reset":
                if ex is not None:
                    ex.close()
                ex = None
                halt = True
            elif kind == "stop":
                if ex is not None:
                    ex.close()
                logger.info("rank %d: stopped by rank 0 after %d steps "
                            "(%d token mismatches)", link.rank, steps,
                            mismatches)
                return steps
        last = {}
        if halt:
            continue
        e = executor()
        b = e.gen._batch
        toks = plan.get("tokens") or []
        if [int(u) for u in b.uids] != [u for u, _ in toks]:
            raise Desync(f"batch rows {list(b.uids)} here, "
                         f"{[u for u, _ in toks]} on rank 0")
        if toks:
            mine = b.t1.tolist()
            theirs = [t for _, t in toks]
            if mine != theirs:
                mismatches += sum(a != c for a, c in zip(mine, theirs))
                logger.warning("rank %d: %d token(s) differ from rank 0's; "
                               "rank 0's are used", link.rank,
                               sum(a != c for a, c in zip(mine, theirs)))
            b.t1 = mx.array(theirs, dtype=mx.int32)
        events = mark.around(e.step)
        steps += 1
        for ev in events:
            if isinstance(ev, Checkpoint):
                last[("checkpoint", ev.uid)] = (ev.tokens, ev.cache)
            elif isinstance(ev, Finished):
                last[("finished", ev.uid)] = (ev.tokens, ev.cache)


# ------------------------------------------------------------ bring-up

def init(link_kind: str) -> Link:
    """Join the ring (MLX_RANK and MLX_HOSTFILE, or the jaccl variables,
    already set) and wait for every rank before anything loads."""
    backend = {"ring": "ring", "jaccl": "jaccl"}[link_kind]
    group = mx.distributed.init(backend=backend, strict=True)
    link = Link(group)
    logger.info("rank %d of %d joined the %s ring", link.rank, link.size,
                backend)
    link.barrier()
    return link


def serve_follower(path: str, *, link_kind: str, working_set: int,
                   prompt_cache_size: int, completion_batch_size: int,
                   prefill_step_size: int,
                   executes_artifact_code: bool = False) -> int:
    """A rank >= 1 from start to stop: join, load its shard, follow."""
    from .host import ModelHost
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    link = init(link_kind)
    host = ModelHost(draft=False, executes_artifact_code=executes_artifact_code,
                     shard=lambda m: shard(m, link.group), vision=False,
                     load_wait_s=3600.0)
    host.load(path)
    if host.state != "ready":
        raise RuntimeError(f"rank {link.rank} could not load {path}: "
                           f"{host.error}")
    logger.info("rank %d: %s loaded, %.1f GiB active", link.rank, path,
                mx.get_active_memory() / GIB)
    return follow(host.model, host.tokenizer, host.model_key, link,
                  prompt_cache_size=prompt_cache_size,
                  completion_batch_size=completion_batch_size,
                  prefill_step_size=prefill_step_size,
                  working_set=working_set)
