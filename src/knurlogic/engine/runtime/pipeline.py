"""Pipeline split: one model served by N ranks, each holding a contiguous
run of layers (docs/design/server.md, "Cluster: pipeline split").

    rank N-1: embed, layers [0, a)       --hidden-->
    rank r:   layers [.., ..)            --hidden-->
    rank 0:   layers [.., L), norm, lm_head -> logits, samples, drafts

Rank 0 holds the LAST layers, so the logits are born on the rank that
samples and nothing is gathered; a follower's trunk returns zeros of the
logits' shape and its lm_head never runs (`Silent`). Written for
knurlogic; the layer slice keeps mlx-lm's PipelineMixin contract without
calling its uniform, all-gathering `pipeline()` (PROVENANCE.md). Hidden
states cross ranks with send / recv in program order; a receive uses the
RECEIVING rank's activation dtype, never a placeholder's. The MTP head
lives on rank 0 alone; followers are told drafts and verdicts through
fixed per-step broadcasts (BA, B0, B1, B2 in `Coord`).
"""

from __future__ import annotations

import contextlib
import logging
from typing import List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

logger = logging.getLogger(__name__)

_DTYPES = (mx.float32, mx.bfloat16, mx.float16)


# ------------------------------------------------------------------ split

def core_of(model):
    """The trunk core holding `layers` (qwen3_5 / glm5_next:
    model.language_model.model; qwen4_exp / deepseek_v4: model.model)."""
    return getattr(model, "language_model", model).model


def family_of(model) -> str:
    core = core_of(model)
    name = type(core).__name__
    if name == "Qwen3_5TextModel":
        return "qwen3_5"
    if name == "Glm5NextModel":
        return "glm5_next"
    if name == "Qwen4ExpModel":
        return "qwen4_exp"
    if name == "DeepseekV4Model":
        return "deepseek_v4"
    raise ValueError(f"a pipeline split knows qwen3_5, qwen3_5_moe, glm5_next, "
                     f"qwen4_exp and deepseek_v4; this trunk is {name}")


def own_dtype(layer) -> mx.Dtype:
    """The activation dtype of a stage that starts at `layer`: its first
    16-bit float parameter (norms and quantized scales carry it), else
    float32. Read off lazy parameters -- nothing is loaded for it."""
    for _, p in tree_flatten(layer.parameters()):
        if isinstance(p, mx.array) and p.dtype in (mx.bfloat16, mx.float16):
            return p.dtype
    return mx.float32


class _Wrap(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if "inner" not in self:
                raise
            return getattr(self["inner"], name)


def unwrap(layer):
    """The layer a stage end wraps (`Recv` / `Send`, possibly both on a
    one-layer stage), or `layer` itself. Attribute reads already see through
    a wrapper; `type(...)` and `isinstance(...)` do not, so code that asks
    what class a layer is asks this first."""
    while isinstance(layer, _Wrap):
        layer = layer["inner"]
    return layer


class Recv(_Wrap):
    """The first layer of a stage that is not the first: its input is the
    previous stage's output, received in this rank's own dtype."""

    def __init__(self, inner, src: int, group, dtype):
        super().__init__(inner)
        self._src, self._group, self._dtype = src, group, dtype

    def __call__(self, x, *a, **kw):
        x = mx.distributed.recv(x.shape, self._dtype, self._src,
                                group=self._group)
        mx.eval(x)
        return self.inner(x, *a, **kw)


class Send(_Wrap):
    """The last layer of a stage that is not the last: its output goes to
    the next stage in that stage's dtype, sent before the forward returns
    -- or, inside `overlapped` (a prompt's prefill chunks), sent while the
    next chunk computes: at most two sends are in flight, and all of them
    have completed before the prefill's last forward, so nothing is in
    flight when the next collective runs."""

    def __init__(self, inner, dst: int, group, dtype):
        super().__init__(inner)
        self._dst, self._group, self._dtype = dst, group, dtype
        self.overlap = False
        self._pending: list = []
        #: sends made asynchronously (the tests read it)
        self.overlapped = 0

    def __call__(self, x, *a, **kw):
        y = self.inner(x, *a, **kw)
        s = mx.distributed.send(y.astype(self._dtype), self._dst,
                                group=self._group)
        if not self.overlap:
            mx.eval(s)
            return y
        # this chunk's layers and send go to the device now; the previous
        # chunk's send is waited for only after, while this one computes
        mx.async_eval(s)
        self._pending.append(s)
        self.overlapped += 1
        if len(self._pending) > 1:
            mx.eval(self._pending[:-1])
            del self._pending[:-1]
        return y

    def flush(self) -> None:
        """Wait for every send in flight."""
        if self._pending:
            mx.eval(self._pending)
            self._pending.clear()


def sends_of(model) -> List[Send]:
    """This stage's Send (none on rank 0), seen through its wrappers."""
    out = []
    for layer in core_of(model).layers:
        while isinstance(layer, _Wrap):
            if isinstance(layer, Send):
                out.append(layer)
            layer = layer["inner"]
    return out


def overlap_on() -> bool:
    """KNURLOGIC_PIPELINE_OVERLAP=off sends each prefill chunk before the
    next one computes (the A/B for the overlap's gain)."""
    import os
    v = os.environ.get("KNURLOGIC_PIPELINE_OVERLAP", "").strip().lower()
    return v not in ("off", "0", "false", "no")


@contextlib.contextmanager
def overlapped(model):
    """Around a prompt's prefill chunks (batch_loop.admit's prefill_ctx):
    each chunk's hidden state is sent while the next chunk computes (the
    exo fork queued its prefill sends for the same reason), and every send
    has completed on the way out -- before the prefill's last forward and
    any collective after it. Nothing but sends and receives happens
    between the chunks, so the point-to-point order is the program order
    on every rank, as without it."""
    sends = sends_of(model) if overlap_on() else []
    for sd in sends:
        sd.overlap = True
    try:
        yield
    finally:
        for sd in sends:
            sd.overlap = False
        for sd in sends:
            sd.flush()


def bounds_of(counts: Sequence[int]) -> List[Tuple[int, int]]:
    """Layer counts per rank -> (start, end) per rank; rank N-1 first."""
    n = len(counts)
    out = [None] * n
    at = 0
    for r in range(n - 1, -1, -1):
        out[r] = (at, at + int(counts[r]))
        at += int(counts[r])
    return out


def _dtype_code(dt) -> int:
    return _DTYPES.index(dt)


def restage(model, keep: list, start: int, end: int) -> None:
    """Make `model` a stage holding `keep` = its layers [start, end), stage
    ends already wrapped: the layer list, and the per-family indices the
    trunk froze from the whole list. Every layer is read through its
    wrapper (attribute reads see through `Recv` / `Send`)."""
    fam = family_of(model)
    core = core_of(model)
    core.layers = keep
    if fam == "qwen3_5":
        # PipelineMixin's contract, uniform split and all_gather not used
        core.start_idx, core.end_idx = 0, None
        core.pipeline_rank, core.pipeline_size = 0, 1
        core.ssm_idx = next((i for i, lyr in enumerate(keep) if lyr.is_linear),
                            None)
        core.fa_idx = next((i for i, lyr in enumerate(keep)
                            if not lyr.is_linear), None)
    elif fam == "glm5_next":
        # frozen from the full list at __init__ (fork commit f3ab3a83)
        core.ssm_idx = next((i for i, lyr in enumerate(keep)
                             if getattr(lyr, "is_linear", False)), 0)
        core.fa_idx = next((i for i, lyr in enumerate(keep)
                            if not getattr(lyr, "is_linear", True)), 0)
    elif fam == "qwen4_exp":
        # full-model indices in ple_layers and a full-length make_cache
        # (fork commit dd946407)
        core.ple_layers = [i - start for i in core.ple_layers
                           if start <= i < end]
        whole = model.make_cache

        def make_cache():
            return whole()[start:end]
        model.make_cache = make_cache
    # deepseek_v4: nothing else. Each block froze its own compress ratio,
    # hash routing and RoPE from its GLOBAL layer_id at __init__, every
    # layer's cache is the same DeepseekV4Cache (make_cache is one per kept
    # layer), and the stream between stages is the [B, S, hc, D]
    # hyper-connection state, which every stage builds from its own
    # embedding (so a Recv's placeholder has the right shape). The block
    # takes the token ids as a third argument (hash routing); the wrappers
    # pass it through.


def split(model, group, bounds: Sequence[Tuple[int, int]]) -> dict:
    """Keep this rank's layers of `model` (bounds[rank]) and wire its stage
    ends, in place, before the weights are read. -> what was done."""
    rank, n = group.rank(), group.size()
    fam = family_of(model)
    core = core_of(model)
    full = list(core.layers)
    if len(bounds) != n:
        raise ValueError(f"{len(bounds)} layer runs for {n} ranks")
    if sorted(bounds, key=lambda b: b[0])[0][0] != 0 or \
            max(b[1] for b in bounds) != len(full):
        raise ValueError(f"layer runs {list(bounds)} do not cover "
                         f"{len(full)} layers")
    start, end = bounds[rank]
    if not 0 <= start < end <= len(full):
        raise ValueError(f"rank {rank}: layers [{start}, {end}) of "
                         f"{len(full)}")
    keep = full[start:end]

    # every rank's stage dtype, so a send is cast to what its receiver
    # expects (one all_gather, on the CPU, while every rank is loading)
    mine = _dtype_code(own_dtype(keep[0]))
    codes = mx.distributed.all_gather(mx.array([mine], dtype=mx.int32),
                                      group=group, stream=mx.cpu).tolist()
    dts = [_DTYPES[c] for c in codes]
    if rank < n - 1:                       # not the first stage: receive
        keep[0] = Recv(keep[0], rank + 1, group, dts[rank])
    if rank > 0:                           # not the last stage: send
        keep[-1] = Send(keep[-1], rank - 1, group, dts[rank - 1])

    restage(model, keep, start, end)
    logger.info("pipeline rank %d of %d (%s): layers [%d, %d) of %d, "
                "stage dtype %s", rank, n, fam, start, end, len(full),
                dts[rank])
    return {"family": fam, "start": start, "end": end, "layers": len(full),
            "dtypes": [str(d) for d in dts]}


# --------------------------------------------------------------- followers

class Silent:
    """A follower's trunk as the batch engine calls it: the layers run (the
    send happens inside), and the "logits" are zeros of their shape. The
    follower's samples are overwritten by rank 0's tokens, so the lm_head is
    never worth computing, and zeros are finite (the NaN guard never fails a
    row on one rank only)."""

    def __init__(self, trunk):
        self.__dict__["_trunk"] = trunk

    def __call__(self, inputs, cache=None, **kw):
        out = self._trunk(inputs, cache=cache, **kw)
        return mx.zeros(out.shape, dtype=mx.float32)

    def __getattr__(self, name):
        return getattr(self._trunk, name)


def silence(gen) -> None:
    """Make a follower's batch engine call its trunk through `Silent`, and
    overlap its prefill sends with the next chunk (`overlapped`)."""
    s = Silent(gen._trunk)
    gen._trunk = s
    gen._batch.model = s
    model = gen.model
    gen._prefill_ctx = lambda: overlapped(model)


# ------------------------------------------------------------ coordinator

class Coord:
    """The per-step control broadcasts of a pipeline (module docstring):
    rank 0's values on every rank, one all_gather on the CPU each.

    `head`: rank 0 drafts (it bound a head; `tensor.agree_head` told every
    rank). The same on every rank, so BA / B1 are made on every rank or on
    none -- a follower holds no head of its own to ask."""

    def __init__(self, group, head: bool = False):
        self.group = group
        self.leader = group.rank() == 0
        self.head = bool(head)
        self.calls = {"b0": 0, "b1": 0, "b2": 0, "ba": 0, "img": 0}
        #: the last b0 found the ranks holding different row counts (one
        #: rank's admission failed): every rank skips that call's decode
        #: step, whose collectives would not line up, and rank 0's next
        #: plan removes the row
        self.diverged = False

    def _bcast(self, vals: List[int]) -> List[int]:
        n = len(vals)
        v = mx.array(vals if self.leader else [0] * n, dtype=mx.int32)
        out = mx.distributed.all_gather(v, group=self.group, stream=mx.cpu)
        return out[:n].tolist()           # rank 0's block is the first

    def b0(self, t1: Optional[mx.array]) -> Optional[mx.array]:
        """After every admission attempt (and a failed decode step): every
        row's next token, rank 0's. Made even when the admission failed on
        one rank, so the ranks' collective counts stay equal; the row counts
        may then differ (rank 0 dropped a row the follower still holds, or
        the reverse), and a rank whose count is not rank 0's keeps its own
        t1 -- the follower's rows and tokens are set from the next plan --
        and `diverged` is set."""
        self.calls["b0"] += 1
        mine = [] if t1 is None else [int(t) for t in t1.tolist()]
        ns = mx.distributed.all_gather(
            mx.array([len(mine)], dtype=mx.int32), group=self.group,
            stream=mx.cpu).tolist()
        m = max(ns)
        self.diverged = len(set(ns)) > 1
        if m == 0:
            return t1
        v = mx.array((mine if self.leader else [0] * len(mine))
                     + [0] * (m - len(mine)), dtype=mx.int32)
        got = mx.distributed.all_gather(v, group=self.group,
                                        stream=mx.cpu).tolist()
        if ns[0] != len(mine):
            return t1
        return mx.array(got[:ns[0]], dtype=mx.int32)

    def ba(self, ok: bool, hit: int, drafts: bool) -> Tuple[int, bool]:
        """Before an admission's prefill, on a drafting pipeline: -> rank
        0's (hit, drafts). Every rank says whether it got this far; if any
        did not, every rank's admission fails here (RuntimeError), before
        a prefill whose sends one rank would never make."""
        self.calls["ba"] += 1
        got = mx.distributed.all_gather(
            mx.array([int(bool(ok)), int(hit), int(bool(drafts))],
                     dtype=mx.int64), group=self.group,
            stream=mx.cpu).tolist()
        if not all(got[0::3]):
            bad = [r for r, o in enumerate(got[0::3]) if not o]
            raise RuntimeError(f"rank(s) {bad} failed this admission before "
                               f"its prefill")
        return int(got[1]), bool(got[2])

    def images(self, key_slice: list, features, refs):
        """An admission whose uncached span holds images, on every rank
        (tensor and pipeline): rank 0's encoded rows of every image in
        `key_slice`, in order of first appearance, reach every rank. -> a
        FeatureLookup over them (rank 0: its own `features`). Rank 0 alone
        runs the tower; a follower's family embeds from these rows exactly
        as rank 0's would from its store (float32 on the wire: exact for
        bf16 rows). `refs`: every rank's RefLookup (a follower's from the
        admit op), which says each image's row count. Raises VisionError
        on every rank when rank 0 could not read its rows."""
        from knurlogic.engine.vision import EncodedImage, VisionError
        from knurlogic.engine.vision.key import images_in
        self.calls["img"] += 1
        imgs = list(dict.fromkeys(images_in(key_slice)))
        ns = [int(refs(sha, ph).n_tokens) for sha, ph in imgs]
        rows, dim, ok = None, 0, 0
        if self.leader:
            try:
                parts = [features(sha, ph).feats[:n]
                         for (sha, ph), n in zip(imgs, ns)]
                rows = mx.concatenate(parts, axis=0).astype(mx.float32)
                dim, ok = int(rows.shape[1]), 1
            except Exception:  # rank 0 must still tell the other ranks it failed (logged)
                logger.exception("rank 0 could not read an image's rows")
        ok, dim = self._bcast([ok, dim])
        if not ok:
            raise VisionError("rank 0 could not read this request's image "
                              "rows; the admission fails on every rank")
        if not self.leader:
            rows = mx.zeros((sum(ns), dim), dtype=mx.float32)
        got = mx.distributed.all_sum(rows, group=self.group, stream=mx.cpu)
        mx.eval(got)
        if self.leader:
            return features
        table, at = {}, 0
        for (sha, ph), n in zip(imgs, ns):
            table[(sha, ph)] = EncodedImage(ref=refs(sha, ph),
                                            feats=got[at:at + n])
            at += n

        def lookup(sha: str, ph: str):
            return table[(sha, ph)]
        return lookup

    def b1(self, drafting: bool, d2: Optional[mx.array], B: int):
        """-> (drafting, d2 [B] int32 or None)."""
        self.calls["b1"] += 1
        vals = [int(bool(drafting))] + (
            [int(t) for t in d2.tolist()] if (self.leader and d2 is not None)
            else [0] * B)
        got = self._bcast(vals)
        if not got[0]:
            return False, None
        return True, mx.array(got[1:], dtype=mx.int32)

    def b2(self, ok: List[bool], t2: mx.array, B: int):
        """-> (ok flags, t2 [B] int32)."""
        self.calls["b2"] += 1
        vals = ([int(bool(o)) for o in ok] + [int(t) for t in t2.tolist()]
                if self.leader else [0] * (2 * B))
        got = self._bcast(vals)
        return [bool(o) for o in got[:B]], mx.array(got[B:], dtype=mx.int32)


def coordinate(gen, group, drafting: Optional[bool] = None) -> Coord:
    """Install a Coord on a batch engine (every rank of a pipeline).
    `drafting`: rank 0 drafts; default, whether this engine holds a head
    (rank 0's own answer)."""
    if drafting is None:
        drafting = gen._head is not None
    c = Coord(group, head=drafting)
    gen._coord = c
    gen._batch.coord = c
    return c


# ---------------------------------------------------------------- bring-up

def agree(group, *, layer_bytes: Sequence[int], other_bytes: int,
          working_set: int, bandwidth_gbs: Optional[float] = None,
          counts: Optional[Sequence[int]] = None,
          leader_bytes: int = 0) -> dict:
    """Every rank's working set and memory bandwidth, gathered, and the
    layer split computed from them the same way on every rank
    (tuning/resolve.pipeline_shares: same inputs, same split; rank 0's
    `leader_bytes` -- the head and the tower -- counted on rank 0 alone). `counts`
    (layers per rank, rank order) overrides the arithmetic. Raises when the
    ranks read different artifacts."""
    import json
    import zlib

    from knurlogic.tuning import resolve as R
    n, rank = group.size(), group.rank()
    sig = zlib.crc32(json.dumps([list(map(int, layer_bytes)),
                                 int(other_bytes), int(leader_bytes)]
                                ).encode()) & 0x7FFFFFFF
    row = [int(working_set), int(round((bandwidth_gbs or 0) * 1000)),
           len(layer_bytes), sig]
    got = mx.distributed.all_gather(mx.array(row, dtype=mx.int64),
                                    group=group, stream=mx.cpu).tolist()
    rows = [got[4 * r:4 * r + 4] for r in range(n)]
    if len({tuple(r[2:]) for r in rows}) != 1:
        raise RuntimeError(
            "the ranks read different artifacts (layer count, layer-bytes "
            f"checksum per rank: {[tuple(r[2:]) for r in rows]})")
    ranks = [{"name": f"rank{r}", "working_set_bytes": rows[r][0],
              "memory_bandwidth_gbs": (rows[r][1] / 1000) or None}
             for r in range(n)]
    if counts:
        counts = [int(c) for c in counts]
        if len(counts) != n or min(counts) < 1 or \
                sum(counts) != len(layer_bytes):
            raise ValueError(f"--layers {counts}: {n} counts of at least 1 "
                             f"summing to {len(layer_bytes)}")
        b = bounds_of(counts)
        out = {"layers": counts, "bounds": b,
               "bytes": [sum(layer_bytes[x:y]) for x, y in b],
               "reason": f"{len(layer_bytes)} layers as given (--layers "
                         f"{','.join(map(str, counts))}); rank 0 holds the "
                         f"last layers and samples"}
    else:
        out = R.pipeline_shares(list(layer_bytes), ranks, int(other_bytes),
                                int(leader_bytes))
    out["ranks"] = ranks
    out["rank"] = rank
    return out
