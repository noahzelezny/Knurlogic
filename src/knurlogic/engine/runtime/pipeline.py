"""Pipeline split: one model served by N ranks, each holding a contiguous
run of layers (docs/SERVER.md, "Cluster: pipeline split").

    rank N-1: embed, layers [0, a)       --hidden-->
    rank r:   layers [.., ..)            --hidden-->
    rank 0:   layers [.., L), norm, lm_head -> logits, samples, drafts

Rank 0 holds the LAST layers, so the logits are born on the rank that
samples and nothing is gathered: mlx-lm's pipeline all_gathers the final
hidden state to every rank so every rank can compute logits; here the
other ranks never need them (rank 0's step plan carries every token --
engine/runtime/plan.py), so a follower's trunk returns zeros of the logits'
shape and its lm_head never runs (`Silent`).

Written for knurlogic, no exo code. The layer slice keeps the contract of
mlx-lm's PipelineMixin (start_idx / end_idx / pipeline_layers, which the
vendored qwen3_5 reads) without calling its `pipeline()`, whose split is
uniform and whose forward all_gathers; the per-family index fixups follow
Noah's own exo fork commits (engine/runtime/PROVENANCE.md).

Hidden states cross ranks with send / recv, each evaluated where it is
built (a follower's step finishes its send inside the forward; rank 0's
receive is waited for inside the forward), so the order of point-to-point
messages and of the CPU collectives (the plan exchange, B0-B2 below) is
the program order on every rank. A receive is made in the RECEIVING rank's
own activation dtype and a send is cast to its receiver's (the dtypes are
agreed when the model is split), never the dtype of a placeholder: an
unloaded embedding is float32, and a float32 receive of bf16 bytes ran a
whole shard in float32 on the 397B (Noah's 574a7bd7).

MTP on pipeline (the head lives on rank 0, which holds the last layers and
so the true final hidden state). A follower runs a head too, on its own
stage's hidden state: its drafts are never used, but its head cache moves
exactly as rank 0's does, so every rank's bookkeeping (prompt-cache entries,
offsets, the replay) is the same code. Per step, fixed and never dependent
on a verdict (`Coord`):

    B1  [drafting, d2 per row]   before the verify forward: whether this
                                 step drafts (rank 0's timing decides) and
                                 the drafted tokens the first stage embeds
    B2  [ok per row, t2 per row] after the verify forward, only when B1
                                 said drafting: the verdicts that drive
                                 every rank's rollback, and the tokens that
                                 commit

and, on the step that admits a row, B0 [t1 per row]: the admitted row's
first token is sampled inside the admission, after the step's plan was
sent, and a follower's own sample is noise.
"""

from __future__ import annotations

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
    model.language_model.model; qwen4_exp: model.model)."""
    return getattr(getattr(model, "language_model", model), "model")


def family_of(model) -> str:
    core = core_of(model)
    name = type(core).__name__
    if name == "Qwen3_5TextModel":
        return "qwen3_5"
    if name == "Glm5NextModel":
        return "glm5_next"
    if name == "Qwen4ExpModel":
        return "qwen4_exp"
    raise ValueError(f"a pipeline split knows qwen3_5, qwen3_5_moe, glm5_next "
                     f"and qwen4_exp; this trunk is {name}")


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
    the next stage in that stage's dtype, sent before the forward returns."""

    def __init__(self, inner, dst: int, group, dtype):
        super().__init__(inner)
        self._dst, self._group, self._dtype = dst, group, dtype

    def __call__(self, x, *a, **kw):
        y = self.inner(x, *a, **kw)
        s = mx.distributed.send(y.astype(self._dtype), self._dst,
                                group=self._group)
        mx.eval(s)
        return y


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

    core.layers = keep
    if fam == "qwen3_5":
        # PipelineMixin's contract, uniform split and all_gather not used
        core.start_idx, core.end_idx = 0, None
        core.pipeline_rank, core.pipeline_size = 0, 1
        core.ssm_idx = next((i for i, l in enumerate(keep) if l.is_linear),
                            None)
        core.fa_idx = next((i for i, l in enumerate(keep)
                            if not l.is_linear), None)
    elif fam == "glm5_next":
        # frozen from the full list at __init__ (Noah's f3ab3a83)
        core.ssm_idx = next((i for i, l in enumerate(keep)
                             if getattr(l, "is_linear", False)), 0)
        core.fa_idx = next((i for i, l in enumerate(keep)
                            if not getattr(l, "is_linear", True)), 0)
    elif fam == "qwen4_exp":
        # full-model indices in ple_layers and a full-length make_cache
        # (Noah's dd946407)
        core.ple_layers = [i - start for i in core.ple_layers
                           if start <= i < end]
        whole = model.make_cache

        def make_cache():
            return whole()[start:end]
        model.make_cache = make_cache
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
    """Make a follower's batch engine call its trunk through `Silent`."""
    s = Silent(gen._trunk)
    gen._trunk = s
    gen._batch.model = s


# ------------------------------------------------------------ coordinator

class Coord:
    """The per-step control broadcasts of a pipeline (module docstring):
    rank 0's values on every rank, one all_gather on the CPU each."""

    def __init__(self, group):
        self.group = group
        self.leader = group.rank() == 0
        self.calls = {"b0": 0, "b1": 0, "b2": 0}

    def _bcast(self, vals: List[int]) -> List[int]:
        n = len(vals)
        v = mx.array(vals if self.leader else [0] * n, dtype=mx.int32)
        out = mx.distributed.all_gather(v, group=self.group, stream=mx.cpu)
        return out[:n].tolist()           # rank 0's block is the first

    def b0(self, t1: mx.array) -> mx.array:
        """After an admission: every row's next token, rank 0's."""
        self.calls["b0"] += 1
        return mx.array(self._bcast([int(t) for t in t1.tolist()]),
                        dtype=mx.int32)

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


def coordinate(gen, group) -> Coord:
    """Install a Coord on a batch engine (every rank of a pipeline)."""
    c = Coord(group)
    gen._coord = c
    gen._batch.coord = c
    return c


# ---------------------------------------------------------------- bring-up

def agree(group, *, layer_bytes: Sequence[int], other_bytes: int,
          working_set: int, bandwidth_gbs: Optional[float] = None,
          counts: Optional[Sequence[int]] = None) -> dict:
    """Every rank's working set and memory bandwidth, gathered, and the
    layer split computed from them the same way on every rank
    (tuning/resolve.pipeline_shares: same inputs, same split). `counts`
    (layers per rank, rank order) overrides the arithmetic. Raises when the
    ranks read different artifacts."""
    import json
    import zlib

    from knurlogic.tuning import resolve as R
    n, rank = group.size(), group.rank()
    sig = zlib.crc32(json.dumps([list(map(int, layer_bytes)),
                                 int(other_bytes)]).encode()) & 0x7FFFFFFF
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
        out = R.pipeline_shares(list(layer_bytes), ranks, int(other_bytes))
    out["ranks"] = ranks
    out["rank"] = rank
    return out
