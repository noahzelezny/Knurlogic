"""Tensor split: one model served by N ranks, every layer's weights split
N ways (docs/design/server.md, "Cluster: tensor split").

    rank 0:   HTTP -> Scheduler -> TensorExecutor --plan--> ranks 1..N-1
    rank r:   follow(): apply the plan, run the same step, never sample

The layer split follows mlx-lm's Qwen3_5 `Model.shard` (MIT;
PROVENANCE.md) with one change: a VQ codebook is REPLICATED, never sliced.
mlx's default predicates split every parameter of a sharded layer; applied
to a VQSwitchLinear they cut the [K, d] codebook or its d axis, and the
latter decodes against half a codebook and emits fluent garbage.

The step protocol is engine/split/plan.py. Everything that moves between
ranks goes through `Link.exchange`, on the scheduler's thread.
"""

from __future__ import annotations

import logging
import re

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map, tree_map_with_path

from .tensor_rules import (
    PARTS,
    RULES,
    module_refusal,
    owner,
    predicate,
    segment_points,
)

logger = logging.getLogger(__name__)




# ------------------------------------------------------------------ split



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
    module.update(split_params(module.parameters(), pred, rank, n, segments))


def _apply(layer, path: str, rule, rank: int, n: int) -> None:
    """Cut the array or module at `path` under `layer` by `rule` (a path
    the layer lacks -- the other attention, the other mlp -- is skipped)."""
    *up, leaf = path.split(".")
    parent = layer
    for p in up:
        parent = getattr(parent, p, None)
        if parent is None:
            return
    obj = getattr(parent, leaf, None)
    if obj is None:
        return
    if rule.kind == PARTS:
        _deal(obj, path, rank, n)
        return
    pred = predicate(rule.kind)
    seg = segment_points(rule, getattr(parent, "key_dim", 0))
    if isinstance(obj, mx.array):
        setattr(parent, leaf, split_params({leaf: obj}, pred, rank, n,
                                           seg)[leaf])
        return
    done = getattr(obj, "_vq_sharded", None)
    if done is not None:
        # the bundled runtime built this rank's rows itself (load_config)
        if tuple(done) != (rank, n):
            raise ValueError(f"{path} was split as rank {done[0]} of "
                             f"{done[1]}, not {rank} of {n}")
        return
    why = module_refusal(path, rule, set(obj.parameters()))
    if why:
        raise ValueError(why)
    kv = _name(parent, _KV_HEADS)
    h = getattr(parent, kv) if kv else n
    if rule.repeat_kv and n > h:
        def rep(p):
            s = p.shape
            p = p.reshape(h, s[0] // h, *s[1:])
            return mx.repeat(p, n // h, axis=0).reshape(-1, *s[1:])
        obj.update(tree_map(rep, obj.parameters()))
    _split_inplace(obj, pred, rank, n, seg)


class _Elsewhere(nn.Module):
    """A part of a PARTS table another rank holds: its rows read as zeros
    here, and the table's Reduce sums the owner's in."""

    def __init__(self, dim: int):
        super().__init__()
        self._dim = dim

    def __call__(self, ids):
        return mx.zeros((*ids.shape, self._dim), dtype=mx.float32)


def _deal(table, path: str, rank: int, n: int) -> None:
    """Keep this rank's parts of `table` (tensor_rules.owner) and put an
    _Elsewhere in place of every other: their weights are never read."""
    parts = sorted((k for k in table if re.fullmatch(r"shard_\d+", k)),
                   key=lambda k: int(k[6:]))
    if len(parts) % n:
        raise ValueError(f"{path}: {len(parts)} parts do not divide by {n}")
    for k in parts:
        if owner(int(k[6:]), len(parts), n) != rank:
            setattr(table, k, _Elsewhere(table.dim))


#: the per-instance sizes each family's modules reshape by, divided by n on
#: every rank: qwen3_5's names, then qwen4_exp's
_LINEAR_SIZES = ("num_k_heads", "num_v_heads", "n_k", "n_v", "key_dim",
                 "value_dim", "conv_dim")


_HEADS = ("num_attention_heads", "n_heads")


_KV_HEADS = ("num_key_value_heads", "n_kv_heads")


def _name(module, names) -> str | None:
    """The first of `names` that `module` has."""
    return next((k for k in names if hasattr(module, k)), None)


class Reduce(nn.Module):
    """`inner`'s partial output summed across the ranks -- in float32.

    Each rank's partial is already rounded to the activation dtype (bf16)
    by its matmul, so the split cannot be bit-identical to the whole layer
    (which rounds once, after a reduction in another order). Summing in
    float32 rounds once after the sum; with two ranks that equals a bf16
    add, with more it saves the intermediate roundings. Measured on an M4 Max (128 GB)
    (35B-A3B VQ, two ranks over the ring on 127.0.0.1): the float32 sum
    decodes at 29.6 tok/s where mlx's in-dtype bf16 all_sum made 13.6 --
    the ring backend reduces float32 far faster than bf16."""

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


def load_config(path, group) -> dict | None:
    """The config a tensor rank loads with: a SKIPZERO build whose runtime
    splits its own gate/up rows (tensor_rules.skipzero_split) is told this
    rank, so it builds only its rows; None otherwise."""
    import json
    from pathlib import Path

    from .tensor_rules import skipzero_split
    cfg = json.loads((Path(path) / "config.json").read_text())
    if not (cfg.get("vq_skipzero") and skipzero_split(path)):
        return None
    return {"vq_skipzero": {**cfg["vq_skipzero"], "shard": {
        "rank": group.rank(), "n": group.size()}}}


def shard(model, group) -> None:
    """Split a qwen3_5 / qwen3_5_moe / qwen4_exp / deepseek_v4 model across
    `group` in place.

    Every cut is a RULES entry (tensor_rules); tuning/resolve checks the
    same rules against the headers before a rank starts. qwen4_exp's
    hyper-connections, QSA indexer and the rest of its PLE run whole on
    every rank, on the whole hidden state the Reduces leave."""
    n, rank = group.size(), group.rank()
    for layer in model.layers:
        for path, rule in RULES.items():
            _apply(layer, path, rule, rank, n)
        if "attn" in layer:
            # deepseek_v4: a rank holds n_heads / n heads in o_groups / n
            # whole groups; its compressor, indexer and the one shared kv
            # head are whole, and so are the hyper-connections around them
            at = layer.attn
            at.n_heads //= n
            at.n_groups //= n
            at._sink_cache = None               # cast lazily from the cut
            layer.attn = Reduce(at, group)
            layer.ffn = Reduce(layer.ffn, group)
            continue
        if "linear_attn" in layer:
            la = layer.linear_attn
            la.conv1d.groups //= n
            for k in _LINEAR_SIZES:
                if hasattr(la, k):
                    setattr(la, k, getattr(la, k) // n)
            la.sharding_group = None            # Reduce sums it, in fp32
            layer.linear_attn = Reduce(la, group)
        else:
            at = layer.self_attn
            h, kv = _name(at, _HEADS), _name(at, _KV_HEADS)
            setattr(at, h, getattr(at, h) // n)
            setattr(at, kv, max(1, getattr(at, kv) // n))
            layer.self_attn = Reduce(at, group)
        ple = getattr(layer, "ple", None)
        if ple is not None:
            e = ple.ple_embedding
            e.ngram_embedding = Reduce(e.ngram_embedding, group)
        mlp = layer.mlp
        if hasattr(mlp, "switch_mlp"):
            mlp.sharding_group = None
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
