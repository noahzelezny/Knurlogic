"""The tensor split's rules: which arrays of a qwen3_5 / qwen4_exp layer
are cut, on which axis, in which segments -- one table that engine/runtime/tensor.py
`shard` applies to loaded arrays and tuning/resolve checks against the
safetensors headers before anything loads, so the refusal and the loader
cannot disagree. No mlx here: the picker asks this of ~80 models.

A rule also names the parameter layouts it knows. A module holding any
other parameter is UNVERIFIED, not refused: the header arithmetic still
applies to it, and a launch runs that one module whole and split
(engine/runtime/viability.py) before the ring starts. VQ SKIPZERO is the
case that made this: `sz_codes` packs the live rows of every expert into
one [NLIVE, W] list, which a row cut does not respect.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import NamedTuple

A2S, S2A, ROWS = "all-to-sharded", "sharded-to-all", "rows"
#: a table stored in parts (`shard_0` .. `shard_{P-1}`): each rank keeps a
#: contiguous P/n of the parts whole and reads the others' rows as zeros;
#: the table's output is summed across the ranks (exact: one rank's rows
#: are non-zero). No axis of a part is cut, so any layout in it holds.
PARTS = "parts"

#: the parameter layouts the split knows: bf16 (weight, bias), mlx affine
#: quant (weight, scales, biases), VQ (codes, vq_scales, codebook -- the
#: codebook is replicated)
LAYOUTS = frozenset({"weight", "bias", "scales", "biases",
                     "codes", "vq_scales", "codebook", "row_table"})
#: kept whole on every rank: the VQ codebook (a lookup table the codes
#: index) and a SKIPZERO row table (rows stay whole under an input cut)
_WHOLE = ("codebook", "row_table")

#: a SKIPZERO module on disk -> the loaded module's names (the bundled
#: runtime's skipzero_weights); the mask and shape become its row_table
_SZ = {"sz_codes": "codes", "sz_scales": "vq_scales",
       "sz_rowmask": "row_table", "sz_shape": "row_table"}
_SZ_SPLIT = re.compile(r"^SKIPZERO_SHARD\s*=\s*1\b", re.M)


class Rule(NamedTuple):
    #: A2S: output axis; S2A: input axis; ROWS: axis 0; PARTS: whole parts
    kind: str
    #: "qkv": cut at [key_dim, 2 * key_dim] first (q, k, v each split);
    #: an int k: k equal parts
    segments: str | int = 1
    #: fewer KV heads than ranks: each rank gets a copy of its head
    repeat_kv: bool = False


#: module path under a decoder layer -> its rule. A path not here (norms,
#: the router `mlp.gate`, `shared_expert_gate`) is replicated whole.
RULES = {
    "linear_attn.conv1d": Rule(ROWS, "qkv"),
    "linear_attn.in_proj_qkv": Rule(A2S, "qkv"),
    "linear_attn.in_proj_z": Rule(A2S),
    "linear_attn.in_proj_b": Rule(A2S),
    "linear_attn.in_proj_a": Rule(A2S),
    "linear_attn.dt_bias": Rule(ROWS),          # bare arrays, not modules
    "linear_attn.A_log": Rule(ROWS),
    "linear_attn.out_proj": Rule(S2A),
    "self_attn.q_proj": Rule(A2S),
    "self_attn.k_proj": Rule(A2S, repeat_kv=True),
    "self_attn.v_proj": Rule(A2S, repeat_kv=True),
    "self_attn.o_proj": Rule(S2A),
    "mlp.shared_expert.gate_proj": Rule(A2S),
    "mlp.shared_expert.up_proj": Rule(A2S),
    "mlp.shared_expert.down_proj": Rule(S2A),
    "mlp.switch_mlp.gate_proj": Rule(A2S),
    "mlp.switch_mlp.up_proj": Rule(A2S),
    "mlp.switch_mlp.down_proj": Rule(S2A),
    "mlp.gate_proj": Rule(A2S),
    "mlp.up_proj": Rule(A2S),
    "mlp.down_proj": Rule(S2A),
    # qwen4_exp's n-gram table (layer 1 of Flash-Next, 128 parts, 9-54 GiB
    # by build): too big to hold whole on every rank, and its quantization
    # groups / VQ rows run along the 160-wide axis, which no cut respects
    "ple.ple_embedding.ngram_embedding": Rule(PARTS),
}

#: Hugging Face spellings the qwen3_5_moe sanitize renames before shard
#: sees them (engine/families/qwen/architecture/qwen3_5_moe.py): the fused
#: [E, 2I, H] gate_up_proj is cut in half on its output axis into gate and
#: up, so its rule is gate_proj's in two segments.
_HF = {
    "mlp.experts.gate_up_proj": ("mlp.switch_mlp.gate_proj", "weight", 2),
    "mlp.experts.down_proj": ("mlp.switch_mlp.down_proj", "weight", None),
}
_HF_EXPERT = re.compile(r"mlp\.experts\.\d+\.(gate|up|down)_proj\.(\w+)$")
#: an array of one part of a PARTS table: (table path, part, leaf)
_PART = re.compile(r"^(.+)\.shard_(\d+)\.(\w+)$")
#: a decoder layer of the trunk (not mtp.layers, not the tower's blocks)
_LAYER = re.compile(r"^(?:model\.|language_model\.)*layers\.(\d+)\.(.+)$")


def owner(part: int, parts: int, n: int) -> int:
    """The rank holding part `part` of a PARTS table of `parts`."""
    return part * n // parts


def predicate(kind: str) -> Callable:
    """The split axis for a parameter of a layer split `kind`
    ("all-to-sharded": output axis; "sharded-to-all": input axis; "rows":
    axis 0), or None to keep it whole. A path ending in `codebook` is
    ALWAYS None. `w` needs only `.ndim`."""
    if kind not in (A2S, S2A, ROWS):
        raise ValueError(kind)

    def pred(path: str, w):
        if path.endswith(_WHOLE):
            return None
        if kind == ROWS:
            return 0
        if kind == A2S:
            return -1 if path.endswith("bias") else max(w.ndim - 2, 0)
        return None if path.endswith("bias") else -1
    return pred


def segment_points(rule: Rule, key_dim: int):
    """`segments` for split_params: an int, or the cut points."""
    return [key_dim, 2 * key_dim] if rule.segments == "qkv" else rule.segments


def unknown(leaves) -> list:
    """The parameters a rule has no layout for, sorted."""
    return sorted(k for k in leaves if k and k not in LAYOUTS)


def module_refusal(path: str, rule: Rule, leaves) -> str | None:
    """Why the module at `path` holding these parameters cannot be cut by
    `rule` whatever its numbers, or None."""
    if rule.kind == S2A and "bias" in leaves:
        # a split input axis gives each rank a partial sum; a bias added on
        # every rank would be summed N times
        return f"{path} has a bias: a sharded-to-all split of it is not built"
    return None


def locate(name: str):
    """A safetensors name -> (layer, rule path, leaf, extra segments) when
    it is under a rule, else None (replicated). `leaf` is None for a bare
    array (dt_bias, A_log); "shard_<i>.<leaf>" under a PARTS table."""
    m = _LAYER.match(name)
    if not m:
        return None
    layer, tail = int(m.group(1)), m.group(2)
    if tail in _HF:
        path, leaf, seg = _HF[tail]
        return layer, path, leaf, seg
    e = _HF_EXPERT.match(tail)
    if e:
        return layer, f"mlp.switch_mlp.{e.group(1)}_proj", e.group(2), None
    if tail in RULES:
        return layer, tail, None, None
    p = _PART.match(tail)
    if p and p.group(1) in RULES and RULES[p.group(1)].kind == PARTS:
        return layer, p.group(1), f"shard_{p.group(2)}.{p.group(3)}", None
    path, _, leaf = tail.rpartition(".")
    if path in RULES:
        return layer, path, leaf, None
    return None


def sharded(name: str) -> bool:
    """Is this weight split across ranks under tensor?"""
    at = locate(name)
    if at is None:
        return False
    _, path, leaf, _ = at
    if RULES[path].kind == PARTS:
        return True
    return predicate(RULES[path].kind)(leaf or path, _Nd(2)) is not None


class _Nd(NamedTuple):
    ndim: int


def modules_of(shapes: dict) -> dict:
    """{safetensors name: shape} -> {(layer, rule path): {leaf: (name,
    shape, extra segments)}} for every array under a rule."""
    out: dict = {}
    for name, shape in shapes.items():
        at = locate(name)
        if at is not None:
            layer, path, leaf, seg = at
            out.setdefault((layer, path), {})[leaf] = (name, list(shape), seg)
    return out


def skipzero_split(path) -> bool:
    """The artifact's bundled runtime splits its own SKIPZERO gate/up rows
    per rank (vqlab's skipzero_shard: model.py declares SKIPZERO_SHARD = 1
    and takes {"vq_skipzero": {..., "shard": {"rank", "n"}}} at load)."""
    from pathlib import Path
    try:
        return bool(_SZ_SPLIT.search((Path(path) / "model.py").read_text()))
    except OSError:
        return False


def _runtime_view(mods: dict, sz_split: bool) -> dict:
    """modules_of as the split meets them after a load whose runtime splits
    SKIPZERO itself: its gate/up arrive already per rank (not cut here);
    down_proj's compact codes/scales are cut on the input axis like any VQ
    module, its row table kept whole."""
    if not sz_split:
        return mods
    out = {}
    for key, leaves in mods.items():
        if not any(k in _SZ for k in leaves):
            out[key] = leaves
        elif RULES[key[1]].kind == S2A:
            out[key] = {_SZ.get(k, k): v for k, v in leaves.items()}
    return out


def unverified(shapes: dict, sz_split: bool = False) -> dict:
    """{(rule path, unknown leaves): first layer} -- one module per
    distinct layout no rule knows, for the launch to run. A PARTS table
    has none: its parts are never cut."""
    out: dict = {}
    for (layer, path), leaves in sorted(
            _runtime_view(modules_of(shapes), sz_split).items()):
        if RULES[path].kind == PARTS:
            continue
        u = tuple(unknown(leaves))
        if u and (path, u) not in out:
            out[(path, u)] = layer
    return out


def refusals(shapes: dict, n: int, key_dim: int, kv_heads: int,
             sz_split: bool = False) -> list:
    """{safetensors name: shape} -> why these arrays cannot be cut `n` ways
    by RULES, one line per (module, reason) with its numbers and how many
    layers repeat it; [] when they can. An unknown layout is not refused
    here (see `unverified`), but its arrays must still divide."""
    seen: dict = {}

    def say(key, line):
        if key in seen:
            seen[key][1] += 1
        else:
            seen[key] = [line, 0]

    for (layer, path), leaves in sorted(
            _runtime_view(modules_of(shapes), sz_split).items()):
        rule = RULES[path]
        where = f"layers.{layer}.{path}"
        if rule.kind == PARTS:
            parts = len({k.split(".")[0] for k in leaves})
            if parts % n:
                say((path, "parts"), f"{where}: {parts} parts do not "
                    f"divide by {n}")
            continue
        why = module_refusal(where, rule, {k for k in leaves if k})
        if why:
            say((path, "bias"), why)
            continue
        pred = predicate(rule.kind)
        for leaf, (_, shape, seg) in sorted(leaves.items(),
                                               key=lambda kv: kv[0] or ""):
            axis = pred(leaf or path, _Nd(len(shape)))
            if axis is None or not shape:
                continue
            size = shape[axis]
            if rule.repeat_kv and 0 < kv_heads < n:
                size = size * (n // kv_heads)
            pts = segment_points(rule, key_dim) if seg is None else seg
            if isinstance(pts, int):
                parts = [size // pts] * pts if size % pts == 0 else [size]
            else:
                edges = [0, *pts, size]
                parts = [b - a for a, b in zip(edges, edges[1:])]
            bad = [p for p in parts if p % n]
            if bad or (isinstance(pts, int) and size % pts):
                what = f"{where}.{leaf}" if leaf else where
                seg_s = "" if len(parts) == 1 else f" (in segments {parts})"
                unit = "rows" if axis == 0 else f"on axis {axis}"
                say((path, leaf, "div"),
                    f"{what}: {shape[axis]} {unit}{seg_s} do not divide "
                    f"by {n}")
    out = []
    for line, more in seen.values():
        out.append(line + (f" (and {more} more layer{'s' * (more > 1)})"
                           if more else ""))
    return out
