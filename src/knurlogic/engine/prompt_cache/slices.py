"""A layer's cache per head range: cut, joined, and which files make a
range (docs/design/prompt-cache-shards.md, "Which axis is heads").

A tensor rank holds every layer but only a share of each cache: the units
[a, b) of the `units` its kind counts (kv heads, deltanet key heads) on
the axes the family's spec names (engine/split/tensor_rules.CACHE_KINDS,
resolved per layer by `cache_cuts`). Such a share is saved as
`L{i:03d}.h{a}-{b}.safetensors`; a whole layer as `L{i:03d}.safetensors`.
A reader needing [c, d) takes the files that cover it (`plan`): the whole
layer, one exact slice, or several slices, each cut to its overlap and
joined in order (`assemble`). A spec is (kind, units, ((field, axis,
widths), ...)); a field not named there is carried whole in every slice.
"""
from __future__ import annotations

import copy
import re

from . import disk

_SLICE = re.compile(r"^L([0-9]{3,4})(?:\.h([0-9]{1,4})-([0-9]{1,4}))?"
                    + re.escape(disk.SUFFIX) + "$")


def slice_name(i: int, a: int, b: int) -> str:
    return f"L{int(i):03d}.h{int(a)}-{int(b)}{disk.SUFFIX}"


def parse_name(name: str):
    """A layer file's name -> (slot, None) for a whole layer, (slot, (a,
    b)) for a slice, or None."""
    m = _SLICE.match(name or "")
    if m is None:
        return None
    if m.group(2) is None:
        return int(m.group(1)), None
    a, b = int(m.group(2)), int(m.group(3))
    return (int(m.group(1)), (a, b)) if 0 <= a < b else None


def pieces_in(names) -> dict:
    """{slot: set of pieces}: a piece is None (the whole layer) or (a, b)."""
    out: dict = {}
    for n in names:
        p = parse_name(n)
        if p is not None:
            out.setdefault(p[0], set()).add(p[1])
    return out


def plan(have, spec, want):
    """The pieces (None: the whole layer; (a, b): a slice) to read for the
    units `want` ((c, d), or None for the whole layer) of a layer whose
    cut is `spec` (None: replicated, never cut), from the pieces `have`.
    The whole layer when it is there; else a chain of slices from c to d,
    each reaching furthest. None when `have` cannot make it."""
    have = set(have or ())
    if None in have:
        return [None]
    if spec is None:
        return None
    c, d = want if want is not None else (0, int(spec[1]))
    out, pos = [], c
    while pos < d:
        nxt = [p for p in have if p is not None and p[0] <= pos < p[1]
               and p[1] <= spec[1]]
        if not nxt:
            return None
        p = max(nxt, key=lambda r: (r[1], -r[0]))
        out.append(p)
        pos = p[1]
    return out


# ---------------------------------------------------------- the arrays

def _get(obj, path: list):
    for p in path:
        if obj is None:
            return None
        obj = obj[int(p)] if isinstance(obj, (list, tuple)) \
            else getattr(obj, p, None)
    return obj


def _set(obj, path: list, value):
    """`obj` with the field at `path` replaced (copies along the path;
    `obj` itself is not changed)."""
    if not path:
        return value
    head, rest = path[0], path[1:]
    if isinstance(obj, (list, tuple)):
        new = list(obj)
        new[int(head)] = _set(obj[int(head)], rest, value)
        return type(obj)(new) if isinstance(obj, tuple) else new
    out = copy.copy(obj)
    setattr(out, head, _set(getattr(obj, head), rest, value))
    return out


def _map(v, fn):
    """`fn` on every array under `v` (an array, or lists/tuples of them)."""
    import mlx.core as mx
    if isinstance(v, mx.array):
        return fn(v)
    if isinstance(v, (list, tuple)):
        out = [_map(x, fn) for x in v]
        return tuple(out) if isinstance(v, tuple) else out
    if v is None:
        return None
    raise ValueError(f"a {type(v).__name__} where a cut field's arrays are")


def _segs(n: int, widths) -> list:
    """(start, unit width) of each segment on an axis of `n` slots."""
    tot = sum(widths)
    if n % tot:
        raise ValueError(f"an axis of {n} is not whole units of {tot}")
    k = n // tot
    out, at = [], 0
    for w in widths:
        out.append((at, w))
        at += k * w
    return out


def _cut_array(a, axis: int, widths, x: int, y: int):
    import mlx.core as mx
    ax = axis % a.ndim
    parts = []
    for start, w in _segs(a.shape[ax], widths):
        idx = [slice(None)] * a.ndim
        idx[ax] = slice(start + x * w, start + y * w)
        parts.append(a[tuple(idx)])
    out = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=ax)
    return mx.contiguous(out)


def units_of(obj, spec) -> int:
    """How many units `obj` holds (from its first cut array), or 0."""
    import mlx.core as mx
    for field, axis, widths in spec[2]:
        v = _get(obj, field.split("."))
        while isinstance(v, (list, tuple)) and v:
            v = v[0]
        if isinstance(v, mx.array):
            return v.shape[axis % v.ndim] // sum(widths)
    return 0


def cut(obj, spec, x: int, y: int):
    """`obj` with every field `spec` cuts keeping its units [x, y)
    (counted from `obj`'s own first unit); other fields shared."""
    n = units_of(obj, spec)
    if x == 0 and y == n:
        return obj
    for field, axis, widths in spec[2]:
        path = field.split(".")
        v = _get(obj, path)
        if v is None:
            continue
        obj = _set(obj, path, _map(v, lambda a, ax=axis, w=widths:
                                   _cut_array(a, ax, w, x, y)))
    return obj


def _join_arrays(arrs, axis: int, widths):
    import mlx.core as mx
    ax = axis % arrs[0].ndim
    segs: list[list] = [[] for _ in widths]
    for a in arrs:
        for j, (start, w) in enumerate(_segs(a.shape[ax], widths)):
            k = a.shape[ax] // sum(widths)
            idx = [slice(None)] * a.ndim
            idx[ax] = slice(start, start + k * w)
            segs[j].append(a[tuple(idx)])
    return mx.contiguous(mx.concatenate(
        [x for s in segs for x in s], axis=ax))


def _join_field(vals, axis, widths):
    import mlx.core as mx
    if all(v is None for v in vals):
        return None
    if any(v is None for v in vals):
        raise ValueError("a cut field is empty in some slices only")
    v0 = vals[0]
    if isinstance(v0, mx.array):
        return _join_arrays(vals, axis, widths)
    if isinstance(v0, (list, tuple)):
        if any(len(v) != len(v0) for v in vals):
            raise ValueError("slices of a field differ in length")
        out = [_join_field([v[i] for v in vals], axis, widths)
               for i in range(len(v0))]
        return tuple(out) if isinstance(v0, tuple) else out
    raise ValueError(f"a {type(v0).__name__} where a cut field's arrays are")


def join(objs: list, spec):
    """The objects of consecutive unit ranges as one: each cut field
    joined segment by segment; the rest from the first."""
    if len(objs) == 1:
        return objs[0]
    if len({type(o) for o in objs}) != 1:
        raise ValueError("slices of one layer are different classes")
    out = objs[0]
    for field, axis, widths in spec[2]:
        path = field.split(".")
        out = _set(out, path, _join_field([_get(o, path) for o in objs],
                                          axis, widths))
    return out


def assemble(pieces: list, spec, want, units: int):
    """[((a, b) or None, obj)] read per `plan` -> the cache of units
    `want` ((c, d), or None for all `units`)."""
    c, d = want if want is not None else (0, units)
    if spec is None:
        (_, obj), = pieces
        return obj
    out = []
    for rng, obj in pieces:
        a, b = rng if rng is not None else (0, units)
        x, y = max(a, c), min(b, d)
        if x >= y:
            continue
        if units_of(obj, spec) not in (0, b - a):
            raise ValueError(f"a piece of units [{a}, {b}) holds "
                             f"{units_of(obj, spec)}")
        out.append(cut(obj, spec, x - a, y - a))
    return join(out, spec)
