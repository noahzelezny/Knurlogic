"""Every mx.array an object holds (`arrays_of`): what a load evaluates,
what a prompt-cache entry saves or compares, what a step forces before the
next one."""

from __future__ import annotations

#: leaves that hold no array: not walked into
_ATOMS = (str, bytes, bytearray, int, float, bool, type(None), type)


def arrays_of(root, unique: bool = False, limit: int = 2_000_000) -> list:
    """Every mx.array reachable from `root`, in a fixed depth-first order:
    the same structure gives the same list, an array held in two places
    listed at both (so two caches compare slot by slot), or once with
    `unique` (to count what is held). Walked through
    dicts (an nn.Module is one), lists, tuples, sets and plain objects'
    attributes -- so also arrays in "_"-named attributes and helper objects
    that nn.Module.parameters() skips (a bundled model.py's VQ codebooks).
    Cycle-safe; `limit` bounds the objects visited."""
    import mlx.core as mx
    seen: set = set()
    found: list = []
    stack = [root]
    while stack and len(seen) < limit:
        o = stack.pop()
        if isinstance(o, mx.array):
            if not unique or id(o) not in seen:
                seen.add(id(o))
                found.append(o)
            continue
        if id(o) in seen or isinstance(o, _ATOMS):
            continue
        seen.add(id(o))
        kids: list = []
        if isinstance(o, dict):
            kids.extend(o.values())       # an nn.Module is a dict too...
        if isinstance(o, (list, tuple, set, frozenset)):
            kids.extend(o)
        if hasattr(o, "__dict__"):        # ...with plain attributes beside
            kids.extend(vars(o).values())
        stack.extend(reversed(kids))
    return found
