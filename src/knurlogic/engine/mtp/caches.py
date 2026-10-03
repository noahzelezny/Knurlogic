"""Snapshot and rollback for a speculative decode step.

Two cache kinds appear in one `model.make_cache()` list and roll back
differently:

  attention caches   expose keys/values and a `trim(n)`. They MUST be trimmed
                     by the offset DELTA, not by a fixed count: trimming a
                     hardcoded 1 leaves a stale key while the recurrent caches
                     roll back 2, and the streams drift silently.
  recurrent caches   expose a `cache` list of state arrays. Rollback is
                     whatever the snapshot held.

A third kind rolls back by state though it looks like attention: a cache
that answers `is_trimmable() == False` (deepseek_v4's DeepseekV4Cache, whose
compressed pools cannot be trimmed; its `trim` is a no-op). Trimming it
would silently keep the rejected positions, so its whole object state is
held instead (`_grab` / `_put`).

A cache that records its speculative forward (`spec_begin`, deepseek_v4's
DeepseekV4Cache, architecture edit 19) rolls back FORWARD too: `rollback`
leaves a list of them at the first `keep` positions the forward fed
without running the model again. Any other list is restored instead, and
the caller replays.

Keeping old references is a free snapshot only for an architecture that
REASSIGNS its slots (mlx arrays are immutable); one that writes in place
would corrupt the saved reference. So the policy is a per-family field,
default "copy", and `check_snapshot_semantics` earns the "reassign" path.
"""
from __future__ import annotations

import mlx.core as mx


def is_untrimmable(c) -> bool:
    f = getattr(c, "is_trimmable", None)
    return callable(f) and hasattr(c, "keys") and not f()


class _Obj:
    """A held object's attributes (see _grab)."""
    __slots__ = ("obj", "attrs")

    def __init__(self, obj, attrs):
        self.obj, self.attrs = obj, attrs


def _fields(o) -> list:
    names = list(getattr(o, "__dict__", {}))
    for klass in type(o).__mro__:
        names += [n for n in getattr(klass, "__slots__", ())
                  if n not in names and hasattr(o, n)]
    return names


def _grab(o):
    """Everything `o` holds, arrays as new handles: mlx's `a[i] = v`
    rebinds the handle it is called on, so a held handle keeps the old
    value (and nothing is copied)."""
    if isinstance(o, mx.array):
        return mx.array(o)
    if isinstance(o, list):
        return [_grab(x) for x in o]
    if isinstance(o, tuple):
        return tuple(_grab(x) for x in o)
    if isinstance(o, dict):
        return {k: _grab(v) for k, v in o.items()}
    if o is None or isinstance(o, (bool, int, float, str, mx.Dtype)):
        return o
    if hasattr(o, "__dict__") or hasattr(type(o), "__slots__"):
        return _Obj(o, {n: _grab(getattr(o, n)) for n in _fields(o)})
    return o


def _put(s):
    """Back to what `_grab` held: the same objects, their old attributes."""
    if isinstance(s, _Obj):
        for n, v in s.attrs.items():
            setattr(s.obj, n, _put(v))
        return s.obj
    if isinstance(s, mx.array):
        return mx.array(s)
    if isinstance(s, list):
        return [_put(x) for x in s]
    if isinstance(s, tuple):
        return tuple(_put(x) for x in s)
    if isinstance(s, dict):
        return {k: _put(v) for k, v in s.items()}
    return s


def is_attention(c) -> bool:
    return hasattr(c, "keys") and hasattr(c, "trim") and not is_batch_attention(c)


def is_batch_attention(c) -> bool:
    """mlx-lm's BatchKVCache (and qwen4_exp's _BatchAttnCache over it): keys
    and a `trim`, but `offset` is one position PER ROW. Every row of a
    batched speculative step advances by the same count, so the rollback
    unit is the shared write index (`size()`), not a per-row offset —
    `trim(n)` there moves every row's offset together."""
    return (hasattr(c, "keys") and hasattr(c, "trim")
            and isinstance(getattr(c, "offset", None), mx.array))


def is_attention_composite(c) -> bool:
    """mlx_vlm's CacheList (glm5_next fa layers: main-KV + indexer-KV).
    No `keys` of its own, but every member is a plain attention cache, so
    the composite rolls back like one: trim each member by its offset
    delta. A CacheList holding anything non-attention falls through to
    the TypeError — its rollback is unproven.
    (Vendored from vqlab's mtp/caches.py, same provenance as
    families/glm5/heads/glm5.py.)"""
    subs = getattr(c, "caches", None)
    return (subs is not None and len(subs) > 0
            and all(is_attention(s) or is_batch_attention(s) for s in subs))


def position(c):
    """The position a single-row cache sits at, or None if it has none.

    A plain cache answers with `offset`; a composite (mlx's CacheList, GLM's
    head and full-attention layers) answers with its members' common
    offset -- it has no `offset` of its own, and reading one gave -1, which
    discarded every GLM prefix hit as "no aligned head cache". Recurrent
    caches carry state, not a position: None."""
    off = getattr(c, "offset", None)
    if off is not None and not isinstance(off, mx.array):
        return int(off)
    subs = getattr(c, "caches", None)
    if subs:
        offs = {position(m) for m in subs}
        if len(offs) == 1:
            return offs.pop()
    return None


def _pos(c):
    """A member's rollback unit: offset for one row, the shared write index
    for a batched cache (see is_batch_attention). In the batch engine a
    CacheList's members are BatchKVCaches."""
    return c.size() if is_batch_attention(c) else c.offset


def can_record(c) -> bool:
    """A cache that records a speculative forward and rolls back to any
    position inside it (architecture edit 19)."""
    return callable(getattr(c, "spec_begin", None))


def snapshot(caches, *, copy: bool = True) -> list:
    snaps: list = []
    for c in caches:
        if can_record(c):
            c.spec_begin()
            snaps.append(("spec", None, None))
        elif is_untrimmable(c):
            snaps.append(("whole", None, _grab(c)))
        elif is_attention(c):
            snaps.append(("attn", c.offset, None))
        elif is_batch_attention(c):
            snaps.append(("battn", c.size(), None))
        elif is_attention_composite(c):
            snaps.append(("attn-list", [_pos(s) for s in c.caches], None))
        elif hasattr(c, "cache"):
            state = list(c.cache)
            if copy:
                state = [mx.array(x) if isinstance(x, mx.array) else x
                         for x in state]
            snaps.append(("state", getattr(c, "offset", None), state))
        else:
            raise TypeError(
                f"cache {type(c).__name__} is neither an attention cache "
                f"(keys/trim) nor a state cache (.cache); speculative "
                f"rollback cannot be proven correct for it")
    return snaps


def restore(caches, snaps) -> None:
    """Back to exactly where the snapshot was taken."""
    for c, s in zip(caches, snaps):
        kind, offset, state = s
        if kind == "spec":
            c.spec_rollback(0)
        elif kind == "whole":
            # a rollback may run twice from one snapshot (the replay)
            _put(state)
        elif kind == "attn":
            n = c.offset - offset
            if n > 0:
                c.trim(n)
            elif n < 0:
                raise RuntimeError(
                    f"attention cache went BACKWARDS since the snapshot "
                    f"({c.offset} < {offset}); rollback would corrupt it")
        elif kind == "battn":
            n = c.size() - offset
            if n > 0:
                c.trim(n)
            elif n < 0:
                raise RuntimeError(
                    f"batched attention cache went BACKWARDS since the "
                    f"snapshot ({c.size()} < {offset}); rollback would corrupt it")
        elif kind == "attn-list":
            for sub, off in zip(c.caches, offset):
                n = _pos(sub) - off
                if n > 0:
                    sub.trim(n)
                elif n < 0:
                    raise RuntimeError(
                        f"attention cache went BACKWARDS since the snapshot "
                        f"({_pos(sub)} < {off}); rollback would corrupt it")
        else:
            c.cache = list(state)
            if offset is not None:
                c.offset = offset


def rollback(caches, snaps, keep: int) -> bool:
    """Leave every cache at `keep` positions past its snapshot -- the first
    `keep` tokens of the forward run since -- without running the model,
    by the recording caches' own bookkeeping (`can_record`). True when
    done; False when the list holds any other kind of cache, or a forward
    a recording cache cannot roll back: then every cache is restored to
    the snapshot and the caller replays the `keep` tokens.

    Only a list of recording caches rolls forward, never a mix with
    trimmed attention caches: a pipeline's stages hold different layers,
    and every rank must take the same path (the replay is a forward every
    rank runs), so the answer has to follow the family, not the stage."""
    ok = bool(snaps) and all(s[0] == "spec" for s in snaps) and all(
        c.spec_can_rollback(keep) for c in caches)
    if not ok:
        restore(caches, snaps)
        return False
    for c in caches:
        c.spec_rollback(keep)
    return True


def release(caches, snaps) -> None:
    """Keep everything fed since the snapshot (every draft accepted): a
    recording cache stops recording."""
    for c, s in zip(caches, snaps):
        if s[0] == "spec":
            c.spec_end()


def check_snapshot_semantics(caches, advance) -> bool:
    """Does `copy=False` snapshotting actually hold for these caches?

    Snapshot without copying, deep-copy the same arrays separately, run
    `advance()` (one forward through the model), and check the snapshot's
    arrays are still what they were. True means the family may use
    cache_semantics="reassign"; False means it writes state in place and must
    copy. Returns True for an all-attention cache list (nothing to alias):
    attention snapshots, batched ("battn") and composite ("attn-list") too,
    hold positions, not arrays.

    This is the gate that lets a new family claim the cheap path, and it fails
    on an in-place cache by construction — see tests/test_mtp_caches.py, where
    it is run against both a reassigning and a mutating cache.
    """
    snaps = snapshot(caches, copy=False)
    witness = [[mx.array(x) if isinstance(x, mx.array) else x for x in s[2]]
               for s in snaps if s[0] == "state"]
    for w in witness:
        mx.eval(*[x for x in w if isinstance(x, mx.array)])
    advance()
    release(caches, snaps)
    held = [s[2] for s in snaps if s[0] == "state"]
    for kept, ref in zip(held, witness):
        for a, b in zip(kept, ref):
            if not isinstance(a, mx.array) or not isinstance(b, mx.array):
                continue
            if a.shape != b.shape or not bool(mx.all(a == b).item()):
                return False
    return True
