"""Per-family capture of the pre-lm_head activation.

The MTP head drafts token t+2 from the trunk's hidden state at t. mlx-lm's
public contract is `model(tokens, cache=cache) -> logits`, so the hidden
state has to be taken from inside: a wrapper module that records its own
input and forwards. The attribute is a registry field, not a hardcoded
name, and the wrap is a context manager, so the trunk is left exactly as it
was found even if generation raises.

The wrapper is an nn.Module so the wrapped submodule stays reachable from
`model.parameters()` while installed; the captured array is kept in a
closure cell, so it never enters the module tree.
"""
from __future__ import annotations

from contextlib import contextmanager

import mlx.nn as nn


def _resolve(root, path: str):
    """(owner, attr) for a dotted path, so nested capture points work."""
    obj = root
    parts = path.split(".")
    for part in parts[:-1]:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj, parts[-1]


class _Capture(nn.Module):
    """A spy: forwards to `inner` (which may itself be a spy)."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner


def _spy(inner, sink):
    class _Spy(_Capture):
        def __call__(self, x, *args, **kwargs):
            sink[:] = [x]
            return self.inner(x, *args, **kwargs)

    return _Spy(inner)


@contextmanager
def capture_input(core, path: str):
    """Yield a getter for the most recent input to `core.<path>`.

    The getter raises if nothing has been captured yet, which is the honest
    failure for a registry entry that names the wrong module: a silently stale
    hidden state would show up only as degraded acceptance."""
    owner, attr = _resolve(core, path)
    # a list member (`layers.41`: DSpark's target layers) is an item
    if attr.isdigit():
        def put(v, i=int(attr)):
            owner[i] = v
        inner = owner[int(attr)]
    else:
        def put(v):
            setattr(owner, attr, v)
        inner = getattr(owner, attr)
    sink: list = []
    spy = _spy(inner, sink)
    put(spy)
    try:
        def get():
            if not sink:
                raise RuntimeError(
                    f"nothing captured at {path!r}: the trunk forward did not "
                    f"call it. Check the family's capture path against the "
                    f"installed architecture.")
            return sink[0]

        yield get
    finally:
        _unwrap(owner, attr, spy, put)


def _unwrap(owner, attr: str, spy, put) -> None:
    """Take `spy` out of the slot's chain of captures, leaving the others.

    Two generators on one model nest their spies (the second wraps the
    first), and they need not close in reverse order: an abandoned one is
    closed whenever it is collected. Putting back the module found at entry
    would then cut a live generator's spy out of the trunk, and its getter
    would hand back a stale hidden state (the last prefill chunk's) to the
    next decode step."""
    cur = owner[int(attr)] if attr.isdigit() else getattr(owner, attr)
    if cur is spy:
        put(spy.inner)
        return
    while isinstance(cur, _Capture):
        if cur.inner is spy:
            cur.inner = spy.inner
            return
        cur = cur.inner
