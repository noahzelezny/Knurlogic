"""Bind Thinking: auto to the loaded model: its probe's tap when one is
fitted ("ready"), onboarding's wider tap when its Thinking setting is auto
and none is ("learning"), nothing otherwise ("off"). ModelHost binds at
load and unbinds at unload; onboarding rebinds when the setting changes or
a probe is fitted.
"""

from __future__ import annotations

import logging

from knurlogic.engine.model import thinking, trunk

from . import store
from .tap import Tap, default_layers

logger = logging.getLogger(__name__)


def wanted(path: str, split: bool) -> tuple[str, str]:
    """(state, why) auto should be in for the model at `path`."""
    if split:
        return "off", "a split model: auto runs on one Mac only so far"
    if store.load(store.base_of(path)) is not None:
        return "ready", ""
    if thinking.server_default() == thinking.AUTO:
        return "learning", ""
    return "off", ("no probe for this model yet: set its Thinking to auto "
                   "to learn one")


def bind(model, path: str, split: bool) -> None:
    unbind()
    state, why = wanted(path, split)
    tap = None
    probe = None
    if state == "ready":
        probe = store.load(store.base_of(path))
        assert probe is not None
        tap = Tap(model, [probe.layer])
    elif state == "learning":
        tap = Tap(model, default_layers(len(trunk.core_of(model).layers)))
    store.ACTIVE.update(probe=probe, tap=tap, state=state, why=why,
                        path=path)
    logger.info("Thinking: auto %s%s", state, f" ({why})" if why else "")


def refresh(model, path: str, split: bool) -> None:
    """Rebind when what auto should be has changed (the setting moved, or
    onboarding fitted a probe)."""
    if wanted(path, split)[0] != store.ACTIVE["state"] \
            or store.ACTIVE.get("path") != path:
        bind(model, path, split)


def unbind() -> None:
    tap = store.ACTIVE["tap"]
    if tap is not None:
        tap.remove()
    store.ACTIVE.update(probe=None, tap=None, state="off", why="",
                        path=None)
