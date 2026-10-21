"""Read decoder layers' outputs as the model decodes, and score a batch.

The hook is on the decoder layer's CLASS, filtered by instance: some
families walk their layers through a property rebuilt on every access
(Qwen3.5-35B's pipeline_layers), so wrapping the instances in a list would
miss calls. Each wanted layer keeps its latest output, lazily -- nothing is
evaluated here. A decode forward's input positions are tokens the step commits
(engine/mtp/batch_loop feeds `t1` first, then a verified `t2`); the state
at one of them is what the probe was fitted on, the last token of the
thinking up to a paragraph break.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mlx.core as mx

from knurlogic.engine.model import trunk

from . import store


class Tap:
    """Hooks on `layers` of `model`; `latest[i]` is layer i's last output."""

    def __init__(self, model, layers: list[int]):
        found = list(trunk.core_of(model).layers)
        self.depth = len(found)
        self.layers = sorted(set(layers))
        self._index = {id(found[i]): i for i in self.layers}
        self.latest: dict[int, mx.array] = {}
        self._classes: dict[type, Callable[..., Any]] = {}
        #: the probe whose arrays are on the device: (probe, mean, scale, w)
        self._bound: tuple = (None,)
        for cls in {type(found[i]) for i in self.layers}:
            inner = cls.__call__
            self._classes[cls] = inner
            cls.__call__ = self._hooked(inner)  # type: ignore[method-assign]

    def _hooked(self, inner):
        tap = self

        def hooked(layer, *a, **k):
            y = inner(layer, *a, **k)
            i = tap._index.get(id(layer))
            if i is not None:
                tap.latest[i] = y[0] if isinstance(y, tuple) else y
            return y
        return hooked

    def at(self, layer: int, pos: int) -> mx.array | None:
        """[B, D] float32: each row's state at input position `pos` of the
        last forward through `layer`, lazy; None before any. A layer with
        several residual streams (DeepSeek V4's and GLM-5.3's hyper-
        connections, 4 x 4096) is read as one vector, as it was fitted."""
        y = self.latest.get(layer)
        if y is None:
            return None
        h = y[:, pos]
        return h.reshape(h.shape[0], -1).astype(mx.float32)

    def states(self, pos: int) -> mx.array | None:
        """[B, L, D]: every tapped layer at `pos` (onboarding keeps them
        all and picks the layer at the fit)."""
        got = [self.at(i, pos) for i in self.layers]
        return None if any(g is None for g in got) else mx.stack(got, axis=1)

    def score(self, probe: store.Probe, pos: int) -> mx.array | None:
        """[B] probabilities at input position `pos`, lazy."""
        h = self.at(probe.layer, pos)
        if h is None:
            return None
        if self._bound[0] is not probe:
            self._bound = (probe, mx.array(probe.mean), mx.array(probe.scale),
                           mx.array(probe.w))
        _, mean, scale, w = self._bound
        return mx.sigmoid(((h - mean) / scale) @ w + probe.b)

    def remove(self) -> None:
        for cls, inner in self._classes.items():
            cls.__call__ = inner  # type: ignore[method-assign]
        self._classes.clear()
        self.latest.clear()


def default_layers(depth: int) -> list[int]:
    """The layers onboarding reads: one-half to two-thirds of the depth,
    where the probe read best in all five families measured."""
    return sorted({round(f * (depth - 1)) for f in (0.5, 0.6, 0.67)})
