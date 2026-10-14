"""Which MLX back end this process computes on: "metal" (Apple silicon),
"cuda" (an Nvidia GPU) or "cpu".

The one place engine code asks, and where it reads the device's own
description (`device_info`). A hand-written kernel exists per back end
(`mx.fast.metal_kernel`, `mx.fast.cuda_kernel`); code that has one only
for some back ends checks `name()` here and takes its portable path on
the others. Design: docs/design/os-agnostic.md.
"""
from __future__ import annotations

from functools import cache


@cache
def name() -> str:
    import mlx.core as mx
    if mx.metal.is_available():
        return "metal"
    if mx.cuda.is_available():
        return "cuda"
    return "cpu"


def metal() -> bool:
    return name() == "metal"


def device_info() -> dict:
    """mx.device_info() (mx.metal.device_info() on an mlx that predates
    it): the device's name, architecture, memory size and recommended
    working set. Raises what mlx raises when there is no device to ask."""
    import mlx.core as mx
    get = getattr(mx, "device_info", None) or mx.metal.device_info
    return get()
