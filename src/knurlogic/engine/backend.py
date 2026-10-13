"""Which MLX back end this process computes on: "metal" (Apple silicon),
"cuda" (an Nvidia GPU) or "cpu".

The one place engine code asks. A hand-written kernel exists per back end
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
