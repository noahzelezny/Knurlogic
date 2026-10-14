"""Layers the vision towers share (mlx): the float32 RMS norms and
rotate_half, each written once.

Both norms compute in float32 and round once to the input's dtype -- the
towers run once per image, so the unfused form costs nothing that matters.
mlx's `x ** 2` and `mx.square(x)` round differently; a norm squares the way
its `square` says, so each tower keeps the form it was matched with.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def _power_square(x: mx.array) -> mx.array:
    return x**2


def _normed(x: mx.array, eps: float, square) -> mx.array:
    """x in float32 over its RMS (last axis)."""
    x_float = x.astype(mx.float32)
    var = mx.mean(square(x_float), axis=-1, keepdims=True)
    return x_float * mx.rsqrt(var + eps)


class RMSNorm(nn.Module):
    """RMSNorm with a gain, float32 throughout, rounded once."""

    square = staticmethod(_power_square)

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return (_normed(x, self.eps, self.square)
                * self.weight.astype(mx.float32)).astype(x.dtype)


class RMSNormNoScale(nn.Module):
    """RMSNorm without a gain (no parameters), float32, rounded once."""

    square = staticmethod(_power_square)

    def __init__(self, dim: int | None = None, *, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return _normed(x, self.eps, self.square).astype(x.dtype)


def rotate_half(x: mx.array) -> mx.array:
    """[x1, x2] -> [-x2, x1] over the last axis's halves (RoPE)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)
