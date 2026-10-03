"""DeepSeek-V4-Flash-Vision-Exp's ViT and aligner in MLX.

Ported from `inference/vision.py` of deepseek-ai/DeepSeek-V4-Flash-Vision-Exp
(MIT License, Copyright (c) 2023 DeepSeek): a 2D-RoPE ViT with full
bidirectional attention over one image, then a 3x3 unfold and a two-layer
GELU MLP into the trunk's width. Parameter names are the reference's, so
the artifact's `vision.*` / `aligner.*` keys load with their prefix cut.
tests/engine/test_vision_deepseek.py holds it to the reference on the real
weights.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .processor import VisionArgs


def vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float):
    """[n_h * n_w, 1, dim] cos and sin: `dim / 2` frequencies for the row,
    then the same for the column (the reference's get_vision_cos_sin)."""
    inv_freq = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    hpos = mx.broadcast_to(mx.arange(n_h)[:, None], (n_h, n_w))
    wpos = mx.broadcast_to(mx.arange(n_w)[None, :], (n_h, n_w))
    pos = mx.stack([hpos, wpos], axis=-1).reshape(-1, 2, 1).astype(mx.float32)
    freqs = (pos * inv_freq).reshape(n_h * n_w, -1)
    return mx.cos(freqs)[:, None], mx.sin(freqs)[:, None]


def apply_rotary(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    dtype = x.dtype
    x1, x2 = mx.split(x.astype(mx.float32), 2, axis=-1)
    return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin],
                          axis=-1).astype(dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        x = x.astype(mx.float32)
        x = x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + self.eps)
        return (self.weight.astype(mx.float32) * x).astype(dtype)


class PatchEmbed(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        self.proj = nn.Linear(3 * args.vision_patch_size ** 2, args.vision_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(x)


class Attention(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        self.n_heads = args.vision_n_heads
        self.head_dim = args.vision_dim // args.vision_n_heads
        self.wqkv = nn.Linear(args.vision_dim, 3 * args.vision_dim)
        self.wo = nn.Linear(args.vision_dim, args.vision_dim)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        n = x.shape[0]
        q, k, v = (t.reshape(n, self.n_heads, self.head_dim)
                   for t in mx.split(self.wqkv(x), 3, axis=-1))
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        q, k, v = (t.transpose(1, 0, 2)[None] for t in (q, k, v))
        o = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim ** -0.5)
        return self.wo(o[0].transpose(1, 0, 2).reshape(n, -1))


class MLP(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        self.w1 = nn.Linear(args.vision_dim, 2 * args.vision_inter_dim, bias=False)
        self.w2 = nn.Linear(args.vision_inter_dim, args.vision_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate, up = mx.split(self.w1(x), 2, axis=-1)
        return self.w2(nn.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        self.norm1 = RMSNorm(args.vision_dim)
        self.attn = Attention(args)
        self.norm2 = RMSNorm(args.vision_dim)
        self.mlp = MLP(args)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    """Full bidirectional attention over one image with 2D RoPE."""

    def __init__(self, args: VisionArgs):
        super().__init__()
        self.rope_dim = args.vision_dim // args.vision_n_heads // 2
        self.rope_theta = args.vision_rope_theta
        self.patch_embed = PatchEmbed(args)
        self.blocks = [Block(args) for _ in range(args.vision_n_layers)]
        self.norm = RMSNorm(args.vision_dim)

    def __call__(self, patches: mx.array, n_h: int, n_w: int) -> mx.array:
        x = self.patch_embed(patches)
        cos, sin = vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


class Aligner(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        self.downsample_ratio = args.vision_downsample_ratio
        in_dim = args.vision_dim * self.downsample_ratio ** 2
        self.w1 = nn.Linear(in_dim, args.dim)
        self.w2 = nn.Linear(args.dim, args.dim)

    def __call__(self, x: mx.array, n_h: int, n_w: int) -> mx.array:
        """[n_h * n_w, C] -> [ceil(n_h/r) * ceil(n_w/r), dim]. The
        reference's zero pad and `F.unfold(r, stride=r)`: each r x r block,
        channels outermost, row-major blocks."""
        r = self.downsample_ratio
        C = x.shape[-1]
        x = x.reshape(n_h, n_w, C)
        x = mx.pad(x, [(0, -n_h % r), (0, -n_w % r), (0, 0)])
        hb, wb = x.shape[0] // r, x.shape[1] // r
        x = x.reshape(hb, r, wb, r, C).transpose(0, 2, 4, 1, 3)
        x = x.reshape(hb * wb, C * r * r)
        return self.w2(nn.gelu(self.w1(x)))


class Tower(nn.Module):
    """`vision` + `aligner`, the artifact's two prefixes."""

    def __init__(self, args: VisionArgs):
        super().__init__()
        self.vision = ViT(args)
        self.aligner = Aligner(args)

    def __call__(self, patches: mx.array, n_vit_h: int, n_vit_w: int) -> mx.array:
        return self.aligner(self.vision(patches, n_vit_h, n_vit_w),
                            n_vit_h, n_vit_w)
