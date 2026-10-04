"""gemma4's vision tower -- standalone, not attached to any trunk (the
trunk's `sanitize` drops vision keys, so this tower loads its own tensors
under `vision_tower.*` / `embed_vision.*`).

Vendored from mlx-vlm 0.6.17 `mlx_vlm/models/gemma4/vision.py` (562 lines,
MIT, Copyright (c) 2025 Prince Canuma), with `from ..base import
ensure_fused_sdpa` changed to `from .._base import ensure_fused_sdpa`
(`engine/vision/_base.py`, so this package never imports mlx-vlm). One
structural trim, documented at `VisionModel.__call__`: the batched
list-of-images and `pixel_position_ids`-supplied branches are dropped,
since `Family.encode` calls the tower with ONE patchified image at a time
(images are encoded and cached individually).
"""
from __future__ import annotations

from functools import partial
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from knurlogic.engine.vision._base import ensure_fused_sdpa
from .config import VisionConfig


class ClippableLinear(nn.Module):
    """Linear with optional input/output clamping (e4b's quantized clip
    params, `use_clipped_linears`); a no-op (±inf bounds) until real clip
    values are loaded, verbatim from mlx-vlm."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 use_clipping: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.use_clipping = use_clipping
        if use_clipping:
            self.input_min = mx.array(float("-inf"))
            self.input_max = mx.array(float("inf"))
            self.output_min = mx.array(float("-inf"))
            self.output_max = mx.array(float("inf"))

    def __call__(self, x: mx.array) -> mx.array:
        if self.use_clipping:
            x = mx.clip(x, self.input_min, self.input_max)
        x = self.linear(x)
        if self.use_clipping:
            x = mx.clip(x, self.output_min, self.output_max)
        return x


def one_hot(indices: mx.array, num_classes: int) -> mx.array:
    return (mx.expand_dims(indices, -1) == mx.arange(num_classes)).astype(mx.float32)


class VisionRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        x_float = x.astype(mx.float32)
        var = mx.mean(x_float**2, axis=-1, keepdims=True)
        normed = x_float * mx.rsqrt(var + self.eps)
        return (normed * self.weight.astype(mx.float32)).astype(x.dtype)


class VisionRMSNormNoScale(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        x_float = x.astype(mx.float32)
        var = mx.mean(x_float**2, axis=-1, keepdims=True)
        return (x_float * mx.rsqrt(var + self.eps)).astype(x.dtype)


class RMSNorm(nn.Module):
    """The encoder layers' norms. knurlogic edit 7: HF's Gemma4RMSNorm --
    (x * rsqrt(mean(x^2) + eps)) * w in float32, rounded once to the
    working dtype; mx.fast.rms_norm rounds x * rsqrt first and again after
    the weight (a quarter of bf16 elements a step off). The tower runs once
    per image, so the unfused form costs nothing that matters."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x.astype(mx.float32),
                                self.weight.astype(mx.float32),
                                self.eps).astype(x.dtype)


@partial(mx.compile, shapeless=True)
def gelu_mul(gate: mx.array, up: mx.array) -> mx.array:
    """knurlogic edit 7: gelu_pytorch_tanh(gate) * up as torch rounds it on
    bf16 -- the gelu in float32, rounded once, then the product."""
    return nn.gelu_approx(gate.astype(mx.float32)).astype(up.dtype) * up


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_multidimensional_rope(inputs, positions, base_frequency=100.0):
    """2D RoPE over the patch grid; splits the head dim into `positions`'
    last axis' parts and rotates each independently (rotate_half must not
    mix features across spatial axes). `inputs`: [B, L, N, H]; `positions`:
    [B, L, 2] (patch grid x/y) or [B, L] (1D fallback, unused here but kept
    for fidelity to the reference)."""
    head_dim = inputs.shape[-1]

    if positions.ndim == 2:
        half = head_dim // 2
        freq_exponents = (2.0 / head_dim) * mx.arange(0, half).astype(mx.float32)
        timescale = mx.power(base_frequency, freq_exponents)
        sinusoid_inp = positions[..., None].astype(mx.float32) / timescale
        cos_val = mx.cos(sinusoid_inp)
        sin_val = mx.sin(sinusoid_inp)
        cos_val = mx.concatenate([cos_val, cos_val], axis=-1).astype(inputs.dtype)
        sin_val = mx.concatenate([sin_val, sin_val], axis=-1).astype(inputs.dtype)
        cos_val = mx.expand_dims(cos_val, axis=2)
        sin_val = mx.expand_dims(sin_val, axis=2)
        return inputs * cos_val + _rotate_half(inputs) * sin_val

    ndim = positions.shape[-1]
    channels_per_dim = 2 * (head_dim // (2 * ndim))
    half_per_dim = channels_per_dim // 2

    result_parts = []
    for d in range(ndim):
        x_part = inputs[..., d * channels_per_dim : (d + 1) * channels_per_dim]
        freq_exponents = (2.0 / channels_per_dim) * mx.arange(
            0, half_per_dim
        ).astype(mx.float32)
        timescale = mx.power(base_frequency, freq_exponents)
        sinusoid_inp = positions[..., d : d + 1].astype(mx.float32) / timescale
        cos_d = mx.cos(sinusoid_inp)
        sin_d = mx.sin(sinusoid_inp)
        cos_d = mx.concatenate([cos_d, cos_d], axis=-1).astype(inputs.dtype)
        sin_d = mx.concatenate([sin_d, sin_d], axis=-1).astype(inputs.dtype)
        cos_d = mx.expand_dims(cos_d, axis=2)
        sin_d = mx.expand_dims(sin_d, axis=2)
        y_part = x_part * cos_d + _rotate_half(x_part) * sin_d
        result_parts.append(y_part)

    return mx.concatenate(result_parts, axis=-1)


class VisionAttention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.rope_base_frequency = config.rope_parameters["rope_theta"]

        clip = getattr(config, "use_clipped_linears", False)
        self.q_proj = ClippableLinear(self.hidden_size, self.num_heads * self.head_dim,
                                      bias=False, use_clipping=clip)
        self.k_proj = ClippableLinear(self.hidden_size, self.num_kv_heads * self.head_dim,
                                      bias=False, use_clipping=clip)
        self.v_proj = ClippableLinear(self.hidden_size, self.num_kv_heads * self.head_dim,
                                      bias=False, use_clipping=clip)
        self.o_proj = ClippableLinear(self.num_heads * self.head_dim, self.hidden_size,
                                      bias=False, use_clipping=clip)

        self.q_norm = VisionRMSNorm(self.head_dim)
        self.k_norm = VisionRMSNorm(self.head_dim)
        self._v_norm = VisionRMSNormNoScale()

    def __call__(self, x: mx.array, positions: mx.array,
                 mask: Optional[mx.array] = None) -> mx.array:
        B, L, _ = x.shape

        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self._v_norm(v)

        q = apply_multidimensional_rope(q, positions, self.rope_base_frequency)
        k = apply_multidimensional_rope(k, positions, self.rope_base_frequency)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        attn_output = ensure_fused_sdpa(q, k, v, scale=1.0, mask=mask)

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(attn_output)


class VisionMLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        clip = getattr(config, "use_clipped_linears", False)
        self.gate_proj = ClippableLinear(config.hidden_size, config.intermediate_size,
                                         bias=False, use_clipping=clip)
        self.up_proj = ClippableLinear(config.hidden_size, config.intermediate_size,
                                       bias=False, use_clipping=clip)
        self.down_proj = ClippableLinear(config.intermediate_size, config.hidden_size,
                                         bias=False, use_clipping=clip)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(gelu_mul(self.gate_proj(x), self.up_proj(x)))


class VisionTransformerBlock(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.self_attn = VisionAttention(config)
        self.mlp = VisionMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size,
                                                 eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size,
                                                  eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, positions: mx.array,
                 mask: Optional[mx.array] = None) -> mx.array:
        normed = self.input_layernorm(x)
        attn_out = self.self_attn(normed, positions, mask)
        attn_out = self.post_attention_layernorm(attn_out)
        h = x + attn_out

        normed_h = self.pre_feedforward_layernorm(h)
        ffw_out = self.mlp(normed_h)
        ffw_out = self.post_feedforward_layernorm(ffw_out)
        return h + ffw_out


class VisionPatchEmbedder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.position_embedding_size = config.position_embedding_size
        self.input_proj = nn.Linear(3 * self.patch_size**2, self.hidden_size,
                                    bias=False)
        self.position_embedding_table = mx.ones(
            (2, self.position_embedding_size, self.hidden_size))

    def _position_embeddings(self, patch_positions: mx.array,
                             padding_positions: mx.array) -> mx.array:
        oh = one_hot(patch_positions, self.position_embedding_size)
        oh = oh.transpose(0, 2, 1, 3).astype(self.position_embedding_table.dtype)
        position_embeddings = oh @ self.position_embedding_table
        position_embeddings = position_embeddings.sum(axis=1)
        position_embeddings = mx.where(
            mx.expand_dims(padding_positions, -1), 0.0, position_embeddings)
        return position_embeddings

    def _patchify(self, pixel_values: mx.array) -> mx.array:
        B, C, H, W = pixel_values.shape
        p = self.patch_size
        pH, pW = H // p, W // p
        patches = pixel_values.reshape(B, C, pH, p, pW, p)
        patches = patches.transpose(0, 2, 4, 3, 5, 1)
        patches = patches.reshape(B, pH * pW, C * p * p)
        patches = 2 * (patches - 0.5)
        return self.input_proj(patches.astype(self.input_proj.weight.dtype))

    def __call__(self, pixel_values: mx.array, patch_positions: mx.array,
                 padding_positions: mx.array) -> mx.array:
        hidden_states = self._patchify(pixel_values)
        position_embeddings = self._position_embeddings(patch_positions,
                                                         padding_positions)
        return hidden_states + position_embeddings


class VisionPooler(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.default_output_length = config.default_output_length
        self.root_hidden_size = self.hidden_size**0.5

    def _avg_pool_by_positions(self, x, patch_positions, length):
        input_seq_len = x.shape[1]
        k = int((input_seq_len // length) ** 0.5)
        k_squared = k**2

        clamped = mx.clip(patch_positions, 0, None)
        max_x = mx.max(clamped[..., 0], axis=-1, keepdims=True) + 1
        kernel_idxs = mx.floor(clamped.astype(mx.float32) / k).astype(mx.int32)
        kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
        weights = one_hot(kernel_idxs, length).astype(mx.float32) / k_squared
        output = mx.einsum("bLl,bLd->bld", weights, x).astype(x.dtype)
        mask = mx.logical_not(mx.all(weights == 0, axis=1))
        return output, mask

    def __call__(self, hidden_states, patch_positions, padding_positions,
                 output_length=None):
        hidden_states = mx.where(mx.expand_dims(padding_positions, -1), 0.0,
                                 hidden_states)
        length = output_length or self.default_output_length
        if hidden_states.shape[1] == length:
            mask = mx.logical_not(padding_positions)  # True = valid, as pooled
        else:
            hidden_states, mask = self._avg_pool_by_positions(
                hidden_states, patch_positions, length)
        # knurlogic edit 6: scale in float32 and return float32, as HF's
        # Gemma4VisionPooler (modeling_gemma4.py:681-688): sqrt(hidden) can
        # push the activations past float16's range, and in bf16 the scale
        # itself rounds (sqrt(1152) = 33.94 -> 34.0); VisionModel
        # standardizes in float32 and casts back
        hidden_states = hidden_states.astype(mx.float32) * self.root_hidden_size
        return hidden_states, mask


class VisionTransformerModel(nn.Module):
    """Weight key path: `vision_tower.encoder.layers.*`."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layers = [VisionTransformerBlock(config)
                       for _ in range(config.num_hidden_layers)]

    def __call__(self, hidden_states: mx.array, positions: mx.array,
                 mask: mx.array) -> mx.array:
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions, mask)
        return hidden_states


class VisionModel(nn.Module):
    """Top-level tower. Weight keys: `vision_tower.patch_embedder.*`,
    `vision_tower.encoder.layers.*`, `vision_tower.std_bias`/`std_scale`.

    `__call__` takes exactly ONE image (`pixel_values` [1, C, H, W],
    channel-first, already resized to a multiple of `patch_size` by
    `Family.preprocess`) and returns [1, n_tokens, hidden_size], n_tokens =
    (H//patch)*(W//patch) // pooling_kernel_size**2 -- aspect-dependent, so
    `VisionSpec.fixed_tokens` stays None. This drops the
    list-of-images and externally-supplied-`pixel_position_ids` branches of
    mlx-vlm's version (see module docstring)."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.patch_size = config.patch_size
        self.pooling_kernel_size = config.pooling_kernel_size
        self.default_output_length = config.default_output_length
        self.max_patches = self.default_output_length * self.pooling_kernel_size**2

        self.patch_embedder = VisionPatchEmbedder(config)
        self.encoder = VisionTransformerModel(config)
        self.pooler = VisionPooler(config)

        if config.standardize:
            self.std_bias = mx.zeros((config.hidden_size,))
            self.std_scale = mx.ones((config.hidden_size,))

    def _patch_positions(self, H, W):
        pH, pW = H // self.patch_size, W // self.patch_size
        grid_x = np.arange(pW)
        grid_y = np.arange(pH)
        gx, gy = np.meshgrid(grid_x, grid_y, indexing="xy")
        real_positions = np.stack([gx.flatten(), gy.flatten()], axis=-1)
        return real_positions.astype(np.int32), pH * pW

    def __call__(self, pixel_values) -> mx.array:
        if not isinstance(pixel_values, mx.array):
            pixel_values = mx.array(pixel_values)
        B, C, H, W = pixel_values.shape
        pool_sq = self.pooling_kernel_size**2

        positions, num_real = self._patch_positions(H, W)
        output_length = num_real // pool_sq
        patch_positions = mx.array(np.tile(positions[None], (B, 1, 1)))
        padding_positions = mx.zeros((B, num_real), dtype=mx.bool_)

        inputs_embeds = self.patch_embedder(pixel_values, patch_positions,
                                            padding_positions)

        valid_mask = ~padding_positions
        attn_mask = mx.expand_dims(valid_mask, 1) * mx.expand_dims(valid_mask, 2)
        mask_fill = mx.array(-1e4, dtype=inputs_embeds.dtype)
        attn_mask = mx.where(attn_mask, mx.array(0.0, dtype=inputs_embeds.dtype),
                             mask_fill)
        attn_mask = mx.expand_dims(attn_mask, 1)

        hidden_states = self.encoder(inputs_embeds, patch_positions, attn_mask)

        pooled, pool_mask = self.pooler(hidden_states, patch_positions,
                                        padding_positions,
                                        output_length=output_length)

        all_real = []
        for i in range(B):
            n_valid = int(pool_mask[i].astype(mx.int32).sum().item())
            all_real.append(pooled[i, :n_valid])
        hidden_states = mx.concatenate(all_real, axis=0)[None]

        # knurlogic edit 6: standardize in float32 (the std_bias subtraction
        # cancels large values) and cast back to the working dtype, as HF's
        # Gemma4VisionModel.forward (modeling_gemma4.py:2056-2060)
        if self.config.standardize:
            hidden_states = (
                (hidden_states - self.std_bias.astype(mx.float32))
                * self.std_scale.astype(mx.float32))
        return hidden_states.astype(inputs_embeds.dtype)

    @staticmethod
    def sanitize(weights):
        return dict(weights)
