# Copyright © 2026 Apple Inc.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients
from mlx.utils import tree_map

from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, KVCache
from .gated_delta import compute_g, gated_delta_kernel, gated_delta_ops
from .pipeline import PipelineMixin
from .qwen3_next import Qwen3NextAttention
from .qwen3_next import Qwen3NextMLP
from .qwen3_next import Qwen3NextRMSNormGated as RMSNormGated
from .qwen3_next import Qwen3NextSparseMoeBlock


# --- knurlogic vendored edit 8: the reference's rounding order ----------------
# In bfloat16 torch computes sigmoid / silu in float32 and rounds the result
# once; MLX's bf16 kernels round inside (silu is x * sigmoid(x), two
# roundings), a bf16 step off on ~1/3 of the elements. These take the
# reference's order; on float32 inputs the casts are no-ops.


def _sigmoid(x: mx.array) -> mx.array:
    return mx.sigmoid(x.astype(mx.float32)).astype(x.dtype)


def _silu(x: mx.array) -> mx.array:
    return nn.silu(x.astype(mx.float32)).astype(x.dtype)


class _SwiGLU(nn.Module):
    """act_fn(gate) * up, the activation rounded once (SwitchGLU's call order)."""

    def __call__(self, x, gate):
        return _silu(gate) * x


class MLP(Qwen3NextMLP):
    def __call__(self, x) -> mx.array:
        return self.down_proj(_silu(self.gate_proj(x)) * self.up_proj(x))


class RMSNorm(nn.RMSNorm):
    """The reference norms in float32 times (1 + weight) in float32
    (Qwen3_5RMSNorm), rounding once. sanitize keeps a checkpoint's
    zero-centred weight + 1 in float32 so this can; a weight already folded
    in bf16 (an artifact converted before edit 8) runs as before."""

    def __call__(self, x):
        if self.weight.dtype == mx.float32 and x.dtype != mx.float32:
            return mx.fast.rms_norm(x.astype(mx.float32), self.weight,
                                    self.eps).astype(x.dtype)
        return mx.fast.rms_norm(x, self.weight, self.eps)


def gated_delta(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, *,
                use_kernel=True):
    """mlx-lm's gated_delta_update with the reference's rounding
    (modeling_qwen3_5 Qwen3_5GatedDeltaNet.forward): beta = sigmoid(b)
    rounded once to b's dtype, g from `a.float() + dt_bias` in float32
    (mlx-lm added a + dt_bias in bf16 before softplus)."""
    beta = _sigmoid(b)
    g = compute_g(A_log, a.astype(mx.float32), dt_bias)
    if state is None:
        B, _, Hk, Dk = q.shape
        Hv, Dv = v.shape[-2:]
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    if (
        not use_kernel
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
        or k.shape[-1] < 32
        or k.shape[-1] % 32 != 0
    ):
        return gated_delta_ops(q, k, v, g, beta, state, mask)
    return gated_delta_kernel(q, k, v, g, beta, state, mask)


class SparseMoeBlock(Qwen3NextSparseMoeBlock):
    """knurlogic vendored edit 7: Qwen3.5-MoE's reference router
    (modeling_qwen3_5_moe Qwen3_5MoeTopKRouter) always renormalizes the
    top-k probabilities; it has no norm_topk_prob and ignores the key.
    mlx-lm's Qwen3-Next block honours it, so a config saying false gave
    un-renormalized expert weights here only.

    knurlogic vendored edit 8: and in the reference's order: softmax of the
    logits in float32, top-k and renormalize in float32, then the weights
    rounded to the logits' dtype (mlx-lm rounded the probabilities to bf16
    first, and picked the top-k among the rounded values: a tie there can
    pick another expert); the experts' and the shared expert's activations
    rounded once."""

    def __init__(self, args):
        super().__init__(args)
        self.norm_topk_prob = True
        self.switch_mlp.activation = _SwiGLU()

    def __call__(self, x: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        logits = self.gate(x)
        probs = mx.softmax(logits.astype(mx.float32), axis=-1)
        k = self.top_k
        inds = mx.stop_gradient(mx.argpartition(probs, kth=-k, axis=-1)[..., -k:])
        scores = mx.take_along_axis(probs, inds, axis=-1)
        scores = (scores / scores.sum(axis=-1, keepdims=True)).astype(logits.dtype)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        se = self.shared_expert  # mlx-lm's MLP, its activation rounded once
        shared_y = se.down_proj(_silu(se.gate_proj(x)) * se.up_proj(x))
        shared_y = _sigmoid(self.shared_expert_gate(x)) * shared_y
        y = y + shared_y

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)

        return y


@dataclass
class TextModelArgs(BaseModelArgs):
    model_type: str = ""
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    num_key_value_heads: int = 8
    max_position_embeddings: int = 131072
    linear_num_value_heads: int = 64
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 192
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    head_dim: Optional[int] = None
    full_attention_interval: int = 4
    # knurlogic vendored edit 6: the layer types are the config's
    # layer_types when it has them (the reference reads nothing else);
    # full_attention_interval only derives them when it has not
    layer_types: Optional[List[str]] = None

    # MoE fields (optional, for Qwen3_5MoeForConditionalGeneration)
    num_experts: int = 0
    num_experts_per_tok: int = 0
    decoder_sparse_step: int = 1
    shared_expert_intermediate_size: int = 0
    moe_intermediate_size: int = 0
    norm_topk_prob: bool = True

    # Rope parameters
    # knurlogic vendored edit 6: absent, as the reference config: no
    # rope_parameters is rope_theta 10000.0 (default_theta), partial 0.25,
    # mrope_section [11, 11, 10] (the rotary embedding's default); a
    # top-level rope_theta / partial_rotary_factor fills a rope_parameters
    # that lacks it. The taken file defaulted to theta 100000.
    rope_parameters: Optional[Dict[str, Union[float, str, bool, List[int]]]] = None

    # Derived from rope_parameters (set in __post_init__)
    partial_rotary_factor: float = 0.25
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        if self.rope_parameters:
            if (
                "type" not in self.rope_parameters
                and "rope_type" in self.rope_parameters
            ):
                self.rope_parameters["type"] = self.rope_parameters.pop("rope_type")

            self.partial_rotary_factor = self.rope_parameters.get(
                "partial_rotary_factor", self.partial_rotary_factor
            )
            self.rope_theta = self.rope_parameters.get("rope_theta", self.rope_theta)
            self.rope_scaling = self.rope_parameters

        # knurlogic vendored edit 6 (Qwen3_5TextConfig.__post_init__)
        if self.layer_types is None:
            k = self.full_attention_interval
            self.layer_types = [
                "linear_attention" if (i + 1) % k else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        else:
            legacy = {"mamba": "linear_attention", "conv": "linear_attention",
                      "attention": "full_attention"}
            self.layer_types = [legacy.get(t, t) for t in self.layer_types]
        bad = sorted(set(self.layer_types) - {"linear_attention", "full_attention"})
        if bad:
            raise ValueError(f"unsupported qwen3_5 layer types: {bad}")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"qwen3_5 config has {len(self.layer_types)} layer_types for "
                f"{self.num_hidden_layers} layers"
            )


# knurlogic vendored edit 6: a key a config leaves out means the reference
# config's default (transformers 5.16.1 Qwen3_5TextConfig /
# Qwen3_5MoeTextConfig), which differ between the two, so they are filled
# per family before TextModelArgs sees the config. The taken file's
# dataclass defaults were neither (vocab 151936, head_dim hidden/heads, ...).
_LINEAR = dict(
    vocab_size=248320, max_position_embeddings=32768, rms_norm_eps=1e-6,
    head_dim=256, linear_conv_kernel_dim=4, linear_key_head_dim=128,
    linear_value_head_dim=128, linear_num_key_heads=16,
    linear_num_value_heads=32, tie_word_embeddings=False, attention_bias=False,
)
REFERENCE_DEFAULTS = {
    "qwen3_5": dict(
        _LINEAR, hidden_size=4096, intermediate_size=12288,
        num_hidden_layers=32, num_attention_heads=16, num_key_value_heads=4,
    ),
    "qwen3_5_moe": dict(
        _LINEAR, hidden_size=2048, num_hidden_layers=40,
        num_attention_heads=16, num_key_value_heads=2, num_experts=256,
        num_experts_per_tok=8, moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
    ),
}


def with_reference_defaults(text_config: dict, model_type: str) -> dict:
    family = (text_config.get("model_type") or model_type or "").removesuffix("_text")
    return {**REFERENCE_DEFAULTS.get(family, {}), **text_config}


# --- MRoPE (knurlogic, vision P1) ---------------------------------------------
# Qwen's positions are 3-D (t, h, w) wherever an image sits in the key; the
# text path this file was vendored with knows only 1-D offsets. Two ways in,
# both optional; with neither, every call below reaches the vendored code
# unchanged (tests/goldens/qwen_g5_text.npz holds that to main's numbers):
#
#   position_ids [3, B, L]  explicit positions of these L tokens: a prefill
#                           chunk holding image tokens, sliced from
#                           Family.positions over the whole key (design D4)
#   rope_delta   [B] | int  text after an image: every axis is offset + delta
#                           (mlx-vlm qwen3_5/language.py:2022-2052). Three
#                           equal axes ARE 1-D rope at a shifted offset, so it
#                           runs through the same fast kernel as text; only
#                           image chunks take the explicit path.
#
# The interleave is mlx-vlm 0.6.17 rope_utils.py:512-517 (MIT, Copyright (c)
# 2025 Prince Canuma), with its non-kernel apply (:655-690): frequency i
# takes axis 1 (h) when i % 3 == 1 and i < 3 * section[1], axis 2 (w) when
# i % 3 == 2 and i < 3 * section[2], else axis 0 (t). Half-split pairing, as
# nn.RoPE with traditional=False. qwen4_exp.py applies the same selector to
# its own rope helper (it may not import this file: engine/arch.py gives it
# no dependency on qwen3_5).


def mrope_selector(mrope_section, freq_dim: int) -> list:
    sel = [0] * freq_dim
    for axis in (1, 2):
        for i in range(axis, min(mrope_section[axis] * 3, freq_dim), 3):
            sel[i] = axis
    return sel


def apply_mrope(x: mx.array, position_ids: mx.array, dims: int, base: float,
                mrope_section, rope=None) -> mx.array:
    """x [B, H, L, D], position_ids [3, B, L]: rotate x[..., :dims], each
    frequency's angle from its section's axis. float32 angles and rotation
    (as mlx-vlm's compute_dtype), cast back to x's dtype. `rope`: the
    layer's own; a YarnRoPE (KNURLOGIC_LONG_CONTEXT=yarn) lends its
    frequencies and mscale so an image chunk ropes as its text does."""
    half = dims // 2
    freqs = getattr(rope, "_freqs", None)
    inv_freq = (1.0 / freqs if freqs is not None else
                base ** (-mx.arange(0, dims, 2, dtype=mx.float32) / dims))
    mscale = float(getattr(rope, "mscale", 1.0) or 1.0)
    if mscale != 1.0:
        x = mx.concatenate([x[..., :dims] * mscale, x[..., dims:]], axis=-1)
    sel = mx.array(mrope_selector(mrope_section, half), dtype=mx.int32)
    pos = mx.take(position_ids, sel, axis=0)                  # [half, B, L]
    angle = pos.transpose(1, 2, 0).astype(mx.float32) * inv_freq
    cos = mx.concatenate([mx.cos(angle)] * 2, axis=-1)[:, None]
    sin = mx.concatenate([mx.sin(angle)] * 2, axis=-1)[:, None]
    xr = x[..., :dims].astype(mx.float32)
    rot = mx.concatenate([-xr[..., half:], xr[..., :half]], axis=-1)
    xr = (xr * cos + rot * sin).astype(x.dtype)
    if x.shape[-1] == dims:
        return xr
    return mx.concatenate([xr, x[..., dims:]], axis=-1)


class Attention(Qwen3NextAttention):
    """mlx-lm's Qwen3NextAttention plus the two optional position inputs.
    knurlogic vendored edit 8: the text path runs this __call__ too (with
    neither input it ropes at the cache offset, as the parent), for the
    reference's rounding of the output gate and the q/k norms."""

    def __init__(self, args):
        super().__init__(args)
        rp = args.rope_parameters or {}
        self.mrope_section = list(rp.get("mrope_section", [11, 11, 10]))
        self.rotary_dims = int(self.head_dim * args.partial_rotary_factor)
        self.rope_base = args.rope_theta
        self.q_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_ids: Optional[mx.array] = None,
        rope_delta: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        q_proj_output = self.q_proj(x)
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)

        keys, values = self.k_proj(x), self.v_proj(x)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        if position_ids is not None:
            queries = apply_mrope(queries, position_ids, self.rotary_dims,
                                  self.rope_base, self.mrope_section,
                                  self.rope)
            keys = apply_mrope(keys, position_ids, self.rotary_dims,
                               self.rope_base, self.mrope_section, self.rope)
        elif rope_delta is not None:
            shifted = (cache.offset if cache is not None else 0) + rope_delta
            queries = self.rope(queries, offset=shifted)
            keys = self.rope(keys, offset=shifted)
        elif cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output * _sigmoid(gate))


class GatedDeltaNet(nn.Module):
    def __init__(self, config: TextModelArgs):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                f"num_v_heads ({self.num_v_heads}) must be divisible by num_k_heads ({self.num_k_heads})"
            )

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_norm_epsilon = config.rms_norm_eps

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )

        self.in_proj_qkv = nn.Linear(
            self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False
        )
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        self.dt_bias = mx.ones(self.num_v_heads)

        A = mx.random.uniform(low=0, high=16, shape=(self.num_v_heads,))
        self.A_log = mx.log(A)

        self.norm = RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.sharding_group = None

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, S, _ = inputs.shape

        if self.sharding_group is not None:
            inputs = sum_gradients(self.sharding_group)(inputs)

        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )

        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        conv_out = _silu(self.conv1d(conv_input))  # knurlogic vendored edit 8

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]

        state = cache[1] if cache else None
        inv_scale = k.shape[-1] ** -0.5
        # knurlogic vendored edit 1: the reference's l2norm adds its eps
        # (1e-6) to sum(x^2); rms_norm adds it to mean(x^2), so scale it
        # by 1/dk (mlx-lm's own normalize_qk does the same).
        l2_eps = 1e-6 * inv_scale**2
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, l2_eps)
        k = inv_scale * mx.fast.rms_norm(k, None, l2_eps)

        # knurlogic vendored edit 8: beta and g in the reference's precision
        out, state = gated_delta(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=not self.training,
        )

        if cache is not None:
            cache[1] = state
            cache.advance(S)

        out = self.norm(out, z)
        out = self.out_proj(out.reshape(B, S, -1))

        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)

        return out


class DecoderLayer(nn.Module):
    def __init__(self, args: TextModelArgs, layer_idx: int):
        super().__init__()
        # knurlogic vendored edit 6: the config's layer_types
        self.is_linear = args.layer_types[layer_idx] == "linear_attention"
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = Attention(args)

        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

        if args.num_experts > 0:
            self.mlp = SparseMoeBlock(args)
        else:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_ids: Optional[mx.array] = None,
        rope_delta: Optional[Any] = None,
    ) -> mx.array:
        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(x), mask, cache)
        elif position_ids is None and rope_delta is None:
            r = self.self_attn(self.input_layernorm(x), mask, cache)
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache,
                               position_ids=position_ids,
                               rope_delta=rope_delta)
        h = x + r
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class Qwen3_5TextModel(PipelineMixin, nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args=args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        # knurlogic vendored edit 6: the first layer of each kind
        lt = args.layer_types
        self.ssm_idx = lt.index("linear_attention") if "linear_attention" in lt else None
        self.fa_idx = lt.index("full_attention") if "full_attention" in lt else None

    def pipeline(self, group):
        super().pipeline(group)
        self.ssm_idx = None
        self.fa_idx = None
        for e, l in enumerate(self.pipeline_layers):
            if self.ssm_idx is None and l.is_linear:
                self.ssm_idx = e
            elif self.fa_idx is None and not l.is_linear:
                self.fa_idx = e
            if self.ssm_idx is not None and self.fa_idx is not None:
                break

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        rope_delta: Optional[Any] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            hidden_states = input_embeddings
        else:
            hidden_states = self.embed_tokens(inputs)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * len(self.pipeline_layers)

        fa_mask = None
        ssm_mask = None
        if self.fa_idx is not None:
            fa_mask = create_attention_mask(hidden_states, cache[self.fa_idx])
        if self.ssm_idx is not None:
            ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])

        # Receive from the previous process in the pipeline
        if pipeline_rank < pipeline_size - 1:
            hidden_states = mx.distributed.recv_like(hidden_states, (pipeline_rank + 1))

        if position_ids is None and rope_delta is None:
            for layer, c in zip(self.pipeline_layers, cache):
                mask = ssm_mask if layer.is_linear else fa_mask
                hidden_states = layer(hidden_states, mask=mask, cache=c)
        else:
            for layer, c in zip(self.pipeline_layers, cache):
                mask = ssm_mask if layer.is_linear else fa_mask
                hidden_states = layer(hidden_states, mask=mask, cache=c,
                                      position_ids=position_ids,
                                      rope_delta=rope_delta)

        # Send to the next process in the pipeline
        if pipeline_rank != 0:
            hidden_states = mx.distributed.send(
                hidden_states, (pipeline_rank - 1) % pipeline_size
            )
            if cache[-1] is not None:
                if hasattr(cache[-1], "keys"):
                    cache[-1].keys = mx.depends(cache[-1].keys, hidden_states)
                else:
                    cache[-1][0] = mx.depends(cache[-1][0], hidden_states)

        # Broadcast h while keeping it in the graph
        if pipeline_size > 1:
            hidden_states = mx.distributed.all_gather(hidden_states)[
                : hidden_states.shape[0]
            ]

        return self.norm(hidden_states)


class TextModel(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen3_5TextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        rope_delta: Optional[Any] = None,
    ) -> mx.array:
        if position_ids is None and rope_delta is None:
            out = self.model(inputs, cache, input_embeddings=input_embeddings)
        else:
            out = self.model(inputs, cache, input_embeddings=input_embeddings,
                             position_ids=position_ids, rope_delta=rope_delta)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    @property
    def layers(self):
        return self.model.pipeline_layers

    def make_cache(self):
        return [ArraysCache(size=2) if l.is_linear else KVCache() for l in self.layers]

    def sanitize(self, weights):
        has_unsanitized_conv1d = any(
            "conv1d.weight" in k and v.shape[-1] != 1 for k, v in weights.items()
        )
        weights = {k: v for k, v in weights.items() if "mtp." not in k}

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        norm_keys = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        )
        for k, v in weights.items():
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
            if has_unsanitized_conv1d and any(k.endswith(sfx) for sfx in norm_keys):
                if v.ndim == 1:
                    # knurlogic vendored edit 8: 1 + weight in float32, as
                    # the reference (Qwen3_5RMSNorm); in bf16 the sum
                    # rounded every scale (see RMSNorm above)
                    weights[k] = v.astype(mx.float32) + 1.0
        return weights

    @property
    def quant_predicate(self):
        if self.args.num_experts <= 0:
            return None

        def predicate(path, _):
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self):
        def predicate(path: str):
            if path.endswith("A_log"):
                return False
            # knurlogic vendored edit 8: the folded 1 + weight stays float32
            if path.endswith(("layernorm.weight", "model.norm.weight",
                              "q_norm.weight", "k_norm.weight")):
                return False
            return True

        return predicate


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict

    @classmethod
    def from_dict(cls, params):
        if "text_config" not in params:
            return cls(model_type=params["model_type"], text_config=params)
        return super().from_dict(params)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        # knurlogic vendored edit 6: absent keys mean the reference's defaults
        self.language_model = TextModel(TextModelArgs.from_dict(
            with_reference_defaults(args.text_config, args.model_type)))

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        rope_delta: Optional[Any] = None,
    ):
        """position_ids [3, B, L] and rope_delta ([B] array or int) are the
        vision path's (MRoPE, above); a text call passes neither and runs the
        vendored code unchanged. position_ids wins when both are given."""
        if position_ids is None and rope_delta is None:
            return self.language_model(
                inputs, cache=cache, input_embeddings=input_embeddings
            )
        if position_ids is not None:
            rope_delta = None
        return self.language_model(
            inputs, cache=cache, input_embeddings=input_embeddings,
            position_ids=position_ids, rope_delta=rope_delta,
        )

    @property
    def model(self):
        return self.language_model.model

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            if key.startswith("vision_tower") or key.startswith("model.visual"):
                continue
            if key.startswith("model.visual"):
                continue
            if key.startswith("model.language_model"):
                key = key.replace("model.language_model", "language_model.model")
            elif key.startswith("language_model."):
                pass
            else:
                key = "language_model." + key
            sanitized[key] = value
        return self.language_model.sanitize(sanitized)

    def shard(self, group=None):
        group = group or mx.distributed.init()
        N = group.size()
        rank = group.rank()

        # A sharding factory for the convolution in gated delta net
        def conv_sharding(key_dim):
            return lambda p, w: (0, [key_dim, 2 * key_dim])

        def repeat_kv_layer_inplace(layer, h):
            # No repeat needed cause we have more heads than nodes
            if N <= h:
                return

            # Repeat function to apply to the layer weights
            def _repeat(p):
                s = p.shape
                p = p.reshape(h, s[0] // h, *s[1:])
                p = mx.repeat(p, N // h, axis=0)
                p = p.reshape(-1, *s[1:])
                return p

            layer.update(tree_map(_repeat, layer.parameters()))

        for layer in self.layers:
            # Linear attention
            if layer.is_linear:
                kd = layer.linear_attn.key_dim
                layer.linear_attn.sharding_group = group
                shard_inplace(layer.linear_attn.conv1d, conv_sharding(kd), group=group)
                layer.linear_attn.conv1d.groups //= N
                shard_inplace(
                    layer.linear_attn.in_proj_qkv,
                    "all-to-sharded",
                    segments=[kd, 2 * kd],
                    group=group,
                )
                shard_inplace(
                    layer.linear_attn.in_proj_z, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.linear_attn.in_proj_b, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.linear_attn.in_proj_a, "all-to-sharded", group=group
                )
                layer.linear_attn.dt_bias = mx.contiguous(
                    mx.split(layer.linear_attn.dt_bias, N)[rank]
                )
                layer.linear_attn.A_log = mx.contiguous(
                    mx.split(layer.linear_attn.A_log, N)[rank]
                )
                shard_inplace(layer.linear_attn.out_proj, "sharded-to-all", group=group)
                layer.linear_attn.num_k_heads //= N
                layer.linear_attn.num_v_heads //= N
                layer.linear_attn.key_dim //= N
                layer.linear_attn.value_dim //= N
                layer.linear_attn.conv_dim //= N

            # Softmax attention
            else:
                layer.self_attn.o_proj = shard_linear(
                    layer.self_attn.o_proj, "sharded-to-all", group=group
                )
                layer.self_attn.q_proj = shard_linear(
                    layer.self_attn.q_proj, "all-to-sharded", group=group
                )
                repeat_kv_layer_inplace(
                    layer.self_attn.k_proj, layer.self_attn.num_key_value_heads
                )
                repeat_kv_layer_inplace(
                    layer.self_attn.v_proj, layer.self_attn.num_key_value_heads
                )
                layer.self_attn.k_proj = shard_linear(
                    layer.self_attn.k_proj, "all-to-sharded", group=group
                )
                layer.self_attn.v_proj = shard_linear(
                    layer.self_attn.v_proj, "all-to-sharded", group=group
                )
                layer.self_attn.num_attention_heads //= N
                layer.self_attn.num_key_value_heads = max(
                    1, layer.self_attn.num_key_value_heads // N
                )

            # MLP
            if isinstance(layer.mlp, MLP):
                layer.mlp.gate_proj = shard_linear(
                    layer.mlp.gate_proj, "all-to-sharded", group=group
                )
                layer.mlp.down_proj = shard_linear(
                    layer.mlp.down_proj, "sharded-to-all", group=group
                )
                layer.mlp.up_proj = shard_linear(
                    layer.mlp.up_proj, "all-to-sharded", group=group
                )

            # MoE
            else:
                layer.mlp.sharding_group = group
                shard_inplace(
                    layer.mlp.shared_expert.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.shared_expert.down_proj, "sharded-to-all", group=group
                )
                shard_inplace(
                    layer.mlp.shared_expert.up_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.down_proj, "sharded-to-all", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.up_proj, "all-to-sharded", group=group
                )

    @property
    def layers(self):
        return self.language_model.model.pipeline_layers

    def make_cache(self):
        return self.language_model.make_cache()

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate
