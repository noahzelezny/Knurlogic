# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache, _BaseCache
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    # knurlogic edit 2: every default is HF transformers 5.16.1
    # configuration_gemma4.py Gemma4TextConfig's (the maker's port), so a
    # config that leaves a key out builds the model the reference builds.
    # mlx-lm 0.32.0 defaulted to an e2b-like shape (hidden 1536, 35 layers,
    # 1 kv head, 20 KV-shared layers, double-wide MLP, softcap 30, a 4:1
    # sliding pattern) -- see PROVENANCE.md.
    model_type: str = "gemma4_text"
    hidden_size: int = 2304
    num_hidden_layers: int = 30
    intermediate_size: int = 9216
    num_attention_heads: int = 8
    head_dim: int = 256
    global_head_dim: int = 512
    global_partial_rotary_factor: float = 0.25
    rms_norm_eps: float = 1e-6
    vocab_size: int = 262144
    vocab_size_per_layer_input: int = 262144
    num_key_value_heads: int = 4
    num_global_key_value_heads: Optional[int] = None
    num_kv_shared_layers: int = 0
    pad_token_id: int = 0
    hidden_size_per_layer_input: int = 256
    rope_traditional: bool = False
    partial_rotary_factor: float = 1.0
    rope_parameters: Optional[Dict] = None
    sliding_window: int = 512
    max_position_embeddings: int = 131072
    attention_k_eq_v: bool = False
    final_logit_softcapping: Optional[float] = None
    use_double_wide_mlp: bool = False
    enable_moe_block: bool = False
    num_experts: Optional[int] = None
    top_k_experts: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
    layer_types: Optional[List[str]] = None
    tie_word_embeddings: bool = True
    # knurlogic (vendored edit 1): "vision" (26B-A4B, 31B) overlays
    # bidirectional image blocks on the sliding layers; None (e2b/e4b)
    # keeps every layer causal
    use_bidirectional_attention: Optional[str] = None
    # knurlogic edit 2: read so a config this file cannot run is refused,
    # not run as another model (the MLP is gelu_pytorch_tanh, no biases)
    hidden_activation: str = "gelu_pytorch_tanh"
    attention_bias: bool = False

    def __post_init__(self):
        if self.hidden_activation != "gelu_pytorch_tanh":
            raise ValueError(
                f"gemma4_text: hidden_activation {self.hidden_activation!r} "
                "is not supported (this build runs gelu_pytorch_tanh, the "
                "released models' activation)")
        if self.attention_bias:
            raise ValueError(
                "gemma4_text: attention_bias true is not supported (this "
                "build's attention projections have no biases, as the "
                "released models')")
        if self.use_bidirectional_attention not in (None, "vision", "all"):
            raise ValueError(
                "gemma4_text: use_bidirectional_attention must be None, "
                f"'vision' or 'all', not {self.use_bidirectional_attention!r}")
        if self.use_bidirectional_attention == "all":
            # knurlogic edit 4: HF's config halves the window for "all"
            # (configuration_gemma4.py: `sliding_window // 2 + 1`); the
            # sliding layers then attend |q - k| <= that window both ways
            self.sliding_window = self.sliding_window // 2 + 1
        if self.rope_parameters is None:
            self.rope_parameters = {
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                },
                "full_attention": {
                    "rope_type": "proportional",
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0,
                },
            }
        if self.layer_types is None:
            # HF: 5 sliding : 1 full
            self.layer_types = [
                "sliding_attention" if (i + 1) % 6 else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        else:
            self.layer_types = list(self.layer_types)
        # HF forces the last layer to full attention, whatever the config
        # says
        if self.layer_types:
            self.layer_types[-1] = "full_attention"


class RMSNormNoScale(nn.Module):
    """RMSNorm without learnable scale."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, None, self.eps)


@partial(mx.compile, shapeless=True)
def logit_softcap(softcap, x):
    return mx.tanh(x / softcap) * softcap


@partial(mx.compile, shapeless=True)
def _complete_square(x2, y2, xy):
    return x2 + mx.expand_dims(y2, -1) - 2 * xy


@partial(mx.compile, shapeless=True)
def geglu(gate, x):
    # knurlogic edit 5: the gelu in float32, rounded once, as torch's
    # gelu_pytorch_tanh does on a bf16 tensor (HF's act_fn); mlx's
    # gelu_approx on bf16 rounds after every step. Fused, so free.
    return nn.gelu_approx(gate.astype(mx.float32)).astype(x.dtype) * x


class MLP(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int = 0):
        super().__init__()
        first_kv_shared_layer_idx = (
            config.num_hidden_layers - config.num_kv_shared_layers
        )
        is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        use_double_wide = config.use_double_wide_mlp and is_kv_shared_layer
        intermediate_size = config.intermediate_size * (2 if use_double_wide else 1)

        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(geglu(self.gate_proj(x), self.up_proj(x)))


class Router(nn.Module):
    """Expert router: norm -> scale -> project -> top-k -> renormalize."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.eps = config.rms_norm_eps
        self.proj = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.scale = mx.ones((config.hidden_size,))
        self.per_expert_scale = mx.ones((config.num_experts,))
        self._root_size = config.hidden_size**-0.5

    def __call__(self, x: mx.array):
        x = mx.fast.rms_norm(x, self.scale * self._root_size, self.eps)

        expert_scores = self.proj(x)

        top_k_indices = mx.argpartition(
            expert_scores, kth=-self.config.top_k_experts, axis=-1
        )
        top_k_indices = top_k_indices[..., -self.config.top_k_experts :]

        top_k_weights = mx.take_along_axis(expert_scores, top_k_indices, axis=-1)
        top_k_weights = mx.softmax(top_k_weights, axis=-1)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_indices]

        return top_k_indices, top_k_weights


class GeGLU(nn.Module):
    """GELU-gated linear unit activation for SwitchGLU."""

    def __call__(self, x, gate):
        return geglu(gate, x)


class Experts(nn.Module):
    """Sparse MoE using SwitchGLU with gather_mm."""

    def __init__(self, config: ModelArgs):
        super().__init__()

        self.switch_glu = SwitchGLU(
            input_dims=config.hidden_size,
            hidden_dims=config.moe_intermediate_size,
            num_experts=config.num_experts,
            activation=GeGLU(),
            bias=False,
        )

    def __call__(
        self, x: mx.array, top_k_indices: mx.array, top_k_weights: mx.array
    ) -> mx.array:
        w = mx.expand_dims(top_k_weights, -1)
        y = self.switch_glu(x, top_k_indices)

        return (w * y).sum(-2)


class Attention(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"
        self.has_kv = layer_idx < config.num_hidden_layers - config.num_kv_shared_layers

        self.head_dim = (
            config.global_head_dim
            if self.layer_type == "full_attention"
            and hasattr(config, "global_head_dim")
            and config.global_head_dim
            else config.head_dim
        )

        dim = config.hidden_size
        self.n_heads = config.num_attention_heads

        # K-eq-V for full attention layers (26B/31B models)
        self.use_k_eq_v = config.attention_k_eq_v and not self.is_sliding
        if self.use_k_eq_v and config.num_global_key_value_heads is not None:
            self.n_kv_heads = config.num_global_key_value_heads
        else:
            self.n_kv_heads = config.num_key_value_heads

        self.scale = 1.0

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        if self.has_kv:
            self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
            if not self.use_k_eq_v:
                self.v_proj = nn.Linear(
                    dim, self.n_kv_heads * self.head_dim, bias=False
                )
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        if self.has_kv:
            self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.v_norm = RMSNormNoScale(self.head_dim, eps=config.rms_norm_eps)

        # RoPE (with partial rotation support)
        layer_key = "sliding_attention" if self.is_sliding else "full_attention"
        rope_params = config.rope_parameters.get(layer_key, {})
        rope_theta = rope_params.get("rope_theta", 10000.0)
        self.rope = initialize_rope(
            dims=self.head_dim,
            traditional=config.rope_traditional,
            base=rope_theta,
            scaling_config=rope_params,
            max_position_embeddings=config.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        shared_kv: Optional[tuple] = None,
        offset: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        queries = self.q_norm(queries)

        if shared_kv is not None:
            keys, values = shared_kv
        elif not self.has_kv:
            raise ValueError(
                f"Layer {self.layer_idx} is a KV-shared layer but received no shared_kv"
            )
        else:
            keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
            values = keys
            if not self.use_k_eq_v:
                values = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

            offset = mx.array(cache.offset) if cache is not None else 0

            keys = self.k_norm(keys)
            keys = keys.transpose(0, 2, 1, 3)
            keys = self.rope(keys, offset=offset)

            values = self.v_norm(values)
            values = values.transpose(0, 2, 1, 3)

        queries = queries.transpose(0, 2, 1, 3)
        queries = self.rope(queries, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output), (keys, values), offset


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config, layer_idx)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # MoE (26B model)
        self.enable_moe = config.enable_moe_block
        if self.enable_moe:
            self.router = Router(config)
            self.experts = Experts(config)
            self.post_feedforward_layernorm_1 = nn.RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_feedforward_layernorm_2 = nn.RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.pre_feedforward_layernorm_2 = nn.RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )

        # Per-layer input gating (2B/4B models)
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.per_layer_input_gate = nn.Linear(
                config.hidden_size, self.hidden_size_per_layer_input, bias=False
            )
            self.per_layer_projection = nn.Linear(
                self.hidden_size_per_layer_input, config.hidden_size, bias=False
            )
            self.post_per_layer_input_norm = nn.RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.per_layer_input_gate = None
            self.per_layer_projection = None
            self.post_per_layer_input_norm = None

        # Layer scalar
        self.layer_scalar = mx.ones((1,))

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        per_layer_input: Optional[mx.array] = None,
        shared_kv: Optional[tuple] = None,
        offset: Optional[Any] = None,
    ) -> mx.array:
        residual = x

        h = self.input_layernorm(x)
        h, shared_kv, offset = self.self_attn(
            h, mask, cache, shared_kv=shared_kv, offset=offset
        )
        h = self.post_attention_layernorm(h)
        h = residual + h

        residual = h

        if self.enable_moe:
            h1 = self.pre_feedforward_layernorm(h)
            h1 = self.mlp(h1)
            h1 = self.post_feedforward_layernorm_1(h1)

            top_k_indices, top_k_weights = self.router(h)
            h2 = self.pre_feedforward_layernorm_2(h)
            h2 = self.experts(h2, top_k_indices, top_k_weights)
            h2 = self.post_feedforward_layernorm_2(h2)

            h = h1 + h2
        else:
            h = self.pre_feedforward_layernorm(h)
            h = self.mlp(h)

        h = self.post_feedforward_layernorm(h)
        h = residual + h

        # Per-layer input gating
        if (
            self.per_layer_input_gate is not None
            and self.per_layer_projection is not None
            and self.post_per_layer_input_norm is not None
            and per_layer_input is not None
        ):
            residual = h
            gate = self.per_layer_input_gate(h)
            # knurlogic edit 5: act_fn(gate) * per_layer_input, rounded
            # where HF rounds (geglu)
            gate = geglu(gate, per_layer_input)
            gate = self.per_layer_projection(gate)
            gate = self.post_per_layer_input_norm(gate)
            h = residual + gate

        if self.layer_scalar is not None:
            h = h * self.layer_scalar

        return h, shared_kv, offset


class Gemma4TextModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.window_size = config.sliding_window
        self.num_hidden_layers = config.num_hidden_layers

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_scale = config.hidden_size**0.5
        self.layers = [
            DecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Per-layer input embeddings (2B/4B models)
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.embed_tokens_per_layer = nn.Embedding(
                config.vocab_size_per_layer_input,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
            )
            self.embed_tokens_per_layer_scale = config.hidden_size_per_layer_input**0.5
            self.per_layer_input_scale = 2.0**-0.5
            self.per_layer_projection_scale = config.hidden_size**-0.5
            self.per_layer_model_projection = nn.Linear(
                config.hidden_size,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                bias=False,
            )
            self.per_layer_projection_norm = nn.RMSNorm(
                config.hidden_size_per_layer_input, eps=config.rms_norm_eps
            )
        else:
            self.embed_tokens_per_layer = None
            self.per_layer_input_scale = None
            self.per_layer_projection_scale = None
            self.per_layer_model_projection = None
            self.per_layer_projection_norm = None

        # Arrange for shared KVs
        self.previous_kvs = list(range(len(self.layers)))
        if config.num_kv_shared_layers > 0:
            N = len(self.layers)
            M = N - config.num_kv_shared_layers
            kvs_by_type = {}
            for i in range(M):
                kvs_by_type[self.layers[i].layer_type] = i
            for j in range(M, N):
                self.previous_kvs[j] = kvs_by_type[self.layers[j].layer_type]

    def _get_per_layer_inputs(
        self,
        input_ids: Optional[mx.array],
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_ids is None:
            if input_embeddings is None:
                raise RuntimeError(
                    "input_embeddings must be provided when input_ids are omitted."
                )

            # Split the sequence dimension if this still holds too much
            # memory. 260k vocab means the distance tensor would be ~1GB
            # per 2k tokens in bf16.
            #
            # If the embedding is quantized we have to dequantize it anyway to
            # perform the match test.
            norms_embedding = self.embed_tokens.weight.square().sum(-1)
            norms_input = input_embeddings.square().sum(-1)
            distance = _complete_square(
                norms_embedding,
                norms_input,
                self.embed_tokens.as_linear(input_embeddings),
            )

            # Checks can be added if needed but they necessarily break the GPU
            # pipelining and force an eval.
            #
            #   match_counts = (distance < eps).sum(-1)
            #
            input_ids = mx.argmin(distance, -1)

        result = self.embed_tokens_per_layer(input_ids)
        result = result * self.embed_tokens_per_layer_scale
        return mx.unflatten(
            result,
            -1,
            (self.config.num_hidden_layers, self.hidden_size_per_layer_input),
        )

    def _project_per_layer_inputs(
        self,
        input_embeddings: mx.array,
        per_layer_inputs: Optional[mx.array] = None,
    ) -> mx.array:
        per_layer_projection = self.per_layer_model_projection(input_embeddings)
        per_layer_projection = per_layer_projection * self.per_layer_projection_scale
        per_layer_projection = mx.unflatten(
            per_layer_projection,
            -1,
            (self.config.num_hidden_layers, self.hidden_size_per_layer_input),
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

        if per_layer_inputs is None:
            return per_layer_projection

        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale

    def _make_masks(self, h, cache, mm_mask: Optional[mx.array] = None):
        """Per-layer-type masks, with gemma4's bidirectional image-block
        overlay where the maker puts it.

        `mm_mask` is [B, L]: -1 outside every image, and for image tokens the
        index (0, 1, 2, ...) of the image that token belongs to along the
        sequence -- so two images back to back still get separate blocks.
        `knurlogic.engine.vision.key.image_spans` already gives that
        grouping exactly (it knows sha/proc_hash, not just "is an image
        token"), so this is the block-id array directly rather than a
        recomputation from a token-type array. `Family.chunk_boundaries`
        (design D5) keeps every such block inside one prefill chunk, so the
        overlay never needs to reach across a chunk boundary.

        Vendored edit 1 (PROVENANCE.md): the overlay goes on the SLIDING
        layers only, as AND(window, OR(causal, same block)), and only when
        the config says `use_bidirectional_attention == "vision"` (26B-A4B,
        31B); full-attention layers stay causal, and e2b/e4b (None) are
        causal everywhere. That is Google's reference
        (google-deepmind/gemma `gm/nn/gemma4/_config.py`: "bidirectional
        for image tokens in sliding layers only, causal for global layers")
        and HF transformers 5.16.1 `modeling_gemma4.py:2101-2145`
        (`create_causal_mask_mapping`). The overlay was first ported from
        mlx-vlm 0.6.17 `gemma4/language.py:486-515`
        (`_apply_blockwise_bidirectional_overlay`), MIT, Copyright (c) 2025
        Prince Canuma, which put it on the full layers, ungated."""
        if self.config.use_bidirectional_attention == "all":
            return self._make_masks_all(h, cache)
        use_bidirectional = (
            self.config.use_bidirectional_attention == "vision"
            and mm_mask is not None
            and h.shape[1] > 1
            and bool((mm_mask >= 0).sum().item())
        )
        mask = {}
        masks = []
        for l, c in zip(self.layers, cache):
            if l.layer_type not in mask:
                if l.layer_type == "full_attention":
                    mask["full_attention"] = create_attention_mask(h, c)
                elif l.layer_type == "sliding_attention":
                    m = create_attention_mask(
                        h, c, window_size=self.window_size,
                        return_array=use_bidirectional)
                    if use_bidirectional and not isinstance(m, str):
                        # m is [L, S]: S - L tokens already in the cache
                        # come first. A prefill chunk may start after
                        # them (the serve path chunks, snapping edges to
                        # image blocks, so a block never reaches back into
                        # the cache): they belong to no block here.
                        L = mm_mask.shape[-1]
                        ctx = m.shape[-1] - L
                        kid = mm_mask
                        if ctx > 0:
                            kid = mx.concatenate(
                                [mx.full((mm_mask.shape[0], ctx), -1,
                                         dtype=mm_mask.dtype), mm_mask],
                                axis=-1)
                        q = mx.expand_dims(mm_mask, -1)
                        k = mx.expand_dims(kid, -2)
                        same_block = (q >= 0) & (q == k)       # [B, L, S]
                        # ...still inside the window: a key more than
                        # window-1 behind the query stays out
                        qpos = mx.arange(L)[:, None]
                        kpos = mx.arange(m.shape[-1])[None] - ctx
                        same_block = same_block & (
                            qpos - kpos < self.window_size)
                        if m.ndim == 4:                        # [B, 1, L, S]
                            same_block = same_block[:, None]
                        m = m | same_block
                    mask["sliding_attention"] = m
            masks.append(mask[l.layer_type])
        return masks

    def _make_masks_all(self, h, cache):
        """knurlogic edit 4: use_bidirectional_attention "all" -- no layer
        is causal. HF transformers 5.16.1: the config sets is_causal False
        (configuration_gemma4.py `__post_init__`), so `create_causal_mask`
        builds a bidirectional mask for the full layers and
        `create_sliding_window_causal_mask` a bidirectional sliding one,
        `abs(q - k) <= sliding_window` (masking_utils.py
        `sliding_window_bidirectional_overlay`), the window already halved
        by ModelArgs.

        Built as the causal mask (which carries the cache's context and any
        batch padding) OR the keys after the query in this chunk -- within
        the window on the sliding layers. The cache keeps what the
        reference's keeps (the sliding layers' RotatingKVCache holds window
        - 1 earlier tokens at a prefill, the window at a decode step, as
        HF's DynamicSlidingWindowLayer), so a chunked prefill and the
        cached decode see what the reference's cache lets them see. A
        decode step has no later key: the plain causal mask is the
        bidirectional one."""
        L = h.shape[1]
        w = self.window_size
        mask = {}
        masks = []
        for l, c in zip(self.layers, cache):
            t = l.layer_type
            if t not in mask:
                sliding = t == "sliding_attention"
                if L == 1:
                    mask[t] = create_attention_mask(
                        h, c, window_size=w if sliding else None)
                else:
                    # causal part: q - k <= w on the sliding layers
                    m = create_attention_mask(
                        h, c, window_size=w + 1 if sliding else None,
                        return_array=True)
                    ctx = m.shape[-1] - L
                    qpos = mx.arange(L)[:, None]
                    kpos = mx.arange(m.shape[-1])[None] - ctx
                    later = kpos > qpos
                    if sliding:
                        later = later & (kpos - qpos <= w)
                    mask[t] = m | later
            masks.append(mask[t])
        return masks

    def __call__(
        self,
        inputs: mx.array = None,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        per_layer_inputs: Optional[mx.array] = None,
        mm_mask: Optional[mx.array] = None,
    ):
        # Make the initial hidden state
        if input_embeddings is None:
            input_embeddings = self.embed_tokens(inputs)
        h = input_embeddings
        h = h * self.embed_scale

        # Get the extra inputs per layer if we have per layer embeddings
        if self.hidden_size_per_layer_input:
            if per_layer_inputs is None:
                per_layer_inputs = self._get_per_layer_inputs(inputs, input_embeddings)
            per_layer_inputs = self._project_per_layer_inputs(h, per_layer_inputs)
        if per_layer_inputs is not None:
            per_layer_inputs = [
                per_layer_inputs[:, :, i, :] for i, _ in enumerate(self.layers)
            ]
        else:
            per_layer_inputs = [None] * len(self.layers)

        # Make the kv cache list, be sure to append None for all the shared kv
        # layers
        if cache is None:
            cache = [None] * len(self.layers)
        else:
            cache = cache + [None] * (len(self.layers) - len(cache))

        # Apply each layer. We save all intermediate kvs and offset and grab
        # the previous one for the shared kv layers.
        masks = self._make_masks(h, cache, mm_mask=mm_mask)
        intermediates = [(None, None)] * len(self.layers)
        for idx, (layer, c, mask, prev_idx, per_layer_input) in enumerate(
            zip(
                self.layers,
                cache,
                masks,
                self.previous_kvs,
                per_layer_inputs,
            )
        ):
            kvs, offset = intermediates[prev_idx]

            h, kvs, offset = layer(
                h,
                mask,
                c,
                per_layer_input=per_layer_input,
                shared_kv=kvs,
                offset=offset,
            )

            intermediates[idx] = (kvs, offset)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Gemma4TextModel(args)
        self.final_logit_softcapping = args.final_logit_softcapping
        self.tie_word_embeddings = args.tie_word_embeddings
        if not self.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        per_layer_inputs: Optional[mx.array] = None,
        mm_mask: Optional[mx.array] = None,
    ):
        out = self.model(
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            per_layer_inputs=per_layer_inputs,
            mm_mask=mm_mask,
        )
        if self.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        if self.final_logit_softcapping is not None:
            out = logit_softcap(self.final_logit_softcapping, out)
        return out

    def sanitize(self, weights):
        sanitized = {}
        first_kv_shared = self.args.num_hidden_layers - self.args.num_kv_shared_layers
        for k, v in weights.items():
            if any(
                s in k
                for s in (
                    "self_attn.rotary_emb",
                    "input_max",
                    "input_min",
                    "output_max",
                    "output_min",
                )
            ):
                continue

            # KV-shared layers reuse K/V from earlier layers — drop their projections
            if any(
                s in k
                for s in (".self_attn.k_proj", ".self_attn.v_proj", ".self_attn.k_norm")
            ):
                try:
                    layer_idx = int(k.split("layers.")[1].split(".")[0])
                    if layer_idx >= first_kv_shared:
                        continue
                except (IndexError, ValueError):
                    pass

            if k.endswith(".experts.gate_up_proj"):
                base = k.removesuffix(".gate_up_proj")
                gate, up = map(mx.contiguous, mx.split(v, 2, axis=-2))
                sanitized[f"{base}.switch_glu.gate_proj.weight"] = gate
                sanitized[f"{base}.switch_glu.up_proj.weight"] = up
                continue

            if k.endswith(".experts.down_proj"):
                base = k.removesuffix(".down_proj")
                sanitized[f"{base}.switch_glu.down_proj.weight"] = v
                continue

            sanitized[k] = v

        return sanitized

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("router.proj"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.head_dim

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads

    def make_cache(self):
        first_kv_shared = self.args.num_hidden_layers - self.args.num_kv_shared_layers
        caches = []
        for i in range(first_kv_shared):
            if self.args.layer_types[i] == "full_attention":
                caches.append(KVCache())
            else:
                caches.append(
                    RotatingKVCache(
                        max_size=self.args.sliding_window,
                        keep=0,
                    )
                )
        return caches
