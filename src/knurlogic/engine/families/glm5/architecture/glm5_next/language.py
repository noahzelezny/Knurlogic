from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from knurlogic.engine.families.glm5.architecture.glm5_next._mlx_vlm.models.base import (
    LanguageModelOutput,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from knurlogic.engine.families.glm5.architecture.glm5_next._mlx_vlm.models.cache import ArraysCache, CacheList, KVCache
from knurlogic.engine.families.glm5.architecture.glm5_next._mlx_vlm.models.deepseek_v32.language import DeepseekV32MoE, MoEGate, group_expert_select
from knurlogic.engine.families.glm5.architecture.glm5_next._mlx_vlm.models.deepseek_v32.language import Model as DSV32Model
from knurlogic.engine.families.glm5.architecture.glm5_next._mlx_vlm.models.deepseek_v4.hyper_connection import HyperConnection, hc_expand
from knurlogic.engine.families.glm5.architecture.glm5_next._mlx_vlm.models.gated_delta import gated_delta_update
from knurlogic.engine.families.glm5.architecture.glm5_next._mlx_vlm.models.mla import MultiLinear
from knurlogic.engine.families.glm5.architecture.glm5_next._mlx_vlm.models.mlp import DeepseekMLP
from knurlogic.engine.families.glm5.architecture.glm5_next._mlx_vlm.models.rope_utils import initialize_rope
from .config import ModelConfig, TextConfig


class Glm5NextRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)

    def __call__(self, hidden_states: mx.array, gate: mx.array) -> mx.array:
        dt = hidden_states.dtype
        x = hidden_states.astype(mx.float32)
        var = (x * x).mean(-1, keepdims=True)
        x = x * mx.rsqrt(var + self.eps)
        x = self.weight.astype(mx.float32) * x
        x = x * mx.sigmoid(gate.astype(mx.float32))
        return x.astype(dt)


def _clamped_swiglu(gate: mx.array, up: mx.array, limit: float) -> mx.array:
    # The reference's SwiGLU (Glm5NextTextMLP / Glm5NextTextExperts._apply_gate):
    # gate clamped above at swiglu_limit, up clamped to +-swiglu_limit.
    return nn.silu(mx.minimum(gate, limit)) * mx.clip(up, -limit, limit)


class _ClampedSwiGLU(nn.Module):
    # SwitchGLU's activation slot: called as activation(x_up, x_gate).
    def __init__(self, limit: float):
        super().__init__()
        self.limit = limit

    def __call__(self, x, gate):
        return _clamped_swiglu(gate, x, self.limit)


class Glm5NextMLP(DeepseekMLP):
    """Dense MLP and shared expert, SwiGLU clamped at swiglu_limit (edit 1)."""

    def __call__(self, x):
        return self.down_proj(
            _clamped_swiglu(self.gate_proj(x), self.up_proj(x),
                            self.config.swiglu_limit)
        )


class Glm5NextMoEGate(MoEGate):
    """Router logits in float32, as the reference (moe_router_dtype) computes
    them: F.linear(x.float(), weight.float()) (edit 2)."""

    def __call__(self, x):
        return group_expert_select(
            x.astype(mx.float32) @ self.weight.astype(mx.float32).T,
            self.e_score_correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


class Glm5NextMoE(DeepseekV32MoE):
    """Routed experts and shared expert SwiGLU-clamped (edit 1), router in
    float32 (edit 2)."""

    def __init__(self, config: TextConfig):
        super().__init__(config)
        self.switch_mlp.activation = _ClampedSwiGLU(config.swiglu_limit)
        self.gate = Glm5NextMoEGate(config)
        if config.n_shared_experts is not None:
            self.shared_experts = Glm5NextMLP(
                config=config,
                intermediate_size=config.moe_intermediate_size
                * config.n_shared_experts,
            )


class Glm5NextForgetGate(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.head_dim = config.linear_head_dim
        self.num_heads = config.linear_num_heads
        self.qkv_dim = self.head_dim * self.num_heads
        self.f_a_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias=False)
        self.dt_bias = mx.zeros(self.qkv_dim)
        self.A_log = mx.zeros(self.num_heads)
        self.safe_gate_lower_bound = config.linear_lower_bound

    def __call__(self, hidden_states: mx.array) -> mx.array:
        B, S, _ = hidden_states.shape
        fg = self.f_b_proj(self.f_a_proj(hidden_states))
        g = (fg.astype(mx.float32) + self.dt_bias.astype(mx.float32)).reshape(
            B, S, self.num_heads, self.head_dim
        )
        decay = mx.exp(self.A_log.astype(mx.float32)).reshape(1, 1, self.num_heads, 1)
        if self.safe_gate_lower_bound is not None:
            return self.safe_gate_lower_bound * mx.sigmoid(decay * g)
        g_softplus = mx.where(g > 20.0, g, mx.log(1.0 + mx.exp(g)))
        return -decay * g_softplus


def _l2norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return x * mx.rsqrt((x * x).sum(axis=-1, keepdims=True) + eps)


def recurrent_kimi_delta(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    g: mx.array,
    beta: mx.array,
    state: Optional[mx.array] = None,
):
    # Reference O(S) recurrence for Kimi Delta Attention, kept as the readable
    # spec and the equivalence oracle for tests. The forward path runs this on
    # the shared fused gated_delta kernel (see Glm5NextLinearAttention).
    dt = query.dtype
    query = _l2norm(query.astype(mx.float32))
    key = _l2norm(key.astype(mx.float32))
    value = value.astype(mx.float32)
    g = g.astype(mx.float32)
    beta = beta.astype(mx.float32)
    B, S, H, Dk = key.shape
    Dv = value.shape[-1]
    query = query * (Dk**-0.5)
    if state is None:
        state = mx.zeros((B, H, Dk, Dv), dtype=mx.float32)
    else:
        state = state.astype(mx.float32)
    outs = []
    for i in range(S):
        q_i = query[:, i]
        k_i = key[:, i]
        v_i = value[:, i]
        g_i = mx.exp(g[:, i])[..., None]
        b_i = beta[:, i][..., None]
        state = state * g_i
        kv_mem = (state * k_i[..., None]).sum(axis=-2)
        delta = (v_i - kv_mem) * b_i
        state = state + k_i[..., None] * delta[..., None, :]
        out_i = (state * q_i[..., None]).sum(axis=-2)
        outs.append(out_i)
    out = mx.stack(outs, axis=1).astype(dt)
    return out, state


class Glm5NextLinearAttention(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.linear_num_heads
        self.head_dim = config.linear_head_dim
        self.qkv_dim = self.num_heads * self.head_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim

        self.q_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)

        self.conv_dim = self.qkv_dim * 3
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )

        self.forget_gate = Glm5NextForgetGate(config)
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias=False)
        self.o_norm = Glm5NextRMSNormGated(self.head_dim, eps=config.rms_norm_eps)
        self.o_proj = nn.Linear(self.qkv_dim, self.hidden_size, bias=False)
        self.fuse_in = True
        self._fused_ready = False

    def _fused_in_proj(self, inputs):
        # q,k,v,f_a,g_a,b all take `inputs`; fuse into one matmul via a lossless
        # output-axis concat of the (quantized) weights, built once and cached.
        if not self._fused_ready:
            mods = [
                self.q_proj,
                self.k_proj,
                self.v_proj,
                self.forget_gate.f_a_proj,
                self.g_a_proj,
                self.b_proj,
            ]
            pts, acc = [], 0
            for m in mods[:-1]:
                acc += m.weight.shape[0]
                pts.append(acc)
            self._split_pts = pts
            self._fq = hasattr(mods[0], "scales")
            self._fw = mx.concatenate([m.weight for m in mods], axis=0)
            if self._fq:
                self._fs = mx.concatenate([m.scales for m in mods], axis=0)
                self._fb = mx.concatenate([m.biases for m in mods], axis=0)
                self._gs, self._bits = mods[0].group_size, mods[0].bits
            self._fused_ready = True
        if self._fq:
            out = mx.quantized_matmul(
                inputs,
                self._fw,
                self._fs,
                self._fb,
                transpose=True,
                group_size=self._gs,
                bits=self._bits,
            )
        else:
            out = inputs @ self._fw.T
        return mx.split(out, self._split_pts, axis=-1)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, S, _ = inputs.shape
        if self.fuse_in:
            q_o, k_o, v_o, fa_o, ga_o, b_o = self._fused_in_proj(inputs)
            mixed = mx.concatenate([q_o, k_o, v_o], axis=-1)
        else:
            mixed = mx.concatenate(
                [self.q_proj(inputs), self.k_proj(inputs), self.v_proj(inputs)], axis=-1
            )
            fa_o = self.forget_gate.f_a_proj(inputs)
            ga_o = self.g_a_proj(inputs)
            b_o = self.b_proj(inputs)
        if mask is not None and mask.dtype == mx.bool_:
            mixed = mx.where(mask[..., None], mixed, 0)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype
            )
        conv_input = mx.concatenate([conv_state, mixed], axis=1)
        if cache is not None:
            cache[0] = mx.contiguous(conv_input[:, -(self.conv_kernel_size - 1) :, :])
        # The conv in float32, silu, then the model dtype: the reference
        # keeps conv1d in float32 (_keep_in_fp32_modules_strict) and casts
        # only causal_conv1d_fn's output (edit 7)
        in_dtype = conv_input.dtype
        conv_out = nn.silu(
            mx.conv1d(conv_input.astype(mx.float32),
                      self.conv1d.weight.astype(mx.float32),
                      groups=self.conv_dim)
        ).astype(in_dtype)

        q, k, v = mx.split(conv_out, [self.qkv_dim, 2 * self.qkv_dim], axis=-1)
        q = q.reshape(B, S, self.num_heads, self.head_dim)
        k = k.reshape(B, S, self.num_heads, self.head_dim)
        v = v.reshape(B, S, self.num_heads, self.head_dim)

        fg = self.forget_gate
        a = fg.f_b_proj(fa_o).reshape(B, S, self.num_heads, self.head_dim)
        # q and k l2-normed and scaled in float32 and kept float32 into the
        # recurrence, as the reference's KDA (it casts q, k, v, g, beta to
        # float32 first); the forget gate's a + dt_bias in float32 (its
        # forget_gate.float() + dt_bias.float()); the output back to the
        # model dtype before the gated norm (core_attn_out.to(initial_dtype))
        # (edit 7)
        q = _l2norm(q.astype(mx.float32)) * (self.head_dim**-0.5)
        k = _l2norm(k.astype(mx.float32))

        state = cache[1] if cache is not None else None
        out, state = gated_delta_update(
            q,
            k,
            v,
            a.astype(mx.float32),
            b_o,
            fg.A_log.reshape(self.num_heads, 1),
            fg.dt_bias.astype(mx.float32).reshape(self.num_heads, self.head_dim),
            state=state,
            lower_bound=fg.safe_gate_lower_bound,
        )
        out = out.astype(in_dtype)
        if cache is not None:
            cache[1] = state
            cache.advance(S)

        gate = self.g_b_proj(ga_o).reshape(B, S, self.num_heads, self.head_dim)
        out = self.o_norm(out, gate).reshape(B, S, -1)
        return self.o_proj(out)


class Glm5NextIndexer(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.dim = args.hidden_size
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.index_topk = args.index_topk
        self.index_kpool = args.index_kpool
        self.index_kpool_always_select_tail = args.index_kpool_always_select_tail
        self.q_lora_rank = args.q_lora_rank
        self.wq_b = nn.Linear(
            self.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(self.dim, self.head_dim, bias=False)
        # eps 1e-6 as the reference's indexer k_norm (edit 4)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim**-0.5
        self.index_kpool_compress_ape = mx.zeros((self.index_kpool, self.head_dim))
        self.index_kpool_compress_gate = mx.zeros((self.head_dim, self.dim))

    def _pooled_states(self, keys, gate_scores, valid):
        B, S, hd = keys.shape
        kp = self.index_kpool
        P = (S + kp - 1) // kp
        any_valid = mx.any(valid, axis=-1)
        first_key = mx.where(
            any_valid, mx.argmax(valid.astype(mx.int32), axis=-1), mx.array(S)
        )
        pool_offsets = mx.arange(P * kp).reshape(1, P, kp)
        pool_indices = first_key[:, None, None] + pool_offsets
        safe = mx.clip(pool_indices, 0, S - 1)
        flat = safe.reshape(B, P * kp)
        idxC = mx.broadcast_to(flat[..., None], (B, P * kp, hd))
        grouped_keys = mx.take_along_axis(keys, idxC, axis=1).reshape(B, P, kp, hd)
        grouped_gate = mx.take_along_axis(gate_scores, idxC, axis=1).reshape(
            B, P, kp, hd
        )
        grouped_valid = (
            mx.take_along_axis(valid.astype(mx.int32), flat, axis=1).reshape(B, P, kp)
            > 0
        )
        grouped_valid = grouped_valid & (pool_indices < S)
        pool_valid = mx.all(grouped_valid, axis=-1)
        pool_indices = mx.where(grouped_valid, pool_indices, -1)
        # pool weights in float32, cast to the keys' dtype (edit 5)
        logits = grouped_gate.astype(mx.float32) + self.index_kpool_compress_ape[
            None, None
        ].astype(mx.float32)
        logits = mx.where(grouped_valid[..., None], logits, -1e30)
        probs = mx.softmax(logits, axis=2)
        probs = mx.where(mx.isnan(probs), 0.0, probs).astype(grouped_keys.dtype)
        pool_keys = mx.sum(probs * grouped_keys, axis=2)
        return pool_keys, pool_indices, pool_valid

    def _visible_tail(self, visible, valid):
        B, S, Kv = visible.shape
        kp = self.index_kpool
        mtw = kp - 1
        any_valid = mx.any(valid, axis=-1)
        first_key = mx.where(
            any_valid, mx.argmax(valid.astype(mx.int32), axis=-1), mx.array(Kv)
        )
        visible_count = mx.sum(visible.astype(mx.int32), axis=-1)
        tail_count = visible_count - (visible_count // kp) * kp
        tail_offsets = mx.arange(mtw)
        tail_start = first_key[:, None] + visible_count - tail_count
        tail_indices = tail_start[..., None] + tail_offsets
        tail_valid = (tail_offsets[None, None, :] < tail_count[..., None]) & (
            tail_indices < Kv
        )
        kv_idx = mx.clip(tail_indices, 0, Kv - 1)
        tail_vis = mx.take_along_axis(visible, kv_idx, axis=-1)
        tail_indices = mx.where(tail_valid & tail_vis, tail_indices, -1)
        return tail_indices

    def __call__(self, x, qr, mask, cache=None):
        B, S, _ = x.shape
        q = self.wq_b(qr).reshape(B, S, self.n_heads, self.head_dim)
        k = self.k_norm(self.wk(x)).reshape(B, S, self.head_dim)
        gate_scores = x @ self.index_kpool_compress_gate.swapaxes(-1, -2)

        if mask is not None and mask.dtype == mx.bool_ and mask.shape == (B, S):
            valid_cur = mask
            valid_cur_all = False
        else:
            valid_cur = mx.ones((B, S), dtype=mx.bool_)
            valid_cur_all = True

        # Pack per-token state and append to the indexer cache so pooling/selection
        # run over the full cached sequence -- unifies prefill and incremental decode.
        packed = mx.concatenate(
            [k, gate_scores, valid_cur.astype(k.dtype)[..., None]], axis=-1
        )
        if cache is not None:
            keys, _ = cache.update_and_fetch(packed[:, None], mx.zeros((B, 1, S, 0)))
            packed_full = keys[:, 0]
        else:
            packed_full = packed
        T = packed_full.shape[1]
        # Short-context bypass: when the whole cache fits within index_topk the indexer
        # would select every token, so skip the O(T) pooling/scoring/topk and let the
        # DSA fall through to dense MLA. The cache is already updated above so state
        # stays consistent; the full pool is rebuilt once when T first exceeds index_topk.
        if getattr(self, "bypass_short", True) and T <= self.index_topk:
            return None
        k_full, gate_full, valid_ch = mx.split(
            packed_full, [self.head_dim, 2 * self.head_dim], axis=-1
        )
        valid = valid_ch[..., 0] > 0

        offset = T - S
        kv_len = T
        kv_pos = mx.arange(T)

        # Incremental pooling at decode: complete pools are stable across steps, so
        # recompute only the suffix (last partial pool + any new pool) and reuse the
        # cached complete pools -- turns the per-step pool cost from O(T) to O(kpool).
        # Exact; falls back to full pooling on prefill, when padding is present, or when
        # the cached pool's batch axis no longer matches the current batch. That last
        # guard matters under continuous batching: BatchGenerator grows/shrinks the
        # batch (extend/filter) on the batch axis but does not carry this per-cache
        # _pool along, so a stale _pool must be discarded and rebuilt for one step.
        # Small S (MTP verify, S<=SMALL_L) takes the same path: the new tokens only
        # touch pools at or after the previous length. After a speculative rollback
        # the cache is trimmed below the length the cached pools were built at, so
        # only pools wholly before min(t_prev, T - S) are reused -- those positions
        # were never rewritten.
        if (
            S <= SMALL_L
            and valid_cur_all
            and cache is not None
            and getattr(cache, "_pool", None) is not None
            and getattr(cache, "_no_pad", False)
            and cache._pool[0].shape[0] == B
        ):
            ck, ci, cv, t_prev = cache._pool
            n_stable = min(t_prev, T - S) // self.index_kpool
            s0 = n_stable * self.index_kpool
            pk_s, pi_s, pv_s = self._pooled_states(
                k_full[:, s0:], gate_full[:, s0:], valid[:, s0:]
            )
            pi_s = mx.where(pi_s >= 0, pi_s + s0, -1)
            pool_keys = mx.concatenate([ck[:, :n_stable], pk_s], axis=1)
            pool_indices = mx.concatenate([ci[:, :n_stable], pi_s], axis=1)
            pool_valid = mx.concatenate([cv[:, :n_stable], pv_s], axis=1)
        else:
            pool_keys, pool_indices, pool_valid = self._pooled_states(
                k_full, gate_full, valid
            )
            if cache is not None:
                cache._no_pad = bool(mx.all(valid))
        if cache is not None:
            cache._pool = (pool_keys, pool_indices, pool_valid, T)
        P = pool_keys.shape[1]
        select_k = min(self.index_topk // self.index_kpool, P)
        pool_end = mx.clip(pool_indices[..., -1], 0, kv_len - 1)
        # scores in float32, as the reference's indexer (edit 5)
        pool_keys_t = pool_keys[:, None].swapaxes(-1, -2).astype(mx.float32)
        tail_on = self.index_kpool_always_select_tail and self.index_kpool > 1
        output_width = self.index_topk + (self.index_kpool - 1 if tail_on else 0)

        # Chunk over the query dimension. A one-shot prefill otherwise materializes
        # [B, S, n_heads, P] scores (O(S*P)) and OOMs at long context; chunking bounds
        # peak to O(chunk*P). Decode (S=1) is a single chunk -> identical to before.
        chunk = 512 if S > 512 else S
        out = []
        for c0 in range(0, S, chunk):
            c1 = min(c0 + chunk, S)
            cs = c1 - c0
            q_pos = offset + mx.arange(c0, c1)
            visible = (kv_pos[None, None, :] <= q_pos[None, :, None]) & valid[:, None, :]
            scores = q[:, c0:c1].astype(mx.float32) @ pool_keys_t
            scores = mx.maximum(scores * self.softmax_scale, 0.0)
            weights = self.weights_proj(x[:, c0:c1]).astype(mx.float32) * (
                self.n_heads**-0.5
            )
            index_scores = mx.sum(weights[..., None] * scores, axis=2)
            pool_visible = mx.take_along_axis(
                visible, mx.broadcast_to(pool_end[:, None, :], (B, cs, P)), axis=-1
            )
            valid_candidates = pool_visible & pool_valid[:, None]
            index_scores = mx.where(valid_candidates, index_scores, -1e30)
            order = mx.argsort(-index_scores, axis=-1)
            selected = order[..., :select_k]
            selected_valid = mx.take_along_axis(valid_candidates, selected, axis=-1)
            pi = mx.broadcast_to(pool_indices[:, None], (B, cs, P, self.index_kpool))
            sel_exp = mx.broadcast_to(
                selected[..., None], (B, cs, select_k, self.index_kpool)
            )
            selected_indices = mx.take_along_axis(pi, sel_exp, axis=2)
            topk = selected_indices.reshape(B, cs, select_k * self.index_kpool)
            sv = mx.broadcast_to(
                selected_valid[..., None], (B, cs, select_k, self.index_kpool)
            ).reshape(B, cs, select_k * self.index_kpool)
            topk = mx.where(sv, topk, -1)
            if tail_on:
                topk = mx.concatenate([topk, self._visible_tail(visible, valid)], axis=-1)
            if topk.shape[-1] < output_width:
                pad = mx.full(
                    (B, cs, output_width - topk.shape[-1]), -1, dtype=topk.dtype
                )
                topk = mx.concatenate([topk, pad], axis=-1)
            topk = topk[..., :output_width]
            topk = mx.where(valid_cur[:, c0:c1][..., None], topk, -1)
            out.append(topk)
        topk = out[0] if len(out) == 1 else mx.concatenate(out, axis=1)
        return topk[:, None].astype(mx.int32)


class _LatentCache(KVCache):
    """The MLA layer's latent cache (K = V = the normed compressed latent),
    stored once as `keys`; `values` is zero-width (edit 6).
    A plain KVCache until engine/kvquant.install swaps it (the manifest's
    `kv_quant.caches`): then the latent is stored quantized
    (kvquant.QuantKVCache) and fetched dequantized, so the absorbed
    (SMALL_L) and expanded paths read it as before. The DSA indexer's cache (the CacheList's second member) is a
    plain KVCache, not named by the manifest: its packed keys pick the
    sparse set and stay exact."""


# Query widths that take the decode-style attention (absorbed MLA + sparse gather):
# single-token decode and the MTP verify forward (1 + draft depth tokens).
SMALL_L = 4

# knurlogic edit 10: once the indexer selects, a prefill chunk attends in
# the latent too (absorbed MLA over each query's own selection,
# _gathered_attention), not over per-head K/V expanded for the WHOLE context
# and masked down to the selection: at 334k tokens that expansion is
# heads x (qk + v) x bf16 for every cached token, every layer, every chunk,
# and the prefill chunk had to shrink to 256 to fit (an M3 Ultra at 96 GB
# was still killed at 98%). Measured on one layer at GLM-5.3-Flash 2.7bpw's
# shapes, chunk 2048, M3 Ultra (tools/bench_glm5_sparse_prefill.py),
# expanded vs latent: 8k 0.181 s / 4.2 GiB vs 0.230 s / 7.4 GiB; 32k
# 0.665 / 11.4 vs 0.297 / 7.4; 131k 2.722 / 42.0 vs 0.473 / 8.1. The
# expanded path's one win, ~0.05 s a chunk at short context, is not worth
# a crossover. True runs mlx-vlm's expanded path (the parity test and the
# bench compare the two).
EXPANDED_PREFILL = False
# query rows per block of _gathered_attention: its gathered latent is
# rows x topk x kv_lora_rank, bounded whatever the chunk
GATHER_ROWS = 128


def _gathered_attention(q, kv_latent, topk, mask, scale):
    """Absorbed MLA over each query row's own selection, a block of rows at
    a time: the same attention as the expanded path's dense scatter mask
    (the same keys, the same softmax), without expanding the context.

    q [B, H, L, D] (already absorbed into the latent: embed_q);
    kv_latent [B, 1, Kv, D]; topk [B, L, K] (-1 = unselected); `mask` the
    forward's mask (bool, as _gather_selected reads it) or None.
    -> [B, H, L, D] in the latent (unembed_out follows)."""
    B, H, L, D = q.shape
    Kv = kv_latent.shape[2]
    K = topk.shape[-1]
    lat = kv_latent[:, 0]                                   # [B, Kv, D]
    valid = topk >= 0
    clamped = mx.clip(topk, 0, Kv - 1)
    if mask is not None and mask.dtype == mx.bool_:
        m = mask
        if m.ndim == 2:
            m = m[None] if (L > 1 and m.shape[0] == L) else m[:, None, :]
        else:
            m = m.reshape(m.shape[0], -1, m.shape[-1])
        m = mx.broadcast_to(m, (B, L, Kv))
        valid = valid & mx.take_along_axis(m, clamped, axis=-1)
    out = []
    for r0 in range(0, L, GATHER_ROWS):
        r1 = min(r0 + GATHER_ROWS, L)
        n = r1 - r0
        idx = clamped[:, r0:r1].reshape(B, n * K, 1)
        keys = mx.take_along_axis(
            lat, mx.broadcast_to(idx, (B, n * K, D)), axis=1
        ).reshape(B, n, K, D)                               # [B, n, K, D]
        qb = q[:, :, r0:r1].transpose(0, 2, 1, 3)           # [B, n, H, D]
        scores = (qb.astype(mx.float32)
                  @ keys.astype(mx.float32).swapaxes(-1, -2)) * scale
        ok = valid[:, r0:r1][:, :, None, :]                 # [B, n, 1, K]
        scores = mx.where(ok, scores, -mx.inf)
        w = mx.softmax(scores, axis=-1, precise=True)
        w = mx.where(ok, w, 0.0)            # a row with nothing valid: 0
        o = (w.astype(keys.dtype) @ keys)                   # [B, n, H, D]
        out.append(o.transpose(0, 2, 1, 3).astype(q.dtype))
    return out[0] if len(out) == 1 else mx.concatenate(out, axis=2)


def _gather_selected(kv_latent, topk, mask, B, L, Kv):
    """Gather the L rows' selected latent keys; mask each row to its own block.

    kv_latent [B, 1, Kv, D]; topk [B, L, K] (-1 = unselected). Returns the gathered
    latent [B, 1, L*K, D] and a bool mask [B, 1, L, L*K] where row i sees only the
    valid entries of its own K-block (and, if a bool `mask` is given, only keys that
    mask allows for row i). Duplicates across rows live in other rows' blocks and are
    masked, so each row attends exactly its own selection -- the same set the dense
    scatter mask picks, without touching the rest of the context.
    """
    K = topk.shape[-1]
    valid = topk >= 0
    clamped = mx.clip(topk, 0, Kv - 1)
    flat = clamped.reshape(B, 1, L * K, 1)
    gathered = mx.take_along_axis(
        kv_latent,
        mx.broadcast_to(flat, (B, 1, L * K, kv_latent.shape[-1])),
        axis=2,
    )
    if mask is not None and mask.dtype == mx.bool_:
        # Bool masks arrive as a causal [L, Kv] (create_causal_mask), a per-key
        # [B, Kv] at L == 1, or [B, 1, 1|L, Kv] from a batch cache: bring to
        # [B, L, Kv] and read each row's own keys.
        m = mask
        if m.ndim == 2:
            m = m[None] if (L > 1 and m.shape[0] == L) else m[:, None, :]
        else:
            m = m.reshape(m.shape[0], -1, m.shape[-1])
        m = mx.broadcast_to(m, (B, L, Kv))
        valid = valid & mx.take_along_axis(m, clamped, axis=-1)
    if L == 1:
        return gathered, valid[:, None]
    eye = mx.eye(L, dtype=mx.bool_)  # [L, L]
    block = eye[:, :, None] & valid[:, None, :, :]  # [B, L(row), L(block), K]
    return gathered, block.reshape(B, 1, L, L * K)



class Glm5NextSparseAttention(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.use_nope = config.mla_use_nope or config.qk_rope_head_dim == 0
        # GLM-5-Next is NoPE by design (qk_rope_head_dim=0, mla_use_nope=True); the
        # config carries no rope parameters. Fail loudly rather than run wrong math
        # if a future config ever requests a RoPE MLA.
        if not self.use_nope:
            raise NotImplementedError(
                "glm5_next implements NoPE MLA only; qk_rope_head_dim>0 with "
                "mla_use_nope=False is not supported."
            )
        self.q_head_dim = config.qk_nope_head_dim
        self.scale = self.q_head_dim**-0.5

        self.q_a_proj = nn.Linear(
            self.hidden_size, self.q_lora_rank, bias=config.attention_bias
        )
        # eps rms_norm_eps (1e-5), as the reference's q_a / kv_a norms (edit 3)
        self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size, self.kv_lora_rank, bias=config.attention_bias
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.embed_q = MultiLinear(
            self.qk_nope_head_dim, self.kv_lora_rank, self.num_heads
        )
        self.unembed_out = MultiLinear(
            self.kv_lora_rank, self.v_head_dim, self.num_heads
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self.indexer = Glm5NextIndexer(config)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        qr = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(qr)
        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        kv_latent = self.kv_a_layernorm(compressed_kv)
        kv_latent = mx.expand_dims(kv_latent, axis=1)

        if cache is not None:
            # K = V = the latent: stored ONCE, as the keys, beside a
            # zero-width values array (edit 6; it was stored as both, twice
            # the bytes). Every cache op -- trim, state, merge, extract,
            # kvquant -- handles the zero-width side as the indexer's.
            kv_latent, _ = cache[0].update_and_fetch(
                kv_latent, mx.zeros((B, 1, L, 0), dtype=kv_latent.dtype)
            )
        else:
            cache = [None] * 2

        topk_indices = self.indexer(x, qr, mask, cache=cache[1])
        # Absorbed MLA (queries into the latent, one shared K=V latent head) for
        # decode and small verify widths; the expanded per-head K/V only for prefill.
        absorbed = L <= SMALL_L
        gathered = (not absorbed and topk_indices is not None
                    and not EXPANDED_PREFILL)
        attn_mask = mask
        if topk_indices is not None and not gathered:
            Kv = kv_latent.shape[2]
            valid_sel = topk_indices >= 0
            if absorbed:
                kv_latent, attn_mask = _gather_selected(
                    kv_latent, topk_indices[:, 0], mask, B, L, Kv
                )
            else:
                shape = list(topk_indices.shape)
                shape[-1] = Kv + 1
                safe_idx = mx.where(valid_sel, topk_indices, Kv)
                sparse_mask = mx.zeros(shape, dtype=mx.bool_)
                sparse_mask = mx.put_along_axis(
                    sparse_mask, safe_idx, mx.array(True), axis=-1
                )
                sparse_mask = sparse_mask[..., :Kv]
                if mask is not None and mask.dtype == mx.bool_:
                    sparse_mask = sparse_mask & mask
                attn_mask = sparse_mask

        if (
            cache is not None
            and cache[0] is not None
            and cache[1] is not None
            and cache[1].keys is not None
        ):
            # a quantized latent's keys are a triple: mx.depends takes and
            # returns the list (engine/kvquant.QuantKVCache)
            cache[0].keys = mx.depends(cache[0].keys, (cache[1].keys, cache[1].values))

        if gathered:
            output = self.unembed_out(_gathered_attention(
                self.embed_q(q), kv_latent, topk_indices[:, 0], mask,
                self.scale))
        else:
            if absorbed:
                q = self.embed_q(q)
                k = v = kv_latent
            else:
                k = self.embed_q(kv_latent, transpose=False)
                v = self.unembed_out(kv_latent)

            output = scaled_dot_product_attention(
                q, k, v, cache=cache, scale=self.scale, mask=attn_mask
            )
            if absorbed:
                output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


@mx.compile
def _hc_expand(x, residual, post, comb):
    """The mHC expand in the model dtype, as the reference's decoder layer
    (`post.to(dtype)`, `comb.to(dtype)`, then bf16 products and sum,
    modeling 1316-1317 / 1325-1326): post and comb are rounded to the
    model dtype first. mlx-vlm's hc_expand kept them float32 throughout
    (edit 8). Exact in float32."""
    dt = x.dtype
    return post.astype(dt)[..., None] * x[:, :, None, :] + mx.matmul(
        comb.astype(dt).swapaxes(-1, -2), residual
    )


class Glm5NextDecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        layer_type = config.layer_types[layer_idx]
        self.is_linear = layer_type == "linear_attention"
        if self.is_linear:
            self.self_attn = Glm5NextLinearAttention(config)
        else:
            self.self_attn = Glm5NextSparseAttention(config)

        is_sparse = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and config.mlp_layer_types[layer_idx] == "sparse"
        )
        self.mlp = Glm5NextMoE(config) if is_sparse else Glm5NextMLP(config)

        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attn_hc = HyperConnection(config)
        self.ffn_hc = HyperConnection(config)
        self.compile_ffn = True
        self._ffn_c = None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        residual = x
        xc, post, comb = self.attn_hc(x)
        r = self.self_attn(self.input_layernorm(xc), mask, cache)
        x = _hc_expand(r, residual, post, comb)
        # Compile the FFN block only for single-stream decode (B=1, S=1) -- the shape it
        # was validated on and where its win lives. Compiling the 288-expert MoE at a
        # batched or prefill shape spikes memory (it can OOM alongside the resident
        # weights), so those shapes take the eager path.
        if self.compile_ffn and x.shape[0] == 1 and x.shape[1] == 1:
            if self._ffn_c is None:
                self._ffn_c = mx.compile(self._ffn_block)
            return self._ffn_c(x)
        return self._ffn_block(x)

    def _ffn_block(self, x: mx.array) -> mx.array:
        # Stateless FFN half (no cache) -> compiles cleanly at a fixed decode shape.
        residual = x
        xc, post, comb = self.ffn_hc(x)
        m = self.mlp(self.post_attention_layernorm(xc))
        return _hc_expand(m, residual, post, comb)


class Glm5NextModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.hc_mult = config.hc_mult
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Glm5NextDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ssm_idx = next((i for i, l in enumerate(self.layers) if l.is_linear), 0)
        self.fa_idx = next((i for i, l in enumerate(self.layers) if not l.is_linear), 0)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        fa_cache = cache[self.fa_idx]
        fa_mask = create_attention_mask(
            h, fa_cache[0] if fa_cache else None, return_array=True
        )
        ssm_mask = create_ssm_mask(h, cache[self.ssm_idx])

        h = mx.broadcast_to(
            h[:, :, None, :], (h.shape[0], h.shape[1], self.hc_mult, h.shape[2])
        )
        h = mx.contiguous(h)

        for layer, c in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            h = layer(h, mask=mask, cache=c)

        h = h.mean(axis=2)
        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, args: TextConfig, config: ModelConfig = None):
        super().__init__()
        self.args = args
        self.config = args
        self.model_type = args.model_type
        self.model = Glm5NextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        mask: Optional[mx.array] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        if inputs is None:
            inputs = kwargs.get("input_ids")
        out = self.model(inputs, cache=cache, inputs_embeds=inputs_embeds)
        # Only the last few positions' logits are ever needed for generation; slicing
        # before the (vocab-wide) projection skips it on discarded prefill positions.
        nlk = kwargs.get("num_logits_to_keep", 0)
        if nlk:
            out = out[:, -nlk:, :]
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return LanguageModelOutput(logits=out)

    def sanitize(self, weights):
        weights = {k: v for k, v in weights.items() if "mtp." not in k}
        weights = DSV32Model.sanitize(self, weights)

        remapped = {}
        conv_parts = {}
        fg_parts = ("A_log", "dt_bias", "f_a_proj.weight", "f_b_proj.weight")
        for k, v in weights.items():
            nk = k.replace(".hc_attn_", ".attn_hc.").replace(".hc_ffn_", ".ffn_hc.")

            fused = False
            for part in ("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight"):
                suffix = ".self_attn." + part
                if nk.endswith(suffix):
                    prefix = nk[: -len(part)]
                    conv_parts.setdefault(prefix, {})[part[0]] = v
                    fused = True
                    break
            if fused:
                continue

            for p in fg_parts:
                suffix = ".self_attn." + p
                if nk.endswith(suffix):
                    nk = nk[: -len(p)] + "forget_gate." + p
                    break

            remapped[nk] = v

        for prefix, parts in conv_parts.items():
            if all(c in parts for c in ("q", "k", "v")):
                remapped[prefix + "conv1d.weight"] = mx.concatenate(
                    [parts["q"], parts["k"], parts["v"]], axis=0
                )
            else:
                for c, w in parts.items():
                    remapped[prefix + c + "_conv1d.weight"] = w

        weights = remapped
        for k, v in list(weights.items()):
            if "conv1d.weight" in k and v.ndim == 3 and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
        return weights

    @property
    def layers(self):
        return self.model.layers

    @property
    def cast_predicate(self):
        # the reference's _keep_in_fp32_modules_strict stay float32 when a
        # checkpoint is converted (edit 7)
        keep = ("e_score_correction_bias", "conv1d", "dt_bias", "A_log")

        def predicate(k):
            return not any(n in k for n in keep)

        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if (
                path.endswith("mlp.gate")
                or "e_score_correction_bias" in path
                or ".indexer" in path
            ):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    def prefill_span(self, ctx: int) -> int:
        """The context a prefill chunk's temporaries span, for the memory
        guard (engine/runtime/scheduler.Scheduler._prefill_x; edit 10): past
        index_topk the latent attention reads at most index_topk keys a
        query, and only the indexer reads the whole context -- its pooled
        keys, ctx / index_kpool, at index_n_heads float32 scores each,
        against the attention's kv_lora_rank float32 latent plus a score
        per head for each key it reads. So a chunk spans
        min(ctx, index_topk) + ctx x (indexer bytes a token / attention
        bytes a key): GLM-5.3-Flash 2048 + ctx / 72 (measured: +0.7 GiB a
        layer from 32k to 131k, about 1/500 -- this is the safe side)."""
        a = self.args
        topk = int(getattr(a, "index_topk", 0) or 0)
        if not topk or EXPANDED_PREFILL:
            return ctx
        per_key = 4 * (int(a.kv_lora_rank) + int(a.num_attention_heads))
        per_tok = 4 * int(a.index_n_heads) / max(int(a.index_kpool or 1), 1)
        return min(ctx, topk) + int(ctx * per_tok / per_key)

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.is_linear:
                caches.append(ArraysCache(size=2))
            else:
                caches.append(CacheList(_LatentCache(), KVCache()))
        return caches
