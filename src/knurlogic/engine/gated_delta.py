"""The gated-delta recurrence (GatedDeltaNet's scan) on every MLX back end:
the one home the families that have it (Qwen3.5, Qwen4-exp, GLM-5-next)
call.

  metal  mlx-lm's kernels, unchanged: the numbers the artifacts were
         validated on.
  cuda   the same kernel ported to CUDA (`mx.fast.cuda_kernel`): one warp
         per value row, as one SIMD-group per row on Metal, and the warp
         reduction in place of simd_sum.
  cpu    mlx-lm's per-token ops, evaluated every CHUNK tokens. Unevaluated,
         a prompt's whole scan is one graph that keeps every token's float32
         state alive (~1 MB per token per layer at 16 heads of 128 x 128),
         which ran a 2B model's warm-up out of memory in 6 GB.

Design: docs/design/os-agnostic.md (gated-delta off Metal).
"""
from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.gated_delta import (
    _gated_delta_step_ops,
    gated_delta_kernel,
)

from knurlogic.engine import backend

#: Tokens of the per-token fallback evaluated at a time.
CHUNK = 64


def recurrence(q, k, v, g, beta, state, mask=None, *, use_kernel=True,
               metal=gated_delta_kernel):
    """y [B, T, Hv, Dv] and the new state [B, Hv, Dv, Dk] for q, k
    [B, T, Hk, Dk], v [B, T, Hv, Dv], g [B, T, Hv] (or [..., Dk]), beta
    [B, T, Hv]. `use_kernel=False` takes the ops on any back end; `metal`
    is the family's own Metal kernel where it vendors one."""
    Dk = k.shape[-1]
    kernel = (use_kernel and mx.default_device() == mx.gpu
              and Dk >= 32 and Dk % 32 == 0)
    if kernel and backend.name() == "metal":
        return metal(q, k, v, g, beta, state, mask)
    if kernel and backend.name() == "cuda":
        return _cuda(q, k, v, g, beta, state, mask)
    return ops(q, k, v, g, beta, state, mask)


def ops(q, k, v, g, beta, state, mask=None):
    """mlx-lm's gated_delta_ops, the state evaluated every CHUNK tokens
    (the same arithmetic; evaluation only bounds what is alive)."""
    _, T, Hk, _ = q.shape
    Hv = v.shape[-2]
    if (r := Hv // Hk) > 1:
        q, k = mx.repeat(q, r, -2), mx.repeat(k, r, -2)
    ys = []
    for t in range(T):
        y, state = _gated_delta_step_ops(
            q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t], state,
            None if mask is None else mask[:, t])
        ys.append(y)
        if T > CHUNK and (t + 1) % CHUNK == 0:
            mx.eval(ys[-CHUNK:], state)
    return mx.stack(ys, axis=1), state


def _cuda_source(masked: bool, vectorized: bool) -> str:
    keep = "mask[b_idx * T + t]" if masked else "true"
    if vectorized:      # g: [B, T, Hv, Dk]
        g_setup = "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
        g_at, g_next = "g_[s_idx]", "g_ += Hv * Dk;"
    else:               # g: [B, T, Hv]
        g_setup = "auto g_ = g + b_idx * T * Hv;"
        g_at, g_next = "g_[hv_idx]", "g_ += Hv;"
    return f"""
        const int T = q_shape[1];
        auto n = blockIdx.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = threadIdx.x;                       // the warp's lane
        auto dv_idx = blockIdx.y * blockDim.y + threadIdx.y;
        if (dv_idx >= Dv) return;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
          state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
        }}
        {g_setup}
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {{
          if ({keep}) {{
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * static_cast<float>({g_at});
              kv_mem += state[i] * static_cast<float>(k_[s_idx]);
            }}
            kv_mem = warp_sum(kv_mem);
            float delta = (static_cast<float>(v_[dv_idx]) - kv_mem) *
                          static_cast<float>(beta_[hv_idx]);
            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + static_cast<float>(k_[s_idx]) * delta;
              out += state[i] * static_cast<float>(q_[s_idx]);
            }}
            out = warp_sum(out);
            if (dk_idx == 0) {{
              y[dv_idx] = static_cast<InT>(out);
            }}
          }} else if (dk_idx == 0) {{
            y[dv_idx] = static_cast<InT>(0.0f);
          }}
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          {g_next}
          beta_ += Hv;
        }}
        for (int i = 0; i < n_per_t; ++i) {{
          o_state[n_per_t * dk_idx + i] = static_cast<StT>(state[i]);
        }}
    """


_CUDA_HEADER = """
__device__ __forceinline__ float warp_sum(float x) {
  for (int o = 16; o > 0; o >>= 1) {
    x += __shfl_xor_sync(0xffffffff, x, o);
  }
  return x;
}
"""

_CUDA_KERNELS: dict = {}


def _cuda(q, k, v, g, beta, state, mask=None):
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    key = (mask is not None, g.ndim == 4)
    if key not in _CUDA_KERNELS:
        masked, vectorized = key
        _CUDA_KERNELS[key] = mx.fast.cuda_kernel(
            name="gated_delta_step" + "_vec" * vectorized + "_mask" * masked,
            input_names=["q", "k", "v", "g", "beta", "state_in"]
            + ["mask"] * masked,
            output_names=["y", "state_out"],
            source=_cuda_source(masked, vectorized), header=_CUDA_HEADER)
    inputs = [q, k, v, g, beta, state] + ([mask] if mask is not None else [])
    return _CUDA_KERNELS[key](
        inputs=inputs,
        template=[("InT", q.dtype), ("StT", state.dtype), ("Dk", Dk),
                  ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
        grid=(32, Dv, B * Hv), threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[q.dtype, state.dtype])
