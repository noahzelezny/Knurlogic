"""Decode attention read straight from an 8-bit KV cache (engine/kvquant).

A Metal kernel reads the packed K/V, their scales and biases directly
(split-K over key blocks, one launch; a second tiny launch combines the
blocks' softmax partials) instead of dequantizing the whole cache to bf16
every decode step. `install` rebinds `scaled_dot_product_attention` in the
model's own modules to `sdpa` below, which takes the kernel when handed
exactly the keys kvquant's cache just returned, and otherwise is mlx-lm's
function unchanged. Prefill stays dequantize+sdpa.

A query row with EVERY key masked returns 0 here, where mlx sdpa returns
NaN. `KNURLOGIC_KV_KERNEL=off` turns it off; `STATS` counts hits and
misses. Design: docs/design/kv-cache.md (decode kernel).
"""
from __future__ import annotations

import logging
import os
import sys

import mlx.core as mx

logger = logging.getLogger(__name__)

#: {on, hits, misses, last_miss}: live counters (state.SERVED["kv_kernel"]
#: is this same dict)
STATS: dict = {"on": False, "hits": 0, "misses": 0, "last_miss": None}
_LOGGED: set = set()


def reset(on: bool) -> None:
    """Zero the counters for a newly installed model and publish them."""
    STATS.update(on=bool(on), hits=0, misses=0, last_miss=None)
    _LOGGED.clear()
    from knurlogic.engine.model import state
    state.SERVED["kv_kernel"] = STATS


def count(hit: bool, why: str = "") -> None:
    """One decode attention: through the kernel (hit) or not (miss)."""
    if hit:
        STATS["hits"] += 1
    else:
        STATS["misses"] += 1
        STATS["last_miss"] = why
    if hit not in _LOGGED:
        _LOGGED.add(hit)
        if hit:
            logger.info("kv kernel: live -- 8-bit decode reads the packed "
                        "K/V (engine/kvattn)")
        else:
            logger.warning("kv kernel: a decode step fell back to "
                           "dequantize + sdpa: %s", why)

#: query rows per KV head (GQA ratio x query length) the kernel takes
MAX_ROWS = 64
#: keys per threadgroup (NB), simdgroups per threadgroup (SG: each walks
#: every SG-th key of the block), and query rows per threadgroup (RC: the
#: R rows of a KV head are split over R / RC threadgroups that each re-read
#: the block's K/V -- fewer registers per thread, more threadgroups in
#: flight). Tuned on an M4 Max (128 GB) with layers chained as in a model:
#: 256/2/2 is 114 / 208 us per layer at 6k / 16k
#: against 230 / 503 for the research kernel's 256/8/all-8 and 146 / 291
#: for dequantize + sdpa.
NB = 256
SG = 2
RC = 2

ENV = "KNURLOGIC_KV_KERNEL"


def enabled() -> bool:
    s = os.environ.get(ENV, "").strip().lower()
    return s not in ("off", "0", "false", "no")


_SRC = r"""
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint blk = threadgroup_position_in_grid.x;
    uint bh = threadgroup_position_in_grid.y / (R / RC);
    int r0 = (threadgroup_position_in_grid.y % (R / RC)) * RC;
    int Hk = kw_shape[1];
    int b = bh / Hk, h = bh % Hk;
    int N = kw_shape[2];
    int n0 = blk * NB, n1 = min(N, n0 + NB);
    constexpr int DPL = D / 32;
    constexpr int WPL = DPL * BITS / 32;
    constexpr int EPW = 32 / BITS;
    constexpr uint MASKB = (1u << BITS) - 1u;
    int d0 = lane * DPL;
    int gi = d0 / G;
    float qv[RC][DPL]; float qs[RC];
    float o[RC][DPL]; float m[RC]; float l[RC];
    float sc0 = scale[0];
    for (int r = 0; r < RC; r++) {
        size_t qo = ((size_t)(b * Hk + h) * R + r0 + r) * D + d0;
        qs[r] = 0.f; m[r] = -INFINITY; l[r] = 0.f;
        for (int i = 0; i < DPL; i++) {
            qv[r][i] = float(q[qo + i]) * sc0; qs[r] += qv[r][i]; o[r][i] = 0.f;
        }
    }
    size_t kb = (size_t)b * kw_strides[0] + (size_t)h * kw_strides[1];
    size_t sb = (size_t)b * ks_strides[0] + (size_t)h * ks_strides[1];
    size_t vb = (size_t)b * vw_strides[0] + (size_t)h * vw_strides[1];
    size_t tb = (size_t)b * vs_strides[0] + (size_t)h * vs_strides[1];
#if SAME_LAYOUT
    // bias laid out exactly like its scale (kvquant's caches, by
    // construction): one index for both -- separate strides cost ~10%
#define KBI(n) kbias[sb + (size_t)(n) * ks_strides[2] + gi]
#define VBI(n) vbias[tb + (size_t)(n) * vs_strides[2] + gi]
#else
    size_t kbb = (size_t)b * kbias_strides[0] + (size_t)h * kbias_strides[1];
    size_t vbb = (size_t)b * vbias_strides[0] + (size_t)h * vbias_strides[1];
#define KBI(n) kbias[kbb + (size_t)(n) * kbias_strides[2] + gi]
#define VBI(n) vbias[vbb + (size_t)(n) * vbias_strides[2] + gi]
#endif
    for (int n = n0 + sg; n < n1; n += SG) {
        bool any = true;
#if HAS_MASK
        any = false;
        for (int r = 0; r < RC; r++) any = any || mask[((size_t)b * L + ((r0 + r) % L)) * N + n];
#endif
        if (!any) continue;
        size_t kr = kb + (size_t)n * kw_strides[2] + lane * WPL;
        float kx[DPL];
        for (int w = 0; w < WPL; w++) {
            uint word = kw[kr + w];
            for (int e = 0; e < EPW; e++) kx[w * EPW + e] = float((word >> (e * BITS)) & MASKB);
        }
        size_t si = sb + (size_t)n * ks_strides[2] + gi;
        float ksc = float(ks[si]);
        float kbi = float(KBI(n));
        size_t vr = vb + (size_t)n * vw_strides[2] + lane * WPL;
        size_t ti = tb + (size_t)n * vs_strides[2] + gi;
        float vsc = float(vs[ti]);
        float vbi = float(VBI(n));
        float vx[DPL];
        for (int w = 0; w < WPL; w++) {
            uint word = vw[vr + w];
            for (int e = 0; e < EPW; e++) vx[w * EPW + e] = vsc * float((word >> (e * BITS)) & MASKB) + vbi;
        }
        for (int r = 0; r < RC; r++) {
#if HAS_MASK
            if (!mask[((size_t)b * L + ((r0 + r) % L)) * N + n]) continue;
#endif
            float acc = 0.f;
            for (int i = 0; i < DPL; i++) acc += qv[r][i] * kx[i];
            float s = simd_sum(ksc * acc + kbi * qs[r]);
            float mn = max(m[r], s);
            float corr = fast::exp(m[r] - mn);
            float p = fast::exp(s - mn);
            l[r] = l[r] * corr + p; m[r] = mn;
            for (int i = 0; i < DPL; i++) o[r][i] = o[r][i] * corr + p * vx[i];
        }
    }
    uint nblk = threadgroups_per_grid.x * SG;
    uint bslot = blk * SG + sg;
    for (int r = 0; r < RC; r++) {
        size_t oi = ((size_t)bh * R + r0 + r) * nblk + bslot;
        for (int i = 0; i < DPL; i++) out_o[oi * D + d0 + i] = o[r][i];
        if (lane == 0) { out_m[oi] = m[r]; out_l[oi] = l[r]; }
    }
"""

_CSRC = r"""
    uint d = thread_position_in_grid.x; uint row = thread_position_in_grid.y;
    size_t base = (size_t)row * NBLK;
    float M = -INFINITY;
    for (int i = 0; i < NBLK; i++) M = max(M, out_m[base + i]);
    float num = 0.f, den = 0.f;
    if (M > -INFINITY) {
        for (int i = 0; i < NBLK; i++) {
            float mi = out_m[base + i];
            if (mi == -INFINITY) continue;
            float w = fast::exp(mi - M);
            den += w * out_l[base + i];
            num += w * out_o[(base + i) * D + d];
        }
    }
    out[(size_t)row * D + d] = static_cast<T>(den > 0.f ? num / den : 0.f);
"""

_K: dict = {}


def _kernel(has_mask: bool, same: bool):
    key = (has_mask, same)
    k = _K.get(key)
    if k is None:
        ins = ["q", "scale", "kw", "ks", "kbias", "vw", "vs", "vbias"]
        if has_mask:
            ins.append("mask")
        k = _K[key] = mx.fast.metal_kernel(
            name=f"kl_kv8_decode_{int(has_mask)}{int(same)}",
            input_names=ins, output_names=["out_o", "out_m", "out_l"],
            header=(f"#define HAS_MASK {int(has_mask)}\n"
                    f"#define SAME_LAYOUT {int(same)}\n"),
            source=_SRC, ensure_row_contiguous=False)
    return k


def _combine():
    k = _K.get("combine")
    if k is None:
        k = _K["combine"] = mx.fast.metal_kernel(
            name="kl_kv8_combine", input_names=["out_o", "out_m", "out_l"],
            output_names=["out"], source=_CSRC)
    return k


def supports(q, K, V, bits: int, group: int, mask=None, sinks=None) -> bool:
    """Whether `decode_sdpa` takes these shapes (else the caller falls
    back to dequantize + sdpa)."""
    if bits != 8 or sinks is not None or q.ndim != 4:
        return False
    B, H, L, D = q.shape
    Hk, Dk = K[0].shape[1], K[0].shape[3] * 32 // bits
    Dv = V[0].shape[3] * 32 // bits
    if Dk != D or Dv != D or D % 128 or D > 512 or H % Hk:
        return False
    if group % (D // 32) or D % group or K[0].shape[0] != B:
        return False
    for w, sc, bi in (K, V):
        if sc.shape != bi.shape or sc.shape[:3] != w.shape[:3]:
            return False
    if (H // Hk) * L > MAX_ROWS:
        return False
    if q.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        return False
    if mask is None or isinstance(mask, str):
        return mask in (None, "causal")
    if mask.dtype != mx.bool_ or mask.ndim > 4:
        return False
    if mask.ndim == 4 and mask.shape[1] != 1:
        return False
    return mask.shape[-1] == K[0].shape[2] and mask.shape[-2] in (1, L)


def _mask_rows(mask, B, L, N):
    """(B, L, N) bool, or None when every key is visible to every row."""
    if mask is None or (isinstance(mask, str) and L == 1):
        return None
    if isinstance(mask, str):                       # "causal", L > 1
        rows = mx.arange(N - L, N)[:, None]
        m = rows >= mx.arange(N)[None]
        return mx.broadcast_to(m[None], (B, L, N))
    if mask.ndim == 4:
        mask = mask[:, 0]
    while mask.ndim < 3:
        mask = mask[None]
    return mx.contiguous(mx.broadcast_to(mask, (B, L, N)))


def decode_sdpa(q, K, V, scale: float, mask=None, group: int = 64,
                bits: int = 8, same_layout: bool = False):
    """Attention of q (B, H, L, D) over K, V -- each the kvquant triple
    (packed uint32, scales, biases) of shape (B, Hk, N, *) -- with mlx
    sdpa's semantics. Returns (B, H, L, D) in q's dtype.

    `same_layout`: each bias is laid out exactly like its scale (same
    shape and strides), so the kernel indexes both with the scale's
    strides. True only for kvquant's caches, which allocate and slice the
    two together; any other caller reads each bias at its own strides."""
    B, H, L, D = q.shape
    Hk, N = K[0].shape[1], K[0].shape[2]
    rep = H // Hk
    R = rep * L
    nblk = (N + NB - 1) // NB
    qq = mx.contiguous(q.reshape(B, Hk, R, D))
    ins = [qq, mx.array([scale], mx.float32), *K, *V]
    m = _mask_rows(mask, B, L, N)
    if m is not None:
        ins.append(mx.contiguous(m))
    slots = nblk * SG
    rc = next(c for c in range(min(RC, R), 0, -1) if R % c == 0)
    o, mm, ll = _kernel(m is not None, same_layout)(
        inputs=ins,
        template=[("BITS", bits), ("D", D), ("G", group), ("R", R),
                  ("L", L), ("NB", NB), ("SG", SG), ("RC", rc)],
        grid=(nblk * 32 * SG, B * Hk * (R // rc), 1), threadgroup=(32 * SG, 1, 1),
        output_shapes=[(B, Hk, R, slots, D), (B, Hk, R, slots),
                       (B, Hk, R, slots)],
        output_dtypes=[mx.float32] * 3)
    out = _combine()(
        inputs=[o, mm, ll], template=[("T", q.dtype), ("NBLK", slots),
                                      ("D", D)],
        grid=(D, B * Hk * R, 1), threadgroup=(min(D, 256), 1, 1),
        output_shapes=[(B, Hk, R, D)], output_dtypes=[q.dtype])[0]
    return out.reshape(B, H, L, D)


# --- the dispatch the model's attention calls ------------------------------

def _base_sdpa():
    from mlx_lm.models import base
    return getattr(base, "_kl_orig_sdpa", base.scaled_dot_product_attention)


def sdpa(queries, keys, values, cache, scale, mask, sinks=None):
    """mlx-lm's scaled_dot_product_attention, taking the 8-bit kernel when
    `cache` is a kvquant cache that just handed back exactly these keys
    for a short query."""
    hit = getattr(cache, "kv8_fetch", None)
    if hit is not None:
        cache.kv8_fetch = None
        kret, K, V, g = hit
        if kret is not keys:
            count(False, "keys changed between the cache and the attention")
        elif not supports(queries, K, V, 8, g, mask, sinks):
            count(False, f"unsupported: q {tuple(queries.shape)}, mask "
                         f"{getattr(mask, 'shape', mask)}, sinks "
                         f"{sinks is not None}")
        else:
            count(True)
            return decode_sdpa(queries, K, V, scale, mask, g, 8,
                               same_layout=True)
    return _base_sdpa()(queries, keys, values, cache=cache, scale=scale,
                        mask=mask, sinks=sinks)


def patch_model(model) -> int:
    """Rebind `scaled_dot_product_attention` in every module the model's
    layers are defined in (where it is mlx-lm's function) to `sdpa`.
    Idempotent; harmless for bf16 caches. Returns how many modules."""
    from mlx_lm.models import base
    orig = _base_sdpa()
    base._kl_orig_sdpa = orig
    names = {type(m).__module__ for _, m in model.named_modules()}
    names.add(type(model).__module__)
    n = 0
    for name in names:
        mod = sys.modules.get(name)
        f = getattr(mod, "scaled_dot_product_attention", None)
        if mod is not None and (f is orig or f is sdpa):
            mod.scaled_dot_product_attention = sdpa  # type: ignore[attr-defined]  # patching a module's own attribute
            n += 1
    return n
