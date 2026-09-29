import os, mlx.core as mx
from fused import _combine
SG = int(os.environ.get("SG", 8))
NB = int(os.environ.get("NB", 256))
SRC = r"""
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint blk = threadgroup_position_in_grid.x;
    uint bh = threadgroup_position_in_grid.y;
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
    float qv[R][DPL]; float qs[R];
    float o[R][DPL]; float m[R]; float l[R];
    float sc0 = scale[0];
    for (int r = 0; r < R; r++) {
        size_t qo = ((size_t)(b * Hk + h) * R + r) * D + d0;
        qs[r] = 0.f; m[r] = -INFINITY; l[r] = 0.f;
        for (int i = 0; i < DPL; i++) { qv[r][i] = float(q[qo + i]) * sc0; qs[r] += qv[r][i]; o[r][i] = 0.f; }
    }
    size_t kb = (size_t)b * kw_strides[0] + (size_t)h * kw_strides[1];
    size_t sb = (size_t)b * ks_strides[0] + (size_t)h * ks_strides[1];
    size_t vb = (size_t)b * vw_strides[0] + (size_t)h * vw_strides[1];
    size_t tb = (size_t)b * vs_strides[0] + (size_t)h * vs_strides[1];
    for (int n = n0 + sg; n < n1; n += SG) {
        MASKLINE
        size_t kr = kb + (size_t)n * kw_strides[2] + lane * WPL;
        float kx[DPL];
        for (int w = 0; w < WPL; w++) {
            uint word = kw[kr + w];
            for (int e = 0; e < EPW; e++) kx[w * EPW + e] = float((word >> (e * BITS)) & MASKB);
        }
        size_t si = sb + (size_t)n * ks_strides[2] + gi;
        float ksc = float(ks[si]), kbi = float(kbias[si]);
        size_t vr = vb + (size_t)n * vw_strides[2] + lane * WPL;
        size_t ti = tb + (size_t)n * vs_strides[2] + gi;
        float vsc = float(vs[ti]), vbi = float(vbias[ti]);
        float vx[DPL];
        for (int w = 0; w < WPL; w++) {
            uint word = vw[vr + w];
            for (int e = 0; e < EPW; e++) vx[w * EPW + e] = vsc * float((word >> (e * BITS)) & MASKB) + vbi;
        }
        for (int r = 0; r < R; r++) {
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
    for (int r = 0; r < R; r++) {
        size_t oi = ((size_t)bh * R + r) * nblk + bslot;
        for (int i = 0; i < DPL; i++) out_o[oi * D + d0 + i] = o[r][i];
        if (lane == 0) { out_m[oi] = m[r]; out_l[oi] = l[r]; }
    }
"""
_K = {}
def _kernel(has_mask):
    k = _K.get(has_mask)
    if k is None:
        ins = ["q", "scale", "kw", "ks", "kbias", "vw", "vs", "vbias"] + (["mask"] if has_mask else [])
        line = "if (!mask[(size_t)b * mask_strides[0] + (size_t)n * mask_strides[1]]) continue;" if has_mask else ""
        k = _K[has_mask] = mx.fast.metal_kernel(name=f"kl_qsdpa2_{int(has_mask)}", input_names=ins,
            output_names=["out_o", "out_m", "out_l"], source=SRC.replace("MASKLINE", line), ensure_row_contiguous=False)
    return k
def decode_sdpa(q, K, V, scale, mask=None, group_size=64, bits=8):
    B, H, _, D = q.shape
    Hk, N = K[0].shape[1], K[0].shape[2]
    R = H // Hk
    nblk = (N + NB - 1) // NB
    ins = [mx.contiguous(q.reshape(B, Hk, R, D)), mx.array([scale], mx.float32), *K, *V]
    if mask is not None: ins.append(mask.reshape(B, N))
    o, m, l = _kernel(mask is not None)(inputs=ins,
        template=[("BITS", bits), ("D", D), ("G", group_size), ("R", R), ("NB", NB), ("SG", SG)],
        grid=(nblk * 32 * SG, B * Hk, 1), threadgroup=(32 * SG, 1, 1),
        output_shapes=[(B, Hk, R, nblk * SG, D), (B, Hk, R, nblk * SG), (B, Hk, R, nblk * SG)],
        output_dtypes=[mx.float32] * 3)
    return _combine()(inputs=[o, m, l], template=[("T", q.dtype), ("NBLK", nblk * SG), ("D", D)],
                      grid=(D, B * H, 1), threadgroup=(min(D, 256), 1, 1),
                      output_shapes=[(B, H, 1, D)], output_dtypes=[q.dtype])[0]
