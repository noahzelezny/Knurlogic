import mlx.core as mx
SRC = r"""
    uint lane = thread_index_in_simdgroup;
    uint r = simdgroup_index_in_threadgroup;
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
    float qv[DPL]; float qs = 0.f;
    size_t qo = ((size_t)(b * Hk + h) * R + r) * D + d0;
    for (int i = 0; i < DPL; i++) { qv[i] = float(q[qo + i]) * scale[0]; qs += qv[i]; }
    float o[DPL]; for (int i = 0; i < DPL; i++) o[i] = 0.f;
    float m = -INFINITY, l = 0.f;
    size_t kb = (size_t)b * kw_strides[0] + (size_t)h * kw_strides[1];
    size_t sb = (size_t)b * ks_strides[0] + (size_t)h * ks_strides[1];
    size_t vb = (size_t)b * vw_strides[0] + (size_t)h * vw_strides[1];
    size_t tb = (size_t)b * vs_strides[0] + (size_t)h * vs_strides[1];
    for (int n = n0; n < n1; n++) {
        MASKLINE
        size_t kr = kb + (size_t)n * kw_strides[2] + lane * WPL;
        float acc = 0.f;
        for (int w = 0; w < WPL; w++) {
            uint word = kw[kr + w];
            for (int e = 0; e < EPW; e++) acc += qv[w * EPW + e] * float((word >> (e * BITS)) & MASKB);
        }
        size_t si = sb + (size_t)n * ks_strides[2] + gi;
        float part = float(ks[si]) * acc + float(kbias[si]) * qs;
        float sc = simd_sum(part);
        float mn = max(m, sc);
        float corr = fast::exp(m - mn);
        float p = fast::exp(sc - mn);
        l = l * corr + p; m = mn;
        size_t vr = vb + (size_t)n * vw_strides[2] + lane * WPL;
        size_t ti = tb + (size_t)n * vs_strides[2] + gi;
        float vsc = float(vs[ti]), vbi = float(vbias[ti]);
        for (int w = 0; w < WPL; w++) {
            uint word = vw[vr + w];
            for (int e = 0; e < EPW; e++) {
                int i = w * EPW + e;
                o[i] = o[i] * corr + p * (vsc * float((word >> (e * BITS)) & MASKB) + vbi);
            }
        }
    }
    uint nblk = threadgroups_per_grid.x;
    size_t oi = ((size_t)bh * R + r) * nblk + blk;
    for (int i = 0; i < DPL; i++) out_o[oi * D + d0 + i] = o[i];
    if (lane == 0) { out_m[oi] = m; out_l[oi] = l; }
"""
_K = {}
def _kernel(has_mask):
    k = _K.get(has_mask)
    if k is None:
        ins = ["q", "scale", "kw", "ks", "kbias", "vw", "vs", "vbias"] + (["mask"] if has_mask else [])
        k = _K[has_mask] = mx.fast.metal_kernel(
            name=f"kl_qsdpa_decode_{int(has_mask)}", input_names=ins,
            output_names=["out_o", "out_m", "out_l"], source=SRC.replace("MASKLINE", "if (!mask[(size_t)b * mask_strides[0] + (size_t)n * mask_strides[1]]) continue;" if has_mask else ""),
            ensure_row_contiguous=False)
    return k

NB = 256
import os; NB = int(os.environ.get("NB", NB))
def decode_sdpa(q, K, V, scale, mask=None, group_size=64, bits=8):
    """q (B,H,1,D); K,V (packed, scales, biases) of shape (B,Hk,N,*)."""
    B, H, _, D = q.shape
    Hk, N = K[0].shape[1], K[0].shape[2]
    R = H // Hk
    nblk = (N + NB - 1) // NB
    qq = mx.contiguous(q.reshape(B, Hk, R, D))
    ins = [qq, mx.array([scale], mx.float32), *K, *V]
    if mask is not None:
        ins.append(mask.reshape(B, N))
    o, m, l = _kernel(mask is not None)(
        inputs=ins,
        template=[("BITS", bits), ("D", D), ("G", group_size), ("R", R), ("NB", NB)],
        grid=(nblk * 32 * R, B * Hk, 1), threadgroup=(32 * R, 1, 1),
        output_shapes=[(B, Hk, R, nblk, D), (B, Hk, R, nblk), (B, Hk, R, nblk)],
        output_dtypes=[mx.float32] * 3)
    return _combine()(inputs=[o, m, l], template=[("T", q.dtype), ("NBLK", nblk), ("D", D)],
                      grid=(D, B * H, 1), threadgroup=(min(D, 256), 1, 1),
                      output_shapes=[(B, H, 1, D)], output_dtypes=[q.dtype])[0]
CSRC = r"""
    uint d = thread_position_in_grid.x; uint row = thread_position_in_grid.y;
    size_t base = (size_t)row * NBLK;
    float M = -INFINITY;
    for (int i = 0; i < NBLK; i++) M = max(M, out_m[base + i]);
    float num = 0.f, den = 0.f;
    for (int i = 0; i < NBLK; i++) { float w = fast::exp(out_m[base + i] - M); den += w * out_l[base + i]; num += w * out_o[(base + i) * D + d]; }
    out[(size_t)row * D + d] = static_cast<T>(num / den);
"""
_C = []
def _combine():
    if not _C:
        _C.append(mx.fast.metal_kernel(name="kl_qsdpa_combine", input_names=["out_o", "out_m", "out_l"],
                                       output_names=["out"], source=CSRC))
    return _C[0]

