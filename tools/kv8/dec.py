import time, mlx.core as mx
from mlx_lm.models.base import quantized_scaled_dot_product_attention as qsdpa
H, Hkv, D, g, b = 16, 2, 256, 64, 8
R = H // Hkv; sc = D ** -0.5
def a_dq(q, K, V): return mx.fast.scaled_dot_product_attention(q, mx.dequantize(*K, group_size=g, bits=b), mx.dequantize(*V, group_size=g, bits=b), scale=sc)
def b_q(q, K, V): return qsdpa(q, K, V, scale=sc, mask=None, group_size=g, bits=b)
def c_mix(q, K, V):
    qq = (q * sc).reshape(1, Hkv, R, 1, D).reshape(1, Hkv, R, D)
    s = mx.quantized_matmul(qq, *K, transpose=True, group_size=g, bits=b)     # (1,Hkv,R,N)
    p = mx.softmax(s.astype(mx.float32), axis=-1).astype(q.dtype)
    Vd = mx.dequantize(*V, group_size=g, bits=b)
    return (p @ Vd).reshape(1, H, 1, D)
def c2(q, K, V):
    qq = (q * sc).reshape(1, Hkv, R, D)
    s = mx.quantized_matmul(qq, *K, transpose=True, group_size=g, bits=b)
    p = mx.softmax(s.astype(mx.float32), axis=-1).astype(q.dtype)
    return mx.quantized_matmul(p, *V, transpose=False, group_size=g, bits=b).reshape(1, H, 1, D)
import fused, fused2
def e2(q, K, V): return fused2.decode_sdpa(q, K, V, sc, None, g, b)
def e_f(q, K, V): return fused.decode_sdpa(q, K, V, sc, None, g, b)
def d_bf(q, K, V): return mx.fast.scaled_dot_product_attention(q, K, V, scale=sc)
for N in (4096, 16384, 32768):
    k = mx.random.normal((1, Hkv, N, D)).astype(mx.bfloat16); v = mx.random.normal((1, Hkv, N, D)).astype(mx.bfloat16)
    K = mx.quantize(k, group_size=g, bits=b); V = mx.quantize(v, group_size=g, bits=b)
    q = mx.random.normal((1, H, 1, D)).astype(mx.bfloat16); mx.eval(k, v, K, V, q)
    ref = d_bf(q, k, v)
    for name, f, KK, VV in (("bf16", d_bf, k, v), ("dequant+sdpa", a_dq, K, V), ("qsdpa", b_q, K, V), ("qmmK+dqV", c_mix, K, V), ("qmmK+qmmV", c2, K, V), ("fused", e_f, K, V), ("fused2", e2, K, V)):
        o = f(q, KK, VV); mx.eval(o); err = float(mx.abs(o - ref).max())
        ts = []
        for rep in range(5):
            t = time.perf_counter()
            for _ in range(20):
                mx.eval([f(q, KK, VV) for _l in range(10)])
            ts.append((time.perf_counter() - t) / 200 * 1e6)
        print(f"N={N:6d} {name:14s} {sorted(ts)[2]:7.1f} us/layer-step  maxerr {err:.4f}", flush=True)
