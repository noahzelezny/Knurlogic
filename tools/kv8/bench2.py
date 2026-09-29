import sys, time, mlx.core as mx
from mlx_lm.models.cache import KVCache
from knurlogic.engine import kvquant as kq
H, Hkv, D, C, L = 16, 2, 256, 512, 10
from mlx_lm.models.base import quantized_scaled_dot_product_attention as qsdpa
def attn(q, k, v, S):
    if isinstance(k, tuple): return qsdpa(q, k, v, scale=D**-0.5, mask=None, group_size=64, bits=8)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=D**-0.5, mask="causal" if S > 1 else None)
def run(kind, N, dec=200, clear=True):
    cs = [kind() for _ in range(L)]; mx.random.seed(0)
    q = mx.random.normal((1, H, C, D)).astype(mx.bfloat16); k = mx.random.normal((1, Hkv, C, D)).astype(mx.bfloat16); mx.eval(q, k)
    t0 = time.perf_counter()
    for i in range(0, N, C):
        x = q
        for c in cs:
            kk, vv = c.update_and_fetch(k + x[:, :2].mean() * 0, k)
            x = attn(x, kk, vv, C)
        mx.eval(x, [c.state for c in cs])
        if clear: mx.clear_cache()
    t1 = time.perf_counter()
    q1 = q[:, :, :1]; k1 = k[:, :, :1]
    for i in range(dec):
        x = q1
        for c in cs:
            kk, vv = c.update_and_fetch(k1 + x[:, :2].mean() * 0, k1); x = attn(x, kk, vv, 1)
        mx.eval(x)
    return t1 - t0, time.perf_counter() - t1
import variants
makers = dict(bf16=KVCache, q8=lambda: kq.QuantKVCache(8), **variants.MAKERS)
sel = sys.argv[1].split(",")
for N in [int(n) for n in sys.argv[2].split(",")]:
    for name in sel:
        mk = makers[name]; run(mk, 1024, 5)
        r = [run(mk, N) for _ in range(3)]
        p = sorted(a for a, b in r)[1]; d = sorted(b for a, b in r)[1]
        print(f"N={N:6d} {name:10s} prefill(10 layers) {p:6.3f}s  decode200 {d:6.3f}s", flush=True)
