import sys, time, mlx.core as mx
from mlx_lm.models.cache import KVCache
from mlx_lm.models.base import scaled_dot_product_attention as sdpa_base
from knurlogic.engine.kvquant import QuantKVCache
H, Hkv, D, C = 16, 2, 256, 512
def attn(q, k, v, S):
    mask = "causal" if S > 1 else None
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=D**-0.5, mask=mask)
def run(make, N, dec=200, part="all"):
    c = make(); mx.random.seed(0)
    x = [mx.random.normal((1, H, C, D)).astype(mx.bfloat16), mx.random.normal((1, Hkv, C, D)).astype(mx.bfloat16)]
    mx.eval(x)
    t0 = time.perf_counter()
    for i in range(0, N, C):
        k, v = c.update_and_fetch(x[1], x[1])
        o = attn(x[0], k, v, C) if part == "all" else k
        mx.eval(o)
    t1 = time.perf_counter()
    q1 = x[0][:, :, :1]; k1 = x[1][:, :, :1]
    for i in range(dec):
        k, v = c.update_and_fetch(k1, k1)
        mx.eval(attn(q1, k, v, 1))
    t2 = time.perf_counter()
    return t1 - t0, t2 - t1
makers = {"bf16": KVCache, "q8": lambda: QuantKVCache(8)}
extra = sys.argv[2:] 
if extra:
    import importlib; m = importlib.import_module(extra[0]); makers.update(m.MAKERS)
for N in (4096, 16384):
    for name, mk in makers.items():
        run(mk, 1024)
        r = [run(mk, N) for _ in range(3)]
        p = sorted(a for a, b in r)[1]; d = sorted(b for a, b in r)[1]
        print(f"N={N:6d} {name:10s} prefill/layer {p*1e3:7.1f} ms  x10 layers {p*10:5.2f}s | decode200/layer {d*1e3:7.1f} ms x10 {d*10:5.2f}s", flush=True)
