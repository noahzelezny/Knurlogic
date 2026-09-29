"""The wired path at layer level: 10 QuantKVCaches filled to N, one decode
step each through update_and_fetch + kvattn.sdpa, kernel on vs off, and the
bare kernel on the cache's sliced views vs contiguous copies.

    python layer.py 6000,16000
"""
import sys
import time

import mlx.core as mx

from knurlogic.engine import kvattn, kvquant

H, Hk, D, LAYERS, IT = 16, 2, 256, 10, 50


def bench(f):
    for _ in range(5):
        mx.eval(f())
    t = time.perf_counter()
    for _ in range(IT):
        mx.eval(f())
    return (time.perf_counter() - t) / IT / LAYERS * 1e6


for N in [int(x) for x in sys.argv[1].split(",")]:
    mx.random.seed(0)
    cs = [kvquant.QuantKVCache(8) for _ in range(LAYERS)]
    for c in cs:
        for i in range(0, N, 1024):
            k = mx.random.normal((1, Hk, min(1024, N - i), D)).astype(mx.bfloat16)
            c.update_and_fetch(k, k)
            mx.eval(c.keys, c.values)
    q = mx.random.normal((1, H, 1, D)).astype(mx.bfloat16)
    k1 = mx.random.normal((1, Hk, 1, D)).astype(mx.bfloat16)
    mx.eval(q, k1)

    def step(kernel):
        kvquant.KERNEL = kernel
        out = []
        for c in cs:
            kk, vv = c.update_and_fetch(k1, k1)
            c.offset -= 1                       # stay at N
            out.append(kvattn.sdpa(q, kk, vv, c, D ** -0.5, None))
        return out
    print(f"N={N} wired off {bench(lambda: step(False)):7.1f} us/layer  "
          f"on {bench(lambda: step(True)):7.1f}", flush=True)
    views = [(kvquant._slice(c.keys, 0, N), kvquant._slice(c.values, 0, N))
             for c in cs]
    copies = [(tuple(mx.contiguous(p) for p in K),
               tuple(mx.contiguous(p) for p in V)) for K, V in views]
    mx.eval(copies)
    for name, kv in (("views", views), ("copies", copies)):
        t = bench(lambda: [kvattn.decode_sdpa(q, K, V, D ** -0.5)
                           for K, V in kv])
        print(f"N={N} bare kernel on {name:6s} {t:7.1f} us/layer", flush=True)
