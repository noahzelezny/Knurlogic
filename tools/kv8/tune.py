"""engine/kvattn's kernel across key-block sizes (NB), simdgroups per
threadgroup (SG) and query rows per threadgroup (RC), against
dequantize+sdpa: 10 layers of one decode step at Qwen3.6-35B-A3B shapes
(16 q heads, 2 KV heads, head dim 256), CHAINED -- each layer's query is
the last one's output, as in a model, so the calls run one at a time (10
independent calls overlap on the GPU and flatter a low-parallelism
kernel).

    python tune.py 16384,32768 [L]
"""
import itertools
import sys
import time

import mlx.core as mx

from knurlogic.engine import kvattn

Ns = [int(x) for x in sys.argv[1].split(",")]
L = int(sys.argv[2]) if len(sys.argv) > 2 else 1
H, Hk, D, LAYERS, IT = 16, 2, 256, 10, 50


def bench(f):
    for _ in range(5):
        mx.eval(f())
    t = time.perf_counter()
    for _ in range(IT):
        mx.eval(f())
    return (time.perf_counter() - t) / IT / LAYERS * 1e6


for N in Ns:
    mx.random.seed(0)
    Ks = [tuple(mx.quantize(mx.random.normal((1, Hk, N, D)).astype(
        mx.bfloat16), group_size=64, bits=8)) for _ in range(LAYERS)]
    q = mx.random.normal((1, H, L, D)).astype(mx.bfloat16)
    mx.eval(Ks, q)
    mask = "causal" if L > 1 else None

    def chain(f):
        x = q
        for K in Ks:
            x = f(x, K)
        return x

    def old():
        return chain(lambda x, K: mx.fast.scaled_dot_product_attention(
            x, mx.dequantize(*K, group_size=64, bits=8).astype(q.dtype),
            mx.dequantize(*K, group_size=64, bits=8).astype(q.dtype),
            scale=D ** -0.5, mask=mask))
    print(f"N={N} L={L} dequant+sdpa {bench(old):7.1f} us/layer", flush=True)
    for nb, sg, rc in itertools.product((32, 64, 128, 256, 512),
                                        (1, 2, 4, 8), (1, 2, 4, 8)):
        if nb < sg:
            continue
        kvattn.NB, kvattn.SG, kvattn.RC = nb, sg, rc

        def new():
            return chain(lambda x, K: kvattn.decode_sdpa(x, K, K, D ** -0.5,
                                                         mask))
        print(f"N={N} L={L} NB={nb:4d} SG={sg} RC={rc} {bench(new):7.1f} "
              f"us/layer", flush=True)
