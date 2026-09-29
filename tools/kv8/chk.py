import mlx.core as mx, fused2 as fused, time
g,b,D,H,Hk=64,8,256,16,2
for N in (100, 1000, 16384):
    k = mx.random.normal((1,Hk,N,D)).astype(mx.bfloat16); v = mx.random.normal((1,Hk,N,D)).astype(mx.bfloat16)
    K = mx.quantize(k,group_size=g,bits=b); V = mx.quantize(v,group_size=g,bits=b)
    kd = mx.dequantize(*K,group_size=g,bits=b).astype(mx.float32); vd = mx.dequantize(*V,group_size=g,bits=b).astype(mx.float32)
    q = mx.random.normal((1,H,1,D)).astype(mx.bfloat16)
    ref = mx.fast.scaled_dot_product_attention(q.astype(mx.float32), kd, vd, scale=D**-.5)
    o = fused.decode_sdpa(q, K, V, D**-.5, None, g, b)
    print(N, float(mx.abs(o.astype(mx.float32)-ref).max()), float(mx.abs(ref).max()))
