# 8-bit KV decode attention (research, 2026-09-28)

Decode over an 8-bit KV cache today dequantizes the whole cache every step
and then runs bf16 sdpa. On the M4 (Qwen3.6-35B-A3B, 10 full-attention
layers, 2 KV heads, head dim 256), microseconds per layer-step:

| context | bf16 sdpa | dequant + sdpa (now) | mlx quantized sdpa | fused.py kernel |
|---|---|---|---|---|
| 16k | ~140 | ~275 | 314 | ~240 |
| 32k | ~280 | ~540 | 732-757 | ~431 |

End to end: 8-bit costs no measurable prefill; decode +7% at 6k, +18% at
16k context. The fused Metal kernel reads 8-bit K/V directly (split-K,
one-launch combine), max error 1.4e-4 vs float32 at 16k; it beats mlx's own
quantized attention by 1.3-1.7x, but is still ~1.7x bf16. Not wired in yet.

- e2e.py: real model, prefill and decode, bf16 vs 8-bit interleaved
- bench.py, bench2.py: 10-layer micro-benchmark
- dec.py, variants.py: decode attention variants
- fused.py, fused2.py: the Metal kernels; chk.py: correctness check
