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

## Wired in (engine/kvattn.py)

Decode steps (query length 1) over an 8-bit cache now take the kernel;
prefill, MTP verifies and 6/4 bits keep dequantize + sdpa.
`KNURLOGIC_KV_KERNEL=off` turns it off.

Tuning (tune.py, layers CHAINED as in a model -- the numbers above ran 10
independent calls, which overlap on the GPU and flatter a kernel with few
threadgroups): the research config (256 keys per block, 8 simdgroups, all
8 query rows per threadgroup) was 230 / 503 us per layer at 6k / 16k, i.e.
slower than dequantize + sdpa (146 / 291) once serial -- end to end it lost
4-7%. Splitting a KV head's query rows over threadgroups two at a time
(RC=2) with 2 simdgroups: 114 / 208 us. RC=1 or 4 and NB 32-512 were all
worse. For 2-4 query rows (MTP verify) the kernel is 1.2-1.7x slower than
dequantize + sdpa at every config tried, so those stay there.

End to end (e2e_kernel.py), M4, Qwen3.6-35B-A3B VQ 3.4, 200 decode tokens,
modes interleaved, 3 reps, decode tok/s median [all]:

| context | bf16 | 8-bit dequantize+sdpa | 8-bit kernel |
|---|---|---|---|
| 6k  | 70.01 [68.93 70.02 70.01] | 63.88 [64.53 63.88 63.81] | 66.29 [66.34 64.53 66.29] |
| 16k | 65.53 [65.64 64.63 65.53] | 54.01 [52.62 54.01 56.65] | 61.92 [60.60 61.92 62.34] |

Needle (a code planted mid-haystack, 9.5k and 26k prompt tokens): the exact
answer in all three modes at both lengths.
