# The 8-bit KV cache

## src/knurlogic/engine/kvattn.py

Without this module, a decode step over kvquant's cache dequantizes the
whole cache to bf16 and runs mx.fast sdpa: every step reads the 8-bit cache
AND writes and reads a bf16 copy of it. The Metal kernel reads the packed
K/V, their scales and biases directly (split-K over key blocks, one launch;
a second tiny launch combines the blocks' softmax partials).

Measured on an M4 Max (128 GB), Qwen3.6-35B-A3B VQ 3.4 (10 attention
layers, 2 KV heads, head dim 256), end-to-end decode tok/s, bf16 / 8-bit
dequantize+sdpa / 8-bit kernel, interleaved, median of 3: 70.0 / 63.9 /
66.3 at 6k context, 65.5 / 54.0 / 61.9 at 16k.

HOW IT IS WIRED, without touching a family's attention: kvquant's caches
still return dequantized arrays from `update_and_fetch` -- lazily, so if
nothing reads them they are never computed (gemma4's KV-shared layers do
read them: they attend over the owner's returned keys with cache=None, so
those layers still pay the dequantize) -- and, for a decode step, remember
(returned keys, packed K, packed V). `install` rebinds the
`scaled_dot_product_attention` name in the model's own modules to `sdpa`,
which takes the kernel when it is handed exactly the keys that cache just
returned, and otherwise is mlx-lm's function unchanged. Prefill (long
queries) stays dequantize+sdpa: 8-bit costs it nothing measurable.

Routed here: decode steps (kvquant.KERNEL_MAX_QUERY; an MTP verify of 2-4
rows measures slower than dequantize+sdpa and stays there). The kernel
itself takes: 8 bits, K and V the same head dim (128, 256 or 512: a
multiple of 128, so each lane reads whole 32-bit words), any GQA ratio, up
to `MAX_ROWS` query rows per KV head, no mask / "causal" / a boolean mask
shared across heads (a batch's left padding, ragged rows). Anything else --
sinks, an additive or per-head mask, head dim 64, GLM's MLA latent (its own
attention function, not rebound) -- takes the dequantize path.

One difference from mlx sdpa: a query row with EVERY key masked returns 0
here, where mlx sdpa returns NaN.

`KNURLOGIC_KV_KERNEL=off` turns it off (tuning/settings.py), for A/B.
Whether it is live is counted, not assumed: `STATS` (hits: decode
attentions it served; misses: a decode step the cache offered it that took
dequantize + sdpa -- keys transformed after the fetch, as GLM's MLA does,
an unsupported shape, or an attention that calls mx.fast directly) is
state.SERVED["kv_kernel"] and /status.json's kv_kernel, and the first hit
and first miss are each logged once.

## src/knurlogic/engine/kvquant.py

What mlx 0.32.3 / mlx-lm 0.32.0 provide, and what they do not:

  mx.quantize / mx.dequantize   affine, per-group scale + bias, bits 2-8
                                including 6 (and 3, 5); groups of 32/64/128
                                along the last axis.
  mlx_lm QuantizedKVCache       ONE sequence. There is no batched quantized
                                cache in mlx-lm 0.32.0 -- BatchKVCache and
                                BatchRotatingKVCache are bf16 only, and
                                `_make_cache` refuses anything else.
  quantized SDPA                mlx_lm.models.base routes to
                                `quantized_scaled_dot_product_attention`
                                (two mx.quantized_matmul) when the cache has
                                a `bits` attribute; mx.fast.scaled_dot_
                                product_attention itself takes no quantized
                                K/V. Only attention written against
                                mlx_lm.models.base takes that path; the
                                vendored families each have their own.

So knurlogic keeps its own pair: a single-row cache (what a prefill and the
prompt cache hold) and a batched one (what the decode loop holds), both
STORING the quantized triple and handing the attention code back
dequantized arrays of the input dtype. Every family's attention is then
untouched -- including gemma4's KV-shared layers, which reuse the returned
arrays -- and neither class has a `bits` attribute, so mlx-lm's base SDPA
does not mistake the dequantized arrays for quantized ones (the attribute
is `kv_bits`).

The cost of that choice: memory is what shrinks (the stored cache, the
prompt cache's entries). Dequantizing per layer per step is MORE memory
traffic than bf16, so at 8 bits a decode step (query length 1) instead goes
through engine/kvattn's Metal kernel, which reads the packed K/V directly.
The dequantized arrays still returned beside it are lazy: computed only if
something reads them -- which gemma4's KV-shared layers do (they attend
over the owner layer's returned keys with cache=None), so there the copy is
still made for those layers. On an M4 Max (128 GB), Qwen3.6-35B-A3B, decode
tok/s bf16 / dequantize / kernel: 70.0 / 63.9 / 66.3 at 6k context, 65.5 /
54.0 / 61.9 at 16k. Prefill and 6/4 bits stay dequantize + sdpa.

What is quantized: mlx-lm's plain `KVCache` (exact type) in the list
`model.make_cache()` returns, and inside a CacheList; and a family's own
cache class its manifest names (`kv_quant.caches`): qwen4_exp's attention
cache (K/V quantized, the sparse indexer's keys kept exact) and GLM's MLA
latent (quantized; the DSA indexer's cache kept exact). Not: recurrent
state (ArraysCache: deltanet/SSM), sliding windows (RotatingKVCache,
bounded by their window), or the MTP head's draft cache (one layer; the
draft stays bf16). Rollback and prompt-cache trims move the write index
only, exactly as for bf16: the stale tail is overwritten by the next
update.
