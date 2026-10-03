# Vendored architecture modules

Each entry is a claim that this file is the arithmetic the
artifacts were validated against -- not merely that it imports.

## deepseek_v4.py

- taken: 2026-09-29
- from: `mlx_lm/models/deepseek_v4.py` of the project's own
  mlx-lm fork (installed from a local `mlx_lm-0.31.9` wheel), vendored with
  its author's permission. No exo code is in it.
- mlx-lm base: 0.31.9 (the fork); runs here on the pinned 0.32.0 (0.31.3
  until 2026-10-02).
- fork file sha256: `78bf144caae1e1067f2910d070e3a71fe6f2d11704691cb2a272c9aebf0a13ef`
- vendored sha256: `d03bbf2c55c5ef9ddba57c56ee06ea602e4963e3d1158454b81eed4e4e9cad7f`
  (the fork's file plus the edits below; every one is marked
  `knurlogic edit` in the source)
- the env also holds `deepseek_v4.py.bak` (byte-identical to the file
  taken) and `.bak2` (differs only in `EXPERTS_CHUNK = 4` in sanitize's
  expert stacking; the file taken has 32).
- import closure: `.base` (BaseModelArgs, scaled_dot_product_attention),
  `.cache` (RotatingKVCache, BatchRotatingKVCache), `.switch_layers`
  (SwitchGLU). All exist in 0.31.3; `base.py` and `switch_layers.py` are
  byte-identical between 0.31.3 and the fork's 0.31.9, and `cache.py`
  differs only in ArraysCache extract/merge None-slot handling and a
  BatchKVCache `mx.depends` -- neither class is used by this module. So no
  fork-only helper is vendored. On 0.32.0: `base.py` adds a mask
  expand_dims in the quantized SDPA (n_repeats > 1, 4-D mask) and
  `switch_layers.py` always stop_gradients the expert indices (no forward
  change); `cache.py` folds meta_state into `state` (from_state takes one
  argument) -- RotatingKVCache's state now carries its scalars. The tiny
  golden still matches the fork to 2.0e-06 (1.7e-06 on mlx 0.31.2).
- not taken: the fork's `utils.py` F8_E8M0 loader shim. It only matters
  for a raw DeepSeek FP8 checkpoint; the mlx-community conversion is
  already MLX-quantized (8-bit affine g64, routed experts mxfp4 g32) and
  holds no F8_E8M0 tensors.
- side effect at import: registers a minimal `deepseek_v4` AutoConfig with
  transformers (exist_ok), so the tokenizer loads.
- validated: tiny random-weight configs only (tests/test_deepseek_v4_arch.py,
  tests/goldens/build_deepseek_v4.py). Not yet run on the real
  DeepSeek-V4-Flash weights: pins.json is empty until `knurlogic smoke
  --pin` passes on it.

### Edits 1-8, each found by a tiny-model test the fork fails

1. **Indexer query RoPE** (`Indexer.__call__`). The indexer's q is
   `[B, S, H, D]`, and `mx.fast.rope` puts positions on axis -2 -- the
   heads: head h was rotated at `offset + h` for every token. The keys in
   its pool are rotated by their true positions, so the top-k rows were
   chosen by scores that depended on the head index, and a prefilled
   prompt chose differently from the same tokens decoded one at a time.
   Now rotated with the sequence on axis -2 (as the attention's own q is).
   This is not only prefill-vs-decode parity: a single-token decode step
   was also rotated at `offset + h` per head, so DECODE output changes too.
   Changes every choice the indexer makes -- prefill and decode -- once a
   pool holds more than `index_topk` (512) rows, i.e. past ~2048 tokens on
   Flash.
2. **Indexer prefill visibility** (`Indexer.__call__`). A prefill query
   ranked all pool rows, its future ones included, then the attention mask
   dropped the future ones -- leaving fewer than top-k visible rows. Now
   the invisible rows are masked before the top-k, as a decode step (all
   rows in its past) and DeepSeek's reference do.
3. **Ragged decode mask** (`V4Attention.__call__`). A decode step skipped
   the mask entirely. With rows of different lengths in one batch (the
   local cache a BatchRotatingKVCache), a short row attended to its empty
   window slots and to the zero rows padding its pools. Now, only in that
   case, the batch cache's own window mask plus per-row pool visibility.
   Single-row and same-length batches take the fork's mask-free path.
4. **Overlap carry per row** (`Compressor.__call__`, `_ragged_prev`). When
   one row completed a ratio-4 window, every row's `prev_kv/prev_gate`
   carry was replaced. Now only rows that emitted move their carry.
5. **Missing carry is -inf, not 0** (`_fill_of`, merge / extend). A row
   with no previous window, merged beside one that has one, got a zero
   gate (a real softmax weight on an empty window) instead of -inf.
6. **Same-count emits onto ragged pools** (`update_pool`). When every row
   emitted one row but the pools already differed in length, the uniform
   concat put a short row's new row after its padding and dropped the
   per-row lengths.
7. **extend read lengths after concatenating** (`extend`). The carry and
   pool lengths were read off the already-concatenated buffers, so the
   shorter row took the longer one's length and emitted a window early.
8. **Hot-path buffer length** (`_settle`, `_count_buffers`; merge / extend
   / extract). The decode hot path keeps a fixed `[B, ratio]` buffer with
   `buffer_count` valid; merge / extend / extract read the shape instead,
   so a row that had decoded carried `ratio` tokens (zeros included), and
   an extracted row restarted its window at 0.

Edits 3-8 only matter to batches (knurlogic's batch engine merges rows
prefilled alone, and rows join a batch that is already decoding); 1-2 to
any prompt past index_topk compressed rows. The golden
(tests/goldens/deepseek_v4_tiny.npz) is computed BY THE FORK, with an
index_topk no pool reaches, so it checks everything the edits leave alone.

### Edits 9-12, needed to load real artifacts

9. `DeepseekV4MoE.__init__` pre-quantized the experts to mxfp4; the fork's
   patched `mlx_lm/utils.py` skips already-quantized modules, stock mlx-lm
   0.31.3 does not ("Unable to quantize ... QuantizedSwitchLinear"). The
   experts are pre-quantized only when the config carries no
   `quantization` (a raw FP4 checkpoint); an MLX-quantized artifact is
   converted by the loader. `ModelArgs.quantization` added.
10. The transformers config shim sets `max_position_embeddings` and
   `rope_theta` before `PretrainedConfig.__init__`; transformers 5.x reads
   them while standardizing rope params and otherwise raises
   AttributeError when loading the tokenizer.
11. `_ragged_prev` (edit 4's per-row carry) is not `@mx.compile`d. Its
   `lens` is a Python list, so a compiled function would trace one graph
   per distinct emit pattern per layer -- at 8 rows, up to hundreds of
   variants x 41 layers -- to save a handful of slices. Arithmetic
   unchanged.
12. Comment-only: two comments say "Fork patch" instead of naming another
   project. No code change.

### Edit 13, needed by MTP drafting

13. **Ragged multi-token mask** (`V4Attention.__call__`). Edit 3's case
   for a step S > 1 wide that continues a merged batch (MTP's 2-wide
   verify): `_build_window_mask` assumes every row's window ends at the
   buffer's end, which a left-padded row's does not, so a short row beside
   a longer one read the wrong window (tests/test_deepseek_v4_mtp.py,
   three rows). Now the batch cache's own `make_mask(S)` there. A
   right-padded prefill (`_lengths` set), a fresh cache and a single row
   keep the fork's mask.

### Edits 14-15, DeepSeek-V4-Flash-Vision-Exp (images)

Both follow the artifact's own reference, `inference/model.py` of
deepseek-ai/DeepSeek-V4-Flash-Vision-Exp (MIT); config-driven, so a
config without `vision_n_layers` (Flash) builds and runs as before.
Design: `docs/design/deepseek-vision.md`. Tests:
`tests/engine/test_vision_deepseek.py` against goldens the reference
itself computes (`tests/support/goldens/build_deepseek_v4_vision.py`).

14. **Image tokens** (`ModelArgs`, `MoEGate`, `DeepseekV4MoE`,
   `DeepseekV4Block`, `DeepseekV4Model`, `Model`, `sanitize`,
   `cast_predicate`). `vision_n_layers` / `vision_max_n_token` read from
   the config. With vision, every gate has `bias_vl` and the hash layers
   a `bias` too (unused, as in the reference); `MoEGate._route_vl` is the
   reference Gate for a call holding image ids (id >= vocab_size): score
   layers take the top-k of `scores + bias_vl` for an image token and of
   `scores + bias` for text, hash layers the top-k of `scores + bias_vl`
   for an image token and `tid2eid` for text (an image id looked up as
   0); weights from the unbiased scores. The model keeps the four learned
   image rows (`image_start/end/newline/pad`), and `DeepseekV4Model.embed`
   never indexes the table with an id >= vocab_size (looked up as 0, the
   row replaced by its type's). `Model.__call__` takes `input_embeddings`
   (the family's merged rows) and `vl_ids` (the ids routing and the
   image-span window read, while `inputs` carries the placeholder ids);
   the image path runs only for a vision config's prefill whose ids hold
   image tokens, so text and every decode step take the fork's path
   (fused gate kernel included). `sanitize` drops `vision.*` /
   `aligner.*` (the family loads them standalone), maps the image rows to
   `model.image_*`, and renames only a `.ffn.gate.bias` suffix (it had
   turned `bias_vl` into `e_score_correction_bias_vl`). `bias_vl` stays
   float32 (`cast_predicate`).
15. **Image-span window** (`image_visible`, `_build_window_mask_visible`,
   `V4Attention.__call__`). The reference's `get_image_visible` +
   `get_window_topk_idxs_visible`: inside an [IMAGE_START, IMAGE_END]
   span a query also sees back to the span's start and forward to its
   end (left clamped to 383, right to 384, at most window + 384 keys), in
   a prefill only; the compressor and the indexer are unchanged. Computed
   per prefill chunk from that chunk's ids: the family's
   `chunk_boundaries` never let a chunk edge fall inside a span (the
   reference prefills a span in one call), so the whole span, and every
   key it reaches, is in the chunk's window.

`rms_norm_eps` (1e-20 on Vision-Exp) was already read from the config by
every norm the reference builds from `norm_eps` (block norms, q/kv norms,
the per-head q norm, both compressors, hyper-connection pre-norms, the
HC head and the final norm); tested, not changed.

### Edit 16, DeepSeek-V4-Flash-Vision-Exp (DSpark drafting)

16. **DSpark config** (`ModelArgs`). `dspark_block_size`,
   `dspark_noise_token_id`, `dspark_target_layer_ids` and
   `dspark_markov_rank` read from the config (defaults: no DSpark), so
   the DSpark head (`heads/deepseek_v4_dspark.py`) finds them on
   `model.args`. Nothing in the trunk reads them; `sanitize` still drops
   `mtp.*` (the head is a sidecar). Design:
   `docs/design/deepseek-vision.md` (DSpark).

### Edit 17, needed by block drafting

17. **Ragged multi-token mask after a 1-wide step**
   (`V4Attention.__call__`, edit 13's case). After a 1-wide decode step
   the batch window cache is a rotated ring; `make_mask(S)` computed its
   left-padding trim from the ring's write index, while the S-wide update
   first puts the ring in temporal order and trims by the buffer length,
   so a row shorter than the window had its oldest key masked. The cache
   is put in temporal order (`_temporal_order`, what the update does
   first) before the mask is made. Found by DSpark's block verify, whose
   verify forward follows 1-wide replays (tests/test_deepseek_v4_dspark.py,
   three rows). The same mask serves an MTP batch's 2-wide verify and
   replay after plain steps.

### Edit 18, a fix (Flash and Vision-Exp)

18. **The shared expert's SwiGLU clamp** (`DeepseekV4MoE.__init__`). The
   fork built the shared expert with `swiglu_limit=0.0` (no clamp);
   DeepSeek's reference builds it with `args.swiglu_limit` (10), as its
   routed experts, in Flash's inference/model.py and Vision-Exp's alike.
   Measured on the Vision-Exp teacher (VQ Lab, 3 corpora x 12288 tokens):
   ~0.026% of shared-expert activations leave +-10, changing the shared
   output by ~2% on average and up to 58% in a chunk
   (tests/engine/test_deepseek_v4_arch.py, the shared-expert clamp test).

### Edit 19, for speculative decoding (DSpark and MTP)

19. **Rolling a verify forward back without a replay**
   (`DeepseekV4Cache.spec_begin` / `spec_can_rollback` / `spec_rollback`
   / `spec_end`, recording hooks in `accumulate_windows`, `update_pool` and
   `Compressor.__call__`). The cache could not be trimmed, so a rejected
   draft cost a restore and a replay forward of the committed tokens
   (~55 ms of a ~163 ms DSpark step on Vision-Exp). `spec_begin` holds the
   cache's state (new array handles, nothing copied) and records what the
   next forward feeds each branch (raw kv / gate rows, start position,
   the pooled rows it emits, the compressor's `ape`); `spec_rollback(p)`
   restores the held state and re-runs the bookkeeping on the first p
   tokens: the window ring takes their keys (the forward appended them in
   order), each branch's `accumulate_windows` runs on their rows, the
   overlap carry is redone from them, and `update_pool` takes the pooled
   rows the forward computed for the windows they complete (a window's
   row depends only on its own tokens and the window before it). Falls
   back (False from `spec_can_rollback`) after more than one forward, a
   1-wide one, or a right-padded prefill. Nothing changes when unused.
   Proven against a cache fed only the p tokens, every state array and the
   following steps' logits within 1e-4 float32 (measured ~1e-6: the pooled
   rows come from sums of another width), across ratio-4 / 128 / 0 layers,
   pool boundaries, the decode hot path's buffer and a merged batch of
   rows of different lengths (tests/engine/test_deepseek_v4_rollback.py).

### Edit 20, a fix (Flash and Vision-Exp)

20. **The reference's low-precision activation simulation** (new
   `fp8_simulate`, `fp4_simulate`, `rotate_activation`,
   `fp8_simulate_nope`, `fp4_simulate_rotated`; applied in
   `_attn_qkv_partial_rope`, `_compressor_norm_strided_rope` /
   `_compressor_qat`, `Compressor.__init__` / `__call__`, `Indexer`).
   DeepSeek's reference inference rounds some activations through FP8 /
   FP4 and back to match its quantization-aware training; the fork did
   none of it. The sites, identical in Flash's and Vision-Exp's
   inference/model.py (line numbers Flash / Vision-Exp):
   - the attention's window kv, non-rope dims: `act_quant(kv[..., :-rd],
     64, scale_fmt, scale_dtype, True)` (506 / 547), so every token's
     window key and the attention's own value;
   - the attention's compressor, each pooled row's non-rope dims, after
     its norm and rope, before it is stored: `act_quant(kv[..., :-rd],
     64, ...)` (372 / 412);
   - the indexer's compressor (`Compressor(..., rotate=True)`), the whole
     row: `rotate_activation(kv)` then `fp4_act_quant(kv, 32, True)`
     (369-370 / 409-410);
   - the indexer's query after its rope: `rotate_activation(q)` then
     `fp4_act_quant(q, 32, True)` (414-416 / 454-456).
   The math is kernel.py's: act_quant_kernel per block of 64, amax
   floored at 1e-4, scale `fast_round_scale(amax * (1/448))` (2 to the
   ceiling of log2, from the float32 bits; both configs run ue8m0, as
   model.py sets scale_fmt = "ue8m0" whenever scale_dtype is "fp8", its
   default), clamp to +-448, e4m3 round-to-nearest-even (`mx.to_fp8`,
   equal to torch's float8_e4m3fn cast on every bf16 value and 1M float32
   ones), times the scale; fp4_quant_kernel per block of 32, amax floored
   at 6 * 2**-126, scale the power-of-two ceiling of amax / 6, clamp to
   +-6, e2m1 round-to-nearest-even; rotate_activation the Sylvester
   Hadamard transform scaled by dim**-0.5 in float32
   (`mx.hadamard_transform` on a float32 copy: on bf16 it accumulates in
   bf16). A last dim that is not whole blocks (only tiny test configs;
   the reference asserts) is left as it is. The model runs the same
   arithmetic as two Metal kernels, one dispatch a site
   (`_fp8_nope_kernel`: a simdgroup per 64-block, writing the roped dims
   after it, so it replaces the concatenate that was there;
   `_rotate_fp4_kernel`: one thread a value, the butterflies in
   mx.hadamard_transform's order, a simdgroup per 32-block), with e4m3 /
   e2m1 rounding as rint(v * 2**-e) * 2**e at the format's spacing:
   equal to `mx.to_fp8` on every bf16 value in range and 2M float32 ones,
   and to the op graphs (on the CPU, the op graphs run). Added decode
   cost, measured on a random 8-layer config with Flash's attention
   shapes (head_dim 512, rope 64, indexer 64 x 128): within noise of
   none per step (~4.0 ms); per site ~0.5 us over the concatenate it
   replaces and ~6 us for the indexer query's kernel (the op graphs: ~25
   and ~40 us). Estimated on Flash: ~0.2 ms a token (43 window sites, 21
   indexer queries, pooled rows every 4 / 128 tokens). Pooled rows are rounded before
   `update_pool`, so edit 19's recorded rows are the rounded ones; the
   rounding is per row, so a rolled-back cache holds what a narrower
   forward would. Flash's MTP block uses the trunk's V4Attention and so
   rounds its kv too, as the reference's MTPBlock (an Attention) does. The DSpark head (heads/deepseek_v4_dspark.py) rounds its
   main_kv and block kv the same way (Vision-Exp 811 / 830).
   Proven against the reference itself under torch (kernels ported in
   tests/support/goldens/build_deepseek_v4_dspark.py): the three kernels
   bit for bit on float32 and bf16 inputs; a tiny trunk with ratio 0 / 4
   (indexer keeping 8 of 31 rows) / 128 / 4 layers plus three DSpark
   stages, 126-token prefill and 5 decode steps: max abs difference on
   the logits 1.7e-6 (0.81 without this edit), main hidden 3.6e-6 (1.7),
   draft logits 2.1e-6 (2.2), confidence 9.5e-6 (2.1), draft ids equal
   (not without) (tests/engine/test_deepseek_v4_dspark.py).
   Edit 21 adds the remaining site, every FP8 / FP4 linear's input.

### Edit 21, a fix (Flash and Vision-Exp)

21. **The linears' activation rounding** (new `fp8_act`, `swiglu_act`,
   `rms_norm_act`, `_moe_sum`; applied in `DeepseekV4Block.__call__`,
   `V4Attention.__call__`, `_attn_wqkv_quant_split_norm` /
   `_attn_qkv_split_norm`, `_attn_wo_chain_quant`, `DeepseekV4MLP`,
   `DeepseekV4MoE.__call__`, `MoEGate`). The reference's `linear()`
   (Flash 108-120 / Vision-Exp 123-135) quantizes x with `act_quant(x,
   128, scale_fmt, scale_dtype)` (ue8m0 scales: model.py sets scale_fmt
   = "ue8m0" whenever scale_dtype is "fp8") before fp8_gemm / fp4_gemm,
   whenever the weight is FP8 or FP4, whatever our weights' format.
   Which linears are: `Linear`'s dtype defaults to `default_dtype`,
   float8_e4m3fn under the config's dtype "fp8" (Transformer.__init__,
   776 / 934), and routed experts are FP4 (expert_dtype "fp4"); the
   checkpoint agrees (F8_E4M3 + F8_E8M0 scales, I8 experts). Rounded:
   attention wq_a and wkv (one x; our fused wqkv_a), wq_b, wo_b, the
   indexer's wq_b (all FP8, 457-463 / 498-504, 393 / 433), the shared
   expert's w1 / w3 / w2 (FP8, its Expert built with dtype None, 627 /
   682), the routed experts' w1 / w3 / w2 (FP4, 591-593 / 646-648),
   Flash's MTP e_proj / h_proj (742-743) and Vision-Exp's DSpark
   main_proj (882). Not rounded: wo_a (FP8 in the checkpoint, but
   convert.py dequantizes it to bf16 and the model declares it bf16,
   462 / 503), the compressors' wkv / wgate (fp32, 297-298 / 337-338),
   the indexer's weights_proj (bf16, 394 / 434), the gate (F.linear in
   float32), the head, the hyper-connections, the confidence head (fp32).
   An expert's w2 input is the reference's: SwiGLU in float32, times the
   routing weight (now float32 out of the gate, as the reference's Gate
   returns it) before w2, back in the activation dtype, rounded; the
   experts' outputs and the shared expert's summed in float32 (its
   MoE.forward); the trunk previously weighted after w2. The block's
   norms (attn_norm, ffn_norm) and q_norm return the rounded copy from
   the same dispatch (`rms_norm_act`), and the SwiGLU, weight and
   rounding are one dispatch (`swiglu_act`); wo_b's input is `fp8_act`
   (a simdgroup per 128-block) inside the compiled wo chain. The heads
   (heads/deepseek_v4.py, heads/deepseek_v4_dspark.py) round the same
   way. A dim that is not whole 128-blocks (tiny test configs) is left
   as it is.
   Proven against the reference under torch: the DSpark golden now runs
   the reference with dtype fp8 / expert_dtype fp4 (its fp8_gemm /
   fp4_gemm ported to torch on the dequantized operands), with every
   linear's input whole 128-blocks (on two tensor ranks too): max abs
   difference on the logits 9.5e-7 (1.61 without this edit), main hidden
   1.1e-5 (13.4), draft logits 2.1e-6 (5.6), confidence 1.5e-5 (24.7),
   draft ids equal (not without); `fp8_act` bit for bit against
   act_quant block 128 on float32 and bf16. Of 12 prompts, 10 hit an
   FP8 rounding a float32 ulp of summation order flips and part ways
   (0.07-0.74 on the logits); the golden's prompt is one that does not.
   A tensor split cuts three rounded inputs (wo_b's, both experts'
   down_proj); tuning/resolve's tensor_refusals refuses a split whose
   slice of one is not whole 128-blocks (Flash and Vision-Exp split 2, 4
   and 8 ways). Edit 20's rounded tensors are never cut. A tensor split
   still sums in another order than the whole model (per-rank partials),
   so the rounding's inputs differ by float32 noise and an input on an
   FP8 boundary can round to the other value: on the tiny DSpark model the
   split's tokens part from the whole model's (tests/engine/
   test_pipeline.py records it: 708 rounding calls' inputs within 7.2e-7
   relative, then one 64-block of the window kv rounding differently in
   the prefill, the tokens parting 4-5 steps later); with the rounding
   stubbed out the two are equal token for token.
   Added decode cost, measured on Flash's shapes (dim 4096, q_lora 1024,
   wo_b input 8192, expert width 2048, top 6), the ops it changes in a
   43-layer chain: ~0.7 ms a token of evaluation (2.7 -> 3.4 ms in the
   chain), ~1.2 ms with graph building; whole random 24-layer configs
   within noise of +0.5 ms.

### Edit 22, a fix (batching)

(Edit 21 is left to another change in flight.)

22. **A merged window is each row's tokens, not its ring buffer**
   (`DeepseekV4Cache._window`, used by `_merge_local` and `extend`'s
   one-offset path). Rows whose windows are plain RotatingKVCaches at one
   offset were merged by concatenating their raw key / value buffers
   (`_temporal_order` cut only to `_idx`). Two rows at one offset can hold
   buffers of different lengths: a 7-token prefill holds 7 keys, a 6-token
   prefill that then decoded a step holds a ring grown to 8 slots, a
   9-token prefill holds 9 keys for an 8-wide window. The concatenate
   raised (8 against 7 on the tiny model, window 8), and the batch engine
   failed that request's admission: on the real model (window 128) any
   two rows at one offset under 128 tokens whose histories differ, a
   common short chat. `_window` takes the last `size()` tokens in
   temporal order; keys older than those are dropped by the next update
   either way, so nothing a row attends to changes. Proven on the tiny
   model in float32: every ordered pair of ten histories (3 to 16 tokens,
   prefilled whole or in chunks of 4, some decoded in place) merged, and
   rows joining a decoding batch, each row's logits within 1e-4 of its
   lone run's for every later step; and the batch engine admits rows
   shorter than, at and past the window beside a decoding row
   (tests/engine/test_deepseek_v4_arch.py).
