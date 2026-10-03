# Vendored architecture modules

Each entry is a claim that this file is the arithmetic the
artifacts were validated against -- not merely that it imports.

## deepseek_v4.py

- taken: 2026-09-29
- from: `mlx_lm/models/deepseek_v4.py` of the project's own
  mlx-lm fork (installed from a local `mlx_lm-0.31.9` wheel), vendored with
  its author's permission. No exo code is in it.
- mlx-lm base: 0.31.9 (the fork); runs here on the pinned 0.31.3.
- fork file sha256: `78bf144caae1e1067f2910d070e3a71fe6f2d11704691cb2a272c9aebf0a13ef`
- vendored sha256: `67b006651f79f7415932a38a27948e97e949fd21fc21cfb4a85e0753037f18ff`
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
  fork-only helper is vendored.
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
