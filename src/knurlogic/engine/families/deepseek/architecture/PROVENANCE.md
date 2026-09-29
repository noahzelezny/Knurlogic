# Vendored architecture modules

Each entry is a claim that this file is the arithmetic the
artifacts were validated against -- not merely that it imports.

## deepseek_v4.py

- taken: 2026-09-29
- from: the M4's `exo` env, `mlx_lm/models/deepseek_v4.py` -- the maintainer's own
  mlx-lm fork (installed from a local `mlx_lm-0.31.9` wheel), vendored with
  his permission. No exo code is in it.
- mlx-lm base: 0.31.9 (the fork); runs here on the pinned 0.31.3.
- fork file sha256: `78bf144caae1e1067f2910d070e3a71fe6f2d11704691cb2a272c9aebf0a13ef`
- vendored sha256: `d4fc963282e1214a6f5f74b21e3874fb0fd341a77cac218600bc89393afc50b0`
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

### Edits (2026-09-29), each found by a tiny-model test the fork fails

1. **Indexer query RoPE** (`Indexer.__call__`). The indexer's q is
   `[B, S, H, D]`, and `mx.fast.rope` puts positions on axis -2 -- the
   heads: head h was rotated at `offset + h` for every token. The keys in
   its pool are rotated by their true positions, so the top-k rows were
   chosen by scores that depended on the head index, and a prefilled
   prompt chose differently from the same tokens decoded one at a time.
   Now rotated with the sequence on axis -2 (as the attention's own q is).
   Changes every choice the indexer makes once a pool holds more than
   `index_topk` (512) rows, i.e. past ~2048 tokens on Flash.
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
   Single-row and same-length batches still take the fork's mask-free path.
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

## Edits 9-10 (2026-09-29, first live load)

9. `DeepseekV4MoE.__init__` pre-quantized the experts to mxfp4; the fork's patched
   `mlx_lm/utils.py` skipped already-quantized modules, stock mlx-lm 0.31.3 does not, so
   every rank failed with "Unable to quantize ... QuantizedSwitchLinear". Now the experts
   are pre-quantized only when the config carries no `quantization` (a raw FP4 checkpoint);
   an MLX-quantized artifact is converted by the loader. `ModelArgs.quantization` added.
10. The transformers config shim sets `max_position_embeddings` and `rope_theta` before
   `PretrainedConfig.__init__`; transformers 5.x reads them while standardizing rope
   params and raised AttributeError when loading the tokenizer.

## Edit 11 (2026-09-29)

11. `_ragged_prev` (edit 4's per-row carry) is no longer `@mx.compile`d. Its `lens`
   is a Python list, so the compiled function traced one graph per distinct emit
   pattern per layer -- at 8 rows, up to hundreds of variants x 41 layers -- to save
   a handful of slices. Arithmetic unchanged.
