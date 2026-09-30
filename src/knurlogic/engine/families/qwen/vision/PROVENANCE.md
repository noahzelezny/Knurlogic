# Qwen vision: where every line came from

Design: docs/design/vision.md. Reference: mlx-vlm 0.6.17 (MIT, Copyright
(c) 2025 Prince Canuma). Held to it by committed goldens made from mlx-vlm's
own classes (`tests/goldens/build_qwen.py`), so the tests run without it.

## Vendored

| here | from (mlx_vlm/models/...) | sha256 of the source file | edits |
|---|---|---|---|
| `vision.py` | `qwen3_vl/vision.py` (447 lines, all) | `0c51d19e208e5d8fa7a6751aee0a9360a5bd8783dcfcf532286d14a0f1fbc7f9` | imports only: `..base.ensure_fused_sdpa` -> `engine.vision._base`; `.config.VisionConfig` -> defined in the file |
| `vision.py` `VisionConfig` | `qwen3_vl/config.py:28-49` + `qwen3_5/config.py:47-58` (deepstack guard) | `8b529cfbd568c7d202140b4d5212be7fe740310c1a80374fa7dd4e59b9763017` | the two merged into one dataclass |
| `processing.py` | `qwen3_vl/processing_qwen3_vl.py:164-227` verbatim; `:230-412` image half of `Qwen3VLImageProcessor`; `:593-626` `_qwen_vl_image_kwargs` | `21d68148d9bd99952445beaee21237993510dc4c3a2c94a2c5cda5f19429b180` | transformers base class dropped; one image per call; `settings()` added (feeds proc_hash); kwargs read from LOCAL files only (no Hub fallback: the server does not fetch); no video |
| `rope_index.py` | `qwen3_5/language.py:1729-1904` `get_rope_index` | `4805ae90fb3bba463512cbce89c9bb7fa78d56b25bd8b541db1392ad34f18ae0` | ported, not copied: one row, no attention mask, numpy, grids per image from refs |
| `family.py` key mapping | `qwen3_5/qwen3_5.py:16-25` `sanitize_key` (vision prefixes only) | `02bb994533e53589d9b1ee3623a79516e2e976bad34ebe533091c4c4af1c2ebd` | only the `model.visual` / `model.language_model.visual` / `vision_tower` prefixes |
| `family.py` merge | `qwen3_5/qwen3_5.py:121-143` `merge_input_ids_with_image_features` | same | replaced by the shared `scatter.merge` (by sentinel; engine/vision/scatter.py holds its `masked_scatter` to the reference) |

## The trunk half (../architecture/, not in this folder)

| file | what | from |
|---|---|---|
| `qwen3_5.py` `mrope_selector`, `apply_mrope`, `Attention` | interleaved MRoPE; `position_ids` [3,B,L] / `rope_delta` [B] threaded model -> text model -> layer -> attention | selector `rope_utils.py:512-517`, apply `:655-690` (sha256 `6944f69a03c41cbb9613afbd2f1b54b71c504840aaeb0eade39c6109dbfdd25b`); delta rule `qwen3_5/language.py:2022-2052` |
| `qwen3_5_moe.py` | nothing: inherits qwen3_5's Model and layers | -- |
| `qwen4_exp.py` `RotaryEmbedding` (MRoPE), `Qwen4ExpModel`/`DecoderLayer`/`Attention` threading, `QSAIndexer` positions, `_IndexerCache.pos` and every `_BatchAttnCache` column op on it | the indexer ropes its query with the MRoPE positions and each pooled block with the stored position of the block's first token | `qwen4_exp/language.py:21-63, 306-348` (sha256 `a64846ff036f3ae47c8f32cdec36433b8ad5c403503e2b350fc9b98271234544`) |

With neither `position_ids` nor `rope_delta` every trunk call runs the code
as vendored before the MRoPE edit: `tests/goldens/qwen_g5_text.npz` (built from those
files, `tests/goldens/build_qwen_g5.py`) holds the logits to the bit.

## Gates (tests/test_vision_qwen.py) and the mutation each was checked with

| gate | holds | broken once, went red |
|---|---|---|
| G1 | tower == mlx-vlm, atol 1e-5, both sidecar namings | position embedding dropped |
| G2 | grid, n_tokens, pixel_values == mlx-vlm's processor | min_pixels ignored |
| G3 | positions + delta of a two-image prompt exact | vision_start = class default 248045; image h/w swapped |
| G4 | prefill logits (atol 1e-4) + 40 greedy tokens | interleave collapsed to axis t (qwen3_5, qwen4_exp); indexer block rope at logical starts (qwen4_exp) |
| G7b | warm text turn 2 on rope_delta == mlx-vlm cold turn 2 | rope_delta dropped in attention; and in-test: the warm turn without the delta must diverge |
| G5 | text path bit-identical to the pre-MRoPE snapshot; MRoPE path with text positions == 1-D path | text routed through the explicit MRoPE path |
| batch | per-row rope_delta [delta, 0] == each row alone | row 0's delta for all rows; qwen4_exp merge without the text row's default positions |
