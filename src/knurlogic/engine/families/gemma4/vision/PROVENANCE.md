# gemma4 vision -- provenance

Design: docs/design/vision.md.

## Vendored from mlx-vlm 0.6.17

- host package: `mlx_vlm` 0.6.17, MIT, Copyright (c) 2025 Prince Canuma

| this file | from | lines |
|---|---|---|
| `vision.py` | `mlx_vlm/models/gemma4/vision.py` | full file (562), one import changed (`from .._base import ensure_fused_sdpa`, see `engine/vision/_base.py`), one structural trim (below) |
| `config.py` | `mlx_vlm/models/gemma4/config.py::VisionConfig` | trimmed to the fields `vision.py` reads (audio/video-only fields dropped: out of scope, vision-only package) |
| `__init__.py::MultimodalEmbedder` | `mlx_vlm/models/gemma4/gemma4.py:22-35` | verbatim |
| `__init__.py::RMSNormNoScale` | `mlx_vlm/models/gemma4/language.py::RMSNormNoScale` | verbatim (duplicated from `vision.VisionRMSNormNoScale` rather than shared, because it sits on `embed_vision` in the weight tree, not the tower) |
| `../architecture/gemma4_text.py::_make_masks` overlay | `mlx_vlm/models/gemma4/language.py:455-515` (`Gemma4TextModel._block_sequence_ids_for_mask`, `_apply_blockwise_bidirectional_overlay`, the `use_bidirectional_vision` gate) | ported, not copied verbatim -- see the deviation below and `../architecture/PROVENANCE.md` |

`scatter.merge` (image features into text embeddings) is shared,
vendored at `engine/vision/scatter.py` (mlx-vlm's `masked_scatter`,
`gemma4/gemma4.py:13-20`) and held to a golden there
(`tests/goldens/p0_masked_scatter.npz`); this package reuses it rather than
porting `gemma4.py`'s `_scatter`/`get_input_embeddings` a second time.

## Structural trim: `vision.py::VisionModel.__call__`

mlx-vlm's version has three call shapes: a list of differently-sized
images (batches a turn's images through one forward pass), an
externally-supplied `pixel_position_ids` path, and the plain
`[B, C, H, W]` path. `Family.preprocess`/`Family.encode` (this package's
`__init__.py`) call the tower once per image (an image is
encoded once and cached by sha, never batched with another image's
pixels), so only the plain path is kept. Padding to `max_patches` is also
dropped for the same reason: mlx-vlm pads so a batch's images share one
tensor width; a single image needs no padding, and `_patch_positions` here
returns exactly `num_real` positions.

## Deviation: the mask overlay uses block ids, not a token-type array

mlx-vlm's `_make_masks` takes `mm_token_type_ids` (0 = text, 1 = image,
2 = audio) and derives contiguous "blocks" from it inside
`_block_sequence_ids_for_mask` (a run of 1s or 2s becomes one block,
`language.py:455-469`). This build has no audio token and already knows,
from `engine/vision/key.image_spans`, exactly which image each run of
sentinels belongs to (by sha, not by "is this a vision token") -- an
importantly stronger fact than mlx-vlm's, which would merge two adjacent
images of the same type into one block if it ever saw two images back to
back with no separator. So `Gemma4Vision._mm_mask` (this package's
`__init__.py`) builds the block-id array directly from `image_spans`, and
`../architecture/gemma4_text.py::Gemma4TextModel._make_masks` takes that
array as `mm_mask` and applies
`_apply_blockwise_bidirectional_overlay`'s boolean-or logic verbatim,
skipping `_block_sequence_ids_for_mask` entirely. `test_g4_mask_overlay_matches_reference`
(`tests/test_vision_gemma4.py`) holds the OVERLAY logic itself to a golden
built from mlx-vlm's own `Gemma4TextModel._block_sequence_ids_for_mask` /
`_apply_blockwise_bidirectional_overlay`
(`tests/goldens/build_gemma4.py`, `gemma4_mask_overlay.npz`), so the boolean
arithmetic is checked against the reference even though the block-id input
is produced a different way.

## Deviation: `encode()` pre-divides by `embed_scale`

See the docstring at the top of `__init__.py` ("WHY encode() PRE-DIVIDES BY
embed_scale"). Short version: mlx-vlm's `gemma4.Model.get_input_embeddings`
scales ONLY the text embeddings before scattering in the (unscaled)
projected image features (`gemma4.py:85-170`); knurlogic's
`../architecture/gemma4_text.py::Gemma4TextModel.__call__` scales whatever
`input_embeddings` it receives, always
(`h = input_embeddings; h = h * self.embed_scale`, unedited -- every
caller shares this line). `Gemma4Vision.encode` divides the tower's
projected features by `embed_scale` before caching them
(`EncodedImage.feats`), so the trunk's later multiply cancels the division
on exactly the image rows and leaves the text rows scaled as it always
did. `tests/test_vision_gemma4.py::test_g5_embed_merges_image_rows_and_runs_through_the_trunk`
and the manual encode->embed->forward roundtrip in the same file exercise
this end to end.

## PLE (per-layer inputs): no trunk edit needed

mlx-vlm computes `per_layer_inputs` from `input_ids` with every
multimodal placeholder zeroed (`gemma4.py:88-100`).
`../architecture/gemma4_text.py::Gemma4TextModel.__call__` already accepts a
precomputed (unprojected) `per_layer_inputs` and, when given one, skips its
own `_get_per_layer_inputs` and only projects
(`gemma4_text.py:530-534`, unedited). So `Gemma4Vision.embed` builds the
zeroed-id array itself (every sentinel position -> 0) and calls the
trunk's own (unedited) `_get_per_layer_inputs` on it, passing the result
back in as `per_layer_inputs` -- the contract's "per_layer_inputs override
from zeroed ids" needed no change to `gemma4_text.py`, only to this
package's `embed`. `test_g5_per_layer_inputs_zero_image_positions` checks
this against the trunk's own method, and against what an UNzeroed id would
have produced (must differ).

## Tower loads standalone

`load_weights` (`__init__.py`) reads `vision_tower.*` / `embed_vision.*`
straight off the model directory (a shard index's `weight_map` when there
is one, or every `*.safetensors` file in the directory otherwise -- a
release's real layout for e4b is the quantized `embed_vision
.embedding_projection` living in the SAME shard as text weights,
`model-00002.safetensors`; 26b's tower is its own 356-tensor sidecar),
never through the text model's
`sanitize`. `test_g1_load_weights_standalone_tower`
gates that the tensor count matches what was written.

## Open issues

- **Quantization (e4b's `embed_vision.embedding_projection` -> `QuantizedLinear`,
  and `ClippableLinear`'s clip params).** `load_weights` reads whatever
  tensors are under the two prefixes and applies them via `tree_unflatten`;
  it has NOT been exercised against a quantized `nn.Linear` (the module tree
  would need `nn.QuantizedLinear` swapped in before `update`, the way
  mlx-lm's own quantized loading does it). The unit tests load no real
  model; the real-model gate (tools/vision_gate.py) is what covers it.
- **`use_clipped_linears` per-tensor clip bounds.** `ClippableLinear` loads
  `input_min`/`input_max`/`output_min`/`output_max` as ordinary buffers when
  `use_clipping=True`; not exercised against a real e4b checkpoint's actual
  clip values, only against random tiny-fixture ones (which are finite but
  not representative of the reference's `±inf`-until-loaded default).
- **26b's standalone sidecar path is not integration-tested** (the unit
  tests load no real weights); `load_weights`' "no shard index"
  branch (a single `*.safetensors` in the directory) is the one path this
  package's own tests exercise for a full tower, and it is also what a
  26b-style single-shard sidecar looks like structurally.
- **Sliding-window vision layers.** `vision_config.layer_types` defaults to
  all `full_attention` (config.py `__post_init__`); the released gemma4
  vision towers were not checked for a sliding pattern, and if one exists
  it is not specially handled (`VisionTransformerModel` treats every layer
  identically, following the vendored source, which also has no
  window-size branch in the vision attention itself).
- **`positions()` returns `(None, 0)`** per the design (gemma: plain 1D
  RoPE for the trunk); this was not re-derived from a real config, only
  taken from `docs/design/vision-contracts.md` (gemma uses plain 1D RoPE).
