# Vendored architecture modules

Each entry is a claim that this file is the arithmetic the
artifacts were validated against -- not merely that it imports.

## gemma4_text.py

- taken: 2026-09-18
- from: `<a venv>`
- interpreter: `<a venv>`
- mlx-lm: 0.32.0
- sha256: `f3f8c047c2ac31306267e8de61bb06d2952f2bf3fd4adbbb8ce25b952ccac01f`
- note: qwen4exp venv (mlx-lm 0.32.0) taken as authoritative: it is where vqlab fits and scores, and it is the SUPERSET -- qwen3_5 here carries PipelineMixin, the exo-env copy does not
- **P2 edit (2026-09-23):** `_make_masks`/`Gemma4TextModel.__call__`/`Model.__call__`
  take an added `mm_mask: Optional[mx.array]` ([B, L], -1 outside an image,
  else the image's index along the sequence) and, on the full-attention
  layers only, overlay bidirectional attention within same-block spans on
  top of the causal mask. Ported from mlx-vlm 0.6.17
  `gemma4/language.py:455-515` (`_block_sequence_ids_for_mask`,
  `_apply_blockwise_bidirectional_overlay`, the `use_bidirectional_vision`
  gate in `_make_masks`), MIT, Copyright (c) 2025 Prince Canuma -- with
  `_block_sequence_ids_for_mask` NOT ported: `engine/families/gemma4/vision` passes
  the block-id array directly (it already has it from
  `engine/vision/key.image_spans`, which distinguishes images by sha, not
  just "is a vision token"), so recomputing block ids from a token-type
  array is unneeded here (no audio token in this build). See
  `src/knurlogic/engine/families/gemma4/vision/PROVENANCE.md`.

## gemma4.py

- taken: 2026-09-23
- from: `/opt/anaconda3/lib/python3.12/site-packages/mlx_lm/models/gemma4.py`
- mlx-lm: 0.31.3 (the pinned install)
- sha256: `4671e4a63cb9849582abac566599a0a85370a46d410f4ad69d81a88788d00fd8`
- note: the multimodal wrapper released gemma rungs load through
  (`model_type: gemma4`, text under `.language_model`). Vendored so it sits
  on the vendored gemma4_text rather than beside it.
- **edit:** `Model.__call__` takes and forwards `mm_mask` to gemma4_text;
  upstream drops it, and a vision prefill through the wrapper raised
  TypeError before any image-block overlay could apply (found by the e4b
  real-model gate, 2026-09-23).
