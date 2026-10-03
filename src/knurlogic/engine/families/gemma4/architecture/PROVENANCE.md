# Vendored architecture modules

Each entry is a claim that this file is the arithmetic the
artifacts were validated against -- not merely that it imports.

## gemma4_text.py

- taken: 2026-09-18
- from: `<a venv>`
- interpreter: `<a venv>`
- mlx-lm: 0.32.0
- sha256: `f3f8c047c2ac31306267e8de61bb06d2952f2bf3fd4adbbb8ce25b952ccac01f`
- note: the mlx-lm 0.32.0 environment is taken as authoritative: it is where the VQ artifacts are fit and scored, and it is the SUPERSET -- its qwen3_5 carries PipelineMixin, the other environment's copy does not
- **edit (vision mask):** `_make_masks`/`Gemma4TextModel.__call__`/`Model.__call__`
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
  Superseded in placement and gate by vendored edit 1 below.

### Audit against the maker's reference (2026-10-03)

Held to HF transformers 5.16.1 `models/gemma4/modeling_gemma4.py` (Google's
port; 5.5.0 agrees) and Google's own `google-deepmind/gemma`
`gemma/gm/nn/gemma4/` (`_config.py`, `_transformer.py`). Golden:
`tests/support/goldens/build_gemma4_text.py` -> `gemma4_text.npz`, held by
`tests/engine/test_gemma4_text_parity.py` (dense e-style with PLE, KV
sharing, double-wide MLP, sliding/full mix, proportional partial RoPE;
26B-A4B-style MoE with K=V full layers; float32; 11-token prefill past a
6-token window, 5 decode steps through make_cache). Text path: max abs
logit diff 9.3e-6 (dense) and 3.3e-6 (moe) before any edit -- norms
(plain w, fp32 inside), v_norm without scale, layer_scalar, attention
scale 1.0, no attention softcap, final softcap, window boundary, rope
thetas and proportional partial rope, KV-sharing source layers, PLE
scales, embed scale, router (softmax over the top-k == renormalized full
softmax), per-expert scale, double-wide MLP all match. One difference,
in the image mask:

- **vendored edit 1 (image-block mask placement):** `_make_masks` put the
  bidirectional image-block overlay on the FULL-attention layers, for
  every config with an image (as mlx-vlm 0.6.17 does). The maker puts it
  on the SLIDING layers only, as AND(window, OR(causal, same block)), and
  only when `use_bidirectional_attention == "vision"` (26B-A4B, 31B);
  full layers stay causal and e2b/e4b (`None`) are causal everywhere.
  Refs: Google `gm/nn/gemma4/_config.py` (`use_bidirectional_attention`
  comment: sliding layers only, causal for global) and `_transformer.py`
  (`sliding_attention_mask` used for LOCAL_SLIDING only, window still
  applied); HF 5.16.1 `modeling_gemma4.py:2093-2145`
  (`create_masks_for_vision_model`) and `:2398-2409` (the gate). Ours
  before: `gemma4_text.py` `_make_masks` (full_attention branch).
  `ModelArgs` gains `use_bidirectional_attention` (default None, read from
  text_config). Severity: every image prompt, every image token and every
  token after it (prefill logits off by up to 3.26 on the 26B-style
  golden, 1.27 on the e-style one); text-only requests unaffected.
  After: 1.2e-5 / 9.3e-6.

## gemma4.py

- taken: 2026-09-23
- from: `<site-packages>/mlx_lm/models/gemma4.py`
- mlx-lm: 0.31.3 (the pinned install)
- sha256: `4671e4a63cb9849582abac566599a0a85370a46d410f4ad69d81a88788d00fd8`
- note: the multimodal wrapper released gemma rungs load through
  (`model_type: gemma4`, text under `.language_model`). Vendored so it sits
  on the vendored gemma4_text rather than beside it.
- **edit:** `Model.__call__` takes and forwards `mm_mask` to gemma4_text;
  upstream drops it, and without the edit a vision prefill through the
  wrapper raises TypeError before any image-block overlay can apply (the
  e4b real-model gate exercises it).
