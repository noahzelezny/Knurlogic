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

### Second audit: configs that omit keys, "all", bf16

Held to HF transformers 5.16.1 `configuration_gemma4.py` as well as
`modeling_gemma4.py`. The golden gains a `defaults` config (only vocab,
hidden, intermediate, layer count and window given), a
`use_bidirectional_attention: "all"` config (prefill, cached decode, and
a prefill in two chunks through the cache), the resolution of a set of
configs (`resolve/*`: keys left out, a last layer given as sliding, "all",
K=V, the gemma4 wrapper, both released artifacts' text configs; read field
by field off a lazily built model), bf16 runs (`dense_bf16`, `moe_bf16`:
HF's eager and sdpa in bf16), and HF's bf16 rounding points of the norm
and the MLP activation (`ops/*`). All held by
`tests/engine/test_gemma4_text_parity.py`.

- **knurlogic edit 2 (config defaults):** `ModelArgs` defaulted to an
  e2b-like shape: hidden 1536, 35 layers, intermediate 6144, 1 kv head,
  20 KV-shared layers, double-wide MLP, final softcap 30, and a
  `sliding_window_pattern` 5 (4 sliding : 1 full) when `layer_types` is
  absent. HF's `Gemma4TextConfig` (`configuration_gemma4.py:169-200`):
  hidden 2304, 30 layers, intermediate 9216, 4 kv heads, no KV sharing, no
  double-wide MLP, no softcap, and `layer_types` 5 sliding : 1 full (it
  never reads a `sliding_window_pattern` key) with the LAST layer forced
  to full attention whatever the config says. The defaults are now HF's,
  the pattern is HF's (the `sliding_window_pattern` field is gone; a
  config's key is ignored, as HF ignores it), the last layer is forced
  full, and the default `rope_parameters` are HF's dict (no
  `partial_rotary_factor` on the sliding entry: the same RoPE). Also read
  now, and refused at load when not what this file runs:
  `hidden_activation` (only gelu_pytorch_tanh), `attention_bias` (only
  false), `use_bidirectional_attention` (None/"vision"/"all", HF's
  Literal). Every other field already had HF's default. The released
  artifacts carry every one of these keys, so they resolve as before
  (`resolve/artifact:*`). Before: the `defaults` golden could not load
  (20 KV-shared layers of 8); after 1.8e-5.
- **knurlogic edit 4 (`use_bidirectional_attention: "all"`):** was read
  and ignored -- the model ran causal. HF: the config halves the window
  (`sliding_window // 2 + 1`, `configuration_gemma4.py:184-186`) and sets
  `is_causal` False, so `create_causal_mask` /
  `create_sliding_window_causal_mask` build bidirectional masks
  (`masking_utils.py` `sliding_window_bidirectional_overlay`:
  `abs(q - k) <= window`). `ModelArgs` halves the window the same way and
  `Gemma4TextModel._make_masks_all` builds the masks: the causal mask OR
  the later keys in the chunk (within the window on the sliding layers).
  Not causal means no unique cached decode; ours keeps what HF's cache
  keeps (RotatingKVCache of the halved window: window - 1 earlier tokens
  at a prefill, the window at a decode step, as DynamicSlidingWindowLayer)
  and matches HF's cached decode and HF's two-chunk prefill (1.0e-5;
  before, causal: off by whole units). A prompt the
  serve path prefills in chunks sees across chunks only what that cache
  keeps -- as HF's own cache. No released rung uses "all".
- **knurlogic edit 5 (MLP activation rounding):** HF's act_fn
  (gelu_pytorch_tanh) on a bf16 tensor computes in float32 and rounds
  once; mlx's `gelu_approx` on bf16 rounds after every step (42% of
  elements a step or more from HF's, `ops/act`). `geglu` (dense MLP,
  experts, and now the per-layer-input gate, which called gelu_approx and
  multiply separately) computes the gelu in float32; it is compiled and
  fused, so it costs nothing measurable (24.7 -> 25.3 us at [4096, 2112]).
  After: under 1% a step or two off (the two tanh implementations), 0 vs
  ~1e-6 where mlx's tanh saturates first.
- **Kept, measured (RMSNorm rounding):** HF's `Gemma4RMSNorm` computes
  `(x * pow(mean(x^2) + eps, -0.5)) * w` in float32 and rounds once;
  `mx.fast.rms_norm` on bf16 rounds `x * rsqrt` to bf16, then multiplies
  by w and rounds again: 26% of elements one bf16 step from HF's, none
  further (`ops/norm`). The unscaled norms (v_norm, the router's norm, the
  embedder's) round once and match exactly. Both exact forms were
  measured on an M3 Ultra, 6 decode layers at 26B-A4B's dense widths, bf16:
  3.20 ms/token as is, 3.48 (+9%) casting to float32 around
  `mx.fast.rms_norm`, 3.39 (+6%) with a custom Metal kernel that rounds
  once; per norm, 9 -> 18 us (casts) and 9 -> 14 us (kernel) at decode.
  Not cheap, so not taken; the difference is held at one step by
  `test_text_rmsnorm_is_within_one_bf16_ulp_of_the_reference`.
- **Kept (other bf16 points):** RoPE is one fused kernel (float32 inside,
  one rounding; HF rounds cos/sin to bf16 and rounds x*cos, rot*sin and
  the sum); attention is fused (the softmax probabilities never rounded to
  bf16 -- HF's eager rounds them, HF's sdpa does not); the router's
  expert weights are bf16 softmax over the top-k (HF: float32 softmax over
  all experts, renormalized, applied in float32 then rounded per expert).
  End to end in bf16, on random tiny weights, ours sits about as far from
  HF's eager logits as HF's own sdpa does: dense_bf16 0.15 from eager
  (0.12 from sdpa) against a 0.10 eager/sdpa spread; moe_bf16 0.35 from
  eager, 0.055 from sdpa, spread 0.37. The test holds ours within twice
  the spread of HF's eager bf16 logits (and the same weights in float32
  to 2e-4).

## gemma4_text.py, gemma4.py: current pins

pins.json holds the sha256 of each file as edited (edits 1-5 above, edit
3 below); `text_path_of` keeps the untouched mlx-lm file's digest.

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
- **knurlogic edit 3 (text_config is the text model's config):**
  `ModelArgs.__post_init__` overwrote `text_config["vocab_size"]` with
  the top-level `vocab_size` (default 262144; HF's `Gemma4Config` never
  reads a top-level one) and filled in `num_key_value_heads` 1 and
  `num_attention_heads` 8 where the text config left them out (HF's
  defaults: 4 and 8). Now `text_config` passes through as is and
  gemma4_text's defaults (HF's, edit 2) apply, as in
  `configuration_gemma4.py` `Gemma4Config.__post_init__`. Held by
  `resolve/wrapper` and `resolve/wrapper_empty`.
