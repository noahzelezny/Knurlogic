# mlx-vlm vision map for knurlogic's five released families

## 0. What was available, and the license

- **mlx-vlm 0.6.17** is at `/opt/anaconda3/envs/exo/lib/python3.13/site-packages/mlx_vlm` (`version.py`: `__version__ = "0.6.17"`). This is the only complete source tree on disk.
- **mlx-vlm 0.7.1** at `/tmp/knur_clean/lib/python3.12/site-packages/mlx_vlm` is an empty skeleton. `find -type f` returns 0 files. Only directories and `__pycache__` dirs remain, and `mlx_vlm-0.7.1.dist-info/licenses/` is empty.
  - The only 0.7.1 source that survives is knurlogic's own vendored `src/knurlogic/engine/architectures/glm5_next/`. `PROVENANCE.md` records it as taken from that 0.7.1 path, sha `9ae6238…`.
  - So 0.6.17 vs 0.7.1 can only be compared for glm5_next (section 4). For the other families the 0.7.1 side is **not verified**.
  - Other installs on disk: 0.6.17 at `~/mlx-vlm/lib/python3.12`, 0.5.0 at `/opt/anaconda3/lib/python3.12`, 0.4.4 in the uv cache.
- **License:** MIT. `mlx_vlm-0.6.17.dist-info/METADATA` says `License: MIT`, and the licenses/LICENSE file says "MIT License / Copyright © 2025 Prince Canuma". knurlogic's `architectures/THIRD-PARTY.md` already lists `glm5_next/` as mlx-vlm 0.7.1 MIT.
- **knurlogic today:**
  - The Qwen and gemma text models are vendored from mlx-lm 0.32.0 as flat files: `architectures/{qwen3_5,qwen3_5_moe,qwen4_exp,gemma4_text}.py` (PROVENANCE.md).
  - glm5_next is vendored from mlx-vlm 0.7.1 as a package and already includes `vision.py` (257 lines) and `processing.py` (831 lines).
  - PROVENANCE.md notes that glm5_next still depends on unvendored mlx_vlm siblings.
  - `pyproject.toml` has the optional extra `vlm = ["mlx-vlm>=0.7.1"]`.

All paths below are relative to `M=/opt/anaconda3/envs/exo/lib/python3.13/site-packages/mlx_vlm/models` unless stated otherwise.

---

## 1. Shared Qwen vision stack (qwen3_5, qwen3_5_moe and qwen4_exp use one tower)

- `qwen3_5/vision.py` (5 lines), `qwen3_5_moe/vision.py` (5) and `qwen4_exp/vision.py` (5) are each `class VisionModel(Qwen3VLVisionModel): pass`, imported from `..qwen3_vl`.
- `qwen3_vl/vision.py:199-209` whitelists the model types `qwen3_vl, qwen3_5, qwen3_5_moe, qwen3_5_vision, qwen3_5_moe_vision, qwen4_exp, qwen4_exp_vision`.
- The released configs (config.json read with stdlib) all use the same tower: depth 27, hidden 1152, patch 16, spatial_merge 2, pos-emb 2304, `deepstack_visual_indexes=[]`. Only `out_hidden_size` differs:

| family | out_hidden_size |
|---|---|
| 397B | 4096 |
| 35B | 2048 |
| 27B | 5120 |
| Flash-Next | 2560 |

### Files in the Qwen closure (0.6.17)

| file | lines | role |
|---|---|---|
| `qwen3_vl/vision.py` | 447 | tower (see below) |
| `qwen3_vl/config.py` | 127 | `VisionConfig` (:30+), `_config_kwargs`, `_maybe_deserialize_config` |
| `qwen3_vl/qwen3_vl.py` | 252 | `masked_scatter` (:14-32), base `Model.get_input_embeddings` (:44-133) |
| `qwen3_vl/processing_qwen3_vl.py` | 902 | numpy/PIL image and video processor plus `Qwen3VLProcessor(ProcessorMixin)` |
| `qwen3_vl/language.py` | 676 | Qwen3-VL LM (imported by `qwen3_vl/__init__`); not needed if you skip the package `__init__` |

`qwen3_vl/vision.py` contents: `VisionRotaryEmbedding` :54, `PatchEmbed`(Conv3d) :69, `PatchMerger` :106, `Attention` :126, `MLP` :164, block :175, `VisionModel` :194, `rot_pos_emb` :240, `fast_pos_embed_interpolate` :301, `__call__` :381-427, `sanitize` :429-447.

### Imports from other mlx_vlm modules, as a closed set for vision only

- `qwen3_vl/vision.py:6-7` imports `..base.ensure_fused_sdpa` (base.py:529-540) and `.config`.
- `qwen3_vl/config.py:5` imports `..base.BaseModelConfig` (base.py:104-120).
- `qwen3_vl/processing_qwen3_vl.py:21,900` imports `..base.load_chat_template, to_mlx, install_auto_processor_patch` (base.py:18, 36, 541). It also imports from transformers (`ImageProcessingMixin`, `ProcessorMixin`, `BaseVideoProcessor`, `BatchFeature`) at :14-19.
- **base.py (605 lines) itself imports `..turboquant` (6607 lines) and `.cache.create_causal_mask` (cache.py, 2788 lines)** (base.py:13-15). Do not vendor base.py wholesale. Copy the three or four symbols (`BaseModelConfig`, `ensure_fused_sdpa`, `InputEmbeddingsFeatures` base.py:70, `check_array_shape` base.py:390).

### Pixels to embeddings (Qwen)

1. **Preprocess** (`processing_qwen3_vl.py`).
   - `_smart_resize_image` (:182-205) rounds height and width to a multiple of `factor = patch*merge = 32`, clamped to min/max pixels.
   - `_process_one` (:302-354) does a bicubic PIL resize, rescale, and mean/std normalize. It repeats the frame `temporal_patch_size=2` times, reshapes to `(grid_t, tps, C, gh/ms, ms, ps, gw/ms, ms, ps)`, transposes `(0,1,4,7,5,8,3,2,6,9)`, and flattens to `[gt*gh*gw, C*tps*ps*ps]` (= 3·2·16·16 = 1536).
   - It returns `pixel_values` plus `image_grid_thw = [1, gh, gw]` (:356-378).
   - Token count is `gh*gw/merge²` (`num_image_tokens` :383-413). The processor expands the image placeholder to that many `image_token_id` tokens.
2. **Tower** (`vision.py:381-427`): Conv3d patch_embed, plus bilinear-interpolated learned `pos_embed` (`fast_pos_embed_interpolate`), plus 2D rotary (`rot_pos_emb`). Then 27 blocks with `cu_seqlens` (per-image attention), then `merger` (LayerNorm, then 2×2 spatial merge to 4608, fc1, GELU, fc2 to `out_hidden_size`).
   - It returns `(hidden, deepstack_list)`. Deepstack is empty for Qwen3.5: `qwen3_5/config.py:47-58` forces `deepstack_visual_indexes=[]` and raises if it is set.
3. **Merge** (`qwen3_5/qwen3_5.py:59-119`):
   - `inputs_embeds = embed_tokens(input_ids)`.
   - A `vision_cache` lookup keyed by `_image_key` (:89-100).
   - `masked_scatter` into every position where `input_ids == image_token_index or video_token_index` (:121-143, using `qwen3_vl.qwen3_vl.masked_scatter`, a flatten plus `np.where` gather).
4. **Positions (MRoPE)**: `get_rope_index` is at `qwen3_5/language.py:1729-1904`.
   - Text runs get `t=h=w` sequential positions.
   - Each image block gets `(t_idx, h_idx, w_idx) + text_len + st_idx` over the merged grid (`gh/2 × gw/2`).
   - The next text resumes at `max+1`. So an image of N tokens advances the position by only `max(gh,gw)/2`.
   - `rope_deltas = max_pos + 1 - len(tokens)` (:1882-1885).
   - Decode positions are `cache_offset + rope_deltas` (:2022-2035).
   - **The state lives on the module**: `self._rope_deltas` and `self._position_ids` (:1418-1419, 1931-1936) are reset whenever `pixel_values` is passed. The server clears them per request (`server/generation.py:2125-2128`, `generate/ar.py:339-342`). Per-module state like this does not survive prefix-cache reuse or batching.
   - Rotary is interleaved MRoPE (`qwen3_5/language.py:27-46`, `_apply_mrope(..., style="interleaved")`; `rope_utils.py:498-539`, 1505 lines total).
   - Released Qwen token ids: image 248056, video 248057, vision_start 248053, vision_end 248054 (config.json). `qwen3_5/config.py:120` defaults `vision_start_token_id=248045`, but config.json overrides it (248053 in the released configs). `get_rope_index` depends on this value (:1762-1766).

### The knurlogic gap for all three Qwen families (inferred)

- knurlogic's text model is mlx-lm's `qwen3_5.py`. It uses `Qwen3NextAttention` (`architectures/qwen3_5.py:19`) with 1D RoPE at cache offset, and accepts `input_embeddings` (:267-274, :328-334).
- With interleaved MRoPE, 1D RoPE equals MRoPE when t=h=w, which is why text-only works. **For images the attention must take a 3×B×L position_ids and apply interleaved MRoPE.** That means patching or vendoring the attention rope path, which mlx-lm does not have.
- Not verified at runtime.

---

## 2. qwen3_5 (Qwen3.8-27B dense and the Qwen3.5 dense family)

- **Files** (`qwen3_5/`, 5193 lines total):

| file | lines |
|---|---|
| `__init__` | 6 |
| `config.py` | 153 |
| `fp8.py` | 114 |
| `gated_delta.py` | 911 |
| `language.py` | 2239 |
| `qwen3_5.py` | 178 |
| `speculative_verifier.py` | 1587 |
| `vision.py` | 5 |

- **Imports:**
  - `__init__.py:1-2` imports `base.install_auto_processor_patch` and `qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor`.
  - `config.py:4-6` imports `base`, `qwen3_vl.config`.
  - `language.py:7-22` imports `activations.swiglu`, `base`, `cache.{ArraysCache,KVCache}`, `rope_utils.{MRoPERotaryEmbedding, apply_multimodal_rotary_pos_emb}`, `.gated_delta`, `.speculative_verifier`.
  - `speculative_verifier.py:7-13` imports `activations`, `base`, `exact_speculative_verify` (262 lines).
  - `qwen3_5.py:6-13` imports `base.InputEmbeddingsFeatures`, `qwen3_vl.Model`, `qwen3_vl.processing_qwen3_vl`, `qwen3_vl.qwen3_vl.masked_scatter`, `.fp8`.
  - Transitive: `base` pulls in `turboquant` (6607) and `cache` (2788); `mlp` and `switch_layers` pull in `activations` (40).
- **Glue:** `Model(Qwen3VLModel)` at `qwen3_5.py:50-178`. `sanitize_key` (:16-25) maps `model.visual.*` and `model.language_model.visual.*` to `vision_tower.*`.
  - The norm `+1` shift (:28-47, :162-166) is keyed on LM suffixes only. Vision `norm1`/`norm2`/`merger.norm` are not shifted.
- **Vendor for knurlogic:**
  - The shared Qwen vision set (§1).
  - `masked_scatter`.
  - `get_rope_index` (about 175 lines, pure mlx and python).
  - MRoPE cos/sin for the attention path. The simple non-kernel path in `rope_utils` is enough; the Metal `_mrope_apply_kernel` (:569+) is optional.
  - The Qwen image processor minus its transformers bases.
  - Do **not** vendor mlx-vlm's `qwen3_5/language.py`; the text model stays mlx-lm's.
- **Released artifact:** `Qwen3.8-27B-VQ-*` has `model_type qwen3_5`, and its sidecar uses MLX-layout keys (§6).

## 3. qwen3_5_moe (Qwen3.5-397B-A17B and Qwen3.6-35B-A3B)

- **Files:** `__init__` 6, `config.py` 125, `language.py` 118, `qwen3_5_moe.py` 75, `vision.py` 5 (329 lines total).
- **Imports:**
  - `config.py:4-7` imports `base`, `qwen3_5.config.{resolve_qwen_eos_token_id, sanitize_quantization_config}`, `qwen3_vl.config`.
  - `language.py:6-12` imports `qwen3_5.language.{LanguageModel, Qwen3_5Attention, Qwen3_5GatedDeltaNet, Qwen3_5MLP, Qwen3_5Model}` and `switch_layers.SwitchGLU` (228 lines).
  - `qwen3_5_moe.py:4-11` imports `qwen3_5.Model` and helpers from `qwen3_5.qwen3_5`.
- **Glue:** inherits `qwen3_5.Model.get_input_embeddings`. Vision, merge and MRoPE are identical to §2. Only the LM and sanitize differ.
- **Vendor:** the same vision set as §2. Nothing MoE-specific is needed for vision.

## 4. qwen4_exp (Qwen3.8-Flash-Next)

- **Files:** `__init__` 17, `config.py` 203, `language.py` 851, `qwen4_exp.py` 62, `vision.py` 5 (1138 lines total).
- **Imports:**
  - `config.py:4-7` imports `base`, `qwen3_5.config`, `qwen3_vl.config`.
  - `language.py:9-18` imports `cache.{ArraysCache,KVCache,QuantizedKVCache}`, `qwen3_5.language.*`, `qwen3_5_moe.language.Qwen3_5MoeSparseMoeBlock`.
  - `qwen4_exp.py:5-9` imports `qwen3_5.Model`, `qwen3_5.qwen3_5.sanitize_key`.
- **Glue:** `Model(Qwen3_5Model)` (`qwen4_exp.py:14-19`) inherits `get_input_embeddings` and the `vision_cache` hook from qwen3_5. `LanguageModel(Qwen3_5LanguageModel)` (`language.py:832`) inherits `get_rope_index`. Config defaults `vision_start_token_id=248053`, `vision_end=248054` (config.py:166-171).
- **Special case, PLE n-gram embeddings:**
  - `Qwen4ExpNGramEmbedding.__call__(input_ids, cache)` (`language.py:607-641`) hashes the raw token ids, including image-placeholder ids, over an (n-1)-token history kept in an `ArraysCache`.
  - knurlogic's mlx-lm `qwen4_exp.py:818-861` likewise passes `ids` alongside `input_embeddings`.
  - Two consequences:
    - A cached prefix must also carry that n-gram history state.
    - Placeholder ids feed PLE; mlx-vlm does not zero them for qwen4_exp, unlike gemma4 (§6). This is inferred and not verified against the reference.
- **Vendor:** the vision set from §2, plus a check that the vendored mlx-lm `qwen4_exp` attention (`RotaryEmbedding` :813, `rope(_positions(offset,S))` :371) can take MRoPE positions. That attention also has a QSA indexer with its own rope (:267-271).

## 5. glm5_next (GLM-5.3-Flash)

### 0.6.17

- **Files:** `__init__` 13, `config.py` 127, `glm5_next.py` 156, `language.py` 776, `vision.py` 357 (1429 lines total). There is **no processor** in 0.6.17.
- **Imports:**
  - `vision.py:6` imports only `.config`; it is self-contained, with its own `check_array_shape` :9.
  - `glm5_next.py:6` imports `base.{InputEmbeddingsFeatures, LanguageModelOutput}`.
  - `language.py:6-20` imports `base`, `cache.{ArraysCache,CacheList,KVCache}`, `deepseek_v32.language.{DeepseekV32MoE, Model}` (627 lines, which in turn imports `base, cache, mla, mlp, rope_utils, switch_layers`), `deepseek_v4.hyper_connection` (285), `gated_delta` (300), `mla` (82), `mlp` (68), `rope_utils`.

### 0.7.1 (knurlogic's vendored copy, the only 0.7.1 source on disk)

- **Files:** `__init__` 15, `config.py` 213, `glm5_next.py` 156, `language.py` 1105, `processing.py` 831, `vision.py` 257.
- **Imports:**
  - `language.py:6-14` imports `base.{LanguageModelOutput, create_ssm_mask, scaled_dot_product_attention}`, `cache.{…, PoolingCache}`, `deepseek_v4.hyper_connection`, `gated_delta`, **`linear`** and **`sparse_attention`**. These last two are new and not present in 0.6.17. `deepseek_v32` is dropped.
  - `processing.py:8-15` imports from transformers and `base.{install_auto_processor_patch, load_chat_template}`, plus `qwen3_vl.processing_qwen3_vl._flatten_images`.
- Every file differs from 0.6.17 (`diff -q`).

### Vision tower (both versions)

- Patch embed is Conv3d `[T=2, 14, 14]`.
- Vision 2D rotary.
- 24 blocks with qkv, q_norm/k_norm, SwiGLU MLP (0.7.1 adds `_limited_swiglu` clip, `swiglu_limit=10`).
- Then `post_layernorm`, Conv2d `downsample` 2×2 to `out_hidden_size=4096`, and `merger` (gate/up/down, proj, post_projection_norm) (0.6.17 `vision.py:230-337`).

### Merge

- **0.6.17** (`glm5_next.py:20-93`): per-row cumsum gather at `input_ids==image_token_id`, falling back to video_token_id. It accepts `cached_image_features` but has **no `vision_cache` hook**.
- **0.7.1** (vendored `glm5_next.py:12-38`, `_replace_features`): separates images from video using `video_start`/`video_end` cumsum (`in_video`). Video frames use `image_token_id` inside video spans. It still accepts `cached_image_features`.

### Positions and processor

- **No MRoPE.** The LM is NoPE MLA by design (`language.py:444-456` in 0.6.17). Image tokens are ordinary 1D slots. This makes it the simplest family for a prefix cache that survives images.
- Token ids: image 154854, video 154855, image_start/end 154830/154831 (config.json).
- Processor (0.7.1 `processing.py:18-105`): `smart_resize` with a token budget of `min_image_tokens=16`, `max_image_tokens=8000`, and `factor = patch 14 × merge 2 × patch_expand_factor`.

### Weight-name mismatch in 0.7.1 (inferred from the code, not verified by loading)

- 0.6.17's attribute is `self.vision_model` (:18), with sanitize mapping `model.visual.` and `visual.` to `vision_model.` (:116-119).
- 0.7.1 renamed it to `self.vision_tower`, and its sanitize maps only `model.visual.` to `vision_tower.`.
- The **released GLM shards store 347 keys as `vision_model.*`** (all in `model-00019.safetensors`, from index.json). With 0.7.1 code those keys would not bind to `vision_tower`.
- knurlogic needs a `vision_model.` → `vision_tower.` remap.

### Vendor

- Already vendored: `vision.py` and `processing.py`.
- Still needed:
  - Drop the transformers bases from `processing.py`.
  - Inline `install_auto_processor_patch`, `load_chat_template` and `_flatten_images`.
  - Add a `vision_cache` hook.
  - Add the key remap.
- The LM still needs `base`/`cache`/`hyper_connection`/`gated_delta`/`linear`/`sparse_attention`/`mla`/`switch_layers`. Those are text-side dependencies, already noted in PROVENANCE.md.
- The LM has `ArraysCache` (gated-delta) state, so a prefix cache must snapshot SSM state and cannot trim it. The same is true of the Qwen GatedDeltaNet layers.

## 6. gemma4 (gemma-4-26b-a4b-it and gemma-4-e4b-it)

- **Files** (`gemma4/`, 3820 lines total):

| file | lines |
|---|---|
| `__init__` | 9 |
| `audio.py` | 509 |
| `audio_feature_extractor.py` | 371 |
| `config.py` | 145 |
| `gemma4.py` | 310 |
| `language.py` | 873 |
| `processing_gemma4.py` | 1041 |
| `vision.py` | 562 |

- **Imports:**
  - `vision.py:7` imports `.config`; `:224` does a lazy `..base.ensure_fused_sdpa`.
  - `gemma4.py:6-10` imports `base.InputEmbeddingsFeatures`, `.audio.AudioEncoder`, `.language.{LanguageModel, RMSNormNoScale}`, `.vision`.
  - `audio.py` imports `.processing_gemma4`, `.config`, `.vision.ClippableLinear`.
  - `language.py:8-16,113` imports `base`, `cache.{KVCache,RotatingKVCache}`, `rope_utils.initialize_rope`, `switch_layers.SwitchGLU`.
  - `processing_gemma4.py:30,1039` imports `base.{load_chat_template, to_mlx, install_auto_processor_patch}`, plus lazy `...prompt_utils` and `...utils` at :804-805 and transformers at :13-28.
  - `config.py:4` imports `base`.
  - Vision-only vendoring can drop audio. The `audio_config=None` path is at `gemma4.py:55-68`; whether the released configs set `audio_config` was not checked.
- **Tower** (`vision.py:407-562`):
  - `patch_embedder` (`input_proj` Linear 768→1152, where 768 = 3·16·16, plus a 2-axis `position_embedding_table [2,10240,1152]`).
  - `encoder.layers` (q/k/v/o `ClippableLinear`, q_norm/k_norm, 4 RMSNorms, gated MLP).
  - `pooler` (3×3 pooling, `pooling_kernel_size=3`).
  - Optional `std_bias`/`std_scale`.
  - Padding to `max_patches = default_output_length(280)·9` with position −1 (:434-456).
  - Then `embed_vision = MultimodalEmbedder` (RMSNormNoScale, then Linear to text hidden size) (`gemma4.py:22-35`).
- **Processor:** `Gemma4ImageProcessor` (`processing_gemma4.py:104-261`), aspect-preserving, `max_soft_tokens=280`, soft tokens = patches/9.
- **Merge** (`gemma4.py:70-170`):
  - `inputs_embeds = embed_tokens(ids) * embed_scale`.
  - **PLE:** `per_layer_inputs` are computed from ids with image/audio/video placeholders zeroed (:92-105). knurlogic's mlx-lm `gemma4_text.py:444-490` accepts `per_layer_inputs`.
  - `_scatter` has a `vision_cache` get/put (:107-126), then `masked_scatter` (:13-20, cumsum-gather).
- **Positions:** plain 1D RoPE (image tokens consume ordinary positions), so this is prefix-cache friendly. However:
  - **Bidirectional attention inside image blocks.** `use_bidirectional_attention="vision"` (config.py:42) drives `_make_masks` (`language.py:486-515`) via `mm_token_type_ids`, and only when `h.shape[1] > 1`. An image block must therefore be prefilled whole, within one chunk, and must not be split across a prefix boundary.
  - Sliding-window layers use `RotatingKVCache`, so prefix reuse is bounded by the window. This is inferred and not verified.
- **Token ids:** image 258880, video 258884, boi 255999, eoi 258882 (config.json).
- **Vendor:** `vision.py`, the vision part of `config.py`, `MultimodalEmbedder` plus `RMSNormNoScale`, the `get_input_embeddings` merge including the PLE zeroing, the `mm_token_type_ids` bidirectional mask overlay grafted onto mlx-lm `gemma4_text`, and the image part of `Gemma4ImageProcessor`.

## 7. Vision weight tensor names vs module parameter names (safetensors headers read with stdlib)

| artifact | where | count | key form | notes |
|---|---|---|---|---|
| Qwen3.5-397B-A17B (×4) | `model-vision-graft.safetensors`, listed in index weight_map (333) | 333 BF16 | **HF** `model.visual.{blocks.N.{attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2,norm1,norm2}, merger.{norm,linear_fc1,linear_fc2}, patch_embed.proj, pos_embed}` | `patch_embed.proj.weight [1152,3,2,16,16]` is PyTorch layout. `sanitize_key` maps to `vision_tower.*`. `VisionModel.sanitize` (vision.py:429-447) transposes `(0,2,3,4,1)` |
| Qwen3.8-Flash-Next (×4) | sidecar, in index (333) | 333 BF16 | HF `model.visual.*` | same transpose; `merger.linear_fc2 [2560,4608]` |
| Qwen3.6-35B-A3B (×4) | sidecar, in index (333) | 333 BF16 | **MLX** `vision_tower.*` | `patch_embed [1152,2,16,16,3]`, already transposed; `check_array_shape` passes it through |
| Qwen3.8-27B (×3) | sidecar, in index (333) | 333 BF16 | MLX `vision_tower.*` | `merger.linear_fc2 [5120,4608]` |
| gemma-4-26b-a4b | sidecar (**356**, not 333), in index | 356 BF16 | MLX `vision_tower.{patch_embedder.{input_proj,position_embedding_table}, encoder.layers.N.{self_attn.{q,k,v,o}_proj.linear, q_norm,k_norm, mlp.{gate,up,down}_proj.linear, 4 norms}, std_bias, std_scale}` + `embed_vision.embedding_projection.weight [2816,1152]` | no clip params, unquantized |
| gemma-4-e4b | main shard `model-00002.safetensors` | 661 | MLX `vision_tower.*` + clip params `{input,output}_{min,max}` per linear | **`embed_vision.embedding_projection` is quantized (weight/scales/biases)**. `gemma4.sanitize` drops clip params unless `use_clipped_linears` is set (`gemma4.py:224+`) |
| GLM-5.3-Flash (×3) | main shard `model-00019.safetensors` | 347 | MLX `vision_model.*` | matches the 0.6.17 attribute name; breaks with 0.7.1's `vision_tower` (§5) |

- Qwen param names match the module tree exactly: `qkv`/`proj` at vision.py:132-133, `linear_fc1`/`linear_fc2` at :167-168, `norm1`/`norm2` at :178-179, `merger.norm`/`linear_fc1`/`linear_fc2` at :111-116.
- The Qwen sidecars come in two naming conventions, so a vendored loader must accept both. The existing `sanitize_key` plus `check_array_shape` already do.
- Each released dir also ships its own `model.py` (a quantlab VQ runtime, e.g. `Qwen3.8-27B-VQ-3.9bpw/model.py:1-30`, about 4.6–5.2k lines each). It is not part of mlx-vlm and was not analyzed further.

## 8. Reusing images across turns: what mlx-vlm 0.6.17 already has

- **Encoded-image cache:** `mlx_vlm/vision_cache.py` (79 lines, standalone, imports only mlx and hashlib).
  - An LRU `VisionFeatureCache` keyed by path/URL string or sha256 of the PIL bytes (:35-50).
  - It stores post-projection features.
  - Consumed by `qwen3_5.Model.get_input_embeddings` (qwen3_5.py:89-100, so also qwen3_5_moe and qwen4_exp) and `gemma4._scatter` (gemma4.py:107-126). Not consumed by glm5_next.
  - Wired in `server/generation.py:1684-1699` (`_image_key = images`); size comes from `MLX_VLM_VISION_CACHE_SIZE` (`server/runtime_config.py:200,230`).
  - This is the "encode an image once" piece, and it is small enough to vendor verbatim.
- **Prefix cache that survives images:** `mlx_vlm/apc.py` (4529 lines; imports `_stream_cleanup`, `apc_storage`, `kv_quant`).
  - `hash_image_payload` (:371-402) hashes `pixel_values` content, and the result is folded into the block hash via `semantic_extra_hash` (:196+).
  - `media_safe_prefix_min` and `adjust_prefix_to_text_suffix_boundary` (:443-503) force a restored prefix to contain **all** media spans, so the suffix is text-only.
  - That is exactly the "turn 5 costs only turn 5's tokens" rule. Borrow the design; don't vendor 4.5k lines.
- **MRoPE caveat for Qwen:** a restored prefix must also restore `rope_deltas`, because new-token position = cache_offset + rope_deltas. mlx-vlm keeps that on the LM module (`_rope_deltas`, language.py:1418-1419) and resets it per request. knurlogic should store `rope_deltas` with the cached prefix entry. For GLM (NoPE) and gemma (1D) positions are just the offset.
- **Recurrent state:** qwen3_5/moe/qwen4_exp (GatedDeltaNet `ArraysCache`), glm5_next (gated_delta) and qwen4_exp's n-gram `ArraysCache` hold state that can't be trimmed. The prefix cache has to snapshot at the reuse boundary rather than trim.

## 9. Minimum vendoring set for knurlogic to own vision (estimates, not verified)

- **Common (about 300 lines):**
  - `vision_cache.py` (79).
  - Extracted `BaseModelConfig` / `InputEmbeddingsFeatures` / `ensure_fused_sdpa` / `check_array_shape` from base.py (about 60).
  - A `masked_scatter` variant (about 20).
  - A text-only-suffix prefix rule modelled on apc.py:443-503 (about 60).
- **Qwen (shared by 3 families, about 1,000 lines):**
  - `qwen3_vl/vision.py` (447).
  - The `VisionConfig` part of `qwen3_vl/config.py` (about 60).
  - The image path of `processing_qwen3_vl.py` (:182-414, about 230, rewritten without transformers bases).
  - `get_rope_index` (about 175).
  - Interleaved MRoPE cos/sin from `rope_utils` (about 60).
  - `sanitize_key` plus the vision sanitize.
  - A patch to the mlx-lm attention in `architectures/qwen3_5.py` (via qwen3_next Attention) and `qwen4_exp.py` so it takes 3D position_ids.
- **glm5_next:** already vendored (0.7.1 `vision.py` 257 + `processing.py` 831). Still needed: the `vision_model.` key remap, a `vision_cache` hook, and removing the processing.py dependencies on transformers, `base` and `qwen3_vl._flatten_images`.
- **gemma4 (about 900 lines):**
  - `vision.py` (562, keeping `ClippableLinear` for e4b).
  - Vision config (about 40).
  - `MultimodalEmbedder` plus `RMSNormNoScale`.
  - The merge with PLE zeroing (about 60).
  - The bidirectional image-block mask from `language.py:455-515`, ported into mlx-lm `gemma4_text`.
  - The image part of `Gemma4ImageProcessor` (:104-261, about 160).
  - Skip audio.
- **Runtime dependencies:** numpy and PIL (both processors resize with PIL). transformers can be dropped if the processor classes are rewritten without the `ImageProcessingMixin`/`ProcessorMixin` bases.