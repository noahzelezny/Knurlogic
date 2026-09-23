# Exo fork (mtp-stage1): how the vision path and image-aware prefix cache work

Base path: `~/exo/src/exo/worker/engines/mlx/` (shortened to `…/mlx/` below). I only read code. I loaded nothing and ran nothing.

## Findings

- **Images are recognised by the SHA-256 of their decoded pixels.** It is `sha256(PIL.Image.tobytes())` (`vision.py:724`, `vision.py:749-754`). The same pixels therefore match even when the base64 differs (for example, the file was re-encoded).
- **Vision-tower output is kept in memory per process, but the key covers the whole image list.** `VisionProcessor._feature_cache` holds up to 32 entries, dropping the oldest (`vision.py:742-743, 764-773`). The key is a hash over *all* images in the request (`vision.py:749-754`). Turn 5 with the same one image hits. Adding a second image misses, and every image is re-encoded. So "encode once per conversation" only holds while the image set stays the same.
- **The whole prompt is re-embedded every turn.** `create_vision_embeddings` runs `embed_tokens` over the entire prompt (`vision.py:659-693`) and `mx.eval`s it (`vision.py:816`), even when the prefix cache will skip most of it. That is O(prompt) work and memory on every turn.
- **Prefix matching** (`cache.py:491-597`):
  1. Take the longest shared token prefix (`get_prefix_length`, `cache.py:767`).
  2. `_validate_media_match` (`cache.py:599-627`) cuts the match back to the start of the first cached image region whose hash differs from the query's region at the same `start_pos`.
  3. Restore: `deepcopy(entry)` (`cache.py:584`), then `trim_cache` down to `restore_pos` (`cache.py:695`). Models with recurrent/SSM layers restore from the nearest snapshot at or before the target (`cache.py:475-489, 552`). With no usable snapshot, the cache starts fresh (`cache.py:569`).
  - Regions come from runs of `image_token_id` (`vision.py:696-727`). The i-th run is paired with the i-th image, which assumes each image is one contiguous run.
  - Edge case: if the query has no region at a cached `start_pos`, the check just skips it (`continue`, `cache.py:616`). Tokens matching makes that unlikely (not verified).
- **A cached prefix does contain the image's effect.** Its KV was computed from the merged embeddings, so a hit skips the image positions entirely. `patch_embed_tokens` receives `start_offset=prefix_hit_length` and slices the precomputed embeddings by absolute position (`generate.py:85-121`, called at `generate.py:799-808` and `mtp_batch_generate.py:257-267`). Sequential path: the pool entry is written back with `media_regions` (`generate.py:864-878`). Batch path: same (`mtp_batch_generate.py:365-373`).
- **Rows with images never draft.**
  - Sequential path: `plan_mtp(..., has_vision=vision is not None)` returns None (`mtp/speculative.py:127`, `generate.py:746`).
  - Sequential path, text-only: when MTP is on, the prefix pool is switched off entirely (`generate.py:747-748`). The batch path lifts this.
  - Batch path: `RowParams(drafts=vision is None)` (`mtp_batch_generate.py:223`).
  - Batch path: a vision row stores a trunk-only pool entry, which a later drafting request cannot use (docstring `mtp_batch_generate.py:14-23`; `split_pool_entry` at `:240`).
- **What actually breaks head seeding with images.** The docstring says seeding runs "under the vision embedding patch". In the code it does not:
  - The `with ctx:` block covers only the chunked trunk prefill (`mtp/batch_loop.py:200-220`).
  - The final `model(ids[:, n-1:])` call (`:222`) and `seed_head` (`:236`) both run after the patch has been removed.
  - The real defect: the head computes `e = norm_e(core.embed_tokens(nxt_id))` (`mtp/heads/qwen35.py:285`, `glm5.py:230`, `qwen4_exp.py:211`). `seed_head` feeds it `ids[:, 1+i:1+j]` (`mtp/seed.py:56`). At image positions the head therefore sees the plain embedding of the image *placeholder token* instead of the image features. That is off-distribution input, not a crash.
  - A second hazard if seeding were ever moved inside the patch: `_inject` keeps one shared `offset` counter (`generate.py:95-101`), so head calls would advance it and misalign the trunk's own splicing.
- **Recent commits on these paths** (`origin/main..HEAD`):
  - `a3f7baec`: MTP batch KV prefix pool, with the head cache stored beside the trunk's.
  - `1e140d4d`: batch MTP.
  - `b6319b11`: MTP stage 0.
  - `5de365f9`, `e0085bb5`: vision weight prefix and sidecar vision weights.
  - `95d79be1`, `62fbc858`, `6e7b96cd`: duck-typed snapshot handling.
  - `2c526bd6`, `b9c32276`: snapshot thinning and capping.
  - `e5d3298b`: chunked-safe pool reuse.
  - `860883a5`, `ce019259`, `1970e593`: pool eviction and `use_prefix_cache`.

## What's needed

**(a) Encode each image once per conversation**

1. Key the feature cache per image, not per request. Cache `sha256(pixels) -> (features[n_i, D], n_tokens_i)` and store it by content hash. Then build the concatenated features from per-image hits and send only the misses to `encode_images`.
   - Batching the misses is fine for Qwen, whose `grid_thw` path runs all images in one tower call (`vision.py:600-606`).
   - Gemma4 already encodes one image at a time (`vision.py:607-619`).
2. Hash the raw base64 bytes as a fast first-level key, so the image isn't decoded twice (`_find_media_regions` and `_image_cache_key` both decode it: `vision.py:723`, `vision.py:751`). Keep the pixel hash as the canonical ID.
3. Build merged embeddings only for the suffix after the prefix hit. Look up the pool first, using regions from the tokens alone. Then call `embed_tokens(prompt[hit:])` and scatter in only the features whose placeholder positions are ≥ `hit`.
   - Today's order is the reverse: embed the whole prompt, then look up the pool.
   - The image-feature index for a position is `cumsum(is_image)[pos] - 1` over the full prompt (`vision.py:684`), so it must stay global (count placeholders before `hit`).
4. Optional: hang the per-image features off the pool entry so they are evicted with it, rather than using a separate 32-entry cache.

**(b) Let rows with images draft**

1. Seed the head from the merged embeddings. Add an `e_override` argument to `head.advance` and `draft_logits`, used instead of `core.embed_tokens(nxt_id)`. `seed_head` then passes `merged[:, 1+i:1+j]`.
   - This keeps the placeholder token IDs for the per-layer or other token-ID paths, as the Gemma per-layer patch does (`generate.py:136-145`).
   - The heads live in knurlogic's own `engine/mtp/heads/*`, so it is a local change (not verified in the knurlogic tree).
2. Gemma: undo the `embed_scale` pre-divide applied to image features (`vision.py:678-680`) for whatever the head's `norm_e` expects (not verified; gemma4 may not have a head at all).
3. Decode positions are never image tokens, so drafting after prefill is unchanged.
4. Keep the head outside the patch. Better, replace the offset-counter monkeypatch with a direct `input_embeddings=` argument. mlx-lm's qwen3_5 and gemma models do take one (not verified for knurlogic's vendored models).
5. With the head seeded correctly, vision rows can store trunk + head entries, so the `drafts=vision is None` guard and the trunk-only pool entry go away.

## Reusable verbatim vs exo-specific

- **Reusable verbatim:**
  - `MediaRegion`, `_find_media_regions`, `_validate_media_match`.
  - `create_vision_embeddings`, including the Gemma `embed_scale` handling.
  - `patch_embed_tokens` (or replace it as in (b) step 4).
  - The per-family branches in `encode_images`, but they depend on mlx_vlm modules loaded via `_import_mlx_vlm` (`vision.py:222`), so vendoring means copying those tower and processor classes.
  - The snapshot, trim and thinning logic in `cache.py` (`CacheSnapshot`, `_thin_snapshots`, `trim_cache`, duck copies).
- **Exo-specific:**
  - `VisionCardConfig`, `ModelId`, `Base64Image`, `TextGenerationTaskParams`, `chat_template_messages`, `build_vision_prompt`/`_format_vlm_messages` (tied to exo's API message shape).
  - `KVPrefixCache`'s distributed `group` and eviction by memory percentage.
  - `remote_prefill` and the prefill endpoint (`generate.py:809-830`).
  - The Kimi-VL processor branch (`vision.py:512-515`).
  - `DeepseekV4Cache` copies (`cache.py:181`).
  - The `EXO_MTP`/`plan_mtp` placement gates.
  - The `ExoBatchGenerator` contract.