# Adversarial review: vision build design (knurlogic, five families)

This is a read-only review. I loaded nothing, wrote nothing and made no POSTs. Findings marked **[read]** were checked in this pass. Findings marked **[inferred]** are my own inference and not verified.

## Verdict

**The design is not ready to build.** The idea is sound: per-image store, positional sentinel key, suffix-only embedding and drafts off in Phase A. But there are blockers:

- Qwen positions go wrong on text-only suffix turns (B1).
- mlx runs on the HTTP thread (B2).
- The load path contradicts itself (B3).
- mlx-lm line citations are pinned to a version that is not the one installed (B4).
- The gemma chunk rule has no enforcement point on the single-request path (B5).

Fix B1–B5 and the package-ownership collisions (C1–C3), then fan out.

---

## Blockers

### B1. Qwen MRoPE is wrong on every turn after the image. The core "real context" case breaks silently. **Severity: critical**

- **Evidence.**
  - Design §1.3 step 4 computes `family.embed` (and so `position_ids` and `rope_delta`) only "if the uncached span `[hit:]` contains sentinels".
  - On turn 2–5 the image is inside the hit, so the suffix is text. By design, no positions are passed. The trunk then falls back to 1D `offset` positions.
  - The correct position is `offset + rope_delta`, where `rope_delta = max+1-len` (vlm-families §1, mlx_vlm `qwen3_5/language.py:1882-1885, 2022-2035`).
  - Text still answers fluently. G4 and G8 as specified run the image turn cold, so they don't catch it.
  - Turn-5 "recalls red" might still pass, because the positions are only shifted. This is the silent-degradation case.
- **Fix.**
  - Positions must be derived from the **full key** on every prefill and decode for any row whose key contains a sentinel, whether or not the suffix does.
  - Make `rope_delta` a function of the key, not of the suffix. `positions(key)` is already specified as pure, so call it unconditionally when `key_has_image(key)`.
  - For the prefill of a text suffix, `position_ids = arange(hit, L) + rope_delta` broadcast to 3 rows.
  - Add gate **G7b**: turn 1 has an image, turn 2 is text only and warm. Compare it against a cold turn 2 through mlx-vlm (tokens identical, last-prefill logits atol 1e-4). This is the test that actually fails if B1 is present.

### B2. Tower encode on the HTTP thread races the generator's mlx stream. **Severity: critical**

- **Evidence.**
  - Design §1.3 step 1 runs `preprocess + encode` in `seam._post` on the HTTP handler thread.
  - mlx-lm does all generation on its own thread (reports: `_generate` on the generator thread; `_tokenize` runs there per knurlogic-serve §2a).
  - I don't know of any safe concurrent mlx evaluation on a shared GPU stream from two Python threads *[inferred]*. The host is also shared, and two uncoordinated allocations add up to peak memory.
- **Fix: move all image work onto the generator thread.**
  - mlx-lm 0.31.3 `ResponseGenerator._tokenize(self, tokenizer, request, args)` has `request.messages` in hand (installed `server.py:516-535` **[read]**).
  - So wrap `_tokenize` to do all of it: pull image parts out of `request.messages`, decode, hash, look up the store, encode on a miss, replace each part with the placeholder, call the real `_tokenize`, expand, and return the key.
  - This also **removes risk #2 (the carrier)** entirely. No side table and no `id(body)` are needed.
  - `_post` only needs to return 400 for images sent to a model without vision. The PIL decode can stay on the HTTP thread (cheap and not mlx), but it's simpler not to split it.

### B3. The load path contradicts itself for VQ artifacts (sidecar glob, sanitize, bundled `model.py`). **Severity: critical**

- **Evidence [read].**
  - Every checked artifact has `"model_file": "model.py"`: Qwen3.8-27B, gemma e4b, gemma 26b, GLM 2.7bpw.
  - Each bundle's `model.py` defines `class Model(_arch.Model)`, where `_arch` comes from `_resolve_arch(model_type, …)` (27B `model.py:4982-5054, 5164`). `_loading_runtime()` inspects the import stack: when the loader is mlx_lm, the bundle binds `mlx_lm.models.<type>`, deliberately text-only (`:5047-5052`).
  - knurlogic's `register` installs vendored architectures into `sys.modules` (`engine/register.py:100-109`). So a P1 edit to `architectures/qwen3_5.py` *does* reach VQ artifacts, but only if `register()` runs before `load`. The design never states this precondition.
  - `model-vision-graft.safetensors` **does match** mlx-lm's `model*.safetensors` glob. The design's framing implies it is loaded separately, and it doesn't say this.
    - mlx-lm already reads the 333 tensors (or 356 for gemma 26b) into memory, and `sanitize` discards them (`architectures/qwen3_5.py:428-432` **[read]**).
    - P1 says "stop dropping vision keys at `qwen3_5.py:430`". Doing that makes `load_weights(strict=True)` fail, because the mlx-lm text `Model` has no `vision_tower`.
    - Or, if the tower becomes a submodule, the VQ `Model.__init__` subclass and `_arrayish(super().__call__)` (`:5164-5198`) must tolerate it. Not verified.
  - Flash-Next bundles are mlx_lm-shaped with PLE (`model.py:5001-5012` comment). The Qwen 16/20 are VLM-layout (`language_model.*` keys). Whether knurlogic's `sanitize` handles both spellings for VQ keys is not verified.
- **Fix: choose one option in P0 and freeze it.**
  - **(a)** Keep `sanitize` dropping the vision keys. `Family.load_weights` reads the vision tensors itself with `safetensors`/`mx.load` filtered by the index `weight_map` (sidecar or main shard) and builds a standalone tower module that is not attached to the trunk. This option is recommended because it leaves the bundled runtime untouched.
    - The cost is double I/O for the sidecar (it is read then discarded). Accept that, or add a `sanitize` hook that stashes the vision dict aside instead of dropping it.
  - **(b)** Attach the tower to the trunk. This needs audits of every bundle's `Model` subclass across ~20 artifacts. Reject it.
  - State the precondition `register(*families)` before `seam.load` in the contract, and add a load test with a fake `model_file` bundle that subclasses `_arch.Model`.
  - GLM: the remap `vision_model.* → vision_tower.*` then lives in `Family.load_weights`, not in the vendored `glm5_next.py`. That also shrinks P3's shared-file edits.

### B4. mlx-lm version skew makes the line-level interface contracts unreliable. **Severity: high**

- **Evidence [read].**
  - The default `python3` resolves `mlx_lm` 0.31.3 at `/opt/anaconda3/lib/python3.12/site-packages`. `pyproject.toml:27` requires `mlx-lm>=0.31.3`.
  - `seam.py:28-31` says 0.31.3 executes `model_file` unconditionally and 0.32.0 gates it.
  - The reports cite `_tokenize :536-553`, `fetch_nearest_cache :753/:965`, `process_message_content :118-144`.
    - In the installed 0.31.3, `_tokenize` is at `:516` and `fetch_nearest_cache` is at `:753/:965`.
    - So `_tokenize` doesn't match, while the other two coincide.
    - The chat-ui report read `/opt/anaconda3/lib/python3.12/.../server.py:134-141`, while knurlogic-serve cites `:118-144`. The reports disagree with each other.
  - Which interpreter serves knurlogic, and which env the P1–P3 references run in, is not established. G1–G4 use `/opt/anaconda3/envs/exo/bin/python` (py3.13). Does that env have knurlogic installed with a matching mlx-lm? Not verified.
- **Fix.**
  - P0 pins `mlx-lm==X` (the one actually serving) and records `mlx_lm.__version__` plus a hash of `server.py` in a test that fails on drift.
  - Every seam wrap resolves methods by name with `hasattr` asserts, never by line.
  - Decide the test env explicitly. Either install knurlogic editable into the exo env, or vendor the mlx-vlm reference outputs as `.npz` goldens generated once. The goldens option is recommended: it removes the cross-env dependency and makes G1–G4 run anywhere.

### B5. The gemma image-block chunk rule has no enforcement point on the single path. **Severity: high**

- **Evidence.**
  - Design §1.3 step 6 passes `input_embeddings` to `stream_generate`. mlx-lm chunks the prefill internally at `prefill_step_size` (installed `server.py:825, 987` **[read]**) with no hook for boundaries.
  - gemma's bidirectional overlay is active only for L>1 chunks that contain the whole block (mlx_vlm `gemma4/language.py:486-515`).
  - `chunk_boundaries` exists only in P4's `batch_loop.admit`.
  - Also, `_is_batchable` is false whenever `args.seed` is set (`server.py:685-686`), so seeded requests take the single path.
- **Fix.** Either route **all** vision requests through `MTPBatchGenerator` (with `head=None`), with the batch engine as the only vision path, or have `_stream` do its own snapped prefill before handing off to decode. Also add a gemma G4 variant with `prefill_step_size=16` on the single path.

---

## Collisions and ownership gaps

| # | Sev | Issue | Evidence | Fix |
|---|---|---|---|---|
| C1 | high | `tests/fixtures_vision.py` is owned by P0, but P4 needs a stub family "in" it and P1–P3 each need family-specific tiny configs and weight mappers. Four packages would edit one file. | design §2 P0 / P4 | P0 ships `fixtures_vision.py` with only the scaling helpers and a `STUB` family. Each package puts its own fixtures in `tests/fixtures_vision_<fam>.py`. |
| C2 | high | Loadlock callers are unassigned. The test plan says `seam.load`/`switch` take `model_load`. `seam.py` is owned by P4, but P4's package list doesn't include it. P5 edits `mcp.load`. P0 creates the lock with no callers. | test-plan §4, design P4/P5 | Assign the `seam.load/switch` lock wiring to P4 explicitly, and the `mcp.py`/`serve.py` wiring to P5. |
| C3 | med | `qwen3_5_moe.py` imports `Model` from `.qwen3_5` (`architectures/qwen3_5_moe.py:6` **[read]**), so MRoPE threading there is transitive. qwen4_exp is independent (it imports only mlx). P1 lists three edits, but the real surface is two plus a check. Low conflict, but G5 for the MoE is then a regression test of the P1 `qwen3_5` edit. | read | Note it in P1, and run the moe G5 inside the qwen3_5 test. |
| C4 | med | P5 depends on `served_vision()`, which P4 creates in `seam.py`. P5's done gate stubs `/status.json`, but `web.py` must import something. | design dep graph | P0 defines `served_vision` as a stub in `engine/vision/__init__.py` (a module-level `_SERVED_SPEC`). P4 sets it, P5 reads it, and `seam.py` isn't imported by P5. |
| C5 | low | `PROVENANCE.md`/`THIRD-PARTY.md` appends | design §2 | Use per-family `engine/vision/<fam>/PROVENANCE.md` (the design's own alternative), and make it mandatory. |
| C6 | med | Phase-B edits to `mtp/heads/*` and `seed.py` sit inside P4, the critical-path package, behind a flag with no gate before merge. | P4 | Move Phase B to a P6 after integration. P4 ships Phase A only. |

---

## Contract and correctness issues

1. **The sentinel omits `proc_hash` and the model identity. Severity: medium.**
   - The store key is `(model_key, sha, proc_hash)`, but the KV key is `("img", sha, k)`.
   - If the processor settings change (the store-size setting, a max_pixels clamp in `request.py`), `n_tokens` may stay the same while the features differ. The KV then hits with stale image features.
   - Fix: `("img", sha, proc_hash, k)`. The trie is already per `model_key`, so that part is fine.

2. **The segment split is ignored. Severity: medium.**
   - The mlx-lm `_tokenize` returns `prompt, segments, segment_types, initial_state` (docstring `server.py:516-528` **[read]**).
   - The design's wrap only rewrites `prompt`. `segments` also carry ids and feed `insert_segments`/checkpoints.
   - Fix: the wrap expands every segment with the same span map and asserts `sum(len(seg)) == len(key)`.

3. **G8 and the real-model turn-N `prompt − cached ≤ new + 32` will fail for template reasons, not vision reasons. Severity: high (the gate is wrong, not the build).**
   - Qwen3-style templates drop earlier turns' `<think>` and `reasoning_content`. The turn-N prompt is then *not* turn N-1's `prompt + generated`: it diverges at the start of the previous assistant turn *[inferred; not verified per template]*.
   - For hybrid GatedDeltaNet caches only "shorter/exact" hits work. The hit then drops to the nearest snapshot, possibly well before the image.
   - The design lists this as risk 9 but also uses it as a pass condition.
   - Fix:
     - G8 must use a tiny template that doesn't rewrite history, plus a separate test with the real template.
     - The real-gate condition should be `cached ≥ end of the last image span` (the image is never re-prefilled) with tower calls == 1.
     - Report `prompt − cached` as a metric, not a gate, until the thinking re-render is handled (for example, by storing a checkpoint at the end of the user turn).

4. **Gates that cannot fail. Severity: medium.**
   - G5 "`position_ids=None` identical" is a tautology: the None branch is the old code. Replace it with: text-only through the MRoPE path with `position_ids = broadcast(arange)` equals the 1D path. That proves the interleave is right.
   - G10 in Phase A with `drafts=False` compares plain decoding with plain decoding. It passes by construction, so label it a smoke test.
   - The P5 done gate ("page loads against a stubbed status") doesn't test attaching, sending or the byte-identity of the resend.
   - The G6 counter is incremented by the code under test. It must wrap the tower's `__call__` from outside.

5. **Image token count varies with resolution, and the chat UI makes it worse. Severity: medium.**
   - The Qwen count is `gh*gw/4` after `smart_resize`. gemma is aspect-preserving with `max_soft_tokens=280`, so the count is not fixed. The design's `VisionSpec.fixed_tokens` and the UI's "gemma fixed" note are unverified.
   - The UI downscales to a 1536 long edge before the server's own resize, so what the UI estimates is not what gets charged.
   - Fix: the chip estimate calls a server endpoint (`/v1/vision/estimate?w=&h=`) that runs the real `smart_resize`, or ports the exact function. Set `fixed_tokens=None` for gemma until the processor is read.

6. **gemma placeholder semantics. Severity: medium.**
   - mlx-vlm gemma expands `<start_of_image>` into `boi + N×image + eoi` (ids 255999, 258880, 258882, per vlm-families §6).
   - The `expand_ids` contract ("1 pad → n pads") doesn't say whether boi and eoi come from the template or from expansion. Getting this wrong shifts every position.
   - Fix: `placeholder_text` and `expand_ids` are specified per family with the exact id sequence before and after. G2 asserts the expanded id list, not just the count.

7. **Per-row `rope_delta` in batched decode has no carrier across prefix-cache reuse. Severity: medium.**
   - The design holds the delta "on the row state". A cache entry restored for turn N+1 carries no delta, and recomputing it from the key requires the store `grid_thw`. Once the store evicts that image, `positions(key)` can't run.
   - Fix: carry `grid_thw` in the sentinel, or make the store's `ImageRef` metadata non-evictable (small) separately from the features. Then `positions(key)` never depends on evictable data.

8. **qwen4_exp n-gram PLE state. Severity: low–medium.** The n-gram `ArraysCache` hashes raw ids, including placeholders. `to_ids` must produce exactly the ids the reference hashes. Include qwen4_exp in G7b.

9. **Flash-Next vision "unreachable" per its own bundle. Severity: low.**
   - `model.py:5001-5012` says the four Flash-Next rungs can't serve vision through mlx_vlm.
   - Under design option (a) the standalone tower is fine. Say so, because the artifact's README and `vision-smoke` report FAIL and will confuse users.

10. **Store memory. Severity: medium.**
    - The 65 MB figure per GLM image is right in magnitude.
    - The store must be byte-bounded **and** default small (for example 256 MB), and counted in `tuning/resolve.py` before load, not after.
    - `EncodedImage.nbytes` must use `feats.nbytes` after eval.

11. **Request size. Severity: low.** Also clamp decoded pixels (a decompression bomb via PIL `MAX_IMAGE_PIXELS`) before hashing.

12. **Pickling for distributed serving. Severity: low.** `_share_object` pickles requests (`server.py:484-492` **[read]**). With B2's fix nothing extra rides on the request, so this is moot. Otherwise PIL objects and mx arrays on the request would break a distributed serve.

---

## Unbacked claims

- "mlx-vlm 0.7.1 at `/tmp/knur_clean` exists": the reports contradict each other. vlm-families found 0 files, while serve and test-plan saw directories only. Treat it as absent. The GLM G1 reference is 0.6.17 only, and the delta must be documented.
- "gemma4 may have no head": check `find_head` on the gemma artifacts before planning Phase B. No gemma artifact lists an mtp sidecar (**[read]**; only GLM shows `mtp-head-q6.safetensors`).
- Phase-B "acceptance ≥80% of text": no baseline exists for the image-row text continuation. Define the measurement.

---

## Changes required before fan-out (P0 amendments)

1. Move all image work into a `_tokenize` wrapper on the generator thread, and drop the `_post` rewrite and the carrier (B2, risk 2).
2. Compute positions from the full key on every Qwen prefill and decode, and add G7b (B1).
3. Load the tower standalone from the index `weight_map`, keep `sanitize` dropping the vision keys, and state the `register()` precondition (B3).
4. Pin mlx-lm and use npz goldens for the mlx-vlm references (B4).
5. Make vision requests batch-engine-only, or snap the prefill in `_stream` (B5).
6. Change the sentinel to `("img", sha, proc_hash, k)`, expand the segments too, and make the metadata non-evictable.
7. Split the fixtures per family, assign the loadlock wiring, and move Phase B to its own package after integration.
8. Rewrite G5, G8, G10 and the real-gate reuse condition as above.

Paths:
- src/knurlogic/engine/seam.py
- src/knurlogic/engine/register.py
- src/knurlogic/engine/architectures/qwen3_5.py
- src/knurlogic/engine/architectures/qwen3_5_moe.py
- ~/.exo/models/TheDrainFlorist--Qwen3.8-27B-VQ-3.9bpw/model.py
- /opt/anaconda3/lib/python3.12/site-packages/mlx_lm/server.py