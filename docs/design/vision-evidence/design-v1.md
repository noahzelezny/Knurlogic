# Build design: vision for knurlogic's five released families, with images reused across turns

Evidence comes from the five reports. I also re-read a few anchors in the knurlogic tree myself (read-only):
- `engine/mtp/batch_loop.py:148` `admit`, with `prefill_ctx` at :158, :179 and :207
- `engine/mtp/batch_generator.py`: `_admit_one` :163, `RowParams` :176, `_entry` :194
- `engine/seam.py`: `load` :101, `serve` :153, `_post` :220, `_stream` :621, `_factory` :673
- `architectures/qwen3_5.py:271-274,332-334,417-420` take `input_embeddings`
- `architectures/gemma4_text.py:447-533` takes `input_embeddings` and builds `per_layer_inputs` from ids at :533

`src/knurlogic/engine/vision/` does not exist yet. Anything else here that was inferred rather than read is marked *not verified*.

## 1. Architecture

### 1.1 Module layout

Only `engine/` imports mlx.

```
src/knurlogic/engine/vision/
  __init__.py      contracts: ImageRef, VisionSpec, EncodedImage, Family protocol, errors   [P0]
  _base.py         extracted from mlx_vlm/base.py: BaseModelConfig(:104-120), ensure_fused_sdpa(:529-540),
                   check_array_shape(:390); NO turboquant/cache import (base.py:13-15)                    [P0]
  key.py           cache-key codec: expand / to_ids / image_spans / is_key                              [P0]
  store.py         encoded-image LRU, per image (vendored idea from mlx_vlm/vision_cache.py, 79 lines)   [P0]
  images.py        decode base64/data-URL/local path -> PIL, sha256(PIL.tobytes()+mode+size)            [P0]
  scatter.py       masked_scatter/ embeds_for(key, start, end, model, store)                             [P0]
  registry.py      family -> "module:attr" lazy table (fixed strings, so packages don't edit it)         [P0]
  qwen/            tower.py (qwen3_vl/vision.py 447), config.py, processor.py (processing_qwen3_vl
                   :182-414 minus transformers), rope_index.py (qwen3_5/language.py:1729-1904), mrope.py [P1]
  gemma4/          tower.py (vision.py 562 incl. ClippableLinear), embedder.py (MultimodalEmbedder,
                   RMSNormNoScale), processor.py (processing_gemma4 :104-261), merge.py (PLE zeroing)   [P2]
  glm5/            adapter over architectures/glm5_next/{vision,processing}.py + key remap              [P3]
  request.py       OpenAI/Anthropic body -> (text-only body, [ImageRef]); pad expansion              [P4]
src/knurlogic/machine/loadlock.py   flock protocol (test-plan §4)                                     [P0]
tools/vision_gate.py                real-model gate (test-plan §3)                                    [P5]
```

### 1.2 Contracts (P0, frozen before the fan-out)

```python
# engine/vision/__init__.py
@dataclass(frozen=True)
class ImageRef:              # one image occurrence in a prompt
    sha: str                 # sha256 hex of decoded pixels + (mode,size)  (exo vision.py:724,749-754)
    n_tokens: int            # placeholder count after merge/pool
    grid_thw: tuple[int,int,int] | None   # Qwen/GLM; None for gemma
@dataclass
class EncodedImage:          # value in store
    feats: "mx.array"        # [n_tokens, text_hidden], post-projection, pre-embed_scale for gemma
    ref: ImageRef
    nbytes: int
@dataclass(frozen=True)
class VisionSpec:            # exposed to web/mcp/UI
    family: str; image_token_id: int; patch: int; merge: int
    min_pixels: int; max_pixels: int; fixed_tokens: int | None; proc_hash: str
class Family(Protocol):
    spec: VisionSpec
    def preprocess(self, img: "PIL.Image") -> tuple[dict, ImageRef]    # pixel_values etc.
    def encode(self, pixels: dict) -> "mx.array"                        # [n_tokens, hidden]
    def placeholder_text(self, ref: ImageRef) -> str                    # single-pad form for the template
    def expand_ids(self, ids: list[int], refs: list[ImageRef]) -> list[int]  # 1 pad -> n_tokens pads
    def embed(self, model, ids: "mx.array", key_slice, store) -> dict   # {"input_embeddings":..., extra kwargs:
                                                                         #  position_ids / per_layer_inputs / mm_mask}
    def positions(self, key: list) -> tuple["mx.array" | None, int]    # (3xL pos or None, rope_delta)
    def load_weights(self, model_path: str, model) -> int               # returns tensor count bound
def build(family: str, model_path: str, text_model) -> Family | None    # via registry.py; None = no vision
```

**Cache key format** (`key.py`). The key is the same length as the KV:

```python
Sentinel = tuple  # ("img", sha:str, k:int)  k in 0..n_tokens-1
def expand(ids: list[int], refs: list[ImageRef], image_token_id: int) -> list[int | Sentinel]
def to_ids(key, image_token_id) -> list[int]
def image_spans(key) -> list[tuple[int, int, str]]   # (start, end, sha); needs each image as ONE contiguous run
def is_key(seq) -> bool
```

- **Why this shape.** mlx-lm's `PromptTrie` accepts any hashable (`mlx_lm/models/cache.py:1536-1552`). Positional sentinels keep length equal to KV length, which the segment trim at `server.py:756-765` needs (knurlogic-serve §2b).
- **The two collision cases.** Two same-size images differ at k=0, so the trie cuts the match at the image start (gate G9). The same image hits all the way through.
- **Failure is loud, not silent.** If a tuple ever reaches `mx.array` it raises. It cannot be read as a wrong id.
- **No separate image hash in the key.** Positions come from the key itself: `positions(key)` is pure. So `rope_deltas` is recomputed from key plus store `grid_thw` instead of held as module state. This avoids mlx-vlm's `_rope_deltas`-on-module hazard (`qwen3_5/language.py:1418-1419`, `generation.py:2125-2128`).

**Store key.** `(model_key, sha, proc_hash)`, LRU, byte-bounded. It is per image, not per image-set. That fixes exo's whole-list key (`exo vision.py:749-754`), where adding a second image re-encoded everything.

### 1.3 Data flow: turn N of a conversation containing an image

1. **HTTP thread, `seam._post` (seam.py:220).**
   - No image parts → fall straight through, byte-identical to today's text path.
   - Image parts present:
     - `request.prepare(body, family)` decodes each image and hashes its pixels (`images.py`).
     - A store miss runs `preprocess` + `encode` once and puts the result in the store. A hit skips both. This makes the tower-call counter the G6 metric.
     - Each part is replaced by `placeholder_text` (one pad), which gets past mlx-lm's `process_message_content` (`server.py:118-144`).
     - The `[ImageRef]` list is attached to the request object carried to the generator thread (precedent: `seam.py:614-619` reads `request[2]`). The exact carrier is *not verified* (`server.py` ~1100+ unread).
   - The model is not vision-capable → return 400.
2. **Generator thread, wrapped `ResponseGenerator._tokenize`.** It calls the real one, runs `family.expand_ids` on the result, then `key.expand`, and returns the **key** as `prompt`. From here, `fetch_nearest_cache` (`server.py:753/965`), segment trim, `insert_segments` and `insert_cache` all operate on keys.
3. **Prefix hit.** Turn N's key = turn N-1's `prompt + generated` + new text, so this is the "shorter" hit, which is the only kind that works for the ArraysCache/GatedDeltaNet families (knurlogic-serve §2c). The image span sits inside the hit, so no image work happens.
4. **Batch admit, `MTPBatchGenerator._admit_one` (:163).**
   - `ids = to_ids(key)`.
   - If the uncached span `[hit:]` contains sentinels, compute `family.embed(model, ids[hit:], key[hit:], store)`. This embeds only the suffix (fixing exo's whole-prompt re-embed, `vision.py:659-693`). The feature index is global, `cumsum` over the full key.
   - Returns `input_embeddings` plus family extras: Qwen `position_ids` [3,1,L] + `rope_delta`; gemma `per_layer_inputs` from ids with pads zeroed + image-block mask; GLM nothing.
   - Calls `batch_loop.admit(..., embeds=..., extras=...)`.
   - `st["fed"]` stays the key, so `all_tokens` at :255 remains a valid key.
5. **Decode.**
   - Qwen: position = `offset + rope_delta_row`, a per-row delta held on the row state, not the module.
   - GLM (NoPE MLA) and gemma (1D RoPE): position = offset.
6. **Single path, `seam._stream` (:621).** Same key → ids/embeds translation, passing `input_embeddings` to `stream_generate` (`mlx_lm/generate.py:321-365`).

**Turn 5 cost.** The tower runs 0 times. Prefill covers turn 5's new tokens plus template overhead only, which is the G8 / vision_gate pass condition.

### 1.4 How drafting interacts

- **Phase A (ships first).**
  - A row whose **uncached span contains an image** gets `RowParams(drafts=False)`. It rides along and replays, matching `batch_loop.py:28-30`.
  - Its stored entry is trunk-only (`_entry` :194).
  - On turn N+1 the suffix is text, but `split_pool_entry`/`hit` (:170-174) finds no head cache. That row therefore does not draft either: a re-prefill would violate the turn-5 goal. **So vision conversations do not draft in Phase A.**
  - Text-only conversations are unchanged.
- **Phase B (P4, feasible but gated on measurement).**
  - Seed the head from merged embeddings rather than `embed_tokens(placeholder)`. That is the actual defect exo has, at `heads/qwen35.py:285`, `glm5.py:230`, `qwen4_exp.py:211` and `seed.py:56`.
  - Then image rows store trunk+head entries and later text turns draft.
  - **Open risk:** the Qwen head's own attention uses 1D positions. After an image the trunk positions are `offset + rope_delta`, so the head must receive the same delta. *Not verified* whether the head's rope is exposed.
  - gemma4 may have no head (*not verified*). For gemma, drafting is moot.
  - **Plain answer:** drafting over the text that follows an image is plausible for Qwen and GLM. For GLM it is likely easiest (NoPE). It ships only if G10 token-identity passes and acceptance measured on a real model is ≥ 80% of the text-only rate. Otherwise Phase A stands.

### 1.5 Batching engine

- `_factory` (seam.py:673) must always build `MTPBatchGenerator` once vision is loaded, with `head=None` allowed. Plain mlx-lm `BatchGenerator` would feed tuples into `mx.array` (knurlogic-serve §2b).
- Mixed rows: decode already does per-row offsets. Qwen additionally needs a per-row `rope_delta` vector added in the attention's position computation (P1 owns this).

## 2. Work packages

Worktrees: one branch per package. File ownership is exclusive. "Reads" means read-only.

### P0: Contracts, key codec, image store, loadlock, fixtures (blocks everything; about 1 day)

- **Owns (creates):**
  - `engine/vision/{__init__,_base,key,store,images,scatter,registry}.py`
  - `machine/loadlock.py`
  - `tests/fixtures_vision.py` (tiny configs for all five families per test-plan §1; special ids remapped into vocab 512, option (a))
  - `tests/test_vision_key.py`, `tests/test_image_store.py`, `tests/test_loadlock.py`
- **Registry table (fixed strings):**
  - `qwen3_5`, `qwen3_5_moe`, `qwen4_exp` → `knurlogic.engine.vision.qwen:build`
  - `gemma4` → `...vision.gemma4:build`
  - `glm5_next` → `...vision.glm5:build`
  - Missing module → `build` returns None (capability off), so packages land independently.
- **Reads:** `mlx_vlm/vision_cache.py`, `mlx_vlm/base.py`, `exo …/mlx/vision.py:696-773`, `mlx_lm/models/cache.py:1536-1690`.
- **`scatter.embeds_for`:**
  - Generic `masked_scatter` (cumsum gather, as `gemma4.py:13-20`) with a global feature index.
  - The `Family.embed` implementations call it.
- **Tests (all tiny, no model):**
  - expand/to_ids round-trip
  - same-size different-sha keys diverge at the image start in a real `LRUPromptCache` (G9 at key level)
  - key length == id length
  - the store LRU evicts by bytes
  - per-image hits when a second image is added
  - `images.py` hash is stable across PNG re-encode of the same pixels
  - loadlock: Busy from a subprocess, release after `kill -9`, no mlx import in `machine/`
- **Done gate:** `pytest tests/test_vision_key.py tests/test_image_store.py tests/test_loadlock.py` green; `python -c "import knurlogic.machine.loadlock"` with mlx absent from `sys.modules`; contracts merged to main before P1–P5 branch.

### P1: Qwen vision (qwen3_5, qwen3_5_moe, qwen4_exp) plus MRoPE in the trunks

- **Owns:**
  - `engine/vision/qwen/*`
  - edits to `engine/architectures/qwen3_5.py`, `qwen3_5_moe.py`, `qwen4_exp.py`:
    - `position_ids: [3,B,L] | None` and `rope_delta: [B] | None` params threaded to attention
    - interleaved MRoPE from `rope_utils.py:498-539` (non-kernel path), with `mrope_section [11,11,10]`
    - None keeps today's 1D path exactly
  - stop dropping vision keys at `qwen3_5.py:430`; sidecar loading moves to `Family.load_weights`
  - `tests/test_vision_qwen.py`
- **Reads:** `mlx_vlm/models/qwen3_vl/{vision,config,processing_qwen3_vl}.py`, `qwen3_5/{qwen3_5,language}.py`, `engine/vision/*` (P0).
- **Contract details:**
  - `load_weights` accepts both the HF `model.visual.*` and MLX `vision_tower.*` sidecars (`sanitize_key` `qwen3_5.py:16-25`, plus the transpose `(0,2,3,4,1)` in `vision.py:429-447` gated by `check_array_shape`). It asserts 333 tensors bound.
  - `positions(key)` reimplements `get_rope_index` as a pure function of key + grid. Its inputs are `vision_start_token_id` from config (248053, not the default 248045, `config.py:120`) and `rope_delta = max+1-len` (:1882-1885).
  - qwen4_exp:
    - apply MRoPE to both the main attention (`qwen4_exp.py:371`) and the QSA indexer rope (:267-271) (*not verified* that the indexer needs it; test decides)
    - the n-gram PLE keeps raw placeholder ids (vlm-families §4, unverified vs reference; G3 decides)
- **Tests:** G1–G5 for all three families against mlx-vlm 0.6.17 (`importorskip`, interpreter `/opt/anaconda3/envs/exo/bin/python`). G5 proves text is unchanged when `position_ids=None`. `test_batch_drafting.py` must still pass untouched.
- **Done gate:** G1 atol 1e-5, G2 exact grid/count, G3 exact positions, G4 40 greedy tokens identical, G5 identical, existing 166 tests green.

### P2: gemma4 vision (e4b and 26b)

- **Owns:**
  - `engine/vision/gemma4/*`
  - edits to `engine/architectures/gemma4_text.py`:
    - an `mm_mask` / `mm_token_type_ids` kwarg adds the bidirectional image-block overlay from mlx-vlm `gemma4/language.py:486-515`
    - `per_layer_inputs` override from zeroed ids (`gemma4.py:92-105`)
  - `tests/test_vision_gemma4.py`
- **Reads:** mlx-vlm `gemma4/{vision,gemma4,processing_gemma4,config,language}.py`. Skip `audio*`.
- **Contract details:**
  - `embed` returns `{"input_embeddings": embed*embed_scale with feats scattered, "per_layer_inputs": ..., "mm_mask": ...}`.
  - `load_weights`:
    - 356-tensor sidecar for 26b
    - 661 main-shard keys for e4b, including quantized `embed_vision.embedding_projection` (weight/scales/biases → `nn.QuantizedLinear`)
    - clip params kept for e4b per `use_clipped_linears` (`gemma4.py:224+`)
  - **Invariant for P4:** `Family.chunk_boundaries(key) -> list[int]` returns image-block spans that must not be split by a prefill chunk.
- **Tests:** G1–G5. Plus: an image block straddling a 16-token chunk size still yields identical tokens (proves chunk alignment). RotatingKVCache prefix reuse inside the window.
- **Done gate:** same thresholds as P1.

### P3: glm5_next vision (runs parallel with P1 and P2)

- **Owns:**
  - `engine/vision/glm5/*`
  - edits to `engine/architectures/glm5_next/{glm5_next.py,processing.py,PROVENANCE.md,THIRD-PARTY.md is shared → append only via P3}`:
    - remove the transformers bases
    - inline `install_auto_processor_patch` / `load_chat_template` / `_flatten_images`
    - remap `vision_model.` → `vision_tower.`: the released shards hold 347 `vision_model.*` keys in `model-00019` (vlm-families §5)
  - `tests/test_vision_glm5.py`
- **Reads:** mlx-vlm 0.6.17 `glm5_next/*` (the 0.7.1 tree at `/tmp/knur_clean` is empty, so the reference for G1 is 0.6.17's tower, or the vendored 0.7.1 itself for G3/G4).
- **Contract:** `positions()` returns `(None, 0)` (NoPE). Patch 14. The processor token budget is 16..8000.
- **Tests:** G1 (tower vs 0.6.17; `_limited_swiglu` differs in 0.7.1, so compare with `swiglu_limit` large or document the delta), G2–G5, and a key-remap test: the 347 names from the index.json fixture list all bind.
- **Done gate:** as P1, with G1 relaxed if 0.6.17 and 0.7.1 differ by design; that must be written down.

### P4: Serve path (key-carrying requests, embeds admit, drafting policy); depends on P0, integrates P1–P3 via the registry only

- **Owns:**
  - `engine/vision/request.py`
  - `engine/seam.py`: `_post` intercept; `_tokenize` wrap; `_stream` key→ids/embeds; `_factory` always `MTPBatchGenerator` with vision; `load`/`_pinned` builds `Family` and loads the tower; `served_vision() -> VisionSpec | None`
  - `engine/mtp/batch_generator.py`: `_admit_one` translation, `drafts=False` for image-in-suffix rows, `head=None`, per-row `rope_delta`
  - `engine/mtp/batch_loop.py`: `admit(..., embeds=None, extras=None, chunk_boundaries=None)`, slicing per chunk, chunk edges snapped to boundaries
  - Phase-B drafting in the same package, to avoid a batch_loop conflict: `engine/mtp/seed.py` and `engine/mtp/heads/*` gain an `e_override` kwarg; `seed_head(..., embeds=)`. Enabled behind `KNURLOGIC_VISION_DRAFT=1` until measured.
  - `tests/test_image_cache.py` (G6–G9 end-to-end through the real `LRUPromptCache` + `MTPBatchGenerator`), `tests/test_vision_batch.py` (G10, G11)
- **Reads:** `mlx_lm/server.py` (`_tokenize` :536-553, `_generate` :688-777, :874-1019, `_next_request` ~1100+), exo `mtp_batch_generate.py:223-373`, `generate.py:85-145`.
- **Interface consumed:** only `vision.build()` / `Family` / `key` / `store`. P4 develops against a **stub family** in `tests/fixtures_vision.py`: identity tower, `positions` returning `(None, 0)`. It never imports P1–P3 directly.
- **Tests:**
  - G6: tower counter == 1 over 2 turns
  - G7: warm == cold tokens, last-prefill logits atol 1e-4
  - G8: hit_len == len(turn1 key)
  - G9: different image → miss + different output
  - G10: image prompt MTP vs plain identical (Phase A: drafts off; Phase B flag on)
  - G11: 3 mixed rows each == solo
  - Text path: a request with no image produces byte-identical HTTP output vs main (golden test using the existing `test_batch_drafting` tiny model)
- **Done gate:** all green against the stub. After merge, re-run with P1–P3 real families (orchestrator).

### P5: Interfaces, chat UI, real-model gate tool; depends on P0 (`VisionSpec`) and P4's `served_vision()` signature only

- **Owns:**
  - `interfaces/web.py`:
    - `/v1/messages` Anthropic image blocks → OpenAI parts before handoff (:351)
    - `vision` on `models_document` (:58) / `loaded_document` (:98) / status
  - `interfaces/web/index.html`: Chat | Bench tabs, the chat spec from exo-chat-ui §3
    - storage: `kn.chats` localStorage plus IndexedDB `kn-img` keyed by sha256 of the final data URL
    - attach pipeline: downscale once, re-encode once, byte-identical resend
    - SSE parser port with `keepalive P/T`
    - `<think>` split, prefill bar, lightbox
    - sanitized markdown (option B in-page, or marked + DOMPurify from jsdelivr)
    - per-turn "new of total" meter from `cached_tokens` (`server.py:1346`)
    - Apache-2.0 header on the ported pieces
  - `interfaces/mcp.py`:
    - `vision` field in `models` / `state` / `fit`
    - a `ready()` blocker from `loadlock.holder()` (:72-133)
    - `load` (:316) takes `model_load`
  - `interfaces/serve.py`: store size setting in `_status` :127
  - `tuning/resolve.py`: tower weights plus store bytes in the working set
  - `tools/vision_gate.py` (test-plan §3), `tests/data/` fixture generators (red square, "42"), `tests/test_web_vision.py`, `tests/test_mcp.py` additions
- **Reads:** `engine/seam.py` (for `served_vision`), exo dashboard files.
- **Tests (tiny, no model):**
  - web documents expose `vision` with the model mocked
  - Anthropic→OpenAI block translation
  - MCP `ready()` shows the lock holder
  - `index.html` JS unit checks are optional (headless not required)
- **Done gate:** tests green. The page loads in the Browser pane against a stubbed `/status.json` (no model) showing the attach button disabled/enabled by `vision`.

### Dependency order

```
P0 ──┬─> P1 (Qwen) ─┐
     ├─> P2 (gemma) ├─> orchestrator integration (registry resolves real families) ─> real gates
     ├─> P3 (GLM) ──┤
     ├─> P4 (serve, stub family) ──┘
     └─> P5 (interfaces; needs only VisionSpec + served_vision signature, fixed in P0 docs)
```

- **Shared-file conflicts:**
  - `architectures/THIRD-PARTY.md` / `PROVENANCE.md`: P1/P2/P3 each append their own section; the orchestrator resolves trivially at merge. Alternatively, each writes `engine/vision/<fam>/PROVENANCE.md` and P0 pre-creates the pointer.
  - Only P4 touches `seam.py` and `mtp/*`.
  - Only P1 touches the Qwen architecture files.

## 3. Verification

**Tiny random fixtures.** Safe to run in parallel in any worktree, float32 and seed 0, each under 1 GB, no loadlock:
- P0: key, store, loadlock tests
- P1/P2/P3: G1–G5 per family (mlx-vlm 0.6.17 reference via the exo env interpreter; `importorskip` elsewhere)
- P4: G6–G11 plus the text golden test
- P5: web/mcp tests
- Always: the existing `tests/*` (166)

**Real model.** Serialized by the orchestrator at the end, one at a time, each behind `loadlock.model_load` after MCP `ready()`, `try/finally` unload, and no exo POST:
1. `gemma-4-e4b-it-VQ-PLE` (7.4G; main-shard vision, quantized embedder)
2. `Qwen3.8-27B-VQ-3.9bpw` (12G; MLX-key sidecar)
3. `Qwen3.6-35B-A3B-VQ-3.4bpw` (14G; MoE)
4. `gemma-4-26b-a4b-it-VQ-6.2bpw` (19G; 356-tensor sidecar)
5. `Qwen3.8-Flash-Next-VQ-2.1bpw` (47G): only if `ready()` and fit ≥ 55G
6. GLM-5.3 (108G) and Qwen3.5-397B (112G, HF-key sidecar): **tower-only** local check (load only the vision keys via safetensors filter, compare to the mlx-vlm tower, one tower at a time). The full gate waits for cluster serving and needs the maintainer's go-ahead.

Per-artifact pass conditions:
- vision tensor count
- "Paris" text answer
- "red" image answer and the "42" OCR answer
- 5-turn run: tower calls == 1; `prompt − cached ≤ new + 32` on turns 2–5; ttft(t5) ≤ 1.5× the text-only equivalent and ≪ cold; turn 5 recalls "red"
- memory back to baseline ±1 GB after unload
- Phase-B drafting acceptance measured on 27B and 35B only

## 4. Risks and open questions, ranked

1. **MRoPE in the mlx-lm-derived Qwen trunks (P1).** The mlx-lm attention is 1D (`qwen3_5.py:19` via Qwen3NextAttention). If it is wrong, grounding degrades silently and text still passes. qwen4_exp's QSA indexer rope (:267-271) is an extra unknown. Mitigation: G3 exact positions plus G4 against mlx-vlm. Per-row `rope_delta` in batched decode is new code with no reference.
2. **Carrying `ImageRef` from the HTTP thread to the generator thread** (`_next_request` / `CompletionRequest`, not read). If mlx-lm offers no clean carrier, P4 falls back to a side table keyed by `id(body)` or an injected field. This is fragile under mlx-lm upgrades, so pin mlx-lm.
3. **Sentinels leaking into mlx-lm code that assumes ints** (stats, `len`, detokenizer, `cache_key.append` at `server.py:1006`). This is not an audit of every use of `prompt`. Mitigation: the G6–G9 end-to-end run through the real server objects. Fallback: encode sentinels as negative ints, `-(hash31(sha,k))`, if tuples break something.
4. **Recurrent/SSM state.** GatedDeltaNet (Qwen, GLM) and the qwen4_exp n-gram ArraysCache can only take "shorter/exact" hits. An edited-and-regenerated turn misses back to the nearest snapshot. That is acceptable, but the UI's "edit" will show full re-prefill.
5. **gemma bidirectional image blocks.** Chunking must snap to `chunk_boundaries` (`language.py:486-515`, active only when L>1). A prefix hit can never end mid-image with the sentinel key, but the chunk snap is required. RotatingKVCache bounds reuse on sliding layers (*not verified*).
6. **Drafting on image rows (Phase B).** The head's positional scheme after MRoPE is unverified, as is whether gemma has a head. If it is not reached, vision conversations run without drafting. Correctness is unaffected; speed on image chats is lower.
7. **GLM reference gap.** The 0.7.1 source is only knurlogic's vendored copy, and it differs from 0.6.17 (`_limited_swiglu`, merge rewrite). G1 cannot be exact against 0.7.1. The full real gate is cluster-only.
8. **Store memory accounting.** Features for an 8000-token GLM image at hidden 4096 in bf16 are about 65 MB each. The store must be counted in `tuning/resolve.py` or it will OOM the shared host.
9. **Wire determinism from clients.** Reuse needs byte-stable images and templates. The server hashes pixels, not base64, so re-encoded bytes still hit. But templates that re-render `reasoning_content` differently break the prefix (*not verified* per family).
10. **Request size.** No body cap is enforced (`server.py:1123-1132`). The chat UI downscales, but agents may post huge images. Add a max-pixels clamp in `request.py`.

Key paths:
- `~/Documents/AgenicAI/knurlogic/src/knurlogic/engine/seam.py`
- `~/Documents/AgenicAI/knurlogic/src/knurlogic/engine/mtp/batch_loop.py`
- `~/Documents/AgenicAI/knurlogic/src/knurlogic/engine/mtp/batch_generator.py`
- `~/Documents/AgenicAI/knurlogic/src/knurlogic/engine/architectures/`
- `~/Documents/AgenicAI/knurlogic/tests/test_batch_drafting.py`
- `/opt/anaconda3/envs/exo/lib/python3.13/site-packages/mlx_vlm/models/`