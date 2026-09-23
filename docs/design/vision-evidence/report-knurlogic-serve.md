# Knurlogic serving path: where vision plugs in and how the prompt cache must change

## 0. Headline findings

1. **mlx-lm's server rejects image parts before they reach the model.** In `mlx_lm/server.py:118-144`, `process_message_content` joins only the `type=="text"` fragments. If there are fewer text fragments than parts, it raises `ValueError("Only 'text' content type is supported.")` (line 141). It runs inside `_tokenize` at `server.py:536`, which is called from both the batch path (`server.py:738`) and the single path (`server.py:933`). A request with an `image_url` part therefore fails at tokenize time. No image data survives into any mlx-lm path.
2. **The prompt cache is a token trie.** `LRUPromptCache` is at `mlx_lm/models/cache.py:1623`, and `PromptTrie.add/get` is at `cache.py:1536-1552`. The trie walks `current[tok]` dicts, so any hashable element can be a key. Nothing in the trie itself requires the key elements to be ints. Callers are `fetch_nearest_cache(model_key, prompt)` at `server.py:753` (batch) and `server.py:965` (single), and `insert_cache` at `server.py:874`, `903` and `1019`.
3. **In the batch path, the cache key and the model's input are the same list.** `insert_segments(segments=…, all_tokens=[prompt[:prompt_cache_count]])` is at `server.py:769-777`. Knurlogic's `_admit_one` (`engine/mtp/batch_generator.py:163-192`) then runs `mx.array(prefix + rest)` straight into `admit`. On completion, `_next` stores `all_tokens = st["fed"]` (`batch_generator.py:255`), and that list goes back to the server as the cache key (`server.py:873-876`, not re-read in detail). So a key built with hash sentinels would be fed to the model as ids unless the model's input is separated from the key.
4. **Knurlogic already has a hook meant for vision prefill, but nothing passes it yet.** `batch_loop.admit(..., prefill_ctx=...)` (`engine/mtp/batch_loop.py:148-200`) documents that "a vision request's embedding patch goes there" (docstring around line 185). The chunked forwards run inside it, and the final single-token forward runs outside it. `MTPBatchGenerator._admit_one` does not pass `prefill_ctx` (`batch_generator.py:176-181`). The only caller that does is exo glue: `speculative.py:381` (`_pipeline_prefill_ctx`, a pipeline all-gather toggle, not vision).
5. **The policy that image requests do not draft already exists, as prose and exo glue.** The `batch_loop.py:28-30` docstring says an image row "rides along", replays, and never drafts. `speculative.py:144-153` refuses a head when `has_vision`. Neither is wired in knurlogic's own serving path, because no vision request reaches it today.
6. **The trunk architectures already accept embeddings.** `qwen3_5.py:271-276` (`input_embeddings` replaces `embed_tokens`) and `:332-334`, `:417-420`. `qwen3_5.py:430` strips `vision_tower`/`model.visual` keys at sanitize, so the vision weights are currently dropped. I did not check the other four families; not verified.
7. **M-RoPE.** `qwen3_5.py` has `"mrope_section": [11, 11, 10]` (grep hit around line 58). Image tokens need 3D position ids, and whether the text trunk's forward computes or accepts them was not verified. That is a correctness risk for the Qwen families: a text-only 1D RoPE over image tokens degrades image grounding silently.
8. **mlx-vlm 0.7.1 at `/tmp/knur_clean/.../mlx_vlm` exists** (it lists `__pycache__`, `evals`, `generate`). I did not inspect it further.

## 1. The serving path as it stands

```
interfaces/serve.py:run() -> engine/seam.py:serve() (153)
  ├─ pins ModelProvider.load -> _pinned (185-197): load_draft_head + install_drafting
  ├─ ModelProvider.__init__ -> captures provider in _SERVED (199-205)
  ├─ routes: APIHandler.do_GET/do_POST overridden (216-281); knurlogic routes
  │    first, else _real_post -> mlx-lm handler
  │    (web.py:309-351 builds routes; "POST /v1/messages" is a raw handler, web.py:351)
  └─ srv.main()
mlx-lm ResponseGenerator._generate (server.py:688+)
  ├─ batchable: _tokenize -> fetch_nearest_cache(model_key, prompt) -> segment trim
  │    -> batch_generator.insert_segments(...)  (738-777)
  │    BatchGenerator = seam._install_batch_drafting._factory (657-689)
  │      -> MTPBatchGenerator (batch_generator.py:135) -> _admit_one -> batch_loop.admit
  └─ single: _serve_single (wrapped by seam.install_drafting 614-619)
       -> fetch_nearest_cache (965) -> stream_generate (swapped to seam._stream 621-652
       -> mtp_stream_generate) -> insert_cache(prompt+generated) (1019)
```

- `_is_batchable` is `model_provider.is_batchable and args.seed is None` (`server.py:685-686`).
- `interfaces/mcp.py` has no chat or generation tool. Its tools are ready, fit, state, models, settings, drafting, load, place, unplace, unload and deps (`mcp.py:72-433`). Agents reach the model through the HTTP OpenAI surface, so MCP needs no inference change. At most it needs a `vision: bool` field in `models`/`fit`/`state` (not verified whether it has one now).

## 2. Proposal

### 2a. Where to intercept

Intercept in `seam.serve`'s `_post` (`seam.py:220`), before `_real_post`, for `POST /v1/chat/completions` and `/v1/messages`:

- Parse the body. If no message has an `image_url`/`image` part (or an Anthropic `{"type":"image","source":…}` block), fall through untouched. The text path stays byte-identical.
- If there are images and the served artifact has no vision tower, return a 400 that names the reason.
- Otherwise, call `vision.prepare(body)`. It returns:
  - the rewritten `messages`, with each image part replaced by the family's image placeholder text (Qwen `<|vision_start|><|image_pad|>…<|vision_end|>`, Gemma `<start_of_image>`, GLM equivalent);
  - an ordered list of `ImageRef(sha256, n_tokens, grid_thw)`.
  
  The images themselves go into an **encoded-image store**, `engine/vision/store.py`. It is an LRU keyed by `(model_key, sha256(decoded bytes), processor_config_hash)` and holds the vision-tower output `[n_tokens, hidden]` plus the grid and position ids. The image is encoded once, on first sight. Later turns re-send the same base64, get the same hash and hit the store.
- Put the parts-free body on a thread-local, or stash it by request id in a side table keyed by an injected field, and call `_real_post` with it. `process_message_content` then passes.

That covers mlx-lm's parse. It does not yet cover the pad-token expansion. `_tokenize` runs `apply_chat_template(tokenize=True)` (`server.py:553`), and the template emits one `<|image_pad|>` per image, where the model needs `n_tokens` of them. Two options:

- **(A)** Wrap `ResponseGenerator._tokenize`, in the same place as `install_drafting` wraps `_serve_single`, so that after the real call it expands each single pad into `n_tokens` pads using the `ImageRef` list from the thread-local. `_tokenize` runs on the generator thread (`server.py:738`/`933`), not the HTTP thread, so the side data has to ride on `request`/`args`. For example, wrap `_next_request`, or attach it to the `CompletionRequest` object. The `_serve_single` wrapper already reads `request[2]` (`seam.py:614-619`), which is the precedent. Recommended.
- **(B)** Text-level expansion: put `n_tokens` pad strings in the message content before templating. This is simpler, but it breaks on templates that escape content. Not recommended.

### 2b. Two parallel sequences: model ids and cache key

After (A), each request carries two sequences of equal length:

- `ids`: real token ids, with pad ids at the image spans. This is what the model sees.
- `key`: the same list, except that each image span `[s, s+n)` is replaced by `n` copies of a hashable sentinel `("img", sha256, k)` for k in 0..n-1. A single sentinel per image would not work, because the key length must equal the KV length for `fetch_nearest_cache`'s prefix arithmetic and the segment trimming at `server.py:756-765`.

Why the key must change: every image's span is the identical `<|image_pad|>` run. With id-only keys, two different images of the same size collide. The cache would hand back KV computed from the wrong image, and the output would silently be wrong. The sentinel makes the trie diverge at the first image token. Same image, same key, cache hit. Different image, the cache hits up to the image start and prefills from there.

The approach:

- The **server sees `key`**. The wrapped `_tokenize` returns `key` as `prompt` and `segments`, so `fetch_nearest_cache`, the segment trim, `insert_segments`, `all_tokens` and `insert_cache` all operate on keys. The trie accepts tuples (`cache.py:1543`).
- **`MTPBatchGenerator._admit_one` translates** `prompt` (a key) back into ids. Sentinel → pad id. From the sentinels it also builds the embedding tensor for the uncached span: embed the ids, then scatter the stored vision features into the image positions. It admits with `input_embeddings` for chunks that overlap an image. This goes through `prefill_ctx`, or better, `admit` gains an `embeds=` argument so that `admit` does `model(chunk, cache=cache, input_embeddings=embeds[:, i:end])`. That is cleaner than monkeypatching `embed_tokens`. `st["fed"]` stays the key list, so what `_next` returns as `all_tokens` (`batch_generator.py:255`) is already a correct key.
- **Plain mlx-lm `BatchGenerator`** (no head, or `--no-draft`): it would feed tuples into `mx.array` and crash. Either route every image request through `MTPBatchGenerator` with drafting off per row (`RowParams(drafts=False)`), which requires the factory at `seam.py:673` to always build ours when vision is present, or wrap real `BatchGenerator._process_prompts` too (not verified where). Recommendation: have `MTPBatchGenerator` accept `head=None` and use it for every model, which makes it the one batch engine.
- **Single path** (`_serve_single` → `seam._stream`, `seam.py:621`): `prompt` there is also the key. `_stream` translates the key to ids plus embeddings, and passes `input_embeddings` to `real_stream`. `mlx_lm.generate.stream_generate` supports that (`generate.py:321-365`, with lengths checked equal). Drafting is skipped for image requests. `cache_key.append(gen.token)` (`server.py:1006`) keeps key semantics.

### 2c. Drafting coexistence

- A row whose uncached span contains an image gets `drafts=False`. It then behaves as `batch_loop.py:28-30` already describes: it rides along, replays, and takes the trunk's tokens.
- Better, as a later option: the head is seeded from the captured hidden states `h`, which are correct post-vision activations regardless of input. So drafting over the text after the image may be fine, and the exo concern was the `embed_tokens` patch interacting with the capture wrapper (`speculative.py:9-10`). With the explicit `input_embeddings=` route, that concern may disappear. Not verified; measure acceptance first.
- **Head cache on prefix reuse.** `split_pool_entry` and `hit` (`batch_generator.py:170-174`) already fall back to a fresh prefill when the head cache is misaligned. An entry stored by a non-drafting image row has no head cache (`_entry`, `batch_generator.py:194-209`), so turn N+1 either drafts from a full re-prefill or doesn't draft. Recommendation: vision conversations don't draft (simplest), so the cached trunk KV is always reusable. That meets the goal of turn 5 paying only for turn 5's tokens.
- **Hybrid/recurrent caches** (Qwen3.5 linear attention): `fetch_nearest_cache`'s "longer" branch trims (`cache.py`, around 1680-1690). Recurrent state cannot be trimmed, so only exact or shorter prefix hits work. That is already true for text and is unchanged here (not verified in detail). A chat's turn N+1 extends turn N's `prompt + generated`, which is the "shorter" hit, so it works.

### 2d. M-RoPE (Qwen families)

The prefill of an image span needs position ids from the grid, not from `cache.offset`. Text after the image resumes at `max(pos)+1`, which is not the raw index. So `rope_deltas` must be stored alongside the cache entry, or be recomputable from the key. It is recomputable: the key contains the sentinels, and the store has `grid_thw`. The trunk forward in `engine/architectures/qwen3_5.py`/`qwen3_5_moe.py`/`qwen4_exp.py` must accept `position_ids`, and decode must add `rope_deltas`. This is the largest model-side change, and I have not verified what the vendored trunks do now. Gemma4 and GLM5 positions: not verified.

## 3. Files that would change

| File | Change |
|---|---|
| `src/knurlogic/engine/vision/__init__.py` (new) | `prepare(body, family) -> (messages, [ImageRef])`; image decode (base64/data-URL/file path; no remote fetch by default) |
| `engine/vision/store.py` (new) | Encoded-image LRU keyed by `(model_key, sha256, proc_hash)`, byte-bounded, counted against `prompt_cache_bytes` |
| `engine/vision/key.py` (new) | `expand(ids, refs) -> key`, `to_ids(key)`, `embeds_for(key, start, end, model)`, `positions_for(key)` |
| `engine/vision/towers/{qwen3_5,gemma4,glm5_next}.py` (new) | Vendored vision towers and processors from mlx-vlm (qwen3_5_moe and qwen4_exp share the Qwen tower, not verified) |
| `engine/architectures/qwen3_5.py`, `qwen3_5_moe.py`, `qwen4_exp.py`, `glm5_next/*`, gemma4 arch | Stop dropping vision keys at sanitize (`qwen3_5.py:430`), or load the `model-vision-graft.safetensors` sidecar separately; accept `position_ids`/rope deltas |
| `engine/seam.py` | `_post` (220): detect images and call `vision.prepare`; wrap `ResponseGenerator._tokenize` to expand pads and return the key; `_stream` (621): key → ids/embeds, no drafting for image rows; `_factory` (673): always `MTPBatchGenerator` when the model has vision; `load()`/`_pinned`: load the tower and processor next to the trunk |
| `engine/mtp/batch_generator.py` | `_admit_one` (163): key → ids plus embeds, `drafts=False` for image rows, pass `embeds`/positions; allow `head=None` |
| `engine/mtp/batch_loop.py` | `admit` (148): `embeds=`/`position_ids=` args, per-chunk slicing; `MTPBatch` plain step unchanged |
| `interfaces/web.py` | `/v1/messages` raw handler (351): translate Anthropic image blocks; `models_document` (58)/`loaded_document` (98) expose `vision: true`; new chat UI page route (exo-dashboard-style) with image upload |
| `interfaces/serve.py` | Wire the vision store size into settings and status (`_status` 127) |
| `interfaces/mcp.py` | `models`/`state`/`fit` report vision capability; no inference tool |
| `tuning/resolve.py` | Budget for tower weights and the image-embedding store in the working set |
| tests (new) | Key collision (two same-size images diverge), turn-N reuse (`prompt_cache_count` covers the image), text path unchanged, image row in a mixed drafting batch |

## 4. Open or unverified items

- How `_next_request` and `CompletionRequest` can carry side data to the generator thread (`server.py` around 1100+). Not read.
- Whether the real `BatchGenerator` can be kept for image requests at all. The recommendation assumes it cannot.
- Positional schemes and the placement of image-token ids in gemma4 and glm5_next.
- Whether the MTP head drafts correctly after vision prefill with explicit `input_embeddings`.