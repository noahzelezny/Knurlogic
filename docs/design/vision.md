# Vision

The released families -- qwen3_5, qwen3_5_moe, qwen4_exp (Flash-Next),
glm5_next (GLM-5.3), gemma4 -- serve text and images from
`pip install knurlogic`, with code knurlogic owns rather than mlx-vlm, and
with **images as real context**: an image is encoded once per conversation,
the prefix cache survives it, and it is never prefilled twice.

Every released rung (one published quantization level of a model) carries
`vision_config` and its vision weights: a 333-tensor
`model-vision-graft.safetensors` sidecar for the Qwen families and gemma
26b; inside the main shards for GLM (347 keys, `vision_model.*`) and gemma
e4b (661). The towers are vendored from mlx-vlm 0.6.17 (MIT), which
implements all five families; mlx-vlm is not a dependency.

Interfaces and data shapes: [vision-contracts.md](vision-contracts.md).

## knurlogic owns the VQ runtime

Every released rung ships its own `model.py`, and each one deliberately
builds a text-only model when mlx-lm loads it. A vision tower cannot be
grafted onto that, so knurlogic carries one VQ runtime of its own
(`engine/vq/`), vendored verbatim and pinned by commit and digest
(`engine/vq/PROVENANCE.md`). It is family-aware and vision-aware; the
bundled `model.py` stays in the Hub uploads for plain mlx-lm users.

* **Per-rung differences are settings, not code.** Within a family the
  published runtimes differ mostly by the defaults of a few numerics flags.
  Each rung's flags are read from its *published* `model.py` and recorded
  in `engine/vq/rungs.json`; knurlogic reproduces what is shipped, not what
  was intended. The table and its findings: [vq-rung-knobs.md](vq-rung-knobs.md).
* **A rung's numerics come from the rung.** The bf16-I/O flags are
  numerics-active (up to +0.97% perplexity), so `tuning/resolve.numerics_for`
  applies the rung's declared knobs, and a runtime profile applies only when
  a person asks for it.
* **The identity gate** (`tools/vq_gate.py`): knurlogic's runtime and the
  rung's bundled runtime, each in its own process, give the same logits on
  the same prompt (atol 1e-5) and 40 identical greedy tokens. A rung that
  has not passed loads the `model.py` it ships and is listed as such, never
  silently served on the new runtime. For an older-generation bundle,
  reproducing its flags on the new runtime is a runtime change, not a
  no-op, which is exactly what the gate checks.

## The stack is pinned

Exact versions of mlx and mlx-lm, and the VQ runtime's commit, are recorded
in `pyproject.toml`, with a test that fails on drift (versions plus a digest
of mlx-lm's `server.py`). mlx-vlm reference outputs for the identity gates
are generated once as `.npz` goldens under an interpreter with mlx-vlm
0.6.17 and committed, so G1-G4 run anywhere without mlx-vlm installed.

## All image work on the scheduler thread

Generation runs on one thread that owns the MLX stream; image work runs
there too, never on an HTTP thread (two threads on one GPU are two
uncoordinated allocations). At tokenize (`engine/vision/request.py`): pull
the image parts out, decode and hash, look up the image store, encode on a
miss, replace each part with the family's placeholder, tokenize, expand
placeholders to the image's token count, and return the cache key as the
prompt. The HTTP layer only refuses images sent to a model without vision
(400).

## The cache key

`key.expand()` turns token ids into a key the same length as the KV: each
image token becomes a sentinel `("img", sha, proc_hash, k)`, with `proc_hash`
so a processor change can never hit a stale entry. The prompt trie accepts
any hashable. Two different images of the same size diverge at the image's
first token; the same image hits all the way through. There is one sentinel
per image token, so key length always equals KV length.

* **Normalise before hashing.** EXIF orientation applied, converted to RGB,
  clamped (max pixels, and PIL's decompression-bomb limit); the hash is of
  the normalised pixels, so one photo cannot hash two ways.
* **The placeholder is a real special-token id**, the tokenizer's own image
  token (Qwen 248056, GLM `<|image|>` 154854), verified present at load: a
  string the tokenizer does not know would be spelled out as text and the
  image would never reach the sequence.
* **The image store** is keyed `(model_key, sha, proc_hash)`, per image (a
  whole-list key would re-encode everything when a second image is added),
  byte-bounded LRU, and counted in the memory budget (`tuning/resolve.py`):
  a GLM image can be ~65 MB of features. An image is pinned from tokenize
  through admission; image metadata (grid, token count) is never evicted.

## Positions come from the whole key

For Qwen (MRoPE), a text-only turn after an image still needs positions
shifted by that image's `rope_delta`; computing positions only when the new
suffix contains an image degrades silently (fluent text, wrong grounding).
So `positions(key)` is a pure function of the full key and runs on every
prefill and decode of any row whose key contains an image. A text suffix
prefills at `arange(hit, L) + rope_delta` on all three axes. G7b guards it.

## Chunk edges never split an image

gemma4 attends bidirectionally within an image block, so a prefill chunk
must never split one. Every request goes through the batch executor, whose
admission snaps chunk edges to `Family.chunk_boundaries(key)`, including
seeded requests and exact cache hits whose remainder holds image tokens.

## Drafting and images

A row whose uncached span contains an image does not draft; text-only
conversations are unchanged. A cache entry with an image is kept rather
than thrown away to draft. Drafting on the text after an image (seeding the
head from merged embeddings) is not implemented.

## Reuse across turns

Every response's usage carries a cache report {prompt tokens, cached,
expanded image spans, tower encodes}, so the cache gates can be checked from
outside; the chat's token meter reads it.

Reuse depends on the chat template keeping earlier reasoning:

* **Qwen3.8 / Flash-Next / 35B-A3B** (`chat_template.jinja:116`) keep
  earlier reasoning unless `preserve_thinking` is false; **GLM-5.3**
  (`chat_template.jinja:149`) keeps it unless `clear_thinking` is true
  (default false). The prefix is stable provided the client sends each
  earlier assistant turn's `reasoning_content` back. knurlogic's chat always
  does; for other clients the metric is reported and the behaviour
  documented ("send reasoning_content back to keep reuse").
* **gemma4** renders earlier reasoning only for the latest turn with tool
  calls (`chat_template.jinja:239`) and strips thinking from earlier content
  (`:319`, `:327`), with no switch. The image is still never re-prefilled
  (it sits in an earlier user turn, before the divergence), but each new
  turn re-prefills the previous answer. A cache checkpoint at the end of
  each user message would recover full reuse; it is not implemented.

## Gates

Tiny random fixtures (no model, no load lock), in `tests/test_vision_*.py`,
`tests/test_image_cache.py` and `tests/test_vision_batch.py`:

* **G1** vendored tower == mlx-vlm golden (atol 1e-5); **G2** grid and
  token count exact; **G3** positions exact; **G4** 40 greedy tokens
  identical to the golden, including gemma with `prefill_step_size=16` so a
  chunk would split an image block; **G5** the text path is unchanged when
  no image is present.
* **G6** the tower is called once across a two-turn conversation; **G7**
  warm turn 2 == cold turn 2 (tokens, last-prefill logits atol 1e-4);
  **G7b** image in turn 1, text-only turn 2, warm == cold (fails if
  positions are computed from the suffix only); **G8** turn 2's cached
  prefix covers all of turn 1, prompt and reply; **G9** a different image of
  the same size misses and answers differently; **G10** an image prompt with
  the drafting machinery == without; **G11** three mixed rows each == solo.
* Every gate is mutated once: break the thing it guards and confirm it goes
  red.

On real models, one at a time behind the load lock: vision tensor count; a
text answer; an image answer (a colour, OCR of "42"); a five-turn
conversation where the tower runs once, the cache covers at least through
the end of the last image span on turns 2-5 (`prompt - cached <= new + 32`
for templates that keep reasoning), and turn 5 recalls the image; memory
back to baseline after unload. `tools/vision_gate.py` runs them.

## Risks

1. MRoPE threaded into mlx-lm-derived trunks, and per-row `rope_delta` in
   batched decode -- no reference implementation. G3, G4, G7b guard it.
2. Sentinels reaching code that assumes int tokens (stats, detokenizer,
   cache-key append). G6-G9 run through the real serve path.
3. Recurrent-state families take only exact or shorter prefix hits; an
   edited turn re-prefills from the nearest snapshot.
4. GLM's vendored tower differs from mlx-vlm 0.6.17 by design
   (`_limited_swiglu`); G1 for GLM documents the delta.

## Module notes

### src/knurlogic/engine/families/gemma4/vision/__init__.py

WHY THE TOWER IS STANDALONE. `load_weights` reads `vision_tower.*` and
`embed_vision.*` straight off the model directory's safetensors (filtered
by the index's weight_map when there is one), into a `VisionModel` +
`MultimodalEmbedder` the module owns -- never through the text model's
`sanitize`, which drops every non-text key (docs/design/vision-contracts.md,
"load_weights").

WHY encode() PRE-DIVIDES BY embed_scale. mlx-vlm's `gemma4.Model
.get_input_embeddings` scales ONLY the text embeddings (`inputs_embeds =
embed_tokens(ids) * embed_scale`) and then scatters the (unscaled)
projected image features in on top, replacing those rows entirely
(mlx-vlm `gemma4.py:85-170`). knurlogic's `gemma4_text.Gemma4TextModel
.__call__` scales whatever `input_embeddings` it is handed -- text or
already-merged -- by `embed_scale` unconditionally
(`../architecture/gemma4_text.py:527-528`, unedited: the vendored edit list
does not include this scaling line, and touching it would change text
behaviour every caller shares). So the Family divides the tower's projected
features by `embed_scale` before they are cached (`encode`) and merges them
into UNSCALED text embeddings (`embed`): the trunk's later `* embed_scale`
then cancels the division on the image rows and applies correctly to the
text rows, exactly matching the reference's order of operations. This is
the one deviation from a byte-for-byte port and is recorded again at
`encode`.

WHY per_layer_inputs USES ZEROED IDS. mlx-vlm's merge computes gemma4's
per-layer inputs (PLE) from `input_ids` with every multimodal placeholder
zeroed (mlx-vlm `gemma4.py:88-100`) -- an image token must not look up a
per-layer embedding as if it were vocabulary id 258880. `embed` builds that
zeroed-id array from the key and calls the trunk's own
`_get_per_layer_inputs` (unedited) on it, then passes the UNPROJECTED
result back in as `per_layer_inputs`; `Gemma4TextModel.__call__` already
takes a precomputed (unprojected) `per_layer_inputs` and only runs
`_project_per_layer_inputs` on it (`gemma4_text.py:530-534`), so no trunk
edit is needed for this half of the contract -- only the mask overlay
needs one.

### src/knurlogic/engine/families/glm5/vision/__init__.py

The GLM Family's tower is
`knurlogic.engine.families.glm5.architecture.glm5_next.vision.VisionModel`,
the SAME class the trunk's `Model.vision_tower` would build. That class does
not need mlx-vlm installed to import, and the Family loads it STANDALONE
(contracts: "load_weights: standalone tower, not attached to the trunk; the
trunk's sanitize keeps dropping vision keys").

Weight keys on disk are `vision_model.*` (347 keys, inside the main
shards); the Family's own tower attribute is "vision_tower" so it never
collides with anything the trunk's own `sanitize()` does with
`vision_model.*` keys (which it drops). The remap therefore lives in
`Family.load_weights`, not in the vendored `VisionModel`, which has its
transformers bases removed and is otherwise unchanged.

NoPE: GLM's trunk gets its positions from its own 1D rope inside
`language.py`, unaffected by an image span -- there is no MRoPE grid to
thread through decode the way Qwen needs. `positions()` therefore always
answers `(None, 0)`: "the trunk's own 1D positions, no rope_delta"
(contracts, Family protocol).

### src/knurlogic/engine/families/qwen/vision/family.py

What each Qwen Family method is, and where it comes from:

  load_weights  the tower's tensors, found through the index weight_map
                (the 333-tensor model-vision-graft.safetensors sidecar on
                every released Qwen rung, or main shards), in EITHER naming:
                HF `model.visual.*` (397B, Flash-Next) or MLX `vision_tower.*`
                (27B, 35B-A3B). Key mapping is mlx-vlm qwen3_5/qwen3_5.py
                `sanitize_key` (:16-25), the patch-embed transpose is the
                tower's own `sanitize`. The tower stands ALONE: the trunk's
                sanitize keeps dropping the vision keys, so nothing about
                loading the text model changes.
  preprocess    processing.ImageProcessor, settings from the rung's own
                preprocessor_config.json (mlx-vlm reads the same file)
  encode        the tower, once; output rows = t*h*w / merge^2 = n_tokens
  placeholder   "<|vision_start|><|image_pad|><|vision_end|>" -- the exact
                text the Qwen chat template emits for an image part, read
                back from the rung's tokenizer files by ID and checked to be
                special added tokens (a string the tokenizer does not know
                is spelled out as text and the image silently never reaches
                the sequence)
  embed         embed_tokens over key[start:] (sentinels back to the pad id,
                which mlx-vlm feeds too), then scatter.merge -- mlx-vlm's
                merge_input_ids_with_image_features by sentinel. Returns
                input_embeddings only: see `positions`.
  positions     rope_index over the WHOLE key, refs from the store (never
                evicted) -- the trunk's `position_ids` for a prefill chunk
                (slice [:, :, a:b]) and `rope_delta` for text after the last
                image and every decode step
  chunk_boundaries  [] -- Qwen attention is causal across an image

WHY embed DOES NOT RETURN position_ids. The contract lets it, but embed only
gets a FeatureLookup, and positions depend on every image in the key --
including ones before `start` whose features the store may have evicted.
Refs are never evicted, so positions come from `positions(key, refs)` alone;
one home for them. This is a clarification of the Family contract in
docs/design/vision-contracts.md.

### src/knurlogic/engine/families/qwen/vision/rope_index.py

The port of mlx-vlm's `get_rope_index` covers ONE row with no attention
mask -- the only shape the serve path asks for -- in numpy, taking each
image's grid from its ref instead of a batch-wide `image_grid_thw`:

  * images are counted the way the reference counts them: an image token
    right after `vision_start_token_id` (:1756-1762). vision_start comes
    from the model's config -- 248053 in every released rung, not the class
    default 248045 -- and getting it wrong silently zeroes the image count;
  * the k-th image is the k-th run found by `index(image_token_id, st)`
    (:1768-1776), given the k-th grid;
  * text before an image continues from the previous segment's max + 1; an
    image's (t, h, w) = its grid indices (h and w after the spatial merge)
    + text_len + st_idx (:1799-1837); trailing text likewise (:1838-1849);
  * rope_delta = max + 1 - len (:1882-1885).

WHY PURE. A text-only turn after an image still needs positions shifted by
that image's delta; computing them only when the new suffix has an image
gets that turn wrong. Nothing is carried between calls: the same key gives
the same positions, cold or warm.

The delta never changes as text is appended after the last image (trailing
text is linear), so a caller may compute it once per row and add it to the
cache offset at every decode step -- the trunk's `rope_delta` input.

Held to the reference by G3 (tests/test_vision_qwen.py: positions and delta
of a two-image prompt exact against mlx-vlm's own get_rope_index).

### src/knurlogic/engine/vision/__init__.py

  __init__.py   the contracts: ImageRef, EncodedImage, VisionSpec, the
                Family protocol, the errors, served_vision()
  key.py        the cache key: token ids with each image token replaced by
                a sentinel ("img", sha, proc_hash, k)
  store.py      encoded images, per image, byte-bounded LRU, counted
  images.py     request bytes -> a clamped RGB image and its pixel hash
  scatter.py    image features into text embeddings, by sentinel (mlx)
  _base.py      the three helpers the vendored towers need from mlx-vlm (mlx)
  registry.py   model_type -> the family package that serves its images
  request.py    a chat request with images -> the cache key (generator thread)
  cachehook.py  pins an image while a cached conversation still holds it
  quant.py      quantizes a tower's layers to match its checkpoint (mlx)
  (each family's tower, preprocessing and embed live in its folder under
   engine/families/<family>/vision/, named by its manifest)

WHY THE FRONT DOOR IS STDLIB ONLY. The page and the MCP (interfaces/) read
`VisionSpec` and `served_vision()` to say whether the served model sees
images, and they may not pay for an mlx import to ask -- the same rule
`engine/mtp`'s front door keeps (tests/test_resolve.py). So nothing there
imports mlx or PIL; arrays appear only as annotations. Where the module and
the design differ, the design says why the module is right or the
difference is a bug.

### src/knurlogic/engine/vision/cachehook.py

A prompt-cache entry whose key holds image sentinels (key.py) is KV computed
from those images. While it lives, a turn that extends it may re-read the
images -- a partial hit that cuts into an image span re-embeds the rest of
that span from the store. So each image stays in the ImageStore while ANY
live entry references its sha: one store pin per (entry, image run), taken
when the entry is inserted, dropped when it leaves the cache by any path.

HOW. mlx-lm's LRUPromptCache (mlx_lm/models/cache.py) removes entries on
four paths: insert_cache's replacement of an equal key, its pop_prefixes of
shorter keys, its size/bytes LRU pops, and trim_to. Intercepting each is
brittle; instead the two MUTATING methods are wrapped by name (with
assertions -- a pinned mlx-lm that renames one fails at install, loudly),
and after either returns the hook RECONCILES: the live entries are read off
`_lru._lrus` (the deques of (model, tokens) every path keeps in step with
the trie), new ones are pinned, gone ones unpinned. Entries are tracked by
the identity of their `tokens` list (the hook holds a reference, so the id
is stable); a replacement of an equal key is a new list, so the old entry's
pins go and the new one's come.

THE ADMIT GAP. VisionServe.tokenize pins a request's images until the batch
generator admits the row. If the scheduler raises between tokenize and
insert (building the state machine, say), nothing admits the row and the
pins would stay forever. `pending()` records the pins a tokenize took;
`claim()` is called once `insert_segments` has queued the row (from then on
the generator's admit/remove releases them); `sweep()` releases whatever
was never claimed. The scheduler (engine/runtime/scheduler.py) sweeps
before every tokenize -- its thread is sequential, so a pending entry still
unclaimed then was abandoned -- and wraps tokenize-to-insert in
`admit_guard()`, which claims on success.

### src/knurlogic/engine/vision/images.py

The hash is of PIXELS, not of the base64 a client sent: the same picture
re-encoded (a client that re-compresses PNGs, a different base64 wrapping)
still hits the store and the prompt cache. Mode and size go into the hash
alongside `tobytes()`, so a 100x1 and a 1x100 image of the same bytes cannot
share a name.

CLAMPS, BEFORE HASHING. An image arrives from an HTTP client on a shared
host:
  * MAX_BYTES of encoded input, checked before decoding anything;
  * BOMB_PIXELS: PIL's own decompression-bomb limit (Image.MAX_IMAGE_PIXELS
    default, 89,478,485 px), checked from the header before pixels are
    decoded; over it is refused (ImageRejected), not shrunk -- shrinking
    would first have to decode it;
  * MAX_DECODE_PIXELS: anything larger but legal is downscaled (aspect kept,
    BICUBIC, deterministic) before hashing, so the hash names what the
    processor actually sees. It is a safety bound, not a quality knob: each
    family's processor applies its own max_pixels after this (VisionSpec).

No URLs are fetched: a server that fetches client-supplied URLs is an SSRF
hole, and reading a local path named by a remote client is worse. Paths are
allowed only when the caller says the request is local (allow_paths).

PIL is imported lazily: the module is imported by the serve path's front
door, and a text-only server should not pay for PIL.

### src/knurlogic/engine/vision/key.py

WHY A SENTINEL PER TOKEN, NOT PER IMAGE. The scheduler does prefix
arithmetic on the prompt (the cached count is `len(prompt) - len(rest)`,
and the segment trim follows from it: engine/runtime/scheduler._insert),
so the key must be exactly as long as the KV it names.

WHY THE TRIE ACCEPTS IT. `mlx_lm.models.cache.PromptTrie` walks
`current[tok]` dicts; any hashable works (read at mlx-lm 0.31.3, the pinned
version -- tests/test_vision_key.py runs the real LRUPromptCache so a
version that stops accepting it goes red).

WHAT IT BUYS. Every image's run is the same pad id, so with plain ids two
different images of the same size collide and the cache hands back KV
computed from the wrong picture -- fluent, wrong, silent. With sentinels
two such images diverge at the image's first token (k=0) and the same image
hits all the way through. proc_hash is in the sentinel so a processor
change that keeps n_tokens cannot hit stale features.

WHY IT FAILS LOUD. A sentinel that reaches mx.array raises; it can never be
read as a wrong id. Sentinels reaching mlx-lm code that assumes ints is
guarded end to end by G6-G9, with negative-int sentinels as the fallback --
a change confined to key.py.

### src/knurlogic/engine/vision/request.py

The scheduler tokenizes on its own thread (engine/runtime/scheduler.py),
and generation runs on that same thread. Encoding images on the HTTP thread
would mean two threads on one GPU, two uncoordinated allocations on a
shared host. So ALL image work happens in request.py, called from the
scheduler's tokenize:

    image parts -> decode + clamp + pixel hash (images.load)
                -> store hit, or preprocess + encode + put (the ONLY tower
                   call; G6 counts it from outside)
                -> each part replaced by Family.placeholder_text(ref)
                -> the prompt stage (engine/runtime/prompt.tokenize:
                   template, segments, thinking state)
                -> key.expand_segments: one pad per image widened to
                   n_tokens, each image token a sentinel
                -> (key, segment keys, types, state) back to the server

The server then does its own prefix arithmetic on the key -- it is the same
length as the KV (key.py) -- and hands it to the batch generator, which
turns it back into ids and embeddings (mtp/batch_generator.py). No side
table carries images between threads; the key names them.

PINS (vision-contracts.md). Between this tokenize and the batch admit,
other requests may be tokenized and a small store could evict this one's
images. Every image of the key is pinned and released by the generator when
the row is admitted (or removed unadmitted). Pins are counted per image, so
two queued requests on one image hold it twice.

### src/knurlogic/engine/vision/store.py

Keyed PER IMAGE by (model_key, sha, proc_hash):

* per image, not per request -- keying on a hash of the whole image LIST
  would make adding a second image to a conversation re-encode the first.
  Here the second image is one miss and the first is still a hit;
* model_key, because two models' features are different spaces;
* proc_hash, so a processor change cannot serve stale features.

BYTE-BOUNDED, NOT COUNT-BOUNDED. mlx-vlm's VisionFeatureCache
(mlx_vlm/vision_cache.py, 0.6.17) bounds by count (20). Counts lie about
memory: a GLM image at up to ~8000 tokens x 4096 hidden in bf16 is ~65 MB
of features, a small gemma image is 280 x 2560 x 2 = 1.4 MB. On a shared
host the bound has to be bytes, it has to default small, and
`tuning/resolve.py` has to count it BEFORE a load -- `max_bytes` is that
number, one home: DEFAULT_MAX_BYTES.

REFS ARE NOT EVICTED. The ImageRef of every image ever put stays (a few
hundred bytes each) after its features go: positions() for a text turn
after an image needs the image's grid, and a prompt-cache hit can outlive
the features. Refs die with the store -- clear() on unload, which is also
when the prompt cache they index dies.

PINNING. Between the tokenize wrap (features ensured) and the batch admit
(features read) other requests can be tokenized; without a pin, a small
store could evict the first request's image in between. `pinned()` holds
entries past the bound until released.

The scatter module takes rows by sentinel rather than by a global feature
index (`cumsum(is_image) - 1` over the full prompt, which has to count the
images before the hit), so nothing before a span's `start` is looked at.
