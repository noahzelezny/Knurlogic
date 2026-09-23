# Vision contracts (P0, frozen)

*2026-09-23. What P1-P5 and P-VQ build against. The code is the authority --
`src/knurlogic/engine/vision/__init__.py` holds every signature below with
its reasons; this page is the map, with data shapes. Design:
`docs/design/vision.md` (v2). A change here is a change to every package:
say so in the commit and re-run all the gates.*

## Modules

| module | imports | what |
|---|---|---|
| `engine/vision/__init__.py` | stdlib | `ImageRef`, `EncodedImage`, `VisionSpec`, `Family`, errors, `proc_hash`, `served_vision` |
| `engine/vision/key.py` | stdlib | the cache key codec |
| `engine/vision/store.py` | stdlib | `ImageStore`: per-image, byte-bounded LRU |
| `engine/vision/images.py` | stdlib + PIL (lazy) | decode, clamp, `pixel_sha` |
| `engine/vision/registry.py` | stdlib | model_type -> family `build` |
| `engine/vision/scatter.py` | mlx | `masked_scatter`, `merge` (features into embeddings by sentinel) |
| `engine/vision/_base.py` | mlx | `BaseModelConfig`, `check_array_shape`, `ensure_fused_sdpa`, vendored from mlx-vlm 0.6.17 |
| `machine/loadlock.py` | stdlib | the model-load lock |
| `tests/fixtures_vision.py` | stdlib + numpy (mlx/PIL lazy) | tiny config scaler, `StubFamily`, goldens |

Importing the package, `key`, `store`, `registry` or `images` loads neither
mlx nor PIL (tested): `interfaces/` may read `served_vision()` freely.

## Data

```python
ImageRef(sha: str, proc_hash: str, n_tokens: int,
         grid_thw: tuple[int, int, int] | None = None)      # frozen
EncodedImage(ref: ImageRef, feats: mx.array [n_tokens, text_hidden],
             extras: dict[str, mx.array] = {})               # .nbytes: read from arrays
VisionSpec(family: str, image_token_id: int, patch: int, merge: int | None,
           min_pixels: int, max_pixels: int, fixed_tokens: int | None,
           proc_hash: str)                                   # frozen; .to_json()
proc_hash(settings: dict) -> str   # canonical JSON, sha256, 16 hex
```

* `sha` = `images.pixel_sha(img)`: sha256 over `f"{mode}:{w}x{h}:"` + raw
  pixels of the decoded, EXIF-transposed, RGB, clamped image.
* `feats` row k is the embedding for sentinel k. Evaluate before `put`.
* `fixed_tokens` stays `None` unless the family's processor was read and
  shows a fixed count (critique issue 5).

## The key (design D6)

```
key = token ids, same length as the KV, each image token replaced by
      ("img", sha, proc_hash, k)        k = 0 .. n_tokens-1
```

* Each image occupies ONE contiguous run of `image_token_id`, exactly
  `n_tokens` long, refs in prompt order. Framing ids (Qwen vision_start/end,
  gemma boi/eoi, GLM image_start/end) are ordinary ids around the run.
* `Family.placeholder_text(ref)` must tokenize to exactly ONE
  `image_token_id` per image (plus framing). The only expansion is generic:

```python
key.expand_pads(ids, refs, image_token_id) -> list[int]      # 1 pad -> n_tokens
key.expand(ids, refs, image_token_id) -> key                 # expanded ids -> key
key.expand_segments(segments, refs, image_token_id) -> (key, segment_keys)
                                   # UNexpanded segments from mlx-lm's _tokenize;
                                   # sum(len(s)) == len(key), checked
key.to_ids(key, image_token_id) -> list[int]
key.image_spans(key) -> [Span(start, end, sha, proc_hash, k0)]   # k0: first k in the slice
key.images_in(key) -> [(sha, proc_hash)]                     # per span, in order
key.has_image(key) -> bool;  key.is_sentinel(x) -> bool;  key.sentinel(ref, k)
```

* Any mismatch between pads and refs raises `KeyMismatch` (a user who typed
  the pad token): the serve path answers 400.
* mlx-lm 0.31.3's `LRUPromptCache` accepts the key as is (tested on the real
  class): two same-size images diverge at k=0; the same image hits through.

## Family protocol

```python
class Family(Protocol):
    spec: VisionSpec
    def load_weights(self, model_path: str) -> int            # tensors loaded
    def preprocess(self, img: PIL.Image, sha: str) -> tuple[dict, ImageRef]
    def encode(self, pixels: dict, ref: ImageRef) -> EncodedImage   # the ONLY tower call
    def placeholder_text(self, ref: ImageRef) -> str
    def embed(self, model, key: list, start: int,
              features: FeatureLookup) -> dict                # over key[start:]
    def positions(self, key: list, refs: RefLookup) -> tuple[mx.array | None, int]
    def chunk_boundaries(self, key: list) -> list[tuple[int, int]]

FeatureLookup = Callable[[sha, proc_hash], EncodedImage]     # raises ImageEvicted
RefLookup     = Callable[[sha, proc_hash], ImageRef]         # never evicted
```

* All calls happen on the generator thread (D3).
* `load_weights`: standalone tower, not attached to the trunk; the trunk's
  sanitize keeps dropping vision keys (critique B3, option a).
* `embed` returns `{"input_embeddings": mx [1, len(key)-start, D], **extras}`
  with extras under the trunk's own kwarg names (`position_ids`,
  `per_layer_inputs`, a mask). Image rows come from `scatter.merge`, which
  takes row k of the sentinel's image: a slice cut mid-image needs no
  global feature index.
* `positions(key, refs)`: pure in the FULL key (D4); called on every
  prefill and decode of a row whose key has an image. `None` = trunk's 1D
  positions (gemma, GLM). Qwen: `[3, 1, len(key)]` and `rope_delta`.
* `chunk_boundaries(key)`: `[start, end)` spans no prefill chunk edge may
  fall strictly inside (D5). gemma: every image span. Causal: `[]`.

Family `build` (the registry target):
`build(model_path: str, text_model, config: dict) -> Family | None`.

## Registry

```python
registry.FAMILIES = {
  "qwen3_5":     "knurlogic.engine.vision.qwen:build",
  "qwen3_5_moe": "knurlogic.engine.vision.qwen:build",
  "qwen4_exp":   "knurlogic.engine.vision.qwen:build",
  "gemma4":      "knurlogic.engine.vision.gemma4:build",
  "glm5_next":   "knurlogic.engine.vision.glm5:build",
}
registry.build(model_type, model_path, text_model, config=None) -> Family | None
registry.has_family(model_type) -> bool
```

`None` when: model_type unregistered, config has no `vision_config`, the
family module is absent, or its build declines. An ImportError raised INSIDE
a present family module propagates (a broken build is not "no vision").

## Store

```python
ImageStore(max_bytes=DEFAULT_MAX_BYTES)        # 256 MiB
  .get(model_key, sha, proc_hash) -> EncodedImage | None     # LRU touch
  .features(model_key, sha, proc_hash) -> EncodedImage       # raises ImageEvicted
  .ref(model_key, sha, proc_hash) -> ImageRef | None         # survives eviction
  .lookup(model_key) -> (FeatureLookup, RefLookup)           # for embed/positions
  .put(model_key, enc) -> bool                               # False: could not stay
  .pinned(model_key, [(sha, proc_hash)])                     # context manager
  .clear(model_key=None)                                     # on unload; refs too
  .nbytes, .max_bytes (settable; shrinking evicts), .budget_bytes()
  .stats() -> {entries, nbytes, max_bytes, over_bound_by_pins, pinned,
               refs, hits, misses, evictions}
estimate_nbytes(n_tokens, text_hidden, dtype_bytes=2) -> int
```

* Keyed `(model_key, sha, proc_hash)` per image; `model_key` is mlx-lm's
  `model_provider.model_key` (any hashable).
* Byte-bounded; eviction is oldest-unpinned-first. Pins hold entries above
  the bound until released (the overshoot shows in `stats()`).
* **For tuning/resolve.py:** reserve `DEFAULT_MAX_BYTES` (or the configured
  `max_bytes`) per served vision model BEFORE the load; a live store answers
  `budget_bytes()`.
* **For P4:** pin every image of a request from the `_tokenize` wrap
  through admit (`with store.pinned(mk, key.images_in(prompt_key))` or an
  explicit pin held on the row), so an image cannot be evicted between the
  two.

## Images

```python
images.decode(src: str | bytes, *, allow_paths=False) -> PIL.Image  # RGB, clamped
images.load(src, *, allow_paths=False) -> (PIL.Image, sha)
images.pixel_sha(img) -> str
images.clamp(img, max_pixels=None) -> PIL.Image
MAX_BYTES = 32 MiB   BOMB_PIXELS = 89_478_485   MAX_DECODE_PIXELS = 4096 * 4096
```

* Accepts bytes, raw base64, `data:...;base64,` URLs. Refuses http(s) URLs
  (the server does not fetch) and local paths unless `allow_paths` (a
  remote client must not name server files). Over `BOMB_PIXELS` (from the
  header, before decode) is refused; over `MAX_DECODE_PIXELS` is
  downscaled (BICUBIC, aspect kept) before hashing. All refusals raise
  `ImageRejected` (400).
* Normalisation = mlx-vlm 0.6.17 `utils.load_image` (EXIF transpose, RGB),
  held by the `p0_load_image` golden.

## Errors

`VisionError` > `KeyMismatch` (400), `ImageRejected` (400, also ValueError),
`NoVision` (400), `ImageEvicted` (500, also KeyError).

## Served spec (critique C4)

`served_vision() -> VisionSpec | None`, `set_served_vision(spec | None)`.
P4 sets it when a family is built and clears it on unload; P5 reads it.

## Load lock

```python
loadlock.model_load(artifact, purpose, wait_s=0, agent=None, path=None)
        # context manager, yields the record; raises Busy(.holder)
loadlock.holder(path=None) -> dict | None   # for ready(): a blocker when not None
loadlock.lock_path()   # $KNURLOGIC_LOADLOCK or ~/.cache/knurlogic/load.lock
loadlock.EXIT_BUSY = 75                     # a gate tool's exit when Busy
```

`flock(LOCK_EX|LOCK_NB)`: the kernel releases a dead holder's lock, SIGKILL
included. Not reentrant. Record `{pid, host, agent, artifact, purpose,
started}` is display only. Callers: seam.load/switch (P4); MCP load, serve,
`ready()` blocker (P5); tools gates. Tiny tests never take it.

## Test fixtures and goldens

```python
fixtures_vision.REAL[family]           # structural config fields, 5 families
fixtures_vision.tiny_config(family, real=None, text_config=None, **overrides)
fixtures_vision.tiny_ids(family)       # special ids packed at the top of vocab 512
fixtures_vision.tiny_image(w, h, seed); png_bytes(img)
fixtures_vision.StubFamily(image_token_id, hidden, patch=4, max_side=16,
                           bidirectional=True, placeholder="<image>", embed_fn=None)
fixtures_vision.stub_build(model_path, text_model, config)
fixtures_vision.save_golden(name, arrays, meta) / load_golden(name) -> (arrays, meta)
fixtures_vision.run_reference(script, *args)   # runs in the mlx-vlm 0.6.17 interpreter
```

* Per-family fixtures: `tests/fixtures_vision_<family>.py`, owned by the
  family package. Goldens: `tests/goldens/<name>.npz`, built by
  `tests/goldens/build_<name>.py` under `REFERENCE_PYTHON`
  (`/opt/anaconda3/envs/exo/bin/python`, override `KNURLOGIC_VLM_PYTHON`);
  tests load them with numpy only. A missing golden fails, never skips.
* `StubFamily`: identity tower (patch pixels padded to `hidden`), one token
  per patch, `positions -> (None, 0)`, image-span `chunk_boundaries` when
  `bidirectional`. Count tower calls by wrapping `fam.tower` from outside.

## Pins (design D2)

`pyproject.toml` pins `mlx==0.31.2`, `mlx-lm==0.31.3`; `[tool.knurlogic.pins]`
holds those and the sha256 of `mlx_lm/server.py`. `tests/test_pins.py` fails
on drift and lists what to re-verify. The VQ runtime's pin is P-VQ's.
