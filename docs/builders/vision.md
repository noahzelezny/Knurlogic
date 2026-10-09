# Building on vision

Images are real context: encoded once per conversation, cached with the
prompt, never evicted while a cache entry uses them. The why:
[vision](../design/vision.md); the contracts and data shapes:
[vision-contracts](../design/vision-contracts.md); DeepSeek's case:
[deepseek-vision](../design/deepseek-vision.md).

## Where the code is

`src/knurlogic/engine/vision/` holds the contracts and the generic half:

| file | what |
|---|---|
| `__init__.py` | the contracts: `ImageRef`, `EncodedImage`, `VisionSpec`, the `Family` protocol, the errors (`VisionError`, `ImageRejected`, `ImageTooLarge`, `ImagesOverBudget`, `NoVision`, ...), `served_vision()`. Stdlib only |
| `images.py` | request bytes to a bounded RGB image, and `pixel_sha` (a hash of pixels, not of base64) |
| `key.py` | the cache key: token ids with image sentinels (`sentinel`, `expand`, `expand_segments`, `image_spans`, `images_in`) |
| `store.py` | `ImageStore`: encoded images keyed by (model_key, sha, proc_hash), byte-bounded |
| `scatter.py` | image features into text embeddings, for the uncached span only (`merge`) |
| `request.py` | `VisionServe` (the served model's family, store and key; `tokenize`), `MirrorVision` (a follower rank's vision on a split model) |
| `cachehook.py` | the prompt cache pins the images it references (`install`, `pending`, `claim`, `sweep`, `admit_guard`) |
| `registry.py` | `model_type` to the family that serves its images, built from the manifests; `build`, `vision_weights`, `unavailable_why` |
| `quant.py` | quantize a vision module to match the checkpoint |
| `_base.py` | the helpers the vendored towers import from mlx-vlm |

Each family's tower and `Family` is under
`engine/families/<family>/vision/`, named by the manifest's `vision`
entry.

Hooks outside it:

- `engine/model/vision.py`: `bind` at load (through `registry.build`),
  `clear` at unload, `vision_status`.
- `engine/runtime/scheduler.py`: images go through tokenize; the
  scheduler wraps tokenize-to-insert in `cachehook.admit_guard()` and
  installs the cache hook.
- `engine/mtp/batch_generator.py`: the batch engine takes `vision=`.
- `engine/split/plan.py`: `admit` carries `images` and `refs` to the
  other ranks of a split (`key_to_wire`, `key_from_wire`).
- `tuning/fit.py`: `vision_budget`, `vision_freed_bytes`,
  `_tower_bytes`; `tuning/measured.py`: `VISION_*` constants;
  `tuning/knobs.py`: `vision_of`.
- `interfaces/http/openai.py` and `messages.py`: image parts in requests.

## Rules that keep it correct

- **The front door is stdlib.** The page and MCP read `VisionSpec` and
  `served_vision()` without mlx or PIL.
- **An image is named by its pixels**, plus mode and size, so the same
  image sent two ways hits the same cache entry.
- **Images in use are never evicted.** Each image stays in the store
  while any live prompt-cache entry references its sha; the two mutating
  methods of mlx-lm's LRU cache are wrapped by name and the wrap asserts
  they exist.
- **The admit gap is covered.** Tokenize pins a request's images until the
  row is admitted; a failed admission's pins are swept.
- **No vision is not an error.** `registry.build` returns None for a
  model without vision; an image request to it is refused at admission.
- **On a split, rank 0 encodes.** Followers get refs and rows through the
  plan and embed with the family's own code (`MirrorVision`).

## Adding vision to a family

The steps are in [families](../design/families.md) (item 3) and
[new-model](../design/new-model.md) ("Images"): a `vision` entry in the
manifest, `vision/` with `build(model_path, text_model, config)`
returning a `Family`, a tiny fixture in
`tests/support/fixtures_vision_<family>.py`, and the real-model gate
`tools/vision_gate.py`.

## Notes

Vision spans `engine/vision/` (generic), `engine/families/*/vision/`
(towers, by design), `engine/model/vision.py` (bind at load),
`tuning/fit.py` and `tuning/measured.py` (memory budget), and the
request parsing in `interfaces/http/`.

## Tests

`tests/engine/test_vision_*.py` (per family, `test_vision_e2e.py`,
`test_vision_batch.py`, `test_vision_key.py`), `test_image_store.py`,
`test_image_cache.py`, `test_cachehook.py`,
`tests/tuning/test_vision_budget.py`,
`tests/interfaces/test_web_vision.py`.
