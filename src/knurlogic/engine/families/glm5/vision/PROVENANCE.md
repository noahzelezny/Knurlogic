# glm5_next vision -- provenance (P3)

> **Superseded 2026-09-23.** `_vendor/` (mlx-vlm 0.7.1 siblings) is gone.
> The released GLM rungs are built on mlx-vlm 0.6.17, which 0.7.1 cannot
> load, so glm5_next and its whole import closure were re-vendored from
> 0.6.17 under `engine/architectures/glm5_next/_mlx_vlm/` -- see
> `engine/architectures/PROVENANCE.md`. The tower this family builds is
> that package's `vision.VisionModel`. Preprocessing now normalizes with the
> artifact's image_mean/image_std (CLIP's by default), as
> Glm5NextImageProcessor does. The record below is kept as history.


*What this package vendors, where each file came from, and every edit made
to a vendored file's imports. Design: `docs/design/vision.md` v2. Contracts:
`docs/design/vision-contracts.md`.*

## Why this exists

`knurlogic.engine.architectures.glm5_next` (already vendored, taken from
mlx-vlm 0.7.1, see `../../architectures/PROVENANCE.md`) is only *registered*
under the fake package name `mlx_vlm.models.glm5_next`
(`knurlogic.engine.register`, `ARCH_HOST = {"glm5_next": "mlx_vlm"}`); its
own source uses relative imports (`from ..cache import ...`) that resolve
against that name. Before this package, those siblings did not exist inside
knurlogic at all, so the registered module reached out through
`sys.modules["mlx_vlm.models.*"]` to the REAL, installed mlx-vlm the moment
Python resolved the parent package -- exactly the dependency the design (v2,
Goal) says knurlogic should not have for GLM.

The fix is not to change the registration trick (P4/other packages still
rely on it for other purposes) but to stop glm5_next's *own* files from
needing any `mlx_vlm.*` sibling at all: every one of its `from ..X import Y`
lines (X = base, cache, mla, gated_delta, linear, sparse_attention,
switch_layers, deepseek_v4.hyper_connection, qwen3_vl.processing_qwen3_vl --
the exact set `knurlogic.machine.deps.glm5_siblings()` derives from the
source, re-run 2026-09-23) becomes an ABSOLUTE import of a copy vendored
here, under `_vendor/`. Absolute, not another relative import: a module
registered under the synthetic name `mlx_vlm.models.glm5_next` has THAT as
its `__name__`, so relative dots inside it climb through the fake namespace
(`..` -> `mlx_vlm.models`, `...` -> `mlx_vlm`), not through this package's
real location on disk. Verified 2026-09-23: `python -c "..."` with a
`sys.meta_path` hook that raises on `import mlx_vlm` (exact name only)
still resolves `importlib.import_module("mlx_vlm.models.glm5_next")` to a
working `Model` class.

## Vendored files (`_vendor/`)

All taken 2026-09-23 from the **mlx-vlm 0.7.1** PyPI wheel
(`mlx_vlm-0.7.1-py3-none-any.whl`, downloaded with
`pip download mlx-vlm==0.7.1 --no-deps`), MIT, Copyright (c) 2025 Prince
Canuma -- the same release `glm5_next/` itself was vendored from
(`../../architectures/PROVENANCE.md`), so tower and siblings are from one
consistent snapshot.

| file | from (`mlx_vlm/`) | edit |
|---|---|---|
| `base.py` | `models/base.py` | `from ..turboquant import ...` -> `from .turboquant import ...` (turboquant is vendored beside it, not a package level up) |
| `cache.py` | `models/cache.py` | none |
| `mla.py` | `models/mla.py` | none |
| `gated_delta.py` | `models/gated_delta.py` | none |
| `linear.py` | `models/linear.py` | none |
| `sparse_attention.py` | `models/sparse_attention.py` | none |
| `switch_layers.py` | `models/switch_layers.py` | none |
| `activations.py` | `models/activations.py` | none -- `switch_layers.py`'s own dependency, not in `glm5_siblings()` (glm5_next does not import it directly) but needed for the vendored copy to import standalone |
| `fast_ops.py` | `models/fast_ops.py` | none -- `deepseek_v4/hyper_connection.py`'s dependency, same reason |
| `quantized_verifier.py` | `models/quantized_verifier.py` | none -- added by the end-to-end pass (2026-09-23): `linear.py` and `switch_layers.py` import it INSIDE functions (`native_batch_linear`, a 2-8 token projection), so no import-only test saw it missing and the first real prefill of 2-8 tokens raised ModuleNotFoundError. Its one relative import (`.switch_layers`) is vendored beside it. sha256 f3f3c485f164fbaf237a7f294b44f7fa0ab3e2078bfcfd71c6fab6bf1a877289, equal to the wheel RECORD's entry |
| `turboquant.py` | `turboquant.py` (package top level, not `models/`) | `from .models.cache import ...` -> `from .cache import ...` (flattened: no `models/` subpackage here) -- `base.py`'s dependency |
| `deepseek_v4/hyper_connection.py` | `models/deepseek_v4/hyper_connection.py` | none (its own `from ..X import` lines are correct as-is: `..` from `_vendor/deepseek_v4/` IS `_vendor/`) |
| `qwen3_vl_processing.py` | NOT a vendored file -- see below | -- |

sha256 of each file as vendored (before the import edits listed above,
where any) matches the wheel's copy; re-verify with
`shasum -a 256 _vendor/<file>` against a fresh
`pip download mlx-vlm==0.7.1 --no-deps` if this ever needs re-checking.

### `qwen3_vl_processing.py` -- one function, not a file copy

`glm5_next/processing.py` imports exactly one thing from its `qwen3_vl`
sibling: `_flatten_images` (`from ..qwen3_vl.processing_qwen3_vl import
_flatten_images`, the 9th entry `glm5_siblings()` derives -- the design v2
work-package table names only 8; the 9th showed up only by deriving it from
source, which is the whole point of `glm5_siblings()` deriving rather than
listing). The file it lives in, `qwen3_vl/processing_qwen3_vl.py`, otherwise
subclasses `transformers.ProcessorMixin` / `ImageProcessingMixin` to build a
full HF image processor -- exactly the "transformers bases" the design says
to remove, and `_flatten_images` itself has no such dependency (pure list
recursion, no `self`, no import at all beyond nothing). So only the
function is copied here, not the file, and `transformers` never becomes an
import of anything under `engine/families/glm5/vision/`.

## The `vision_model.*` -> `vision_tower.*` remap

Per design v2 (Goal: "347 keys, `vision_model.*`... inside the main
shards") and this package's brief ("lives in `Family.load_weights`, not the
vendored model"): `glm5/__init__.py`'s `Glm5VisionFamily.load_weights` reads
every `vision_model.*` key straight out of the artifact's own
`model.safetensors.index.json` shards and strips the prefix before handing
the dict to a STANDALONE `VisionModel(vision_config)` (never attached to
any trunk `Model`) via `VisionModel.sanitize` + `tree_unflatten`. The
vendored `engine/architectures/glm5_next/vision.py` itself does no
remapping and knows nothing about being loaded this way -- it is the exact
class `Model.vision_tower` would have built, just built and loaded outside
the trunk (contracts: "standalone tower... the trunk's sanitize keeps
dropping vision keys").

## Where 0.7.1 differs from mlx-vlm 0.6.17 by design (risk #5)

`docs/design/vision.md` risk 5 says G1 for GLM must document this rather
than hide it: `engine/architectures/glm5_next/vision.py`'s
`_limited_swiglu` (clip-then-silu on both gate and up, `swiglu_limit`) and
its patch/merge path do not exist in mlx-vlm 0.6.17's glm5_next at all --
0.6.17 predates this GLM release; 0.7.1 is the first version that carries
it (`../../architectures/PROVENANCE.md`: "upstream was AHEAD, not behind").
There is therefore no 0.6.17 golden to diff against for GLM's tower; G1 for
this family has no upstream reference to run in the exo interpreter (open
issue below).

## `positions()` -> `(None, 0)` (NoPE)

GLM's trunk (`language.py`) computes its own 1D rotary positions inside its
attention layers; there is no MRoPE grid analogous to Qwen's that an image
span needs to shift (D4 in the design is a Qwen-specific concern). So
`Glm5VisionFamily.positions` always returns `(None, 0)`: "the trunk's own
1D positions, no rope_delta" per the Family protocol.

## Deviations from the design (report to the integrator)

1. **`fixed_tokens` is `None`, unverified.** The design (critique issue 5,
   folded into contracts) requires `fixed_tokens` to stay `None` "unless
   the family's processor was read and shows a fixed count." GLM's real
   HF processor was not read (no network fetch of the actual
   `preprocessor_config.json` for a released GLM-5.3 rung was done in this
   pass); `preprocess()` here computes grid size from `image_size` and
   `patch*merge` directly, matching the vendored tower's own patch/merge
   arithmetic, but has NOT been checked against GLM's actual smart-resize
   rules (which may clamp differently). Open issue for the integrator.
2. **No G1/G4 golden for GLM's tower.** No mlx-vlm 0.6.17 golden exists for
   GLM (see "0.7.1 vs 0.6.17" above -- 0.6.17 has no glm5_next at all), and
   this pass did not build a 0.7.1-based golden either (no real-weight
   fixture; the hard rule against loading real models applies, and a
   *tiny random* fixture run through both this vendored tower and a
   0.7.1 `mlx_vlm.models.glm5_next.vision.VisionModel` built from the SAME
   tiny config would be the right G1, comparing two random-init towers
   under a fixed seed -- not built in this pass). Open issue.
3. **`tests/test_vision_glm5.py` in this pass is structural**, not a full
   G1-G11 sweep: it verifies (a) the architecture imports and builds with
   `mlx_vlm` blocked from ever resolving, (b) the vendored 8+1 siblings
   import with no reference to the real `mlx_vlm` package, (c) a tiny
   random-weight `Glm5VisionFamily` encodes a tiny image to the right
   token count and dtype, embeds it into a placeholder run via
   `scatter.merge`, and returns `(None, 0)` from `positions`. It does not
   exercise `load_weights` against a real safetensors shard (no such
   fixture exists without a real checkpoint) -- `load_weights` is
   exercised structurally (empty directory -> 0 tensors) but its remap
   logic against actual `vision_model.*` keys is unverified until the
   orchestrator's real-model gate for GLM (per the design's real-model
   gate list, GLM gets "a tower-only local check" first).
4. **`processing.py` still imports `transformers`** for its
   `ProcessorMixin`/`ImageProcessingMixin` bases (unchanged from vendoring;
   this package did not rewrite `Glm5NextImageProcessor`). This is a
   `transformers` dependency, not an `mlx-vlm` one -- outside this
   package's brief ("knurlogic stops depending on mlx-vlm for GLM") -- and
   `Glm5VisionFamily.preprocess` here does not call into it at all (it
   builds patches itself). Flagged since the design's "remove the
   transformers bases" language could be read to cover it; smallest thing
   that works was to leave the HF-processor class alone and not use it.
