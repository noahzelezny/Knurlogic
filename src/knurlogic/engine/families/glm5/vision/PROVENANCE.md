# glm5_next vision -- provenance

This package vendors no model code of its own. The tower it builds is the
vendored `glm5_next` package's `vision.VisionModel`
(`engine/families/glm5/architecture/glm5_next/`, taken from mlx-vlm 0.6.17
together with its whole import closure under `_mlx_vlm/` -- see
`../architecture/PROVENANCE.md`). The released GLM rungs are built on
mlx-vlm 0.6.17, which 0.7.1 cannot load, so 0.6.17 is the reference.
Design: `docs/design/vision.md`. Contracts:
`docs/design/vision-contracts.md`.

## No mlx-vlm at import

`glm5_next` is registered under the name `mlx_vlm.models.glm5_next`
(`knurlogic.engine.register`); its own relative imports would otherwise
resolve through that name to an installed mlx-vlm. Its import lines point
at the vendored closure (`knurlogic.engine.families.glm5.architecture.
glm5_next._mlx_vlm.models.X`) instead, so GLM serves images with no mlx-vlm
installed (tools/vision_gate.py passes through `knurlogic serve` that way).

## Preprocessing

`Glm5VisionFamily.preprocess` builds patches itself, normalizing with the
artifact's `image_mean`/`image_std` (CLIP's by default), as
Glm5NextImageProcessor does. It does not call into the HF processor class.

## The `vision_model.*` -> `vision_tower.*` remap

Weight keys on disk are `vision_model.*` (347 keys, inside the main
shards). `Glm5VisionFamily.load_weights` reads every `vision_model.*` key
straight out of the artifact's own `model.safetensors.index.json` shards and
strips the prefix before handing the dict to a STANDALONE
`VisionModel(vision_config)` (never attached to any trunk `Model`) via
`VisionModel.sanitize` + `tree_unflatten`. The vendored `vision.py` does no
remapping and knows nothing about being loaded this way -- it is the exact
class `Model.vision_tower` would build, loaded outside the trunk
(contracts: "standalone tower... the trunk's sanitize keeps dropping
vision keys").

## `positions()` -> `(None, 0)` (NoPE)

GLM's trunk (`language.py`) computes its own 1D rotary positions inside its
attention layers; there is no MRoPE grid analogous to Qwen's that an image
span needs to shift. So `Glm5VisionFamily.positions` always returns
`(None, 0)`: "the trunk's own 1D positions, no rope_delta" per the Family
protocol.

## Open issues

1. **`fixed_tokens` is `None`, unverified.** The vision contract requires
   `fixed_tokens` to stay `None` "unless the family's processor was read and
   shows a fixed count." `preprocess()` computes the grid from `image_size`
   and `patch*merge`, matching the vendored tower's own patch/merge
   arithmetic, but has not been checked against GLM's actual smart-resize
   rules (which may clamp differently).
2. **`tests/test_vision_glm5.py` is structural**, not a full gate sweep: it
   verifies that the architecture imports and builds with `mlx_vlm` blocked,
   and that a tiny random-weight `Glm5VisionFamily` encodes a tiny image to
   the right token count and dtype, embeds it via `scatter.merge`, and
   returns `(None, 0)` from `positions`. The remap against real
   `vision_model.*` keys is covered only by the real-model gate
   (tools/vision_gate.py).
