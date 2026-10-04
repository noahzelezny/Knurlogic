"""GLM-5.3 (glm5_next) vision.

The Family that `registry.build("glm5_next", ...)` resolves to
(`docs/design/vision-contracts.md`). Its tower is the vendored
`knurlogic.engine.families.glm5.architecture.glm5_next.vision.VisionModel`,
the SAME class the trunk's `Model.vision_tower` would build, importable
without mlx-vlm and loaded STANDALONE (the trunk's sanitize drops vision
keys). On-disk keys are `vision_model.*`; the remap to this Family's
"vision_tower" lives in `load_weights`, not in the vendored model.

NoPE: GLM's trunk takes its positions from its own 1D rope, unaffected by
an image span, so `positions()` always answers `(None, 0)`.
Design: docs/design/vision.md (GLM).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knurlogic.engine.vision import EncodedImage, ImageRef, VisionSpec, proc_hash

_PATCH_PREFIX = "vision_model."
_TOWER_PREFIX = "vision_tower."


class Glm5VisionFamily:
    """`Family` (vision-contracts.md) for glm5_next."""

    #: CLIP's, the Glm5NextImageProcessor defaults; the artifact's own
    #: processor_config.json overrides them (see `build`).
    IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
    IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, config: dict[str, Any], image_mean=None,
                 image_std=None):
        from knurlogic.engine.families.glm5.architecture.glm5_next.config import (
            VisionConfig,
        )

        self.image_mean = tuple(image_mean or self.IMAGE_MEAN)
        self.image_std = tuple(image_std or self.IMAGE_STD)

        vc = config.get("vision_config") or {}
        self.vision_config = VisionConfig.from_dict(vc)
        self.image_token_id = int(config["image_token_id"])
        self.image_start_token_id = config.get("image_start_token_id")
        self.image_end_token_id = config.get("image_end_token_id")

        patch = self.vision_config.patch_size
        merge = self.vision_config.spatial_merge_size
        # GLM's processor resizes to a multiple of patch*merge and has no
        # fixed token budget (aspect-dependent, like gemma's):
        # fixed_tokens stays None until a real processor read
        # says otherwise (open issue, see PROVENANCE.md).
        self.spec = VisionSpec(
            family="glm5_next",
            image_token_id=self.image_token_id,
            patch=patch,
            merge=merge,
            min_pixels=(patch * merge) ** 2,
            max_pixels=self.vision_config.image_size ** 2 * 4,
            fixed_tokens=None,
            proc_hash=proc_hash({
                "family": "glm5_next", "patch": patch, "merge": merge,
                "temporal_patch": self.vision_config.temporal_patch_size,
                "image_size": self.vision_config.image_size,
                "mean": list(self.image_mean), "std": list(self.image_std),
            }),
        )
        self.tower_model = None  # built lazily, mx is engine-only

    # -- weights ---------------------------------------------------------

    def _build_tower(self):
        if self.tower_model is None:
            from knurlogic.engine.families.glm5.architecture.glm5_next.vision import (
                VisionModel,
            )
            self.tower_model = VisionModel(self.vision_config)
        return self.tower_model

    def load_weights(self, model_path: str) -> int:
        """Load the `vision_model.*` keys out of the artifact's own shards
        into a STANDALONE tower (contracts). The remap to this tower's own
        parameter names (no `vision_model.` prefix at all -- the class is
        not nested under a `vision_tower` attribute of anything) happens
        here, once, from the safetensors index; the trunk's `sanitize`
        never sees these keys."""
        import mlx.core as mx
        from mlx.utils import tree_unflatten

        tower = self._build_tower()
        weights: dict[str, Any] = {}
        root = Path(model_path)
        index = root / "model.safetensors.index.json"
        if index.is_file():
            manifest = json.loads(index.read_text())
            shards = sorted(set(manifest.get("weight_map", {}).values()))
        else:
            shards = [p.name for p in root.glob("*.safetensors")]
        for shard in shards:
            p = root / shard
            if not p.is_file():
                continue
            for k, v in mx.load(str(p)).items():
                if k.startswith(_PATCH_PREFIX):
                    weights[k[len(_PATCH_PREFIX):]] = v
                elif k.startswith(_TOWER_PREFIX):
                    weights[k[len(_TOWER_PREFIX):]] = v
        if not weights:
            return 0
        weights = tower.sanitize(weights)
        from knurlogic.engine.vision.quant import artifact_quantization, quantize_like
        quantize_like(tower, weights, artifact_quantization(model_path))
        tower.update(tree_unflatten(list(weights.items())))
        mx.eval(tower.parameters())
        return len(weights)

    # -- preprocess / encode ----------------------------------------------

    def preprocess(self, img, sha: str):
        """Resize to a multiple of `patch * merge` on each side (GLM has no
        fixed token budget) and lay pixels out the way
        `VisionPatchEmbed.__call__` reshapes them: `[-1, in_channels,
        temporal_patch, patch, patch]` before its own `moveaxis`, so this
        hands it `[n_patches, temporal_patch * patch * patch * in_channels]`
        with the temporal axis repeated (`temporal_patch_size`, a still
        image has no time axis to sample)."""
        import numpy as np
        from PIL import Image


        vc = self.vision_config
        p, merge = vc.patch_size, vc.spatial_merge_size
        step = p * merge
        w, h = img.size
        gw = max(merge, round(w / step) * merge)
        gh = max(merge, round(h / step) * merge)
        im = img.convert("RGB").resize((gw * p, gh * p), Image.BICUBIC)
        arr: Any = np.asarray(im, dtype=np.float32) / 255.0       # [H, W, 3]
        # Normalized as Glm5NextImageProcessor does (do_normalize defaults
        # on). Found on GLM-5.3-Flash 2.7: without it a pure red square was
        # seen as "salmon/coral" -- every colour shifted.
        arr = (arr - np.asarray(self.image_mean, np.float32)) \
            / np.asarray(self.image_std, np.float32)
        # Patches in merge-window order, as Glm5NextImageProcessor.patchify
        # lays them out: each merge x merge window's patches consecutive,
        # windows row-major. The tower's rope (rot_pos_emb) and downsample
        # both assume it; raster order scrambled every image's layout.
        arr = arr.reshape(gh // merge, merge, p, gw // merge, merge, p, 3)
        arr = arr.transpose(0, 3, 1, 4, 2, 5, 6)
        arr = arr.reshape(gh * gw, p, p, 3)
        tp = vc.temporal_patch_size
        arr = np.repeat(arr[:, None], tp, axis=1)                  # [-, tp, p, p, 3]
        arr = arr.transpose(0, 4, 1, 2, 3).reshape(gh * gw, -1)    # channel-first, flat

        n_tokens = (gh // merge) * (gw // merge)
        ref = ImageRef(sha=sha, proc_hash=self.spec.proc_hash,
                       n_tokens=n_tokens, grid_thw=(1, gh, gw))
        return {"pixel_values": arr, "grid_thw": (1, gh, gw)}, ref

    def encode(self, pixels: dict[str, Any], ref) -> EncodedImage:
        import mlx.core as mx

        from knurlogic.engine.vision import EncodedImage

        tower = self._build_tower()
        px = mx.array(pixels["pixel_values"], dtype=mx.float32)
        grid = mx.array([pixels["grid_thw"]], dtype=mx.int32)
        feats = tower(px, grid)
        mx.eval(feats)
        if feats.shape[0] != ref.n_tokens:
            raise ValueError(
                f"glm5_next tower produced {feats.shape[0]} tokens for a "
                f"{ref.grid_thw} grid; expected {ref.n_tokens}")
        return EncodedImage(ref=ref, feats=feats)

    def placeholder_text(self, ref) -> str:
        return "<|image|>"

    # -- embed / positions / chunking -------------------------------------

    @staticmethod
    def _embed_tokens(model):
        """`model` is whatever `text_model` the caller loaded -- the plain
        `LanguageModel` (`.model.embed_tokens`) when knurlogic serves
        GLM text-only, or the multimodal `Model` (`.language_model.model.
        embed_tokens`) if ever built directly. Try both rather than assume
        one, the same defensiveness `fixtures_vision.StubFamily` uses."""
        for path in ("language_model.model.embed_tokens",
                     "model.embed_tokens", "embed_tokens"):
            obj = model
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue
        raise AttributeError("no embed_tokens found on the glm5_next model")

    def embed(self, model, key, start, features) -> dict[str, Any]:
        import mlx.core as mx

        from knurlogic.engine.vision import key as K
        from knurlogic.engine.vision.scatter import merge

        sl = list(key[start:])
        ids = mx.array(K.to_ids(sl, self.image_token_id))[None]
        text = self._embed_tokens(model)(ids)
        return {"input_embeddings": merge(text, sl, features)}

    def positions(self, key, refs) -> tuple[Any | None, int]:
        # NoPE: the trunk computes its own 1D positions; no MRoPE grid, no
        # rope_delta (see module docstring).
        return None, 0

    def chunk_boundaries(self, key):
        # GLM's attention is causal (unlike gemma's bidirectional image
        # block); no chunk edge needs protecting.
        return []


def build(model_path: str, text_model: Any, config: dict[str, Any]):
    if not config.get("vision_config"):
        return None
    mean = std = None
    pc = Path(model_path) / "processor_config.json"
    if pc.is_file():
        ip = json.loads(pc.read_text())
        ip = ip.get("image_processor", ip)
        if ip.get("do_normalize", True) is not False:
            mean, std = ip.get("image_mean"), ip.get("image_std")
    # the tower is read by serve/vision.bind (fam.load_weights), not here:
    # it was read twice per load, and a follower rank builds the family
    # without one
    return Glm5VisionFamily(config, image_mean=mean, image_std=std)
