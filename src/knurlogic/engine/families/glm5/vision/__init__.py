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
import math
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

    #: Glm5NextImageProcessor's token budget defaults (merged tokens per
    #: image); the artifact's processor_config.json overrides them.
    MIN_IMAGE_TOKENS = 16
    MAX_IMAGE_TOKENS = 8000

    def __init__(self, config: dict[str, Any], image_mean=None,
                 image_std=None, min_image_tokens=None,
                 max_image_tokens=None):
        from knurlogic.engine.families.glm5.architecture.glm5_next.config import (
            VisionConfig,
        )

        self.image_mean = tuple(image_mean or self.IMAGE_MEAN)
        self.image_std = tuple(image_std or self.IMAGE_STD)
        # `is None`, not `or`: a configured 0 is a value, not "unset"
        self.min_image_tokens = int(self.MIN_IMAGE_TOKENS
                                    if min_image_tokens is None
                                    else min_image_tokens)
        self.max_image_tokens = int(self.MAX_IMAGE_TOKENS
                                    if max_image_tokens is None
                                    else max_image_tokens)

        vc = config.get("vision_config") or {}
        self.vision_config = VisionConfig.from_dict(vc)
        self.image_token_id = int(config["image_token_id"])
        self.image_start_token_id = config.get("image_start_token_id")
        self.image_end_token_id = config.get("image_end_token_id")

        patch = self.vision_config.patch_size
        merge = self.vision_config.spatial_merge_size
        # GLM's processor keeps the aspect ratio, pads to a multiple of
        # patch*merge and holds each image to min..max_image_tokens merged
        # tokens (see `smart_resize`); the count is aspect-dependent, so
        # fixed_tokens is None.
        self.spec = VisionSpec(
            family="glm5_next",
            image_token_id=self.image_token_id,
            patch=patch,
            merge=merge,
            min_pixels=self.min_image_tokens * (patch * merge) ** 2,
            max_pixels=self.max_image_tokens * (patch * merge) ** 2,
            fixed_tokens=None,
            proc_hash=proc_hash({
                "family": "glm5_next", "patch": patch, "merge": merge,
                "temporal_patch": self.vision_config.temporal_patch_size,
                "image_size": self.vision_config.image_size,
                "mean": list(self.image_mean), "std": list(self.image_std),
                "resize": "pad-aspect",
                "tokens": [self.min_image_tokens, self.max_image_tokens],
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

    def target_size(self, w: int, h: int) -> tuple[int, int, int, int]:
        """(content_w, content_h, canvas_w, canvas_h) exactly as
        Glm5NextImageProcessor.resize (transformers 5.16.1): the canvas is
        `smart_resize`'s aligned size within min..max_image_tokens, the
        image is scaled to fit it keeping its aspect ratio (never up, unless
        it is under the minimum) and the rest is zero padding, right and
        bottom."""
        vc = self.vision_config
        tf = vc.temporal_patch_size
        factor = vc.patch_size * vc.spatial_merge_size
        th, tw = smart_resize(tf, h, w, tf, factor, self.min_image_tokens,
                              self.max_image_tokens)
        scale = min(th / h, tw / w)
        if tf * h * w >= tf * factor ** 2 * self.min_image_tokens:
            scale = min(1.0, scale)
        ch = max(1, min(th, math.floor(h * scale)))
        cw = max(1, min(tw, math.floor(w * scale)))
        return cw, ch, tw, th

    def preprocess(self, img, sha: str):
        """Resize and pad as `target_size` says and lay pixels out the way
        `VisionPatchEmbed.__call__` reshapes them: `[-1, in_channels,
        temporal_patch, patch, patch]` before its own `moveaxis`, so this
        hands it `[n_patches, temporal_patch * patch * patch * in_channels]`
        with the temporal axis repeated (`temporal_patch_size`, a still
        image has no time axis to sample)."""
        import numpy as np
        from PIL import Image

        vc = self.vision_config
        p, merge = vc.patch_size, vc.spatial_merge_size
        im = img.convert("RGB")
        cw, ch, tw, th = self.target_size(*im.size)
        if (cw, ch) != im.size:
            im = im.resize((cw, ch), Image.BICUBIC)
        canvas = np.zeros((th, tw, 3), np.uint8)
        canvas[:ch, :cw] = np.asarray(im)
        gh, gw = th // p, tw // p
        arr: Any = canvas.astype(np.float32) / 255.0       # [H, W, 3]
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
        # What the maker's template emits for an image part (its emit_image
        # macro): <|begin_of_image|>, ONE <|image|> (key.expand_pads widens
        # it to n_tokens), <|end_of_image|>. The placeholder reaches the
        # template as text, which bypasses that macro, so the framing is
        # ours to add, as gemma4's is.
        return "<|begin_of_image|><|image|><|end_of_image|>"

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


def smart_resize(num_frames: int, height: int, width: int,
                 temporal_factor: int = 2, factor: int = 28,
                 min_pixels: int = 16, max_pixels: int = 8000):
    """Glm5NextImageProcessor's `smart_resize` (transformers 5.16.1,
    image_processing_glm5_next.py), unchanged: the aligned (height, width)
    canvas within min..max tokens (`min_pixels`/`max_pixels` are token
    counts, scaled to pixels here as the reference does)."""
    pixels_per_token = temporal_factor * factor ** 2
    min_pixels *= pixels_per_token
    max_pixels *= pixels_per_token

    def align(value, f):
        return math.ceil(value / f) * f

    def fit_within_budget(aligned_frames):
        if max_pixels < aligned_frames * factor ** 2:
            raise ValueError(f"max_pixels={max_pixels} is too small")
        low, high = 1, height
        best_height, best_width = factor, factor
        while low <= high:
            content_height = (low + high) // 2
            content_width = max(1, math.floor(width * content_height / height))
            candidate_height = align(content_height, factor)
            candidate_width = align(content_width, factor)
            if aligned_frames * candidate_height * candidate_width <= max_pixels:
                best_height, best_width = candidate_height, candidate_width
                low = content_height + 1
            else:
                high = content_height - 1
        return best_height, best_width

    aligned_frames = max(temporal_factor,
                         round(num_frames / temporal_factor) * temporal_factor)
    aligned_height = align(height, factor)
    aligned_width = align(width, factor)
    budget = aligned_frames * aligned_height * aligned_width
    if budget < min_pixels:
        scale = math.sqrt(min_pixels / (num_frames * height * width))
        aligned_height = align(max(1, math.ceil(height * scale)), factor)
        aligned_width = align(max(1, math.ceil(width * scale)), factor)
        budget = aligned_frames * aligned_height * aligned_width
    if budget > max_pixels:
        aligned_height, aligned_width = fit_within_budget(aligned_frames)
    return aligned_height, aligned_width


#: PILImageResampling.BICUBIC, Glm5NextImageProcessor's default and the
#: only resample `preprocess` implements
_BICUBIC = 3


def _refuse_unimplemented(ip: dict) -> None:
    """An artifact whose processor asks for a resample filter or a patch
    expansion `preprocess` does not implement is refused at load, not
    served with silently different pixels or token counts."""
    r = ip.get("resample")
    if r is not None and r != _BICUBIC and str(r).lower() != "bicubic":
        raise ValueError(
            f"glm5_next processor_config.json asks for resample {r!r}; "
            f"knurlogic implements bicubic ({_BICUBIC}) only")
    f = ip.get("patch_expand_factor")
    if f is not None and int(f) != 1:
        raise ValueError(
            f"glm5_next processor_config.json asks for patch_expand_factor "
            f"{f}; knurlogic implements 1 only")


def build(model_path: str, text_model: Any, config: dict[str, Any]):
    if not config.get("vision_config"):
        return None
    mean = std = lo = hi = None
    pc = Path(model_path) / "processor_config.json"
    if pc.is_file():
        ip = json.loads(pc.read_text())
        ip = ip.get("image_processor", ip)
        if ip.get("do_normalize", True) is not False:
            mean, std = ip.get("image_mean"), ip.get("image_std")
        lo, hi = ip.get("min_image_tokens"), ip.get("max_image_tokens")
        _refuse_unimplemented(ip)
    # the tower is read by serve/vision.bind (fam.load_weights), not here:
    # it was read twice per load, and a follower rank builds the family
    # without one
    return Glm5VisionFamily(config, image_mean=mean, image_std=std,
                            min_image_tokens=lo, max_image_tokens=hi)
