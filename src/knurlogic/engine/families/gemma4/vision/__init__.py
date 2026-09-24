"""gemma4 vision (e4b, 26b): the Family (engine/vision/__init__.py) built
against the vendored tower (`vision.py`) and the trunk's bidirectional
image-block mask (`architectures/gemma4_text.py`, P2 edit -- see this
package's PROVENANCE.md).

WHY THE TOWER IS STANDALONE. `load_weights` reads `vision_tower.*` and
`embed_vision.*` straight off the model directory's safetensors (filtered
by the index's weight_map when there is one), into a `VisionModel` +
`MultimodalEmbedder` this module owns -- never through the text model's
`sanitize`, which drops every non-text key (design critique B3 option (a);
`docs/design/vision-contracts.md` "load_weights").

WHY encode() PRE-DIVIDES BY embed_scale. mlx-vlm's `gemma4.Model
.get_input_embeddings` scales ONLY the text embeddings
(`inputs_embeds = embed_tokens(ids) * embed_scale`) and then scatters the
(unscaled) projected image features in on top, replacing those rows
entirely (`gemma4.py:85-170`). knurlogic's `gemma4_text.Gemma4TextModel
.__call__` scales whatever `input_embeddings` it is handed -- text or
already-merged -- by `embed_scale` unconditionally
(`architectures/gemma4_text.py:527-528`, unedited: P2's edit list does not
include this scaling line, and touching it would move a P1/text behaviour
every family shares). So this Family divides the tower's projected features
by `embed_scale` before they are cached (`encode`, below) and merges them
into UNSCALED text embeddings (`embed`, below): the trunk's later
`* embed_scale` then cancels the division on the image rows and applies
correctly to the text rows, exactly matching the reference's order of
operations. This is the one deviation from a byte-for-byte port and is
recorded again at `encode`.

WHY per_layer_inputs USES ZEROED IDS. mlx-vlm's merge computes gemma4's
per-layer inputs (PLE) from `input_ids` with every multimodal placeholder
zeroed (`gemma4.py:88-100`) -- an image token must not look up a per-layer
embedding as if it were vocabulary id 258880. `embed` (below) builds that
zeroed-id array from the key and calls the trunk's own
`_get_per_layer_inputs` (unedited) on it, then passes the UNPROJECTED
result back in as `per_layer_inputs`; `Gemma4TextModel.__call__` already
takes a precomputed (unprojected) `per_layer_inputs` and only runs
`_project_per_layer_inputs` on it (`gemma4_text.py:530-534`), so no trunk
edit was needed for this half of the contract's edit list -- only the mask
overlay needed one.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from knurlogic.engine.vision import EncodedImage, ImageRef, VisionSpec, proc_hash
from knurlogic.engine.vision.key import Span, image_spans, is_sentinel, to_ids
from knurlogic.engine.vision.scatter import merge as scatter_merge
from .config import VisionConfig
from knurlogic.engine.vision.quant import artifact_quantization, quantize_like
from .vision import VisionModel

PATCH = 16
POOL = 3  # pooling_kernel_size


class RMSNormNoScale(nn.Module):
    """`embed_vision`'s pre-projection norm. Vendored from mlx-vlm 0.6.17
    `gemma4/language.py` (`RMSNormNoScale`), MIT, Copyright (c) 2025 Prince
    Canuma -- identical to `vision.VisionRMSNormNoScale`, kept as a separate
    class because it lives on `embed_vision`, not the tower, in the weight
    tree."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        x_float = x.astype(mx.float32)
        var = mx.mean(x_float**2, axis=-1, keepdims=True)
        return (x_float * mx.rsqrt(var + self.eps)).astype(x.dtype)


class MultimodalEmbedder(nn.Module):
    """Projects the tower's pooled features into the text model's hidden
    size. Vendored from mlx-vlm 0.6.17 `gemma4/gemma4.py:22-35`
    (`MultimodalEmbedder`), MIT, Copyright (c) 2025 Prince Canuma."""

    def __init__(self, embedding_dim: int, text_hidden_size: int,
                 eps: float = 1e-6):
        super().__init__()
        self.embedding_projection = nn.Linear(embedding_dim, text_hidden_size,
                                              bias=False)
        self.embedding_pre_projection_norm = RMSNormNoScale(embedding_dim, eps=eps)

    def __call__(self, inputs_embeds: mx.array) -> mx.array:
        normed = self.embedding_pre_projection_norm(inputs_embeds)
        return self.embedding_projection(normed)


def _load_index(model_path: Path) -> Optional[Dict[str, str]]:
    idx = model_path / "model.safetensors.index.json"
    if not idx.is_file():
        return None
    return json.loads(idx.read_text()).get("weight_map", {})


def _load_filtered(model_path: Path, prefixes: Tuple[str, ...]) -> Dict[str, mx.array]:
    """Every tensor under `model_path` whose key starts with one of
    `prefixes`. Uses the shard index's weight_map when present (a
    multi-shard release) and falls back to every `*.safetensors` file in
    the directory (a single-file fixture) otherwise."""
    weight_map = _load_index(model_path)
    out: Dict[str, mx.array] = {}
    if weight_map:
        shards = {}
        for key, shard_name in weight_map.items():
            if not key.startswith(prefixes):
                continue
            if shard_name not in shards:
                shards[shard_name] = mx.load(str(model_path / shard_name))
            out[key] = shards[shard_name][key]
    else:
        for f in sorted(model_path.glob("*.safetensors")):
            weights = mx.load(str(f))
            for key, v in weights.items():
                if key.startswith(prefixes):
                    out[key] = v
    return out


class Gemma4Vision:
    """The `Family` (engine/vision/__init__.py) for gemma4."""

    def __init__(self, vision_config: Dict[str, Any], text_hidden_size: int,
                 image_token_id: int, boi_token_id: Optional[int],
                 eoi_token_id: Optional[int]):
        vc = VisionConfig.from_dict(vision_config)
        self.vision_tower = VisionModel(vc)
        self.embed_vision = MultimodalEmbedder(
            embedding_dim=vc.hidden_size, text_hidden_size=text_hidden_size,
            eps=vc.rms_norm_eps)
        self.patch_size = vc.patch_size
        self.pool = vc.pooling_kernel_size
        self.image_token_id = image_token_id
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
        self.text_hidden_size = text_hidden_size
        # Filled from the artifact's tokenizer.json by build(); these are
        # the e4b / 26b strings, kept for fixtures built without one.
        self.token_text = {"boi": "<|image>", "image": "<|image|>",
                           "eoi": "<image|>"}

        self.spec = VisionSpec(
            family="gemma4", image_token_id=image_token_id, patch=vc.patch_size,
            merge=None, min_pixels=vc.patch_size * vc.patch_size,
            max_pixels=vc.default_output_length * vc.pooling_kernel_size**2
                       * vc.patch_size**2,
            fixed_tokens=None,  # aspect-dependent (critique issue 5)
            proc_hash=proc_hash({
                "family": "gemma4", "patch_size": vc.patch_size,
                "pooling_kernel_size": vc.pooling_kernel_size,
                "default_output_length": vc.default_output_length,
            }))

    # -- Family protocol ------------------------------------------------------

    def load_weights(self, model_path: str) -> int:
        p = Path(model_path)
        weights = _load_filtered(p, ("vision_tower.", "embed_vision."))
        tower_w = {k[len("vision_tower."):]: v for k, v in weights.items()
                  if k.startswith("vision_tower.")}
        embed_w = {k[len("embed_vision."):]: v for k, v in weights.items()
                  if k.startswith("embed_vision.")}
        tower_w = self.vision_tower.sanitize(tower_w)
        quant = artifact_quantization(model_path)
        quantize_like(self.vision_tower, tower_w, quant, "vision_tower.")
        quantize_like(self.embed_vision, embed_w, quant, "embed_vision.")
        if tower_w:
            self.vision_tower.update(tree_unflatten(list(tower_w.items())))
        if embed_w:
            self.embed_vision.update(tree_unflatten(list(embed_w.items())))
        mx.eval(self.vision_tower.parameters(), self.embed_vision.parameters())
        return len(tower_w) + len(embed_w)

    def preprocess(self, img: Any, sha: str) -> Tuple[Dict[str, Any], ImageRef]:
        import numpy as np
        p = self.patch_size
        w, h = img.size
        gw, gh = max(1, w // p), max(1, h // p)
        im = img.convert("RGB").resize((gw * p, gh * p))
        arr = np.asarray(im, dtype=np.float32) / 255.0          # [H, W, 3]
        arr = arr.transpose(2, 0, 1)[None]                      # [1, 3, H, W]
        n_tokens = (gh * gw) // (self.pool * self.pool)
        if n_tokens < 1:
            raise ValueError(
                f"image too small for gemma4's {self.pool}x{self.pool} "
                f"pool: {gh}x{gw} patches")
        ref = ImageRef(sha=sha, proc_hash=self.spec.proc_hash, n_tokens=n_tokens)
        return {"pixel_values": arr}, ref

    def encode(self, pixels: Dict[str, Any], ref: ImageRef) -> EncodedImage:
        pooled = self.vision_tower(pixels["pixel_values"])       # [1, n, hidden]
        projected = self.embed_vision(pooled)[0]                 # [n, text_hidden]
        # Pre-divide by embed_scale: see the module docstring ("WHY encode()
        # PRE-DIVIDES BY embed_scale").
        scaled = projected / self._embed_scale()
        mx.eval(scaled)
        if scaled.shape[0] != ref.n_tokens:
            raise ValueError(
                f"tower produced {scaled.shape[0]} rows; ref.n_tokens is "
                f"{ref.n_tokens}")
        return EncodedImage(ref=ref, feats=scaled)

    def placeholder_text(self, ref: ImageRef) -> str:
        # What gemma4's processor puts in place of the template's image
        # token: boi, ONE pad (key.expand_pads widens it to n_tokens), eoi
        # (mlx-vlm processing_gemma4.py full_image_sequence). The template
        # itself emits only the pad, so the framing is ours to add.
        return self.token_text["boi"] + self.token_text["image"] \
            + self.token_text["eoi"]

    def embed(self, model: Any, key: List[Any], start: int,
              features) -> Dict[str, Any]:
        sl = list(key[start:])
        ids = mx.array(to_ids(sl, self.image_token_id))[None]
        # The served model is the gemma4 wrapper (text under
        # .language_model); tests hand the bare gemma4_text Model.
        core = getattr(model, "language_model", model).model
        embed_tokens = core.embed_tokens
        text_embeds = embed_tokens(ids)                          # unscaled
        input_embeddings = scatter_merge(text_embeds, sl, features)

        extras: Dict[str, Any] = {"input_embeddings": input_embeddings}
        if getattr(core, "hidden_size_per_layer_input", 0):
            zeroed = mx.array([0 if is_sentinel(x) else x for x in sl])[None]
            extras["per_layer_inputs"] = core._get_per_layer_inputs(
                zeroed, text_embeds)

        mm_mask = self._mm_mask(sl)
        if mm_mask is not None:
            extras["mm_mask"] = mm_mask
        return extras

    def positions(self, key: List[Any], refs) -> Tuple[Any, int]:
        return None, 0  # gemma: plain 1D RoPE, image tokens ordinary positions

    def chunk_boundaries(self, key: List[Any]) -> List[Tuple[int, int]]:
        return [(s.start, s.end) for s in image_spans(key)]

    # -- internals --------------------------------------------------------------

    def _embed_scale(self) -> float:
        return float(self.text_hidden_size) ** 0.5

    def _mm_mask(self, key_slice: List[Any]) -> Optional[mx.array]:
        """[1, len(key_slice)] int32: -1 outside an image, else the index of
        the image that token belongs to along the slice (see the P2 note in
        `architectures/PROVENANCE.md`)."""
        spans = image_spans(key_slice)
        if not spans:
            return None
        out = [-1] * len(key_slice)
        for i, s in enumerate(spans):
            for j in range(s.start, s.end):
                out[j] = i
        return mx.array(out, dtype=mx.int32)[None]


def _added_tokens(model_path: str) -> Dict[int, str]:
    """id -> text for the artifact's added (special) tokens."""
    p = Path(model_path) / "tokenizer.json"
    if not p.is_file():
        return {}
    return {t["id"]: t["content"]
            for t in json.loads(p.read_text()).get("added_tokens", [])}


def build(model_path: str, text_model: Any, config: Dict[str, Any]):
    vision_config = config.get("vision_config")
    if not vision_config:
        return None
    text_config = config.get("text_config", {})
    fam = Gemma4Vision(
        vision_config=vision_config,
        text_hidden_size=text_config.get("hidden_size", 2560),
        image_token_id=config.get("image_token_id", 258880),
        boi_token_id=config.get("boi_token_id"),
        eoi_token_id=config.get("eoi_token_id"),
    )
    added = _added_tokens(model_path)
    for name, tid in (("boi", fam.boi_token_id), ("image", fam.image_token_id),
                      ("eoi", fam.eoi_token_id)):
        if tid in added:
            fam.token_text[name] = added[tid]
    return fam
