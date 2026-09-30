"""The Qwen vision family: qwen3_5, qwen3_5_moe and qwen4_exp (one tower,
one processor, one position rule).

The tower's tensors load STANDALONE through the index weight_map, in either
naming (HF `model.visual.*` or MLX `vision_tower.*`); the trunk's sanitize
keeps dropping vision keys. The placeholder is read back from the rung's
tokenizer by ID and checked to be special added tokens -- a string the
tokenizer does not know is spelled out as text and the image silently
never reaches the sequence. `embed` returns input_embeddings only:
positions depend on every image in the key, including ones the store may
have evicted, so they come from `positions(key, refs)` alone.
Design: docs/design/vision.md (Qwen).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np

from knurlogic.engine.vision import (EncodedImage, ImageRef, VisionError,
                                     VisionSpec, proc_hash)
from knurlogic.engine.vision import key as K
from knurlogic.engine.vision.quant import artifact_quantization, quantize_like
from knurlogic.engine.vision.scatter import merge

from .processing import ImageProcessor, image_kwargs
from .rope_index import rope_index
from .vision import VisionConfig, VisionModel

#: the sidecar every released Qwen rung carries (333 tensors); the family
#: reads whatever file the index names, this is only for messages
SIDECAR = "model-vision-graft.safetensors"

#: vision-key prefixes, both namings, mlx-vlm qwen3_5.py sanitize_key :16-25
_PREFIXES = (("model.language_model.visual.", ""), ("model.visual.", ""),
             ("vision_tower.", ""), ("visual.", ""))

#: the three special tokens an image occupies in a Qwen prompt, by config key
_TOKEN_KEYS = ("vision_start_token_id", "image_token_id", "vision_end_token_id")


def _tower_key(k: str) -> Optional[str]:
    for pre, _ in _PREFIXES:
        if k.startswith(pre):
            return k[len(pre):]
    return None


def _special_tokens(model_path: str) -> Dict[int, Tuple[str, bool]]:
    """id -> (content, special) for the rung's added tokens, from
    tokenizer.json `added_tokens` or tokenizer_config.json
    `added_tokens_decoder` (the released Qwen rungs carry the former)."""
    out: Dict[int, Tuple[str, bool]] = {}
    p = Path(model_path)
    tc = p / "tokenizer_config.json"
    if tc.is_file():
        for i, d in (json.loads(tc.read_text()).get("added_tokens_decoder")
                     or {}).items():
            out[int(i)] = (d.get("content"), bool(d.get("special", False)))
    tj = p / "tokenizer.json"
    if tj.is_file():
        for d in json.loads(tj.read_text()).get("added_tokens") or []:
            out[int(d["id"])] = (d.get("content"), bool(d.get("special", False)))
    return out


class QwenFamily:
    """engine.vision.Family for the three Qwen families."""

    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.model_type = config.get("model_type", "qwen3_5")
        tc = config.get("text_config", {}) or {}
        ids = {}
        for k in _TOKEN_KEYS:
            v = config.get(k, tc.get(k))
            if v is None:
                raise VisionError(f"{self.model_type} config has no {k}")
            ids[k] = int(v)
        self.image_token_id = ids["image_token_id"]
        self.vision_start_token_id = ids["vision_start_token_id"]
        self.vision_end_token_id = ids["vision_end_token_id"]

        toks = _special_tokens(model_path)
        names = []
        for k in _TOKEN_KEYS:
            got = toks.get(ids[k])
            if got is None or not got[0] or not got[1]:
                raise VisionError(
                    f"{k} {ids[k]} is not a special added token of the "
                    f"tokenizer in {model_path}; the image placeholder would "
                    f"be spelled out as text and never reach the sequence")
            names.append(got[0])
        self._placeholder = "".join(names)

        self.vision_config = VisionConfig.from_dict(
            dict(config["vision_config"]))
        self.merge_size = self.vision_config.spatial_merge_size
        self.processor = ImageProcessor(**image_kwargs(model_path))
        if (self.processor.patch_size != self.vision_config.patch_size
                or self.processor.merge_size != self.merge_size
                or self.processor.temporal_patch_size
                != self.vision_config.temporal_patch_size):
            raise VisionError(
                f"preprocessor (patch {self.processor.patch_size}, merge "
                f"{self.processor.merge_size}) does not match the tower "
                f"(patch {self.vision_config.patch_size}, merge "
                f"{self.merge_size})")
        self.spec = VisionSpec(
            family=self.model_type, image_token_id=self.image_token_id,
            patch=self.processor.patch_size, merge=self.merge_size,
            min_pixels=int(self.processor.min_pixels),
            max_pixels=int(self.processor.max_pixels), fixed_tokens=None,
            proc_hash=proc_hash(dict(self.processor.settings(),
                                     family="qwen")))
        self.tower = None          # built by load_weights

    # --- weights -------------------------------------------------------------

    def _vision_files(self, model_path: str) -> Dict[str, List[str]]:
        """file -> [tensor names] for every vision tensor the rung has."""
        p = Path(model_path)
        idx = p / "model.safetensors.index.json"
        files: Dict[str, List[str]] = {}
        if idx.is_file():
            wm = json.loads(idx.read_text()).get("weight_map", {})
            for k, f in wm.items():
                if _tower_key(k) is not None:
                    files.setdefault(f, []).append(k)
            return files
        for f in sorted(p.glob("*.safetensors")):
            names = [k for k in mx.load(str(f)) if _tower_key(k) is not None]
            if names:
                files[f.name] = names
        return files

    def load_weights(self, model_path: str) -> int:
        files = self._vision_files(model_path)
        if not files:
            raise VisionError(f"no vision tensors in {model_path} (looked for "
                              f"{[p for p, _ in _PREFIXES]} in the index "
                              f"weight_map; released rungs keep them in "
                              f"{SIDECAR})")
        weights: Dict[str, Any] = {}
        n = 0
        for f, names in files.items():
            arrays = mx.load(str(Path(model_path) / f))
            for k in names:
                weights[_tower_key(k)] = arrays[k]
                n += 1
            del arrays
        tower = VisionModel(self.vision_config)
        weights = tower.sanitize(weights)
        dtype = next(iter(weights.values())).dtype
        tower.set_dtype(dtype)
        quantize_like(tower, weights, artifact_quantization(model_path))
        tower.load_weights(list(weights.items()), strict=True)
        tower.eval()
        mx.eval(tower.parameters())
        self.tower = tower
        return n

    # --- per image -------------------------------------------------------------

    def preprocess(self, img, sha: str):
        pv, grid = self.processor(img)
        t, h, w = (int(x) for x in grid)
        ref = ImageRef(sha=sha, proc_hash=self.spec.proc_hash,
                       n_tokens=t * h * w // (self.merge_size ** 2),
                       grid_thw=(t, h, w))
        return {"pixel_values": pv, "grid_thw": (t, h, w)}, ref

    def encode(self, pixels: Dict[str, Any], ref: ImageRef) -> EncodedImage:
        if self.tower is None:
            raise VisionError("encode before load_weights")
        dtype = self.tower.patch_embed.proj.weight.dtype
        pv = mx.array(np.asarray(pixels["pixel_values"])).astype(dtype)
        grid = mx.array([list(pixels["grid_thw"])])
        feats, _ = self.tower(pv, grid)
        mx.eval(feats)
        if feats.shape[0] != ref.n_tokens:
            raise VisionError(f"tower gave {feats.shape[0]} rows for an image "
                              f"of {ref.n_tokens} tokens")
        return EncodedImage(ref=ref, feats=feats)

    def placeholder_text(self, ref: ImageRef) -> str:
        return self._placeholder

    # --- per prompt ------------------------------------------------------------

    @staticmethod
    def _embed_tokens(model):
        for path in ("language_model.model.embed_tokens",   # qwen3_5, _moe
                     "model.embed_tokens"):                  # qwen4_exp
            obj = model
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue
        raise AttributeError("no embed_tokens on the Qwen trunk")

    def embed(self, model, key: List[Any], start: int, features):
        sl = list(key[start:])
        ids = mx.array(K.to_ids(sl, self.image_token_id))[None]
        text = self._embed_tokens(model)(ids)
        return {"input_embeddings": merge(text, sl, features)}

    def positions(self, key: List[Any], refs) -> Tuple[Any, int]:
        """(mx int32 [3, 1, len(key)], rope_delta), or (None, 0) for a key
        with no image (the trunk's own 1-D positions are then exact)."""
        spans = K.image_spans(key)
        if not spans:
            return None, 0
        grids = []
        for s in spans:
            ref = refs(s.sha, s.proc_hash)
            if ref.grid_thw is None:
                raise VisionError(f"image {s.sha[:12]} has no grid")
            grids.append(ref.grid_thw)
        pos, delta = rope_index(K.to_ids(key, self.image_token_id), grids,
                                self.image_token_id,
                                self.vision_start_token_id, self.merge_size)
        return mx.array(pos)[:, None, :], delta

    def chunk_boundaries(self, key: List[Any]) -> List[Tuple[int, int]]:
        return []
