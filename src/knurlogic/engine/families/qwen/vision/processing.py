"""Qwen image preprocessing: PIL image -> (pixel_values, grid_thw).

Vendored from mlx-vlm 0.6.17 `mlx_vlm/models/qwen3_vl/processing_qwen3_vl.py`
(sha256 21d68148d9bd99952445beaee21237993510dc4c3a2c94a2c5cda5f19429b180),
MIT, Copyright (c) 2025 Prince Canuma, itself a numpy port of HF's
qwen2_vl image processor:
  _smart_resize_image    :182-205   verbatim
  _resize_video_frames   :164-179   verbatim (the image path resizes through it)
  _to_numpy_image        :208-227   verbatim
  ImageProcessor         :230-412   the image half of Qwen3VLImageProcessor:
                                    __init__, _resolved_size, _process_one,
                                    num_image_tokens verbatim; the
                                    transformers base class (ImageProcessingMixin)
                                    dropped -- it only carried from_pretrained
  image_kwargs           :593-626   _qwen_vl_image_kwargs, LOCAL files only
                                    (the Hub fallback is dropped: the server
                                    does not fetch)
Video is not served (design: images only), so none of the video half.

WHY NUMPY AND PIL, NOT mlx. Preprocessing is pixel shuffling; the arrays go
to the tower as mx arrays at encode time. Held to the reference by G2
(tests/test_vision_qwen.py: grid, token count and pixel_values against
mlx-vlm's own processor on the same image).
"""
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _resize_video_frames(video: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Bicubic resize each frame of a ``(T, C, H, W)`` video."""
    from PIL import Image

    T, C, H, W = video.shape
    if target_h == H and target_w == W:
        return video
    out = np.empty((T, C, target_h, target_w), dtype=video.dtype)
    for i, frame in enumerate(video):
        arr = np.transpose(frame, (1, 2, 0))
        if arr.dtype in (np.float32, np.float64):
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(arr)
        pil = pil.resize((target_w, target_h), resample=Image.BICUBIC)
        out[i] = np.transpose(np.array(pil), (2, 0, 1))
    return out


def _smart_resize_image(
    height: int,
    width: int,
    factor: int = 32,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
) -> Tuple[int, int]:
    """Image variant of ``smart_resize`` — ports HF's qwen2_vl ``smart_resize``."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got "
            f"{max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _to_numpy_image(img) -> np.ndarray:
    """Coerce a PIL.Image / path / numpy to a ``(C, H, W)`` uint8 array."""
    from PIL import Image

    if isinstance(img, str):
        img = Image.open(img)
    if hasattr(img, "convert"):
        img = img.convert("RGB")
        arr = np.array(img)  # (H, W, C)
    elif isinstance(img, np.ndarray):
        arr = img
    else:
        arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] in (1, 3, 4) and arr.ndim == 3:
        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    if arr.shape[0] == 4:
        arr = arr[:3]
    return arr


class ImageProcessor:
    """Numpy port of Qwen2/3-VL image processor (torch-free).

    Produces, per image:
      - ``pixel_values``: ``(grid_t*grid_h*grid_w, C * tps * ps * ps)``
        where images have ``grid_t=1`` (duplicated along the temporal axis to
        match the model's ``temporal_patch_size``)
      - ``grid_thw``: ``[grid_t, grid_h, grid_w]``
    """

    def __init__(
        self,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        min_pixels: int = 56 * 56,
        max_pixels: int = 14 * 14 * 4 * 1280,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        do_convert_rgb: bool = True,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean or [0.5, 0.5, 0.5]
        self.image_std = image_std or [0.5, 0.5, 0.5]
        self.do_convert_rgb = do_convert_rgb

    def settings(self) -> Dict[str, Any]:
        """Every setting that changes an image's pixels or grid -- what the
        family's proc_hash covers (knurlogic addition)."""
        return {"patch_size": self.patch_size,
                "temporal_patch_size": self.temporal_patch_size,
                "merge_size": self.merge_size, "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels, "do_rescale": self.do_rescale,
                "rescale_factor": self.rescale_factor,
                "do_normalize": self.do_normalize,
                "image_mean": list(self.image_mean),
                "image_std": list(self.image_std), "resample": "bicubic"}

    def _resolved_size(
        self,
        height: int,
        width: int,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        resized_height: Optional[int] = None,
        resized_width: Optional[int] = None,
    ) -> Tuple[int, int]:
        """Resolve the post-resize ``(height, width)`` for a single image.

        Shared by ``_process_one`` and ``num_image_tokens`` so a token
        estimate cannot drift from the size preprocessing actually uses.
        """
        factor = self.patch_size * self.merge_size
        if (resized_height is None) != (resized_width is None):
            raise ValueError(
                "resized_height and resized_width must be provided together."
            )
        if resized_height is not None:
            return _smart_resize_image(resized_height, resized_width, factor=factor)
        return _smart_resize_image(
            height,
            width,
            factor=factor,
            min_pixels=self.min_pixels if min_pixels is None else min_pixels,
            max_pixels=self.max_pixels if max_pixels is None else max_pixels,
        )

    def _process_one(
        self,
        image: np.ndarray,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        resized_height: Optional[int] = None,
        resized_width: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[int]]:
        C, H, W = image.shape
        resized_h, resized_w = self._resolved_size(
            H,
            W,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            resized_height=resized_height,
            resized_width=resized_width,
        )
        # Bicubic resize via PIL (same pattern as the video path).
        frame = _resize_video_frames(image[None, ...], resized_h, resized_w)[0]

        img = frame.astype(np.float32)
        if self.do_rescale and image.dtype == np.uint8:
            img = img * self.rescale_factor
        if self.do_normalize:
            mean = np.array(self.image_mean, dtype=np.float32)[:, None, None]
            std = np.array(self.image_std, dtype=np.float32)[:, None, None]
            img = (img - mean) / std

        # Duplicate along T so grid_t * tps frames match the model's expectation.
        patches = np.repeat(img[None, None, ...], self.temporal_patch_size, axis=1)

        ps = self.patch_size
        tps = self.temporal_patch_size
        ms = self.merge_size
        grid_t = 1
        grid_h = resized_h // ps
        grid_w = resized_w // ps

        patches = patches.reshape(
            1,
            grid_t,
            tps,
            C,
            grid_h // ms,
            ms,
            ps,
            grid_w // ms,
            ms,
            ps,
        )
        patches = patches.transpose(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        flatten = patches.reshape(1, grid_t * grid_h * grid_w, C * tps * ps * ps)
        return flatten[0], [grid_t, grid_h, grid_w]

    def __call__(self, img) -> Tuple[np.ndarray, List[int]]:
        """One PIL image -> (pixel_values, [t, h, w]) (knurlogic: one image
        per call; the store holds images one by one)."""
        return self._process_one(_to_numpy_image(img))

    def num_image_tokens(
        self,
        height: int,
        width: int,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        resized_height: Optional[int] = None,
        resized_width: Optional[int] = None,
    ) -> int:
        """Number of language-model image tokens an image of the given size
        will produce, computed without processing any pixels."""
        resized_h, resized_w = self._resolved_size(
            height,
            width,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            resized_height=resized_height,
            resized_width=resized_width,
        )
        grid_h = resized_h // self.patch_size
        grid_w = resized_w // self.patch_size
        return (grid_h * grid_w) // self.merge_size**2


def _load_json(model_path, name: str):
    p = Path(model_path) / name
    return json.loads(p.read_text()) if p.is_file() else None


def image_kwargs(model_path, default_patch_size: int = 16) -> Dict[str, Any]:
    """Read Qwen-VL image processor kwargs out of a checkpoint (local only)."""
    proc_cfg = _load_json(model_path, "processor_config.json") or {}
    raw = _load_json(model_path, "preprocessor_config.json") or {}
    raw.update(proc_cfg.get("image_processor", {}) or {})
    out = {"patch_size": default_patch_size}
    for k in (
        "patch_size",
        "temporal_patch_size",
        "merge_size",
        "image_mean",
        "image_std",
        "rescale_factor",
        "do_rescale",
        "do_normalize",
        "do_convert_rgb",
    ):
        if k in raw:
            out[k] = raw[k]
    size = raw.get("size") or {}
    if size.get("shortest_edge") is not None:
        out["min_pixels"] = size["shortest_edge"]
    if size.get("longest_edge") is not None:
        out["max_pixels"] = size["longest_edge"]
    if raw.get("min_pixels") is not None:
        out["min_pixels"] = raw["min_pixels"]
    if raw.get("max_pixels") is not None:
        out["max_pixels"] = raw["max_pixels"]
    return out
