"""DeepSeek-V4-Flash-Vision-Exp's image processor, numpy + PIL.

Ported from `inference/image_processor.py` of
deepseek-ai/DeepSeek-V4-Flash-Vision-Exp (MIT License, Copyright (c) 2023
DeepSeek): `grid_tokens`, `solve_resize_ratio` and `safe_resize`
unchanged, `load_image` without the record loading (engine/vision/images
decodes) and with numpy for torch, `build_image_block` split in two: the
image's own block (IMAGE_START .. IMAGE_END, the same wherever it sits) and
the IMAGE_PAD run before it, which depends on the block's position (see
`compress_pad`). tests/engine/test_vision_deepseek.py holds both to the
reference.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)
COMPRESS_PAD_TO = 4


@dataclass(frozen=True)
class VisionArgs:
    """The config.json fields the processor and the tower read."""
    vocab_size: int = 129280
    dim: int = 4096
    vision_n_layers: int = 32
    vision_dim: int = 1024
    vision_n_heads: int = 16
    vision_inter_dim: int = 2816
    vision_patch_size: int = 14
    vision_rope_theta: float = 10000.0
    vision_downsample_ratio: int = 3
    vision_max_n_token: int = 384
    vision_min_pixels: int = 147456
    vision_max_wh_ratio: float | None = 8

    @classmethod
    def from_config(cls, config: dict) -> "VisionArgs":
        kw = {k: config[k] for k in cls.__dataclass_fields__ if k in config}
        if "hidden_size" in config:
            kw["dim"] = config["hidden_size"]
        return cls(**kw)


def grid_tokens(best_height, best_width, patch_size, downsample_ratio):
    """Number of LLM tokens the aligner grid occupies (N-layout, incl. row/align padding)."""
    n_llm_h = math.ceil((best_height // patch_size) / downsample_ratio)
    n_llm_w = math.ceil((best_width // patch_size) / downsample_ratio)
    num_tokens = n_llm_h * (n_llm_w + 1) + 2
    if n_llm_h % 2 == 1:
        num_tokens += n_llm_w + 1
    num_tokens += (n_llm_h + 1) // 2 * (n_llm_w + 1) % 2 * 2
    return n_llm_h, n_llm_w, num_tokens


def solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token):
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    if max_w_float < 1.0:
        max_w = 1
        max_h = (max_n_token - 2) // (max_w + 1)
        if max_h % 2 == 1:
            max_h -= 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    elif max_h_float < 2.0:
        max_h = 2
        max_w = ((max_n_token - 2) // max_h) - 1
        assert max_w > 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    else:
        max_w = math.floor(max_w_float)
        max_h = math.floor(max_h_float)
        if max_h % 2 == 1:
            max_h -= 1
        beta = min(max_w * patch_size * downsample_ratio / width, max_h * patch_size * downsample_ratio / height)
        best_width = math.floor(width * beta / patch_size) * patch_size
        best_height = math.floor(height * beta / patch_size) * patch_size
    n_llm_h, n_llm_w, num_tokens = grid_tokens(best_height, best_width, patch_size, downsample_ratio)
    return n_llm_h, n_llm_w, best_height, best_width, num_tokens


def safe_resize(height, width, best_height, best_width, patch_size, downsample_ratio, max_n_token):
    max_n_token -= COMPRESS_PAD_TO - 1
    n_llm_h, n_llm_w, num_tokens = grid_tokens(best_height, best_width, patch_size, downsample_ratio)
    budget = max_n_token
    while num_tokens > max_n_token:
        n_llm_h, n_llm_w, best_height, best_width, num_tokens = solve_resize_ratio(
            height, width, patch_size, downsample_ratio, budget)
        budget -= 1
    return n_llm_h, n_llm_w, best_height, best_width


def to_bf16(x: np.ndarray) -> np.ndarray:
    """float32 rounded to bfloat16 (round to nearest even), kept float32:
    the reference casts the normalized pixels to bf16 before the tower."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    u = (u + (0x7FFF + ((u >> 16) & 1))) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32)


def load_image(image, args: VisionArgs):
    """An RGB PIL image -> (patches [n_vit_h * n_vit_w, 3 * p * p] float32
    holding bf16 values, n_vit_h, n_vit_w, n_llm_h, n_llm_w)."""
    from PIL import Image, ImageOps

    p = args.vision_patch_size
    image = image.convert("RGB")
    width, height = image.size
    if args.vision_max_wh_ratio is not None and width > height * args.vision_max_wh_ratio:
        width = height * args.vision_max_wh_ratio
    if 0 < width * height < args.vision_min_pixels:
        ratio = (args.vision_min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    n_llm_h, n_llm_w, best_height, best_width = safe_resize(
        height, width, best_height, best_width, p, args.vision_downsample_ratio, args.vision_max_n_token)
    n_vit_h, n_vit_w = best_height // p, best_width // p
    if args.vision_max_wh_ratio is not None and image.width >= args.vision_max_wh_ratio * image.height:
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    x = np.asarray(image, dtype=np.float32) / 255           # [H, W, 3]
    x = to_bf16((x - 0.5) / 0.5)
    # the reference's [3, H, W] -> [n_h * n_w, 3, p, p], flattened (c, y, x)
    patches = x.reshape(n_vit_h, p, n_vit_w, p, 3).transpose(0, 2, 4, 1, 3)
    patches = patches.reshape(n_vit_h * n_vit_w, 3 * p * p)
    return np.ascontiguousarray(patches), n_vit_h, n_vit_w, n_llm_h, n_llm_w


def image_block(n_llm_h: int, n_llm_w: int):
    """The image's own block: (types from IMAGE_START to IMAGE_END, perm),
    `build_image_block` without the leading IMAGE_PAD run. `perm[i]` is
    the aligner row of the i-th IMAGE token."""
    pad_h = n_llm_h % 2
    rows = n_llm_h + pad_h
    row_len = n_llm_w + 1
    pad_last = rows // 2 * row_len % 2 * 2
    types = np.array(([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
                     + [IMAGE_PAD] * (row_len * pad_h), dtype=np.int64)
    order = np.arange(rows * row_len).reshape(rows // 2, 2, row_len) \
        .transpose(0, 2, 1).reshape(-1)
    image_idx = np.full((rows * row_len,), -1, dtype=np.int64)
    image_idx.reshape(rows, row_len)[:n_llm_h, :n_llm_w] = \
        np.arange(n_llm_h * n_llm_w).reshape(n_llm_h, n_llm_w)
    perm = image_idx[order]
    perm = perm[perm >= 0]
    types = np.concatenate([
        [IMAGE_START], types[order], [IMAGE_PAD] * pad_last, [IMAGE_END],
    ]).astype(np.int64)
    return types, perm


def compress_pad(start_pos: int) -> int:
    """How many IMAGE_PAD tokens go before an image whose block would start
    at `start_pos`: they put its IMAGE_START at a position == 3 (mod 4)."""
    return COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO


def build_image_block(n_llm_h: int, n_llm_w: int, start_pos: int):
    """The reference's `build_image_block`: (types, perm) with the leading
    IMAGE_PAD run."""
    types, perm = image_block(n_llm_h, n_llm_w)
    pads = np.full((compress_pad(start_pos),), IMAGE_PAD, dtype=np.int64)
    return np.concatenate([pads, types]), perm
