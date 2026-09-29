"""Qwen MRoPE positions as a pure function of the whole prompt.

A port of mlx-vlm 0.6.17 `qwen3_5/language.py` `get_rope_index`
(:1729-1904, sha256 4805ae90fb3bba463512cbce89c9bb7fa78d56b25bd8b541db1392ad34f18ae0,
MIT, Copyright (c) 2025 Prince Canuma) for ONE row with no attention mask,
in numpy, taking each image's grid from its ref. `vision_start_token_id`
comes from the model's config (248053 in every released rung, not the
class default 248045); getting it wrong silently zeroes the image count.

Pure: nothing is carried between calls, so a text-only turn after an image
still gets positions shifted by that image's delta, cold or warm. The delta
never changes as text is appended after the last image, so a caller may
compute it once per row. Held to the reference by G3
(tests/test_vision_qwen.py). Design: docs/design/vision.md (Qwen MRoPE).
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def rope_index(ids: Sequence[int], grids: Sequence[Tuple[int, int, int]],
               image_token_id: int, vision_start_token_id: int,
               spatial_merge_size: int) -> Tuple[np.ndarray, int]:
    """(positions int32 [3, len(ids)], rope_delta) for one prompt.

    grids: (t, h, w) per image in prompt order, in patches (before merge)."""
    tokens = list(ids)
    n = len(tokens)
    vision_tokens = [tokens[i + 1] for i, t in enumerate(tokens[:-1])
                     if t == vision_start_token_id]
    image_nums = sum(t == image_token_id for t in vision_tokens)
    if image_nums > len(grids):
        raise ValueError(f"{image_nums} images in the prompt, "
                         f"{len(grids)} grids")
    segs: List[np.ndarray] = []
    st = 0
    for k in range(image_nums):
        ed = tokens.index(image_token_id, st)
        t, h, w = grids[k]
        gt, gh, gw = int(t), int(h) // spatial_merge_size, \
            int(w) // spatial_merge_size
        text_len = ed - st
        st_idx = int(segs[-1].max()) + 1 if segs else 0
        segs.append(np.broadcast_to(np.arange(text_len), (3, text_len))
                    + st_idx)
        t_index = np.repeat(np.arange(gt), gh * gw)
        h_index = np.tile(np.repeat(np.arange(gh), gw), gt)
        w_index = np.tile(np.arange(gw), gt * gh)
        segs.append(np.stack([t_index, h_index, w_index]) + text_len + st_idx)
        st = ed + gt * gh * gw
    if st < n:
        st_idx = int(segs[-1].max()) + 1 if segs else 0
        text_len = n - st
        segs.append(np.broadcast_to(np.arange(text_len), (3, text_len))
                    + st_idx)
    if not segs:
        return np.zeros((3, 0), dtype=np.int32), 0
    pos = np.concatenate(segs, axis=1).astype(np.int32)
    return pos, int(pos.max()) + 1 - n
