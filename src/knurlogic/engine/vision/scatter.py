"""Image features into text embeddings, for the uncached span only.

The sentinel ("img", sha, proc_hash, k) says WHICH image and WHICH row, so
a span the prefix hit cut into takes rows k..n-1 of its own image and
nothing before `start` is looked at -- no global feature index over the
full prompt, no re-embedding of the whole prompt every turn.

`masked_scatter` is vendored verbatim from mlx-vlm 0.6.17
(`mlx_vlm/models/gemma4/gemma4.py:13-20`, MIT, Copyright (c) 2025 Prince
Canuma), the helper every vendored family's embed uses; a committed golden
(tests/goldens/p0_masked_scatter.npz, made by the mlx-vlm function itself)
holds it to the reference.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import mlx.core as mx

from . import FeatureLookup
from .key import image_spans


def masked_scatter(input_tensor, mask, source):
    mask_flat = mask.flatten().astype(mx.int32)
    indices = mx.cumsum(mask_flat) - 1
    aligned = source.flatten()[indices % source.size]
    return mx.where(mask_flat, aligned, input_tensor.flatten()).reshape(
        input_tensor.shape
    )


def image_mask(key_slice: Sequence[Any]) -> list[bool]:
    """True at every sentinel -- where features replace text embeddings."""
    return [type(x) is tuple for x in key_slice]


def rows_for(key_slice: Sequence[Any], features: FeatureLookup):
    """The feature rows for every sentinel in key_slice, in order, as one
    [n_image_tokens, D] array (None if the slice has no image)."""
    parts = []
    for s in image_spans(key_slice):
        enc = features(s.sha, s.proc_hash)
        n = s.end - s.start
        feats = enc.feats
        if s.k0 + n > feats.shape[0]:
            raise ValueError(f"image {s.sha[:12]} has {feats.shape[0]} rows; "
                             f"the key asks for {s.k0}..{s.k0 + n}")
        parts.append(feats[s.k0:s.k0 + n])
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)


def merge(embeds, key_slice: Sequence[Any], features: FeatureLookup):
    """embeds: the text embeddings of key_slice, [L, D] or [1, L, D], taken
    from the family's embed_tokens over key.to_ids(key_slice) (so any
    family scaling of text embeddings is already applied). Returns the same
    shape with each sentinel position replaced by its image row, cast to
    embeds' dtype. key_slice must be the same length as embeds' L."""
    L = embeds.shape[-2]
    if L != len(key_slice):
        raise ValueError(f"{L} embeddings for a key slice of {len(key_slice)}")
    rows = rows_for(key_slice, features)
    if rows is None:
        return embeds
    mask = mx.array(image_mask(key_slice))
    mask = mx.broadcast_to(mask[:, None], embeds.shape[-2:])
    if embeds.ndim == 3:
        mask = mask[None]
    return masked_scatter(embeds, mask, rows.astype(embeds.dtype))
