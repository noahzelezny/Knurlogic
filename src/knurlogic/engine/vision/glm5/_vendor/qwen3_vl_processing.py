"""`_flatten_images`, vendored from mlx-vlm 0.7.1
`mlx_vlm/models/qwen3_vl/processing_qwen3_vl.py`.

glm5_next's own `processing.py` imports only this one helper from its
qwen3_vl sibling (`from ..qwen3_vl.processing_qwen3_vl import
_flatten_images`, caught by `deps.glm5_siblings()`). The rest of that
upstream file subclasses `transformers.ProcessorMixin` /
`ImageProcessingMixin` to build a full HF processor, which the design (v2,
"Remove the transformers bases") rules out -- glm5_next has no reason to
pull in `transformers` for one list-flattening helper. So only the function
is vendored, not the file it lived in.

Verified byte-identical to mlx-vlm 0.7.1's copy (see PROVENANCE.md).
"""
from __future__ import annotations


def _flatten_images(images):
    """Flatten grouped and array-batched images while retaining path support."""
    if isinstance(images, (list, tuple)):
        return [image for group in images for image in _flatten_images(group)]
    if getattr(images, "ndim", None) == 4:
        return list(images)
    return [images]
