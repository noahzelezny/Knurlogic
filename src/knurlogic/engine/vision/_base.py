"""The three helpers the vendored vision towers import from mlx-vlm's
`models/base.py`, and nothing else from it.

Vendored verbatim from mlx-vlm 0.6.17, `mlx_vlm/models/base.py`
(sha256 7e61bb9bfc8cf8faec64023c42d15d89ecdc03ad84b7a265290ac728ffd647fa),
MIT, Copyright (c) 2025 Prince Canuma: BaseModelConfig (base.py:104-120),
check_array_shape (:390-410), ensure_fused_sdpa (:528-538).

Not an import of base.py: it imports turboquant, mlx-vlm's cache and PIL
at module top, and `pip install knurlogic` serves images without mlx-vlm.
The family packages change `from ..base import X` to
`from knurlogic.engine.vision._base import X` and record that edit in
their PROVENANCE.md.
"""
import inspect
from dataclasses import dataclass

import mlx.core as mx


@dataclass
class BaseModelConfig:
    @classmethod
    def from_dict(cls, params):
        if not params:
            return cls()
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}


def check_array_shape(arr):
    shape = arr.shape

    # Check if the shape has 4 dimensions
    if len(shape) == 4:
        out_channels, kH, KW, _ = shape
        # Check if out_channels is the largest, and kH and KW are the same
        if (out_channels >= kH) and (out_channels >= KW) and (kH == KW):
            return True
        else:
            return False
    # Check if the shape has 3 dimensions
    elif len(shape) == 3:
        _, kW, out_channels = shape
        # Check if out_channels is the largest
        if kW >= out_channels:
            return True
        else:
            return False
    else:
        return False


@mx.compile
def ensure_fused_sdpa(q, k, v, scale, mask=None):
    fused_dims = (64, 80, 128)  # supported by MLX's fused SDPA kernel
    d = q.shape[-1]
    target = next((t for t in fused_dims if d <= t), d)
    if target != d:
        pad = [(0, 0)] * (q.ndim - 1) + [(0, target - d)]
        q, k, v = mx.pad(q, pad), mx.pad(k, pad), mx.pad(v, pad)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)[
        ..., :d
    ]
