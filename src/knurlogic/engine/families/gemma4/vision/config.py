"""gemma4's vision config -- the fields the tower reads, structural only.

Vendored from mlx-vlm 0.6.17 `mlx_vlm/models/gemma4/config.py::VisionConfig`
(MIT, Copyright (c) 2025 Prince Canuma), trimmed to the fields
`vision.py` actually uses (audio/video fields on the sibling `AudioConfig`
and `ModelConfig` dataclasses are out of scope -- P2 is vision only, per the
package's OWNS list). `layer_types`/`rope_parameters` post-init defaults are
kept because `vision.py` reads `rope_parameters["rope_theta"]`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from knurlogic.engine.vision._base import BaseModelConfig


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "gemma4_vision"
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 16
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
    head_dim: int = 64
    hidden_activation: str = "gelu_pytorch_tanh"
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    use_bidirectional_attention: str = "vision"
    layer_types: Optional[List[str]] = None
    rope_parameters: Optional[Dict] = None
    default_output_length: int = 280
    patch_size: int = 16
    position_embedding_size: int = 10240
    pooling_kernel_size: int = 3
    use_clipped_linears: bool = False
    standardize: bool = False

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_theta": 100.0, "rope_type": "default"}
