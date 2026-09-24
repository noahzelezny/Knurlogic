"""engine/families/qwen/vision/ -- images for qwen3_5, qwen3_5_moe and qwen4_exp.

  vision.py      the tower, vendored from mlx-vlm 0.6.17 qwen3_vl
  processing.py  PIL image -> pixel_values + grid, vendored (numpy, no torch)
  rope_index.py  MRoPE positions, a pure function of the whole prompt (D4)
  family.py      QwenFamily: the engine.vision.Family the serve path calls

The trunk half -- MRoPE threaded through attention as `position_ids`
[3, B, L] and per-row `rope_delta` -- lives in the architecture files
(engine/architectures/qwen3_5.py, qwen4_exp.py; qwen3_5_moe inherits).
PROVENANCE.md records every vendored line. Design: docs/design/vision.md D4.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def build(model_path: str, text_model: Any,
          config: Optional[Dict[str, Any]] = None):
    """The registry target (engine/vision/registry.py): a QwenFamily, or None
    when the config has no vision_config. Weights are NOT read here -- the
    serve path calls load_weights under the load lock."""
    if not config or not config.get("vision_config"):
        return None
    from .family import QwenFamily
    return QwenFamily(model_path, config)
