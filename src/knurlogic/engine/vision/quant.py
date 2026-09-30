"""Quantize a vision module to match the checkpoint before loading it.

A converted artifact can store some vision tensors quantized. gemma e4b's
`embed_vision.embedding_projection`, for example, is 8-bit affine: its
checkpoint holds `weight`, `scales` and `biases`, while the rest of the
tower is plain. A freshly built module has plain Linear/Embedding layers
and refuses those keys ("Module does not have parameter named 'scales'").
So every layer that has a `<path>.scales` in the checkpoint is swapped for
its quantized form first, with the bits and group size the artifact's own
config.json records. mlx-lm's loader does the same for the text trunk
(`mlx_lm.utils.load_model`, its class_predicate).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.nn as nn


def artifact_quantization(model_path: str) -> dict[str, Any]:
    p = Path(model_path) / "config.json"
    if not p.is_file():
        return {}
    cfg = json.loads(p.read_text())
    return cfg.get("quantization") or cfg.get("quantization_config") or {}


def quantize_like(module: nn.Module, weights: dict[str, Any],
                  quant: dict[str, Any], prefix: str = "") -> int:
    """Quantize the layers of `module` whose `<path>.scales` is in `weights`
    (keys relative to `module`). `prefix` is the module's path in the
    artifact, used to find per-layer overrides in `quant`. Returns how many
    layers were quantized."""
    if not any(k.endswith(".scales") for k in weights):
        return 0
    n = 0

    def pick(path: str, m: nn.Module):
        nonlocal n
        if f"{path}.scales" not in weights or not hasattr(m, "to_quantized"):
            return False
        override = quant.get(prefix + path)
        n += 1
        return override if isinstance(override, dict) else True

    nn.quantize(module, group_size=quant.get("group_size", 64),
                bits=quant.get("bits", 4), mode=quant.get("mode", "affine"),
                class_predicate=pick)
    return n
