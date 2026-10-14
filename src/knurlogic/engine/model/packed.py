"""Quantize a freshly built module to match packed tensors before loading
them.

A checkpoint can store some layers quantized (`<m>.weight` packed beside
`<m>.scales`, plus `<m>.biases` when affine) while a module built from its
config has plain Linear / Embedding layers that refuse those keys. Every
layer with a `<m>.scales` is swapped for its quantized form first, with the
recipe read off the tensors themselves: bits from the packed weight's width,
group size from the scales', affine when there are biases and mxfp4 /
mxfp8 otherwise. One reader for drafting heads and vision towers alike;
mlx-lm's loader does the same for the text trunk.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: the mode of a packed layer without biases, by bit width
_SCALED_MODES = {4: "mxfp4", 8: "mxfp8"}


def artifact_quantization(model_path) -> dict[str, Any]:
    """The `quantization` (or `quantization_config`) block of an artifact's
    config.json, {} without one."""
    p = Path(str(model_path)) / "config.json"
    if not p.is_file():
        return {}
    cfg = json.loads(p.read_text())
    return cfg.get("quantization") or cfg.get("quantization_config") or {}


def recipe(weights: dict, path: str, mod, quant: dict | None = None,
           prefix: str = "") -> dict | None:
    """The to_quantized kwargs the packed `path` in `weights` was written
    with, or None when it is not packed (or `mod` cannot quantize). `quant`
    is the artifact's quantization block: a layer without biases takes the
    mode of its own entry there (`prefix + path`, e.g. an nvfp4 layer under
    an mxfp4 top level), else the block's top-level mode, else mxfp4 /
    mxfp8 by bit width. `weights` may hold anything with a `.shape` (a
    header's shapes check a file without loading it)."""
    s = weights.get(f"{path}.scales")
    if s is None or not hasattr(mod, "to_quantized"):
        return None
    in_dims = int(mod.weight.shape[-1])
    bits = int(weights[f"{path}.weight"].shape[-1]) * 32 // in_dims
    gs = in_dims // int(s.shape[-1])
    if f"{path}.biases" in weights:
        kind = "affine"
    else:
        quant = quant or {}
        layer = quant.get(prefix + path)
        mode = ((layer.get("mode") if isinstance(layer, dict) else None)
                or quant.get("mode"))
        kind = (mode if mode not in (None, "affine")
                else _SCALED_MODES.get(bits))
        if kind is None:
            raise ValueError(f"{path}: {bits}-bit without biases is neither "
                             f"affine nor mxfp4/mxfp8")
    return {"group_size": gs, "bits": bits, "mode": kind}


def quantize_like(module, weights: dict, model_path=None,
                  prefix: str = "") -> int:
    """Quantize the layers of `module` that `weights` (keys relative to
    `module`) holds packed, each to its own recipe, the scaled modes read
    from the config.json of the artifact at `model_path` (`prefix`: the
    module's path there, for per-layer entries). -> how many."""
    from mlx import nn
    if not any(k.endswith(".scales") for k in weights):
        return 0
    quant = artifact_quantization(model_path) if model_path else {}
    n = 0

    def pick(path, m):
        nonlocal n
        r = recipe(weights, path, m, quant, prefix)
        n += r is not None
        return r or False

    nn.quantize(module, class_predicate=pick)
    return n
