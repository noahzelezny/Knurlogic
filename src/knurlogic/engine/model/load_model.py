"""mlx-lm's `load_model`, with the two changes that make knurlogic the
opener of a model's weight files: `mx.load(wf)` is
`open_weight_file(wf, deferred=True)` (engine/model/weight_files.load), so
where each array's bytes come from has one home -- the file, or a peer page
on a split rank (docs/design/weight-streaming.md); and
`model.sanitize(weights)` is `sanitize(model, weights)`
(weight_files.sanitize), which records what the model's sanitize made from
those files, so a split rank can make it again from fetched bytes. Its
relative import of bitnet_quantize names mlx_lm. Everything else is mlx-lm
0.32.0's, verbatim (MIT, NOTICE).

tests/engine/test_load_model_pin.py holds it to the installed mlx-lm: the
upstream function's source must hash to UPSTREAM_SHA256, so an mlx-lm that
changes it fails there until this copy is brought up to date.
"""
# ruff: noqa -- mlx-lm's code, verbatim: kept as upstream writes it, so a
# re-copy is a plain diff against the new version
from __future__ import annotations

import glob
import importlib
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Type

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
from mlx_lm.utils import (
    _compressed_tensors_quantization,
    _get_classes,
    _transform_awq_weights,
    infer_quant_config,
    load_config,
)

from .weight_files import load as open_weight_file
from .weight_files import sanitize

#: sha256 of mlx_lm.utils.load_model's source this copy was made from
UPSTREAM_SHA256 = "1f769f19e63b4e81f3192670e4958a711fae1fd453bfb5b475f523a663c6e550"
#: the mlx-lm version it was copied from
UPSTREAM_VERSION = "0.32.0"


def load_model(
    model_path: Path,
    lazy: bool = False,
    strict: bool = True,
    model_config: Optional[Dict[str, Any]] = None,
    get_model_classes: Callable[[dict], Tuple[Type[nn.Module], Type]] = _get_classes,
    trust_remote_code: bool = False,
) -> Tuple[nn.Module, dict]:
    """
    Load and initialize the model from a given path.

    Args:
        model_path (Path): The path to load the model from.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``
        strict (bool): Whether or not to raise an exception if weights don't
            match. Default: ``True``
        model_config (dict, optional): Optional configuration parameters for the
            model. Defaults to an empty dictionary.
        get_model_classes (Callable[[dict], Tuple[Type[nn.Module], Type]], optional):
            A function that returns the model class and model args class given a config.
            Defaults to the ``_get_classes`` function.
        trust_remote_code (bool): If ``True``, allow executing a custom model
            architecture file specified by the config's ``model_file`` key.
            Default: ``False``.

    Returns:
        Tuple[nn.Module, dict[str, Any]]: The loaded and initialized model and config.

    Raises:
        FileNotFoundError: If the weight files (.safetensors) are not found.
        ValueError: If the model class or args class are not found or cannot be
            instantiated, or if the config requests a custom ``model_file`` and
            ``trust_remote_code`` is not enabled.
    """
    config = load_config(model_path)
    if model_config is not None:
        config.update(model_config)

    weight_files = glob.glob(str(model_path / "model*.safetensors"))

    if not weight_files and strict:
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    weights = {}
    for wf in weight_files:
        weights.update(open_weight_file(wf, deferred=True))

    if (model_file := config.get("model_file")) is not None:
        if not trust_remote_code:
            raise ValueError(
                f"The model at {model_path} requires importing and running a "
                f"custom module ({model_file!r}) to build its architecture. This "
                "is disabled by default. Pass trust_remote_code=True if you "
                "trust this model."
            )
        spec = importlib.util.spec_from_file_location(
            "custom_model",
            model_path / model_file,
        )
        arch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(arch)
        model_class, model_args_class = arch.Model, arch.ModelArgs
    else:
        model_class, model_args_class = get_model_classes(config=config)

    if "quantization_config" not in config:
        text_config = config.get("text_config", {})
        if "quantization_config" in text_config:
            config["quantization_config"] = text_config["quantization_config"]

    model_args = model_args_class.from_dict(config)

    model = model_class(model_args)

    if hasattr(model, "sanitize"):
        weights = sanitize(model, weights)

    def _quantize(quantization):
        def class_predicate(p, m):
            # Handle custom per layer quantizations
            if p in config["quantization"]:
                return config["quantization"][p]
            if not hasattr(m, "to_quantized"):
                return False
            if f"{p}.scales" not in weights:
                return False
            return infer_quant_config(p, m, weights)

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            mode=quantization.get("mode", "affine"),
            class_predicate=class_predicate,
        )

    if (quantization := config.get("quantization", None)) is not None:
        _quantize(quantization)

    elif quantization_config := config.get("quantization_config", False):
        # Handle legacy quantization config
        quant_method = quantization_config["quant_method"]
        if quant_method == "bitnet":
            from mlx_lm.models.bitlinear_layers import bitnet_quantize

            model = bitnet_quantize(model, quantization_config)
        elif quant_method == "mxfp4":
            quantization = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
            config["quantization"] = quantization
            config["quantization_config"] = quantization
            _quantize(quantization)
        elif quant_method == "mxfp8":
            quantization = {"group_size": 32, "bits": 8, "mode": "mxfp8"}
            config["quantization"] = quantization
            config["quantization_config"] = quantization
            _quantize(quantization)
        elif (
            quant_method == "fp8"
            and quantization_config.get("scale_fmt") == "ue8m0"
            and list(quantization_config.get("weight_block_size", ())) == [32, 32]
        ):
            # Power-of-two scales over 32x32 blocks are mxfp8, so the weights
            # stay packed. Other fp8 checkpoints dequantize in their sanitize.
            quantization = {"group_size": 32, "bits": 8, "mode": "mxfp8"}
            config["quantization"] = quantization
            config["quantization_config"] = quantization
            _quantize(quantization)
        elif quant_method == "compressed-tensors":
            quantization = _compressed_tensors_quantization(quantization_config)
            config["quantization"] = quantization
            config["quantization_config"] = quantization
            _quantize(quantization)
        elif quant_method in ("awq", "gptq"):
            # Transform AutoAWQ/GPTQ packed weights to MLX format
            weights, quantization = _transform_awq_weights(weights, quantization_config)
            config["quantization"] = quantization
            config["quantization_config"] = quantization
            _quantize(quantization)

    if config.get("quantize_activations", False):

        def _maybe_qq(m):
            if isinstance(m, nn.QuantizedLinear):
                if m.mode not in ("nvfp4", "mxfp8"):
                    raise ValueError(
                        "Mode ({m.mode}) does not support activation quantization"
                    )
                if m.get("bias", False):
                    raise ValueError(
                        "Linear layer with bias does not support activation quantization"
                    )
                out_dims, in_dims = m.weight.shape
                in_dims *= 32 // m.bits
                return nn.QQLinear(in_dims, out_dims, m.group_size, m.bits, m.mode)
            else:
                return m

        leaves = tree_map(_maybe_qq, model.leaf_modules(), is_leaf=nn.Module.is_module)

        model.update_modules(leaves)

    model.eval()
    model.load_weights(list(weights.items()), strict=strict)

    if not lazy:
        mx.eval(model.parameters())

    return model, config
