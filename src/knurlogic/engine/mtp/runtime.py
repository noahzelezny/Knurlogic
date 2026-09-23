"""Trunk loading for the MTP tools, across BOTH runtimes.

mlx-lm serves most families; glm5_next exists only in mlx_vlm (which is
also exo's serving path for it). One loader keeps every mtp-* entry
point runtime-agnostic: it returns (model-to-bind, tokenizer), where the
model is the object the registry contract expects — mlx-lm's Model, or
the mlx_vlm LanguageModel (never the VLM wrapper).
"""
from __future__ import annotations

import importlib
import json
import pathlib


def load_trunk(model_path, lazy: bool = False):
    model_path = pathlib.Path(model_path)
    model_type = json.load(open(model_path / "config.json")).get("model_type")
    try:
        importlib.import_module(f"mlx_lm.models.{model_type}")
        have_mlx_lm = True
    except ImportError:
        have_mlx_lm = False
    if have_mlx_lm:
        from mlx_lm.utils import load
        try:
            return load(model_path, lazy=lazy, trust_remote_code=True)
        except TypeError:  # older mlx-lm: no trust_remote_code kwarg
            return load(model_path, lazy=lazy)
    from mlx_vlm.utils import load as vlm_load
    try:
        model, processor = vlm_load(str(model_path), lazy=lazy)
        tok = getattr(processor, "tokenizer", processor)
    except OSError:
        # VQ artifacts ship no preprocessor_config.json (they are served
        # text-only), and AutoProcessor refuses to build without the image
        # half. MTP needs only the tokenizer, so load the model and the
        # tokenizer separately.
        from mlx_vlm.utils import load_model as vlm_load_model
        from transformers import AutoTokenizer
        model = vlm_load_model(model_path, lazy=lazy, trust_remote_code=True)
        tok = AutoTokenizer.from_pretrained(str(model_path))
    lang = getattr(model, "language_model", model)
    return _LogitsAdapter(lang), tok


def encode_chat(tok, text):
    """Chat-template a single user message to token ids, whichever of the
    four shapes this tokenizer's apply_chat_template returns (ids, a BatchEncoding,
    rendered string, or a list of rendered strings — bare HF tokenizers
    from mlx_vlm do the latter two)."""
    ids = tok.apply_chat_template([{"role": "user", "content": text}],
                                  add_generation_prompt=True)
    if hasattr(ids, "get") and "input_ids" in ids:  # BatchEncoding
        return list(ids["input_ids"])
    if isinstance(ids, str):
        return tok.encode(ids)
    if ids and isinstance(ids[0], str):
        return tok.encode("".join(ids))
    return ids


class _LogitsAdapter:
    """mlx_vlm LanguageModels return LanguageModelOutput(logits=...); the
    loop's contract is `model(tokens, cache=...) -> logits`. Unwrap at the
    call and delegate everything else, so capture paths (model.model.*),
    make_cache, args and lm_head all reach the real module untouched."""

    def __init__(self, lang):
        self._lang = lang

    def __call__(self, *args, **kwargs):
        out = self._lang(*args, **kwargs)
        return getattr(out, "logits", out)

    def __getattr__(self, name):
        return getattr(self._lang, name)
