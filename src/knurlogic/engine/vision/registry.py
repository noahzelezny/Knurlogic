"""model_type -> the family package that serves its images.

FIXED STRINGS, SO NO PACKAGE EDITS THIS FILE. The table names modules that
may not exist yet; P1-P3 each create theirs and it starts resolving. A
module that is not there means the capability is off (`build` -> None), so
the packages land independently and a text-only install is the default.

What is NOT swallowed: an ImportError raised from INSIDE a family module
that does exist (a bad vendored import, a missing dependency). That is a
broken build, and returning None would turn it into "this model has no
vision" -- silent, exactly the failure mode this build exists to avoid.

Stdlib only; the family module is imported only when `build` is called.
"""
from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, Optional

#: model_type (config.json top level) -> "module:attr". The attr is
#: `build(model_path: str, text_model, config: dict) -> Family | None`.
FAMILIES: Dict[str, str] = {
    "qwen3_5": "knurlogic.engine.vision.qwen:build",
    "qwen3_5_moe": "knurlogic.engine.vision.qwen:build",
    "qwen4_exp": "knurlogic.engine.vision.qwen:build",
    "gemma4": "knurlogic.engine.vision.gemma4:build",
    "glm5_next": "knurlogic.engine.vision.glm5:build",
}


def resolve(target: str) -> Optional[Callable[..., Any]]:
    """"module:attr" -> the callable, or None when the module is absent.
    Only a ModuleNotFoundError naming the target module itself (or a parent
    package) counts as absent; any other import failure propagates."""
    mod_name, _, attr = target.partition(":")
    try:
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError as e:
        missing = e.name or ""
        if missing and (mod_name == missing
                        or mod_name.startswith(missing + ".")):
            return None
        raise
    return getattr(mod, attr)


def has_family(model_type: str) -> bool:
    """Is there a registered family for this model_type whose package is
    present? (Imports the family module; engine-side callers only.)"""
    t = FAMILIES.get(model_type)
    return t is not None and resolve(t) is not None


def build(model_type: str, model_path: str, text_model: Any,
          config: Optional[dict] = None):
    """The Family for a loaded model, or None when it has no vision here:
    an unregistered model_type, a family package not present, a config with
    no vision_config, or the family's own build declining."""
    t = FAMILIES.get(model_type)
    if t is None:
        return None
    if config is None:
        import json
        from pathlib import Path
        p = Path(model_path) / "config.json"
        config = json.loads(p.read_text()) if p.is_file() else {}
    if not config.get("vision_config"):
        return None
    fn = resolve(t)
    if fn is None:
        return None
    return fn(model_path, text_model, config)
