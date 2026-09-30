"""model_type -> the family package that serves its images.

NO PACKAGE EDITS THIS FILE: the table is built from the family manifests
(engine/families/), and names "module:attr" strings that may not exist
yet; a family that creates its module makes it resolve. A module that
is not there means the capability is off (`build` -> None), so the
packages land independently and a text-only install is the default.

What is NOT swallowed: an ImportError raised from INSIDE a family module
that does exist (a bad vendored import, a missing dependency). That is a
broken build, and returning None would turn it into "this model has no
vision" -- silent, exactly the failure mode this build exists to avoid.

Stdlib only; the family module is imported only when `build` is called.
"""
from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any


#: architecture module -> "module:attr", from each family's manifest
#: (engine/families/). The attr is
#: `build(model_path: str, text_model, config: dict) -> Family | None`.
def _families() -> dict[str, str]:
    from knurlogic.engine import families
    return families.build_maps()["vision"]


FAMILIES: dict[str, str] = _families()


def resolve(target: str) -> Callable[..., Any] | None:
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


def family_of(model_type: str) -> str:
    """The registry name for a config's model_type.

    Through arch.ARCH_FOR_MODEL_TYPE, the one home for how configs spell a
    family: the released rungs report the TEXT config's type
    (`qwen3_5_text`, `gemma4_text`, ...), so looking that up directly
    reported no vision for all 20 of them. Third time this spelling bit."""
    from knurlogic.engine.arch import ARCH_FOR_MODEL_TYPE
    return ARCH_FOR_MODEL_TYPE.get(model_type, model_type)


def registered(model_type: str) -> bool:
    """Has this model_type a registered vision family whose module is
    installed? Finds the module without running it, so interfaces/ can ask
    without importing mlx (the family modules do)."""
    import importlib.util
    t = FAMILIES.get(family_of(model_type))
    if t is None:
        return False
    try:
        return importlib.util.find_spec(t.split(":")[0]) is not None
    except (ImportError, ValueError):
        return False


def has_family(model_type: str) -> bool:
    """Is there a registered family for this model_type whose package is
    present? (Imports the family module; engine-side callers only.)"""
    t = FAMILIES.get(family_of(model_type))
    return t is not None and resolve(t) is not None


def build(model_type: str, model_path: str, text_model: Any,
          config: dict | None = None):
    """The Family for a loaded model, or None when it has no vision here:
    an unregistered model_type, a family package not present, a config with
    no vision_config, or the family's own build declining."""
    t = FAMILIES.get(family_of(model_type))
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
