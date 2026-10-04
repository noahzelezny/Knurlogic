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
import json
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


def registered(model_type: str, path=None) -> bool:
    """Has this model_type a registered vision family whose module is
    installed? Finds the module without running it, so interfaces/ can ask
    without importing mlx (the family modules do). With `path`, the
    artifact's config.json must also describe a tower: DeepSeek-V4-Flash
    and its Vision-Exp share a model_type."""
    import importlib.util
    t = FAMILIES.get(family_of(model_type))
    if t is None:
        return False
    if path is not None:
        from pathlib import Path
        try:
            config = json.loads((Path(path) / "config.json").read_text())
        except (OSError, ValueError):
            return False
        if not has_vision_config(config, path):
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


def _signatures() -> list[dict]:
    from knurlogic.engine import families
    return families.build_maps()["vision_signatures"]


#: each family's vision signature: how its config.json shows a tower
#: ("config": a nested key; "config_layers": a flat layer count > 0) and
#: where its tower's tensors live (engine/families/<family>, `vision`)
SIGNATURES: list[dict] = _signatures()


def has_vision_config(config: dict, path=None) -> bool:
    """Does this config.json describe a vision tower, by any family's
    signature? A nested config (`vision_config`, every mlx-vlm family) is
    enough; a flat layer count (DeepSeek-V4-Flash-Vision-Exp's
    `vision_n_layers > 0`) needs, with `path`, the tower's own tensors
    (`tower_in_weights`) in the artifact too: a text-only conversion keeps
    the config fields but not the tower."""
    for sig in SIGNATURES:
        if sig.get("config") and config.get(sig["config"]):
            return True
    for sig in SIGNATURES:
        key = sig.get("config_layers")
        if not key:
            continue
        try:
            if int(config.get(key) or 0) <= 0:
                continue
        except (TypeError, ValueError):
            continue
        if path is None or not sig.get("tower_in_weights") or \
                _names_tower(path, sig["tower_in_weights"]):
            return True
    return False


def _names_tower(path, prefix: str) -> bool:
    """Does the artifact's weight index (or, without one, a shard header)
    name a tensor under `prefix`?"""
    import struct
    from pathlib import Path
    root = Path(path)
    try:
        index = root / "model.safetensors.index.json"
        if index.is_file():
            names = json.loads(index.read_text()).get("weight_map", {})
            return any(k.startswith(prefix) for k in names)
        for f in sorted(root.glob("*.safetensors")):
            with open(f, "rb") as fh:
                (n,) = struct.unpack("<Q", fh.read(8))
                if any(k.startswith(prefix) for k in json.loads(fh.read(n))):
                    return True
    except (OSError, ValueError, struct.error):
        return False
    return False


def build(model_type: str, model_path: str, text_model: Any,
          config: dict | None = None):
    """The Family for a loaded model, or None when it has no vision here:
    an unregistered model_type, a family package not present, a config with
    no vision config, or the family's own build declining."""
    t = FAMILIES.get(family_of(model_type))
    if t is None:
        return None
    if config is None:
        import json
        from pathlib import Path
        p = Path(model_path) / "config.json"
        config = json.loads(p.read_text()) if p.is_file() else {}
    if not has_vision_config(config, model_path):
        return None
    fn = resolve(t)
    if fn is None:
        return None
    return fn(model_path, text_model, config)
