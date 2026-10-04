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
    signature, that this artifact can serve? A nested config
    (`vision_config`, every mlx-vlm family) or a flat layer count
    (DeepSeek-V4-Flash-Vision-Exp's `vision_n_layers > 0`); with `path`,
    the artifact's weights must carry the vision tensors too
    (`vision_weights`): a text-only conversion keeps the config fields but
    not the tower."""
    return vision_weights(config, path)["state"] == "full"


#: why the picker's vision switch is grayed out, and why an image request
#: is refused, for a vision family's conversion with no vision weights
NO_VISION_WEIGHTS = "this conversion has no vision weights"


def _config_signature(config: dict) -> dict | None:
    """The signature whose config fields this config.json carries."""
    for sig in SIGNATURES:
        if sig.get("config") and config.get(sig["config"]):
            return sig
    for sig in SIGNATURES:
        key = sig.get("config_layers")
        if not key:
            continue
        try:
            if int(config.get(key) or 0) > 0:
                return sig
        except (TypeError, ValueError):
            continue
    return None


def vision_weights(config: dict, path=None) -> dict:
    """What the artifact's WEIGHTS say about the vision its config.json
    describes. The one place that decides, from the safetensors index (or
    the shard headers), family-agnostic through each family's signature:

    - "none": the config describes no tower (not a vision model);
    - "full": the tower's tensors (the signature's `tower_in_weights`, or
      any of its `tower_prefixes`) and every vision-only tensor of the
      trunk (`trunk_keys`, name suffixes) are there; also the answer
      without `path`, and for a nested vision_config whose weights
      cannot be listed;
    - "text_only": none of them are there. The model loads as its text
      model: `text_config` is the config overlay that builds it so (a
      flat layer count set to 0); images are refused with `why`;
    - "partial": some are there and some are not. Refused, `why` says
      which are missing.

    Returns {"state", "why", "text_config"}."""
    sig = _config_signature(config)
    if sig is None:
        return {"state": "none", "why": "", "text_config": {}}
    full = {"state": "full", "why": "", "text_config": {}}
    if path is None:
        return full
    key = sig.get("config_layers")
    text_only = {"state": "text_only", "why": NO_VISION_WEIGHTS,
                 "text_config": {key: 0} if key else {}}
    names = weight_names(path)
    if names is None:
        # a flat layer count alone is no tower (a text-only conversion
        # keeps it); a nested vision_config is taken at its word
        return full if sig.get("config") else text_only
    prefixes = tuple(sig.get("tower_prefixes") or ())
    if sig.get("config"):
        # a nested vision_config says nothing of WHICH family it is (Qwen's
        # and GLM's are both `vision_config`, and the first signature
        # matched is Qwen's): any family's tower names under it count
        prefixes = tuple(dict.fromkeys(
            p for s in SIGNATURES if s.get("config") == sig["config"]
            for p in s.get("tower_prefixes") or ()))
    need = sig.get("tower_in_weights")
    under = any(k.startswith(prefixes) for k in names) if prefixes else False
    tower = any(k.startswith(need) for k in names) if need else under
    trunk = {s: any(k.endswith(s) for k in names)
             for s in sig.get("trunk_keys") or ()}
    if tower and all(trunk.values()):
        return full
    if not (tower or under or any(trunk.values())):
        return text_only
    if not (tower or under) and trunk and all(trunk.values()):
        # no tower, but every vision-only trunk tensor kept (a conversion
        # that dropped the tower alone): the trunk is built as the config
        # says, so those load, and text never reads them; images refused
        return {**text_only, "text_config": {}}
    missing = ([f"the vision tower ({need or ', '.join(prefixes)})"]
               if not tower else [])
    missing += [s.lstrip(".") for s, ok in trunk.items() if not ok]
    return {"state": "partial", "text_config": {},
            "why": ("this conversion has only part of its vision weights "
                    f"(missing: {'; '.join(missing)}); convert it again "
                    "with all of them, or with none to serve it text only")}


def unavailable_why(model_type: str, path) -> str:
    """Why a model whose family reads images has no vision in THIS
    artifact (its config.json describes a tower its weights do not
    carry), or "": vision there, or not a vision model at all. The picker
    grays its vision switch with it; the MCP says it beside
    vision_capable."""
    from pathlib import Path
    if FAMILIES.get(family_of(model_type)) is None:
        return ""
    try:
        config = json.loads((Path(path) / "config.json").read_text())
    except (OSError, ValueError):
        return ""
    v = vision_weights(config, path)
    return v["why"] if v["state"] in ("text_only", "partial") else ""


def weight_names(path) -> list[str] | None:
    """Every tensor name the artifact's weights hold: its index's
    weight_map, or without one each shard's header. None when they cannot
    be read."""
    import struct
    from pathlib import Path
    root = Path(path)
    try:
        index = root / "model.safetensors.index.json"
        if index.is_file():
            return list(json.loads(index.read_text()).get("weight_map", {}))
        shards = sorted(root.glob("*.safetensors"))
        if not shards:
            return None
        names: list[str] = []
        for f in shards:
            with open(f, "rb") as fh:
                (n,) = struct.unpack("<Q", fh.read(8))
                names += [k for k in json.loads(fh.read(n))
                          if k != "__metadata__"]
        return names
    except (OSError, ValueError, struct.error):
        return None


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
