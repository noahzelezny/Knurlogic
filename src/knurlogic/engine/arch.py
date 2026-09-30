"""The architecture layer: which architecture module a model_type needs.

A VQ artifact ships its own kernels (a bundled `model.py`), but not the
ARCHITECTURE model file -- qwen4_exp, qwen3_5, gemma4_text, glm5_next --
which otherwise has to be grafted into mlx_lm/models/, unversioned and
unpinned. Those grafted files drift with the install's mlx-lm version, so
this module treats them as versioned source and verifies them instead.
Design: docs/design/engine.md (architecture layer).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

#: TWO HOST PACKAGES. mlx_lm.models
#: holds flat text-model files (qwen4_exp.py is 1136 lines -- the real
#: language model). mlx_vlm.models holds PACKAGES of the same names
#: (`<arch>/{__init__,config,language,vision,<arch>}.py`, ~1400 lines) where
#: the top-level file is multimodal glue and `language.py` carries the text
#: half. glm5_next exists ONLY in mlx_vlm, which is why it read as "missing"
#: when only mlx_lm was searched.
#:
#: Which host actually loads a given artifact depends on the loader, not only
#: on the config: Flash-Next declares `language_model_only: false` yet its
#: quantizer scores it through mlx_lm. So `host` here records where a
#: module LIVES; choosing the host per artifact is still an open question.
from knurlogic.engine.serve import HOST_PACKAGES

# THE FAMILY MAPS are built from each family's manifest (engine/families/),
# the one home for what a family is; nothing here is hand-kept.
from knurlogic.engine import families as _families

_MAPS = _families.build_maps()

#: Which host a module must be registered UNDER. Not cosmetic: a module's
#: relative imports resolve against its registered parent.
ARCH_HOST = _MAPS["arch_host"]


def host_for(module: str) -> str:
    return ARCH_HOST.get(module, "mlx_lm")

#: model_type (from config.json) -> the module that must exist. Multimodal
#: configs spell the text half with a `_text` suffix; every spelling is
#: listed in the family's manifest.
ARCH_FOR_MODEL_TYPE = _MAPS["arch_for_model_type"]

#: Modules that inherit another's arithmetic. A drift in the base reaches
#: every artifact of the subclass, which is how one file came to cover 11.
ARCH_DEPENDS_ON = _MAPS["arch_depends_on"]

def _load_pins() -> dict:
    """Digests recorded by `knurlogic smoke --pin` on a clean pass.

    A pin is written only when a model actually generated a token AND every
    architecture resolved from a place a downloader would have. Copying
    whatever happens to be installed is exactly the habit this replaces:
    "it imports" is not the claim, "it ran and came from here" is.
    """
    import json
    out = {}
    for f in (d / "pins.json" for d in _families.architecture_dirs()):
        if not f.is_file():
            continue
        try:
            rows = json.loads(f.read_text())
        except Exception:
            continue
        for k, v in (rows.items() if isinstance(rows, dict) else ()):
            if isinstance(v, dict) and isinstance(v.get("sha256"), str):
                out[k] = v["sha256"]  # a malformed entry costs only itself
    return out


PINNED_SHA256 = _load_pins()

#: The mlx-lm the vendored architecture set was validated against. Pinning the
#: FILE does not pin the library it calls into, so this is checked separately.
PINNED_MLX_LM = None  # set when the set is validated end to end


@dataclass
class ArchStatus:
    module: str
    present: bool
    path: Path | None
    sha256: str | None
    pinned: str | None
    vendored: bool = False

    @property
    def state(self) -> str:
        if not self.present:
            return "MISSING"
        if self.pinned is None:
            return "UNPINNED"
        return "OK" if self.sha256 == self.pinned else "DRIFTED"

    @property
    def origin(self) -> str:
        return "vendored" if self.vendored else "site-packages"


def _models_dir(host: str = "mlx_lm") -> Path | None:
    from knurlogic.engine.serve import models_module
    try:
        return Path(models_module(host).__file__).parent
    except Exception:
        return None


def _module_file(models_dir: Path, name: str) -> Path | None:
    """A module may be a flat file OR a package. mlx_vlm uses packages."""
    flat = models_dir / f"{name}.py"
    if flat.is_file():
        return flat
    pkg = models_dir / name / f"{name}.py"
    if pkg.is_file():
        return pkg
    init = models_dir / name / "__init__.py"
    return init if init.is_file() else None


def locate(name: str) -> tuple:
    """(host, path) for the first host package that has this module."""
    for host in HOST_PACKAGES:
        d = _models_dir(host)
        if d is None:
            continue
        p = _module_file(d, name)
        if p is not None:
            return host, p
    return "", None


def supported(model_type: str) -> bool:
    """Can an artifact of this model_type be run here: a family of ours
    claims it, or an installed host package has a module for it. Looks at
    files; imports no mlx."""
    if not model_type:
        return False
    if model_type in ARCH_FOR_MODEL_TYPE:
        return True
    for host in HOST_PACKAGES:
        d = _models_dir(host)
        if d is not None and _module_file(d, model_type) is not None:
            return True
    return False


def required_modules(model_type: str) -> list:
    """Every architecture module an artifact of this type needs present."""
    base = ARCH_FOR_MODEL_TYPE.get(model_type)
    if base is None:
        return []
    out = [base]
    for mod in out:  # grows as it goes: dependencies of dependencies too
        for dep in ARCH_DEPENDS_ON.get(mod, []):
            if dep not in out:
                out.append(dep)
    return out


def modules_for_artifact(a) -> list:
    """required_modules for the text model AND the top-level config type.
    A multimodal rung's config says `gemma4` on the outside and
    `gemma4_text` inside; the runtime loads the OUTER one, so a vendored
    wrapper is only used if it is registered too."""
    out = list(required_modules(a.model_type))
    outer = (getattr(a, "raw_config", None) or {}).get("model_type")
    for m in required_modules(outer) if outer else []:
        if m not in out:
            out.append(m)
    return out


def check(model_type: str, models_dir: Path | None = None) -> list:
    """Report the state of every architecture file this artifact needs.

    A file vendored in this package WINS over one installed in site-packages:
    that is the whole point of vendoring, and `register()` makes it the one
    mlx-lm actually imports. The installed copy is reported only as a
    fallback, so a user without the vendored set still gets a useful answer.
    """
    from knurlogic.engine.register import source_for

    d = models_dir or _models_dir()
    rows = []
    for mod in required_modules(model_type):
        vsrc, is_pkg = source_for(mod)
        if vsrc is not None:
            p, vendored = (vsrc.parent if is_pkg else vsrc), True
        else:
            _host, p = locate(mod)
            vendored = False
        present = bool(p and p.exists())
        if not present:
            sha = None
        elif p.is_dir():
            h = hashlib.sha256()
            for f in sorted(p.rglob("*.py")):
                h.update(str(f.relative_to(p)).encode()); h.update(f.read_bytes())
            sha = h.hexdigest()
        else:
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
        rows.append(ArchStatus(mod, present, p if present else None, sha,
                               PINNED_SHA256.get(mod), vendored))
    return rows
