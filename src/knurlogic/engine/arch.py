"""The architecture layer -- the part that is actually missing.

A VQ artifact ships its own kernels: config.json names a bundled `model.py`
(the VQ runtime), so that half is correct as downloaded. What it does NOT
ship is the ARCHITECTURE model file -- qwen4_exp, qwen3_5, gemma4_text,
glm5_next -- which is a graft into mlx_lm/models/. Those are unversioned,
unpinned, and in practice they DRIFT.

Measured 2026-09-18 across one lab's two envs -- which turned out to be on
DIFFERENT mlx-lm versions (0.32.0 and 0.31.9), and that is the point rather
than a caveat: a grafted file inherits its install's version, so nothing
answers "which arithmetic is this":

    qwen4_exp    1136 vs 1138 lines   cosmetic predicate-arity shim, safe
    qwen3_5       574 vs  535 lines   QK-norm rewrite: algebraically identical,
                                      but 1.2e-02 max rel in bf16; and
                                      PipelineMixin present in only one
    gemma4_text   688 vs  675 lines   different parameter set loaded
    glm5_next    MISSING in both      3 artifacts load in neither env

Comparing a number measured in one env against the other on the qwen3_5
family (11 artifacts, since qwen3_5_moe subclasses it) is a one-harness
violation. That is the whole reason this module exists: the fix is to treat
these files as versioned source and verify them, not to graft them by hand.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

#: TWO HOST PACKAGES, and this was not obvious (2026-09-18). mlx_lm.models
#: holds flat text-model files (qwen4_exp.py is 1136 lines -- the real
#: language model). mlx_vlm.models holds PACKAGES of the same names
#: (`<arch>/{__init__,config,language,vision,<arch>}.py`, ~1400 lines) where
#: the top-level file is multimodal glue and `language.py` carries the text
#: half. glm5_next exists ONLY in mlx_vlm, which is why it read as "missing"
#: when only mlx_lm was searched.
#:
#: Which host actually loads a given artifact depends on the loader, not only
#: on the config: Flash-Next declares `language_model_only: false` yet vqlab
#: scores it through mlx_lm. So `host` here records where a module LIVES;
#: choosing the host per artifact is still an open question.
from knurlogic.engine.seam import HOST_PACKAGES

#: Which host a module must be registered UNDER. This is not cosmetic: a
#: module's relative imports resolve against its registered parent, and
#: glm5_next reaches for eight siblings (..cache, ..base, ..mla, ..mlp,
#: ..gated_delta, ..rope_utils, ..deepseek_v32.language,
#: ..deepseek_v4.hyper_connection). Registered under the wrong parent it
#: cannot import at all.
ARCH_HOST = {"glm5_next": "mlx_vlm"}


def host_for(module: str) -> str:
    return ARCH_HOST.get(module, "mlx_lm")

#: model_type (from config.json) -> the module that must exist.
#: model_type strings are suffixed `_text` on multimodal configs.
ARCH_FOR_MODEL_TYPE = {
    "qwen4_exp_text": "qwen4_exp",
    "qwen4_exp": "qwen4_exp",
    "qwen3_5_text": "qwen3_5",
    "qwen3_5": "qwen3_5",
    "qwen3_5_moe_text": "qwen3_5_moe",   # subclasses qwen3_5
    "qwen3_5_moe": "qwen3_5_moe",
    "gemma4_text": "gemma4_text",
    "glm5_next_text": "glm5_next",
    "glm5_next": "glm5_next",
}

#: Modules that inherit another's arithmetic. A drift in the base reaches
#: every artifact of the subclass, which is how one file came to cover 11.
ARCH_DEPENDS_ON = {"qwen3_5_moe": ["qwen3_5"]}

def _load_pins() -> dict:
    """Digests recorded by `knurlogic smoke --pin` on a clean pass.

    A pin is written only when a model actually generated a token AND every
    architecture resolved from a place a downloader would have. Copying
    whatever happens to be installed is exactly the habit this replaces:
    "it imports" is not the claim, "it ran and came from here" is.
    """
    from knurlogic.engine.register import ARCH_DIR
    f = ARCH_DIR / "PINS.json"
    if not f.is_file():
        return {}
    import json
    try:
        return {k: v["sha256"] for k, v in json.loads(f.read_text()).items()}
    except Exception:
        return {}


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
    from knurlogic.engine.seam import models_module
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


def required_modules(model_type: str) -> list:
    """Every architecture module an artifact of this type needs present."""
    base = ARCH_FOR_MODEL_TYPE.get(model_type)
    if base is None:
        return []
    out = [base]
    for dep in ARCH_DEPENDS_ON.get(base, []):
        if dep not in out:
            out.append(dep)
    return out


def check(model_type: str, models_dir: Path | None = None) -> list:
    """Report the state of every architecture file this artifact needs.

    A file vendored in this package WINS over one installed in site-packages:
    that is the whole point of vendoring, and `register()` makes it the one
    mlx-lm actually imports. The installed copy is reported only as a
    fallback, so a user without the vendored set still gets a useful answer.
    """
    from knurlogic.engine.register import ARCH_DIR, source_for

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
