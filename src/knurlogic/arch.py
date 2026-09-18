"""The architecture layer -- the part that is actually missing.

A VQ artifact ships its own kernels: config.json names a bundled `model.py`
(the VQ runtime), so that half is correct as downloaded. What it does NOT
ship is the ARCHITECTURE model file -- qwen4_exp, qwen3_5, gemma4_text,
glm5_next -- which is a graft into mlx_lm/models/. Those are unversioned,
unpinned, and in practice they DRIFT.

Measured 2026-09-18 across one lab's two envs, both mlx-lm 0.31.3:

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

#: model_type (from config.json) -> the mlx_lm.models module that must exist.
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

#: Pinned digests. EMPTY ON PURPOSE -- populate with `knurlogic pin` against
#: an env that has been validated, and never by copying whatever is installed.
#: An unpinned architecture is reported as UNPINNED, not as OK: "it imports"
#: is not the same claim as "it is the arithmetic we measured."
PINNED_SHA256: dict = {}

#: The mlx-lm this set of architecture files was validated against.
PINNED_MLX_LM = "0.31.3"


@dataclass
class ArchStatus:
    module: str
    present: bool
    path: Path | None
    sha256: str | None
    pinned: str | None

    @property
    def state(self) -> str:
        if not self.present:
            return "MISSING"
        if self.pinned is None:
            return "UNPINNED"
        return "OK" if self.sha256 == self.pinned else "DRIFTED"


def _models_dir() -> Path | None:
    try:
        import mlx_lm.models as m
        return Path(m.__file__).parent
    except Exception:
        return None


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
    """Report the state of every architecture file this artifact needs."""
    d = models_dir or _models_dir()
    rows = []
    for mod in required_modules(model_type):
        p = (d / f"{mod}.py") if d else None
        present = bool(p and p.is_file())
        sha = (hashlib.sha256(p.read_bytes()).hexdigest() if present else None)
        rows.append(ArchStatus(mod, present, p if present else None, sha,
                               PINNED_SHA256.get(mod)))
    return rows
