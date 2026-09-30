"""Tiny glm5_next fixture (tiny fixtures are one file per family).

Builds a config small enough to construct a real `VisionModel` +
`Glm5VisionFamily` with float32, seed-0 random weights, no artifact on
disk -- `fixtures_vision.tiny_config("glm5_next")` already scales the
structural fields (`REAL["glm5_next"]` in `tests/support/fixtures_vision.py`);
this file only adds what the family package itself needs to drive it.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fixtures_vision import tiny_config, tiny_image  # noqa: E402


def glm5_tiny_config(**overrides: Any) -> dict[str, Any]:
    """A tiny glm5_next config: real structure (patch 14, merge 2, silu,
    swiglu_limit 10.0), tiny sizes (`_SCALE_VISION` / `_SCALE_TEXT`)."""
    return tiny_config("glm5_next", **overrides)


def glm5_family(config: dict[str, Any] | None = None):
    """A `Glm5VisionFamily` built from a tiny config, with random float32
    tower weights (never loaded from disk -- `load_weights` is not called;
    `mx.nn.Module.__init__`'s own default init stands in for it, which is
    exactly what a "no real model" tiny-fixture gate needs: SOME weights,
    not zero weights, so a broken wire-up still produces non-constant
    output)."""
    import mlx.core as mx

    from knurlogic.engine.families.glm5.vision import Glm5VisionFamily

    mx.random.seed(0)
    cfg = config if config is not None else glm5_tiny_config()
    fam = Glm5VisionFamily(cfg)
    fam._build_tower()
    return fam


def glm5_tiny_image(seed: int = 0):
    """An image sized so it survives `preprocess`'s patch*merge rounding
    (28x28 with patch=14, merge=2: exactly one merged token)."""
    return tiny_image(w=28, h=28, seed=seed)
