"""gemma4's own tiny-fixture builders (P2). Beside `tests/fixtures_vision.py`
(shared, P0-owned) per its own convention -- one file per family
.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import fixtures_vision as fv  # noqa: E402


def tiny_gemma4_config(**overrides: Any) -> Dict[str, Any]:
    return fv.tiny_config("gemma4", **overrides)


def tiny_text_model(text_config: Dict[str, Any]):
    """A real (registered) gemma4_text.Model at fixture size."""
    import mlx.core as mx
    from knurlogic.engine import register
    register.register("gemma4_text")
    from mlx_lm.models import gemma4_text as arch

    mx.random.seed(0)
    args = arch.ModelArgs.from_dict(text_config)
    model = arch.Model(args)
    model.set_dtype(mx.float32) if hasattr(model, "set_dtype") else None
    return model, arch


def build_gemma4_family(config: Dict[str, Any]):
    from knurlogic.engine.families.gemma4.vision import build as gemma4_build
    return gemma4_build(model_path="/nonexistent", text_model=None, config=config)


def write_tiny_tower_safetensors(path: Path, family) -> Tuple[int, Dict[str, Any]]:
    """Seed the family's tower + embed_vision with tiny random weights and
    write them to `path` as a single safetensors shard, `vision_tower.*` /
    `embed_vision.*` keyed, the way `Family.load_weights` reads them back.
    Returns (tensor count written, the flat weight dict actually used)."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    mx.random.seed(0)
    rng = np.random.default_rng(0)
    flat: Dict[str, Any] = {}
    for prefix, module in (("vision_tower", family.vision_tower),
                           ("embed_vision", family.embed_vision)):
        for k, v in tree_flatten(module.parameters()):
            flat[f"{prefix}.{k}"] = mx.array(
                (rng.standard_normal(v.shape) * 0.02).astype(np.float32))
    mx.save_safetensors(str(path), flat)
    return len(flat), flat
