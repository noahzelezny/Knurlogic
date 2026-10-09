"""GLM-5.3 (glm5_next) loads through mlx-lm's own loader on every rank.

`mlx_lm.utils.load` resolves `(Model, ModelArgs)` from the module
registered as `mlx_lm.models.<model_type>`. Rank 0 and a pipeline follower
both reach it through `model.load.load_unlocked`, so the vendored package
must answer that contract: a follower died with "module
'mlx_lm.models.glm5_next' has no attribute 'ModelArgs'".

Driven from the real config.json (skipped where the artifact is absent),
with the depth and expert count cut so nothing heavy is built.
"""

import json
from pathlib import Path

import pytest

REAL = Path.home() / ".exo/models/zai-org--GLM-5.3-Flash-4bit/config.json"

pytestmark = pytest.mark.skipif(not REAL.is_file(),
                                reason="GLM-5.3-Flash-4bit not on this machine")


def _small(cfg: dict) -> dict:
    cfg = json.loads(json.dumps(cfg))
    t = cfg["text_config"]
    n = 4
    t["num_hidden_layers"] = n
    t["layer_types"] = t["layer_types"][:n]
    t["mlp_layer_types"] = t["mlp_layer_types"][:n]
    t["n_routed_experts"] = 8
    t["num_experts_per_tok"] = 2
    return cfg


def test_follower_resolves_rank0_class_and_builds_it():
    from mlx_lm.utils import _get_classes

    from knurlogic.engine import register
    register.register("glm5_next")
    import sys
    pkg = sys.modules["mlx_lm.models.glm5_next"]
    cfg = json.loads(REAL.read_text())
    model_cls, args_cls = _get_classes(cfg)
    assert model_cls is pkg.Model
    assert args_cls is pkg.ModelConfig
    args = args_cls.from_dict(_small(cfg))
    assert isinstance(args.text_config, pkg.TextConfig)
    assert isinstance(args.vision_config, pkg.VisionConfig)
    model = model_cls(args)
    assert len(model.layers) == 4
