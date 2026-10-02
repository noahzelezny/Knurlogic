"""register() is public: vqlab's check-release calls it to load a
deepseek_v4 tokenizer. Breaking these breaks that release check."""
import importlib
import inspect
import sys

from knurlogic.engine import register as reg


def test_the_public_names_and_signatures_hold():
    assert {"register", "unregister"} <= set(reg.__all__)
    p = inspect.signature(reg.register).parameters
    assert p["names"].kind is inspect.Parameter.VAR_POSITIONAL
    assert p["override"].default is False
    assert not inspect.signature(reg.unregister).parameters


def test_register_deepseek_v4_makes_it_importable_and_its_config_known():
    reg.register("deepseek_v4")
    try:
        assert "deepseek_v4" in reg.available()
        m = importlib.import_module("mlx_lm.models.deepseek_v4")
        assert hasattr(m, "Model") and hasattr(m, "ModelArgs")
        from transformers import AutoConfig
        cfg = AutoConfig.for_model("deepseek_v4")
        assert cfg.max_position_embeddings          # edit 10's tokenizer gap
    finally:
        reg.unregister()
    assert "mlx_lm.models.deepseek_v4" not in sys.modules
