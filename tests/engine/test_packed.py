"""engine/model/packed: a packed layer's recipe read off its tensors."""
from types import SimpleNamespace

from mlx import nn

from knurlogic.engine.model.packed import recipe


def _shapes(rows, packed, groups):
    return {"proj.weight": SimpleNamespace(shape=(rows, packed)),
            "proj.scales": SimpleNamespace(shape=(rows, groups))}


def test_per_layer_mode_overrides_top_level():
    mod = nn.Linear(64, 32)
    w = _shapes(32, 8, 4)                     # 4-bit, 16-wide groups
    quant = {"mode": "mxfp4", "tower.proj": {"mode": "nvfp4"}}
    assert recipe(w, "proj", mod, quant, "tower.") == {
        "group_size": 16, "bits": 4, "mode": "nvfp4"}
    assert recipe(w, "proj", mod, quant)["mode"] == "mxfp4"


def test_mode_falls_back_to_bit_width():
    mod = nn.Linear(64, 32)
    assert recipe(_shapes(32, 8, 2), "proj", mod)["mode"] == "mxfp4"
    w = {**_shapes(32, 8, 2), "proj.biases": None}
    assert recipe(w, "proj", mod, {"mode": "mxfp4"})["mode"] == "affine"
