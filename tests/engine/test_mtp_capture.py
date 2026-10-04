"""engine/mtp/capture.py: the hidden-state spy every MTP head reads
(qwen3_5, glm5, qwen4_exp and DeepSeek MTP through one capture path,
DSpark's block head through several)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")


class _Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.Identity()
        self.layers = [nn.Identity(), nn.Identity()]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


def _slot(core, path):
    return core.layers[1] if path == "layers.1" else core.norm


@pytest.mark.parametrize("path", ["norm", "layers.1"])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
def test_a_capture_closed_out_of_order_leaves_the_others_live(path, dtype):
    """Two generators on one model nest their spies; the outer one closing
    first (a collected generator) must not cut the inner one out, or its
    getter hands the next decode step a stale hidden state."""
    from contextlib import ExitStack

    from knurlogic.engine.mtp.capture import capture_input
    core = _Core()
    original = _slot(core, path)
    a, b = ExitStack(), ExitStack()
    get_a = a.enter_context(capture_input(core, path))
    get_b = b.enter_context(capture_input(core, path))
    core(mx.zeros((1, 16, 4), dtype=dtype))
    assert get_a().shape[1] == 16 and get_b().shape[1] == 16
    a.close()
    core(mx.ones((1, 2, 4), dtype=dtype))
    assert get_b().shape[1] == 2
    b.close()
    assert _slot(core, path) is original


def test_a_capture_closed_in_order_restores_the_module():
    from knurlogic.engine.mtp.capture import capture_input
    core = _Core()
    original = core.norm
    with capture_input(core, "norm") as get_a:
        with capture_input(core, "norm") as get_b:
            core(mx.ones((1, 3, 4)))
            assert get_a().shape[1] == get_b().shape[1] == 3
        core(mx.ones((1, 5, 4)))
        assert get_a().shape[1] == 5
    assert core.norm is original
