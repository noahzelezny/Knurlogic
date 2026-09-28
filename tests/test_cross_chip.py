"""KNURLOGIC_CROSS_CHIP (off by default): 9-31-row quantized matmuls padded to 32 rows so
M3 and M4 round alike; auto on for a job across GPU architectures, and
the setting and every rank's chip passed ring-wide."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from knurlogic.engine import crosschip
from knurlogic.tuning import settings as S

M3 = {"name": "M3 Ultra", "arch": "applegpu_g15d"}
M4 = {"name": "M4 Max", "arch": "applegpu_g16s"}


# --- the wrapper ---------------------------------------------------------------

@pytest.fixture
def qm():
    mx = pytest.importorskip("mlx.core")
    mx.random.seed(0)
    w = mx.random.normal((256, 512)).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, group_size=64, bits=4)
    return mx, wq, s, b


@pytest.mark.parametrize("m", [10, 20, 27, 31])
def test_padded_rows_equal_a_real_32_row_call(qm, m):
    """Row independence: rows of the padded call are those rows of a real
    32-row call (the qmm kernel), bit for bit."""
    mx, wq, s, b = qm
    orig = mx.quantized_matmul
    x32 = mx.random.normal((32, 512)).astype(mx.bfloat16)
    ref = orig(x32, wq, s, b, transpose=True, group_size=64, bits=4)
    f = crosschip.padded(orig)
    got = f(x32[:m][None], wq, s, b, transpose=True, group_size=64, bits=4)
    assert got.shape == (1, m, 256)
    assert mx.array_equal(got[0], ref[:m]).item()


@pytest.mark.parametrize("m", [1, 8, 32, 64])
def test_other_row_counts_are_untouched(qm, m):
    mx, wq, s, b = qm
    calls = []

    def spy(x, *a, **k):
        calls.append(x.shape)
        return mx.quantized_matmul(x, *a, **k)
    f = crosschip.padded(spy)
    x = mx.random.normal((m, 512)).astype(mx.bfloat16)
    f(x, wq, s, b, transpose=True, group_size=64, bits=4)
    assert calls == [(m, 512)]


def test_install_is_idempotent(qm):
    mx = qm[0]
    orig = mx.quantized_matmul
    try:
        crosschip.install()
        once = mx.quantized_matmul
        crosschip.install()
        assert mx.quantized_matmul is once and once is not orig
        assert crosschip.installed()
    finally:
        crosschip.uninstall()
    assert mx.quantized_matmul is orig


def test_the_host_installs_it_only_when_on(qm, monkeypatch):
    from knurlogic.engine.runtime.host import ModelHost
    from knurlogic.engine.serve import state
    monkeypatch.setattr(crosschip, "install", lambda: True)
    monkeypatch.setattr(crosschip, "installed", lambda: True)
    ModelHost(cross_chip=crosschip.resolve("auto", [M3, M4]))._cross_chip()
    assert state.SERVED["cross_chip"]["on"] is True
    hit = []
    monkeypatch.setattr(crosschip, "install", lambda: hit.append(1))
    ModelHost(cross_chip=crosschip.resolve("off", [M3, M4]))._cross_chip()
    assert not hit and state.SERVED["cross_chip"]["on"] is False


# --- auto --------------------------------------------------------------------

def test_auto_is_on_for_mixed_chips():
    r = crosschip.resolve("auto", [M3, M4])
    assert r["on"] and crosschip.describe(r) == "on (M3 Ultra + M4 Max)"


@pytest.mark.parametrize("chips", [None, [M3], [M3, dict(M3)],
                                   [M4, M4, M4]])
def test_auto_is_off_for_one_architecture(chips):
    assert crosschip.resolve("auto", chips)["on"] is False


def test_on_and_off_force():
    assert crosschip.resolve("on", None)["on"] is True
    assert crosschip.resolve("off", [M3, M4])["on"] is False


def test_a_chip_without_its_architecture_is_compared_by_name():
    assert crosschip.resolve("auto", [{"name": "M3 Ultra"},
                                  {"name": "M4 Max"}])["on"]


def test_the_default_is_off():
    assert crosschip.parse(None) == "off" and crosschip.parse("") == "off"
    assert crosschip.resolve(None, [M3, M4])["on"] is False


def test_a_bad_value_is_refused():
    with pytest.raises(ValueError):
        crosschip.parse("maybe")


# --- settings and ring-wide passthrough -------------------------------------

def test_it_is_a_model_launch_setting():
    assert "KNURLOGIC_CROSS_CHIP" in S.MODEL_KNOBS
    assert S.KNOB_RANGE["KNURLOGIC_CROSS_CHIP"][0] == ["off", "on", "auto"]
    assert S.engine_settings({"KNURLOGIC_CROSS_CHIP": "on"}) == \
        {"cross_chip": "on"}
    from knurlogic.interfaces.page.server import clean_sets
    assert clean_sets({"KNURLOGIC_CROSS_CHIP": "auto"})[0]


def test_every_rank_gets_the_setting_and_the_chips():
    from knurlogic.cluster.launch import check_spec, rank_argv
    chips = [M3, M4]
    for r in (0, 1):
        spec = {"rank": r, "world": 2, "split": "tensor", "link": "ring",
                "job": "ab", "hosts": ["a", "b"], "prefill_chunk": 512,
                "sets": {"KNURLOGIC_CROSS_CHIP": "auto"}, "chips": chips}
        a = rank_argv("/m", spec, {})
        assert "KNURLOGIC_CROSS_CHIP=auto" in a
        assert json.loads(a[a.index("--ring-chips") + 1]) == chips
    assert "chips is" in check_spec({**spec, "job": "0123456789abcdef",
                                    "nodes": [{"name": "a", "id": "a"}, {"name": "b", "id": "b"}],
                                    "chips": "x"})


def test_each_rank_resolves_auto_from_the_ring_chips(monkeypatch):
    """A rank's argv -> serve.main -> run(ring=...) carries the chips and
    the setting; every rank resolves them to the same answer."""
    from knurlogic.interfaces import serve
    from knurlogic.cluster.launch import rank_argv
    got = []
    monkeypatch.setattr(serve, "run", lambda *a, **k: got.append((a, k)) or 0)
    for r in (0, 1):
        spec = {"rank": r, "world": 2, "split": "tensor", "link": "ring",
                "job": "ab", "hosts": ["a:1", "b:2"], "prefill_chunk": 512,
                "sets": {"KNURLOGIC_CROSS_CHIP": "auto"}, "chips": [M3, M4]}
        serve.main(rank_argv("/m", spec, {})[4:])
    for a, k in got:
        launch = S.engine_settings(a[6])
        assert crosschip.resolve(launch["cross_chip"],
                                 k["ring"]["chips"])["on"] is True
    assert len(got) == 2
    assert serve._ring_chips("nonsense") is None
