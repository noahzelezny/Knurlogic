"""KNURLOGIC_PER_CHIP_ROUNDING (on by default): off pads 9-31-row quantized
matmuls to 32 rows so M3 and M4 round alike; auto is off for a job across
GPU architectures; the setting and every rank's chip pass ring-wide."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from knurlogic.engine import matmul_padding as mp
from knurlogic.tuning import knobs

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
    f = mp.padded(orig)
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
    f = mp.padded(spy)
    x = mx.random.normal((m, 512)).astype(mx.bfloat16)
    f(x, wq, s, b, transpose=True, group_size=64, bits=4)
    assert calls == [(m, 512)]


def test_install_is_idempotent(qm):
    mx = qm[0]
    orig = mx.quantized_matmul
    try:
        mp.install()
        once = mx.quantized_matmul
        mp.install()
        assert mx.quantized_matmul is once and once is not orig
        assert mp.installed()
    finally:
        mp.uninstall()
    assert mx.quantized_matmul is orig


def test_the_host_pads_only_when_rounding_is_off(qm, monkeypatch):
    from knurlogic.engine.model import state
    from knurlogic.engine.runtime.model_host import ModelHost
    monkeypatch.setattr(mp, "install", lambda: True)
    monkeypatch.setattr(mp, "installed", lambda: True)
    ModelHost(rounding=mp.resolve("auto", [M3, M4]))._pad_matmuls()
    got = state.SERVED["per_chip_rounding"]
    assert got["padded"] is True and got["per_chip_rounding"] == "off"
    hit = []
    monkeypatch.setattr(mp, "install", lambda: hit.append(1))
    ModelHost(rounding=mp.resolve("on", [M3, M4]))._pad_matmuls()
    got = state.SERVED["per_chip_rounding"]
    assert not hit and got["padded"] is False
    assert got["per_chip_rounding"] == "on"


# --- auto --------------------------------------------------------------------

def test_auto_pads_for_mixed_chips():
    r = mp.resolve("auto", [M3, M4])
    assert r["on"] and mp.describe(r) == "off (M3 Ultra + M4 Max)"
    assert mp.public(r)["per_chip_rounding"] == "off"


@pytest.mark.parametrize("chips", [None, [M3], [M3, dict(M3)],
                                   [M4, M4, M4]])
def test_auto_does_not_pad_for_one_architecture(chips):
    assert mp.resolve("auto", chips)["on"] is False


def test_off_pads_and_on_does_not():
    assert mp.resolve("off", None)["on"] is True
    assert mp.resolve("on", [M3, M4])["on"] is False
    assert mp.describe(mp.resolve("on", None)) == "on (set on)"


def test_a_chip_without_its_architecture_is_compared_by_name():
    assert mp.resolve("auto", [{"name": "M3 Ultra"},
                                  {"name": "M4 Max"}])["on"]


def test_the_default_is_on():
    assert mp.parse(None) == "on" and mp.parse("") == "on"
    assert mp.resolve(None, [M3, M4])["on"] is False


def test_a_bad_value_is_refused():
    with pytest.raises(ValueError):
        mp.parse("maybe")


# --- settings and ring-wide passthrough -------------------------------------

def test_it_is_a_model_launch_setting():
    assert "KNURLOGIC_PER_CHIP_ROUNDING" in knobs.MODEL_KNOBS
    assert knobs.KNOB_RANGE["KNURLOGIC_PER_CHIP_ROUNDING"][0] == \
        ["on", "off", "auto"]
    assert knobs.engine_settings({"KNURLOGIC_PER_CHIP_ROUNDING": "off"}) == \
        {"per_chip_rounding": "off"}
    from knurlogic.tuning.checks import clean_sets
    assert clean_sets({"KNURLOGIC_PER_CHIP_ROUNDING": "auto"})[0]


def test_every_rank_gets_the_setting_and_the_chips():
    from knurlogic.cluster.launch import check_spec, rank_argv
    chips = [M3, M4]
    for r in (0, 1):
        spec = {"rank": r, "world": 2, "split": "tensor", "link": "ring",
                "job": "ab", "hosts": ["a", "b"], "prefill_chunk": 512,
                "sets": {"KNURLOGIC_PER_CHIP_ROUNDING": "auto"}, "chips": chips}
        a = rank_argv("/m", spec, {})
        assert "KNURLOGIC_PER_CHIP_ROUNDING=auto" in a
        assert json.loads(a[a.index("--ring-chips") + 1]) == chips
    assert "chips is" in check_spec({**spec, "job": "0123456789abcdef",
                                    "nodes": [{"name": "a", "id": "a"},
                                              {"name": "b", "id": "b"}],
                                    "chips": "x"})


def test_each_rank_resolves_auto_from_the_ring_chips(monkeypatch):
    """A rank's argv -> serve.main -> run(ring=...) carries the chips and
    the setting; every rank resolves them to the same answer."""
    from knurlogic.cluster.launch import rank_argv
    from knurlogic.interfaces import serve
    got = []
    monkeypatch.setattr(serve, "run", lambda *a, **k: got.append((a, k)) or 0)
    for r in (0, 1):
        spec = {"rank": r, "world": 2, "split": "tensor", "link": "ring",
                "job": "ab", "hosts": ["a:1", "b:2"], "prefill_chunk": 512,
                "sets": {"KNURLOGIC_PER_CHIP_ROUNDING": "auto"}, "chips": [M3, M4]}
        serve.main(rank_argv("/m", spec, {})[4:])
    for a, k in got:
        launch = knobs.engine_settings(a[6])
        assert mp.resolve(launch["per_chip_rounding"],
                                 k["ring"]["chips"])["on"] is True
    assert len(got) == 2
    assert serve._ring_chips("nonsense") is None

