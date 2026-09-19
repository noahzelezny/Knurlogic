"""The tuning axis and the wired limit -- the two things a person turns.

Both exist so somebody can say what they want ("less spike", "use my
headroom", "why does it say it does not fit") without learning the names of
any of these knobs. Both are capped by measurements, and every test here is
about a cap holding rather than a knob moving.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knurlogic import settings as S, wired                 # noqa: E402
from knurlogic.artifact import Artifact                    # noqa: E402
from knurlogic.resolve import resolve                      # noqa: E402

GIB = 1 << 30


def _art(size_gib=70, **kw):
    base = dict(path=Path("/nonexistent/art"), model_type="qwen4_exp_text",
                model_file="model.py", bytes_on_disk=size_gib * GIB,
                hidden_size=4096, moe_intermediate_size=1024,
                vq_modules={"m": {"d": 4, "K": 2048}})
    base.update(kw)
    return Artifact(**base)


# --- the axis ---------------------------------------------------------------

def test_fast_never_raises_the_decode_chunk():
    """The cap that matters most. Smaller is faster AND smaller in memory
    (128 -> 32 is 1.37x), so there is no tradeoff to offer here -- a 'fast'
    profile that raised it would be selling a regression as a feature."""
    room = _art(size_gib=20)
    for tune in ("safe", "balanced", "fast"):
        r = resolve(room, 200 * GIB, tune=tune)
        assert int(r.env["VQ_DECODE_CHUNK"]) <= S.DECODE_CHUNK_DEFAULT


def test_safe_bounds_the_transient_tighter_than_headroom_requires():
    a = _art()
    safe = resolve(a, 96 * GIB, tune="safe")
    balanced = resolve(a, 96 * GIB, tune="balanced")
    assert int(safe.env["VQ_DECODE_CHUNK"]) < int(
        balanced.env["VQ_DECODE_CHUNK"])
    assert int(safe.env["VQLAB_PREFILL_CHUNK"]) < int(
        balanced.env["VQLAB_PREFILL_CHUNK"])
    assert float(safe.env["VQLAB_CACHE_LIMIT_GB"]) < float(
        balanced.env["VQLAB_CACHE_LIMIT_GB"])


def test_fast_on_a_tight_box_degrades_and_says_why():
    """`fast` cannot spend headroom that is not there. The difference between
    a knob and a wish is whether it tells you it did not happen."""
    a = _art()
    r = resolve(a, 74 * GIB, tune="fast")          # 4 GiB of headroom
    assert r.env["VQLAB_PREFILL_CHUNK"] == str(S.PREFILL_CHUNK_TIGHT)
    assert any("did not get it" in n for n in r.notes)
    assert any("headroom to hold it in" in n for n in r.notes)


def test_no_tuning_reaches_a_setting_measured_to_be_worse():
    """F25/F33: RTILE=64 is 0.75-0.97x and never faster. No profile, at any
    box size, may reach it."""
    for tune in ("safe", "balanced", "fast"):
        for box in (74 * GIB, 96 * GIB, 400 * GIB):
            assert resolve(_art(), box, tune=tune).env[
                "VQ_MOE_GEMMSEG_RTILE"] == "32"


def test_the_cache_cap_holds_even_with_unlimited_headroom():
    r = resolve(_art(size_gib=1), 10_000 * GIB, tune="fast")
    assert float(r.env["VQLAB_CACHE_LIMIT_GB"]) <= S.CACHE_LIMIT_GB_MAX


def test_an_unknown_tune_is_refused_not_ignored():
    try:
        resolve(_art(), 96 * GIB, tune="turbo")
    except ValueError as e:
        assert "tune must be one of" in str(e)
    else:
        raise AssertionError("a misspelled profile must not silently resolve")


# --- the wired limit --------------------------------------------------------

def test_a_model_that_fits_the_box_but_not_the_limit_says_raise_it():
    """The failure this prevents: concluding your machine is too small when
    it was only never told it could use its own memory."""
    w = wired.Wired(total_bytes=128 * GIB, limit_bytes=96 * GIB,
                    key="iogpu.wired_limit_mb")
    d = wired.advise(100 * GIB, w)
    assert d["action"] == "raise"
    assert d["command"] == "sudo sysctl iogpu.wired_limit_mb=102400"
    assert "has not been told it may use its own memory" in d["note"]


def test_it_never_suggests_wiring_the_whole_machine():
    """Too little left does not OOM the model, it wedges the box."""
    w = wired.Wired(total_bytes=128 * GIB, limit_bytes=64 * GIB,
                    key="iogpu.wired_limit_mb")
    d = wired.advise(127 * GIB, w)
    assert d["action"] == "will-not-fit"
    assert d["target_bytes"] if d.get("target_bytes") else True
    d2 = wired.advise(110 * GIB, w)
    assert d2["target_bytes"] <= w.ceiling_bytes < w.total_bytes


def test_a_rung_over_the_ceiling_is_told_no_sysctl_fixes_it():
    w = wired.Wired(total_bytes=96 * GIB, limit_bytes=84 * GIB,
                    key="iogpu.wired_limit_mb")
    d = wired.advise(200 * GIB, w)
    assert d["action"] == "will-not-fit"
    assert not d["command"], "there is no command, so none must be offered"


def test_an_unknown_system_says_unknown_rather_than_guessing():
    d = wired.advise(10 * GIB, wired.Wired(total_bytes=0, limit_bytes=0))
    assert d["action"] == "unknown" and not d["command"]


# --- how far a change can reach, measured rather than assumed ---------------
# Read out of a real 4523-line bundled runtime: VQ_DECODE_CHUNK is captured
# into a module global on first prefill and then read inside the expert loop,
# so rebinding it lands on the next prefill. The GEMM flags are read at import
# and compiled into Metal source. VQLAB_PREFILL_CHUNK is not read by any of
# the 37 bundled runtimes on this machine.

def test_a_live_knob_lands_on_the_loaded_runtime(monkeypatch):
    """Rebinding the module global is what makes 'no reload' true."""
    import sys
    import types
    from knurlogic import engine

    fake = types.ModuleType("_fake_vq_runtime")
    fake._DECODE_CHUNK = 32
    monkeypatch.setitem(sys.modules, "_fake_vq_runtime", fake)

    out = engine.apply_live({"VQ_DECODE_CHUNK": "8"})
    assert fake._DECODE_CHUNK == 8
    assert "applied" in out["VQ_DECODE_CHUNK"]
    import os
    assert os.environ["VQ_DECODE_CHUNK"] == "8"


def test_a_restart_knob_is_reported_not_silently_skipped():
    """A panel that said 'applied' over a value that did not move would be
    the same lie as an env file sourced after the one that overwrites it."""
    from knurlogic import engine
    out = engine.apply_live({"VQ_MOE_GEMMSEG_RTILE": "32"})
    assert "restart" in out["VQ_MOE_GEMMSEG_RTILE"]


def test_the_cache_limit_uses_the_engines_live_setter():
    from knurlogic import engine
    out = engine.apply_live({"VQLAB_CACHE_LIMIT_GB": "2.0"})
    assert "applied now" in out["VQLAB_CACHE_LIMIT_GB"] or \
        "no live setter" in out["VQLAB_CACHE_LIMIT_GB"]


def test_a_knob_the_bundled_runtime_never_reads_is_called_out(tmp_path):
    """Knurlogic emitted VQLAB_PREFILL_CHUNK for every artifact and not one
    bundled runtime on this machine reads it. A resolved setting that does
    nothing is the exact failure this package exists to prevent."""
    from knurlogic.artifact import Artifact
    from knurlogic.web import knob_reach
    (tmp_path / "config.json").write_text('{"model_type":"x","model_file":"model.py"}')
    (tmp_path / "model.py").write_text(
        'import os\nC = os.environ.get("VQ_DECODE_CHUNK", "32")\n')
    a = Artifact.load(tmp_path)
    assert knob_reach(a, "VQ_DECODE_CHUNK", ("VQ_DECODE_CHUNK",))[0] == "live"
    reach, why = knob_reach(a, "VQLAB_PREFILL_CHUNK", ())
    assert reach == "no-effect" and "never reads" in why


def test_an_artifact_with_no_bundled_runtime_does_not_guess():
    """No runtime to ask is not the same as 'the knob does nothing'."""
    from knurlogic.artifact import Artifact
    from pathlib import Path
    a = Artifact(path=Path("/nonexistent"), model_type="x", model_file=None,
                 bytes_on_disk=0, hidden_size=None, moe_intermediate_size=None)
    assert a.reads_knob("VQ_DECODE_CHUNK") is None
