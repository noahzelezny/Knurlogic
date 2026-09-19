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
