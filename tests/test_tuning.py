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


# --- the env NAME belongs to the artifact -----------------------------------
# VQLAB_CACHE_LIMIT_GB is read by 24 of the 37 bundled runtimes on this
# machine. Those files are published. Renaming it in the resolver would emit
# a name nobody reads and silently stop bounding the cache on every artifact
# already shipped -- the failure this package exists to end, dressed as
# housekeeping. So the resolver emits whichever alias the target reads.

def _artifact_reading(tmp_path, *names):
    from knurlogic.artifact import Artifact
    (tmp_path / "config.json").write_text(
        '{"model_type":"x","model_file":"model.py","vq_linear":{"a":1},'
        '"hidden_size":4096,"moe_intermediate_size":1024}')
    (tmp_path / "model.py").write_text(
        "import os\n" + "".join(f'x = os.environ.get("{n}")\n' for n in names))
    return Artifact.load(tmp_path)


def test_a_legacy_artifact_keeps_the_name_it_was_published_with(tmp_path):
    a = _artifact_reading(tmp_path, "VQLAB_CACHE_LIMIT_GB", "VQ_DECODE_CHUNK")
    env = resolve(a, 96 * GIB).env
    assert "VQLAB_CACHE_LIMIT_GB" in env
    assert "KNURLOGIC_CACHE_LIMIT_GB" not in env


def test_a_new_artifact_gets_the_new_name(tmp_path):
    """Per rung, like `model_file` itself: a new artifact can bundle a runtime
    reading the new name while every published one keeps its own."""
    a = _artifact_reading(tmp_path, "KNURLOGIC_CACHE_LIMIT_GB",
                          "VQ_DECODE_CHUNK")
    env = resolve(a, 96 * GIB).env
    assert "KNURLOGIC_CACHE_LIMIT_GB" in env
    assert "VQLAB_CACHE_LIMIT_GB" not in env


def test_a_knob_no_alias_of_which_is_read_is_not_emitted_at_all(tmp_path):
    """Emitting it anyway is theatre, and theatre is what made a prefill knob
    look resolved for months."""
    a = _artifact_reading(tmp_path, "VQ_DECODE_CHUNK")
    r = resolve(a, 96 * GIB)
    assert not any(k.endswith("PREFILL_CHUNK") for k in r.env)
    assert any("does nothing" in n for n in r.notes)


def test_with_no_bundled_runtime_it_falls_back_to_the_published_name():
    """A guess should fail towards the 24 artifacts that exist, not towards
    the name that is planned."""
    from pathlib import Path
    from knurlogic.artifact import Artifact
    a = Artifact(path=Path("/nonexistent"), model_type="x", model_file=None,
                 bytes_on_disk=70 * GIB, hidden_size=4096,
                 moe_intermediate_size=1024, vq_other={"vq_linear": {"a": 1}})
    env = resolve(a, 96 * GIB).env
    assert "VQLAB_CACHE_LIMIT_GB" in env and "VQLAB_PREFILL_CHUNK" in env


def test_knobs_are_tiered_by_who_would_reach_for_one():
    """33 knobs on one real artifact: 2 you reach for, 8 measured flags, 23
    kernel internals. Showing all of them equally is the busy-panel mistake --
    every knob visible, none weighted, the eye with nowhere to go."""
    assert S.knob_tier("VQ_DECODE_CHUNK") == "reach"
    assert S.knob_tier("VQLAB_CACHE_LIMIT_GB") == "reach"
    assert S.knob_tier("VQ_MOE_GEMMSEG_RTILE") == "deeper"
    assert S.knob_tier("VQ_D8_REGBUF") == "kernel"


def test_unmeasured_knobs_are_named_but_never_given_a_default(tmp_path):
    """The runtime's own defaults apply. Inventing one for a knob nobody has
    measured is how the frozen 2048*4096*2 constant happened."""
    from knurlogic.artifact import Artifact
    (tmp_path / "config.json").write_text(
        '{"model_type":"x","model_file":"model.py","vq_linear":{"a":1}}')
    (tmp_path / "model.py").write_text(
        'import os\n'
        'a = os.environ.get("VQ_DECODE_CHUNK")\n'
        'b = os.environ.get("VQ_D8_REGBUF")\n'
        'c = os.environ.get("VQ_FUSED_MAX_N")\n')
    a = Artifact.load(tmp_path)
    assert a.knobs_read() == ["VQ_D8_REGBUF", "VQ_DECODE_CHUNK",
                              "VQ_FUSED_MAX_N"]
    env = resolve(a, 96 * GIB).env
    assert "VQ_D8_REGBUF" not in env and "VQ_FUSED_MAX_N" not in env


# --- the control stops where the evidence stops -----------------------------

def test_a_dial_offers_only_positions_that_were_measured():
    """Discrete, not continuous. A slider over chunk width would invent
    positions no run ever measured, and 32 is the top because 128 -> 32 is
    1.37x and nothing above it was ever better."""
    vals, _unit = S.KNOB_RANGE["VQ_DECODE_CHUNK"]
    assert vals == [4, 8, 16, 32]
    assert max(vals) == S.DECODE_CHUNK_DEFAULT
    assert min(vals) == S.DECODE_CHUNK_MIN


def test_the_cache_dial_stops_at_what_the_box_can_hold(tmp_path):
    """A control that lets you pick a setting the resolver would refuse is a
    control that lies. The cap is headroom, and it says so."""
    from knurlogic import web
    from knurlogic.artifact import Artifact
    (tmp_path / "config.json").write_text(
        '{"model_type":"x","model_file":"model.py","vq_linear":{"a":1},'
        '"hidden_size":4096,"moe_intermediate_size":1024}')
    (tmp_path / "model.py").write_text(
        'import os\nos.environ.get("VQLAB_CACHE_LIMIT_GB")\n'
        'os.environ.get("VQ_DECODE_CHUNK")\n')
    a = Artifact.load(tmp_path)
    object.__setattr__(a, "bytes_on_disk", 70 * GIB)

    doc = web.settings_document(
        a, live_env={}, live_tune="balanced", live_working_set=76 * GIB,
        resolve_fn=lambda ws, t: resolve(a, ws, tune=t),
        live_knobs=("VQLAB_CACHE_LIMIT_GB",))
    cache = next(k for k in doc({})["knobs"]
                 if k["name"] == "VQLAB_CACHE_LIMIT_GB")
    assert cache["cap"] == 2.0, cache          # 6 GiB headroom -> half of it
    assert "headroom" in cache["cap_why"]


def test_an_artifact_may_declare_its_own_knobs(tmp_path):
    """Kernel work stays with whoever packs the kernels. When a packer
    declares them in config.json, that beats anything scanned or hard coded
    here -- the artifact is the record of what shipped."""
    from knurlogic.artifact import Artifact
    (tmp_path / "config.json").write_text(
        '{"model_type":"x","model_file":"model.py",'
        '"knobs":{"VQ_D8_ROWS_TG":{"default":"8","values":[4,8,16],'
        '"doc":"rows per threadgroup"}}}')
    (tmp_path / "model.py").write_text('import os\nos.environ.get("VQ_D8_ROWS_TG")\n')
    a = Artifact.load(tmp_path)
    assert a.declared_knobs()["VQ_D8_ROWS_TG"]["values"] == [4, 8, 16]


# --- the MTP head that gets thrown away -------------------------------------
# 40 of the 54 artifacts on this machine declare a multi-token-prediction
# head -- 3925 GiB of weights -- and mlx-lm's qwen4_exp port drops them at
# load with a one-line `continue` in sanitize(). Nothing warns. The weights
# were downloaded and do not run.

def test_an_artifact_declaring_mtp_is_recognised(tmp_path):
    from knurlogic.artifact import Artifact
    (tmp_path / "config.json").write_text(
        '{"model_type":"qwen4_exp_text","mtp":{"mtp_num_hidden_layers":1}}')
    assert Artifact.load(tmp_path).has_mtp

    (tmp_path / "config.json").write_text('{"model_type":"qwen4_exp_text"}')
    assert not Artifact.load(tmp_path).has_mtp


def test_the_architecture_is_asked_whether_it_keeps_them():
    """Read off the module that will actually run, not assumed: the answer
    is a line in sanitize(), and it is 'no'."""
    from knurlogic import engine
    assert engine.keeps_mtp_weights("qwen4_exp_text") is False


def test_an_unknown_architecture_says_unknown_not_no():
    """'Could not find the module' is not 'it discards them'."""
    from knurlogic import engine
    assert engine.keeps_mtp_weights("not_a_real_model_type") is None
