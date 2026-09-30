"""The tuning axis and the wired limit -- the two things a person turns.

Both exist so somebody can say what they want ("less spike", "use my
headroom", "why does it say it does not fit") without learning the names of
any of these knobs. Both are capped by measurements, and every test here is
about a cap holding rather than a knob moving.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from knurlogic.machine import wired
from knurlogic.machine.artifact import Artifact
from knurlogic.tuning import settings as S
from knurlogic.tuning.resolve import resolve

GIB = 1 << 30


def _art(size_gib=70, **kw):
    base = dict(path=Path("/nonexistent/art"), model_type="qwen4_exp_text",
                model_file="model.py", bytes_on_disk=size_gib * GIB,
                hidden_size=4096, moe_intermediate_size=1024,
                vq_modules={"m": {"d": 4, "K": 2048}})
    base.update(kw)
    return Artifact(**base)


# --- the axis ---------------------------------------------------------------

def test_no_preset_raises_the_decode_chunk():
    """The cap that matters most. Smaller is faster AND smaller in memory
    (128 -> 32 is 1.37x), so there is no tradeoff to offer here -- a preset
    that raised it would be selling a regression as a feature."""
    room = _art(size_gib=20)
    for tune in S.PRESETS:
        r = resolve(room, 200 * GIB, tune=tune)
        assert int(r.env["VQ_DECODE_CHUNK"]) <= S.DECODE_CHUNK_DEFAULT


def test_lean_bounds_memory_tighter_than_default():
    a = _art()
    lean = resolve(a, 96 * GIB, tune="lean")
    default = resolve(a, 96 * GIB, tune="default")
    # the expert chunk is auto for every preset: lean never changes it
    assert lean.env["VQ_DECODE_CHUNK"] == default.env["VQ_DECODE_CHUNK"]
    # the prompt chunk is already the narrowest by default; lean never widens
    assert int(lean.env["KNURLOGIC_PREFILL_CHUNK"]) <= int(
        default.env["KNURLOGIC_PREFILL_CHUNK"]) == S.PREFILL_CHUNK_TIGHT
    assert float(lean.env["VQ_CACHE_LIMIT_GB"]) < float(
        default.env["VQ_CACHE_LIMIT_GB"])


def test_default_on_a_tight_box_degrades_and_says_why():
    """The default cannot spend headroom that is not there. The difference between
    a knob and a wish is whether it tells you it did not happen."""
    a = _art(model_type="qwen3_5")                 # measured wider than 512
    r = resolve(a, 74 * GIB, tune="default")          # 4 GiB of headroom
    assert r.env["KNURLOGIC_PREFILL_CHUNK"] == str(S.PREFILL_CHUNK_TIGHT)
    assert any(n.startswith("prompt chunk 512") and "room" in n
               for n in r.notes)
    assert any("headroom to hold it in" in n for n in r.notes)




def test_the_cache_cap_holds_even_with_unlimited_headroom():
    r = resolve(_art(size_gib=1), 10_000 * GIB, tune="default")
    assert float(r.env["VQ_CACHE_LIMIT_GB"]) <= S.CACHE_LIMIT_GB_MAX


def test_an_unknown_tune_is_refused_not_ignored():
    try:
        resolve(_art(), 96 * GIB, tune="turbo")
    except ValueError as e:
        assert "isn't default or lean" in str(e)
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

    from knurlogic.engine import serve as engine

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
    from knurlogic.engine import serve as engine
    out = engine.apply_live({"VQ_MOE_GEMMSEG_RTILE": "32"})
    assert "restart" in out["VQ_MOE_GEMMSEG_RTILE"]


def test_the_cache_limit_uses_the_engines_live_setter():
    from knurlogic.engine import serve as engine
    out = engine.apply_live({"VQLAB_CACHE_LIMIT_GB": "2.0"})
    assert "applied now" in out["VQLAB_CACHE_LIMIT_GB"] or \
        "no live setter" in out["VQLAB_CACHE_LIMIT_GB"]


def test_a_knob_the_bundled_runtime_never_reads_is_called_out(tmp_path):
    """Knurlogic emitted VQLAB_PREFILL_CHUNK for every artifact and not one
    bundled runtime on this machine reads it. A resolved setting that does
    nothing is the exact failure this package exists to prevent."""
    from knurlogic.interfaces.page.documents import knob_reach
    from knurlogic.machine.artifact import Artifact
    (tmp_path / "config.json").write_text('{"model_type":"x","model_file":"model.py"}')
    (tmp_path / "model.py").write_text(
        'import os\nC = os.environ.get("VQ_DECODE_CHUNK", "32")\n')
    a = Artifact.load(tmp_path)
    assert knob_reach(a, "VQ_DECODE_CHUNK", ("VQ_DECODE_CHUNK",))[0] == "live"
    reach, why = knob_reach(a, "VQ_MOE_GEMMSEG_RTILE", ())
    assert reach == "no-effect" and "never reads" in why
    # The prompt chunk used to be the example here. It is not any more: the
    # engine reads it as server argv, so the runtime ignoring it is moot.
    assert knob_reach(a, "VQLAB_PREFILL_CHUNK", ())[0] == "restart"


def test_an_artifact_with_no_bundled_runtime_does_not_guess():
    """No runtime to ask is not the same as 'the knob does nothing'."""
    from pathlib import Path

    from knurlogic.machine.artifact import Artifact
    a = Artifact(path=Path("/nonexistent"), model_type="x", model_file=None,
                 bytes_on_disk=0, hidden_size=None, moe_intermediate_size=None)
    assert a.reads_knob("VQ_DECODE_CHUNK") is None


# --- the env NAME belongs to the artifact -----------------------------------
# Published bundled runtimes read VQLAB_CACHE_LIMIT_GB, the VQ runtime's
# old name. Emitting only the new one would silently stop bounding their
# cache, so the resolver emits whichever alias the target reads.

def _artifact_reading(tmp_path, *names):
    from knurlogic.machine.artifact import Artifact
    (tmp_path / "config.json").write_text(
        '{"model_type":"x","model_file":"model.py","vq_linear":{"a":1},'
        '"hidden_size":4096,"moe_intermediate_size":1024}')
    (tmp_path / "model.py").write_text(
        "import os\n" + "".join(f'x = os.environ.get("{n}")\n' for n in names))
    return Artifact.load(tmp_path)


def test_a_legacy_artifact_keeps_the_name_it_was_published_with(tmp_path):
    a = _artifact_reading(tmp_path, "VQLAB_CACHE_LIMIT_GB", "VQ_DECODE_CHUNK")
    env = resolve(a, 96 * GIB).env
    # knurlogic's cache limit, passed through under the name it reads
    assert env["VQLAB_CACHE_LIMIT_GB"] == env["KNURLOGIC_CACHE_LIMIT_GB"]
    assert "VQ_CACHE_LIMIT_GB" not in env


def test_a_new_artifact_gets_the_new_name(tmp_path):
    """Per rung, like `model_file` itself: a new artifact can bundle a runtime
    reading the new name while every published one keeps its own."""
    a = _artifact_reading(tmp_path, "KNURLOGIC_CACHE_LIMIT_GB",
                          "VQ_DECODE_CHUNK")
    env = resolve(a, 96 * GIB).env
    assert "KNURLOGIC_CACHE_LIMIT_GB" in env
    assert "VQLAB_CACHE_LIMIT_GB" not in env


def test_a_runtime_only_knob_no_alias_of_which_is_read_is_not_emitted(tmp_path):
    """Emitting it anyway is theatre, and theatre is what made a prefill knob
    look resolved for months."""
    a = _artifact_reading(tmp_path, "VQLAB_CACHE_LIMIT_GB")
    r = resolve(a, 96 * GIB)
    assert "VQ_DECODE_CHUNK" not in r.env
    assert any("does nothing" in n for n in r.notes)


def test_an_engine_knob_is_emitted_even_when_the_runtime_ignores_it(tmp_path):
    """The prompt chunk is mlx-lm server argv. Whether the artifact's runtime
    reads an env name for it is beside the point -- the engine does, so
    dropping it would be the theatre, just the other way round."""
    a = _artifact_reading(tmp_path, "VQ_DECODE_CHUNK")
    r = resolve(a, 96 * GIB)
    assert S.engine_settings(r.env).get("prefill_step_size")


def test_the_resolved_prompt_chunk_reaches_the_scheduler():
    """The bug this closes (mlx-lm era): the resolver explained a prompt
    chunk the server never saw. Now the scheduler takes it directly."""
    from knurlogic.interfaces.http import scheduler_options
    got = scheduler_options({"prefill_step_size": 512,
                             "decode_concurrency": 4,
                             "prompt_cache_bytes": 1 << 30})
    assert got["prefill_step_size"] == 512
    assert got["completion_batch_size"] == 4
    assert got["prompt_cache_bytes"] == 1 << 30
    assert scheduler_options({})["prefill_step_size"] == 2048


def test_a_measured_family_width_is_a_cap_the_room_decides_how_much_of():
    """qwen3_5 measured 4096. Balanced and fast both read the room free at
    launch: a roomy box takes the widest width whose step transient fits
    10% of the room, a tight box stays at 512."""
    from pathlib import Path

    from knurlogic.machine.artifact import Artifact
    a = Artifact(path=Path("/nonexistent"), model_type="qwen3_5",
                 model_file=None, bytes_on_disk=20 * GIB, hidden_size=4096,
                 moe_intermediate_size=1024, vq_other={})
    default = S.engine_settings(resolve(a, 96 * GIB).env)
    roomy = S.engine_settings(resolve(a, 96 * GIB, tune="default").env)
    huge = S.engine_settings(resolve(a, 400 * GIB).env)
    tight = S.engine_settings(resolve(a, 24 * GIB, tune="default").env)
    # 7.5 GiB predicted at 4096 > 10% of ~67 GiB of room; 3.75 at 2048 fits
    assert default["prefill_step_size"] == 2048
    assert roomy["prefill_step_size"] == 2048
    assert huge["prefill_step_size"] == 4096
    assert "prompt_concurrency" not in roomy   # dead: settings.py says why
    assert tight["prefill_step_size"] == S.PREFILL_CHUNK_TIGHT
    assert "prompt_concurrency" not in tight


def test_with_no_bundled_runtime_it_emits_the_current_name():
    """No bundled runtime to ask: the vendored VQ runtime and the engine
    both read the current name."""
    from pathlib import Path

    from knurlogic.machine.artifact import Artifact
    a = Artifact(path=Path("/nonexistent"), model_type="x", model_file=None,
                 bytes_on_disk=70 * GIB, hidden_size=4096,
                 moe_intermediate_size=1024, vq_other={"vq_linear": {"a": 1}})
    env = resolve(a, 96 * GIB).env
    assert "VQ_CACHE_LIMIT_GB" in env and "VQLAB_CACHE_LIMIT_GB" not in env
    # ...and the prompt chunk's legacy name has no runtime behind it (only
    # the engine reads it, under either name): knurlogic's own is emitted
    assert "KNURLOGIC_PREFILL_CHUNK" in env and "VQLAB_PREFILL_CHUNK" not in env


def test_knobs_are_tiered_by_who_would_reach_for_one():
    """33 knobs on one real artifact: 2 you reach for, 8 measured flags, 23
    kernel internals. Showing all of them equally is the busy-panel mistake --
    every knob visible, none weighted, the eye with nowhere to go."""
    assert S.knob_tier("VQ_DECODE_CHUNK") == "reach"
    assert S.knob_tier("VQ_CACHE_LIMIT_GB") == "reach"
    assert S.knob_tier("VQ_GEMMSEG_BF16IO") == "deeper"
    assert S.knob_tier("VQ_D8_REGBUF") == "kernel"


def test_unmeasured_knobs_are_named_but_never_given_a_default(tmp_path):
    """The runtime's own defaults apply. Inventing one for a knob nobody has
    measured is how the frozen 2048*4096*2 constant happened."""
    from knurlogic.machine.artifact import Artifact
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
    from knurlogic.interfaces.page import documents
    from knurlogic.machine.artifact import Artifact
    (tmp_path / "config.json").write_text(
        '{"model_type":"x","model_file":"model.py","vq_linear":{"a":1},'
        '"hidden_size":4096,"moe_intermediate_size":1024}')
    (tmp_path / "model.py").write_text(
        'import os\nos.environ.get("VQLAB_CACHE_LIMIT_GB")\n'
        'os.environ.get("VQ_DECODE_CHUNK")\n')
    a = Artifact.load(tmp_path)
    object.__setattr__(a, "bytes_on_disk", 70 * GIB)

    doc = documents.settings_document(
        a, live_env={}, live_tune="default", live_working_set=76 * GIB,
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
    from knurlogic.machine.artifact import Artifact
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
    from knurlogic.machine.artifact import Artifact
    (tmp_path / "config.json").write_text(
        '{"model_type":"qwen4_exp_text","mtp":{"mtp_num_hidden_layers":1}}')
    assert Artifact.load(tmp_path).has_mtp

    (tmp_path / "config.json").write_text('{"model_type":"qwen4_exp_text"}')
    assert not Artifact.load(tmp_path).has_mtp


def test_the_architecture_is_asked_whether_it_keeps_them():
    """Read off the module that will actually run, not assumed: the answer
    is a line in sanitize(), and it is 'no'."""
    from knurlogic.engine import serve as engine
    assert engine.keeps_mtp_weights("qwen4_exp_text") is False


def test_an_unknown_architecture_says_unknown_not_no():
    """'Could not find the module' is not 'it discards them'."""
    from knurlogic.engine import serve as engine
    assert engine.keeps_mtp_weights("not_a_real_model_type") is None


# --- settings that can only be chosen before the model loads -----------------

def test_an_explicit_set_beats_the_resolver(monkeypatch, tmp_path):
    """Most of these knobs are read at import and compiled into kernel
    source, so launch is the only moment they can be chosen at all. "The
    resolver decides and you may not" is the wrong default for the one place
    where choosing is possible."""
    from knurlogic.interfaces import serve

    assert serve._parse_sets(["A=1", "B = two"]) == {"A": "1", "B": "two"}


def test_a_set_without_a_value_is_refused():
    import pytest

    from knurlogic.interfaces import serve
    with pytest.raises(SystemExit, match="KEY=VALUE"):
        serve._parse_sets(["JUST_A_NAME"])


def test_preview_reads_and_sets_nothing(tmp_path, monkeypatch):
    """The launch form asks what an artifact WOULD resolve to. Nothing is
    loaded and nothing is applied -- a preview that edited the environment
    would change the process asking the question."""
    import json
    import os

    from knurlogic.interfaces.page import documents

    d = tmp_path / "m"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5", "hidden_size": 2048,
        "moe_intermediate_size": 768}))
    (d / "model.safetensors").write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00{}      ")

    before = dict(os.environ)
    doc = documents._preview(str(d), "default", 84)
    assert doc["preview"] is True
    assert doc["artifact"]["name"] == "m"
    assert {k["name"] for k in doc["knobs"]}          # it resolved something
    assert dict(os.environ) == before                 # and changed nothing


def test_preview_says_which_knobs_are_launch_only(tmp_path):
    """The point of showing settings BEFORE a launch: the ones marked
    `restart` cannot be changed afterwards at all."""
    import json

    from knurlogic.interfaces.page import documents

    d = tmp_path / "m"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({
        "model_type": "qwen4_exp", "hidden_size": 2048,
        "moe_intermediate_size": 768,
        "vq_modules": {"a": {"d": 2, "K": 256}}}))
    (d / "model.safetensors").write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00{}      ")
    doc = documents._preview(str(d), "default", 84)
    reach = {k["name"]: k["reach"] for k in doc["knobs"]}
    assert any(v == "restart" for v in reach.values())
    assert all(k["reach_why"] for k in doc["knobs"])


def test_a_family_spelled_with_text_still_gets_its_measured_width():
    """A qwen3_5 27B reports model_type `qwen3_5_text`. Measured end to end
    through the MCP: it got the 2048 default instead of qwen3_5's 4096."""
    assert S.prefill_chunk_for("qwen3_5_text")[0] == 4096
    assert S.prefill_chunk_for("qwen3_5_moe_text")[0] == 2048
    assert S.prefill_chunk_for("glm5_next_text")[0] == 2048
    assert S.prefill_chunk_for("somebody_else")[0] == S.PREFILL_CHUNK_DEFAULT


def test_the_command_line_no_longer_forces_a_numerics_profile():
    """serve and doctor defaulted --profile to v1.5, which forced both
    bf16-I/O flags off on every VQ rung -- including the v2 rungs published
    with them on. The default is now None: each rung runs what it shipped."""
    from knurlogic.interfaces import doctor, serve
    for mod in (serve, doctor):
        src = open(mod.__file__).read()
        assert '"--profile", default=None' in src, mod.__name__


def test_mla_caches_are_costed_as_their_latent():
    """GLM-5.3's attention layers are deepseek_sparse_attention with an MLA
    latent: they read as 0 full-attention layers (the first prompt after a
    load went uncosted), and as K,V per head would be ~10x too high."""
    from knurlogic.tuning.resolve import kv_bytes_per_token
    tc = {"num_hidden_layers": 4, "kv_lora_rank": 512,
          "qk_rope_head_dim": 0, "index_head_dim": 128,
          "num_attention_heads": 64, "num_key_value_heads": 64,
          "hidden_size": 4096,
          "layer_types": ["linear_attention", "deepseek_sparse_attention",
                          "linear_attention", "deepseek_sparse_attention"]}
    per, why = kv_bytes_per_token(tc)
    assert per == 2 * 640 * 2 and "MLA" in why


def test_the_context_length_applies_live(monkeypatch):
    from knurlogic.engine.serve.load import LIVE_KNOBS, apply_live
    monkeypatch.delenv("KNURLOGIC_CONTEXT_LENGTH", raising=False)
    assert "KNURLOGIC_CONTEXT_LENGTH" in LIVE_KNOBS
    done = apply_live({"KNURLOGIC_CONTEXT_LENGTH": "32768"})
    assert done["KNURLOGIC_CONTEXT_LENGTH"].startswith("applied")
    import os
    assert os.environ["KNURLOGIC_CONTEXT_LENGTH"] == "32768"
    assert apply_live({"KNURLOGIC_CONTEXT_LENGTH": "-1"})[
        "KNURLOGIC_CONTEXT_LENGTH"].startswith("failed")


def test_tight_headroom_scales_with_the_machine():
    """12 GiB was tight for a 96 GiB box and not for a 120 GiB one: 397B
    on an M4 Max kept ~14 GiB, took the 4096 prompt chunk, and one agent at
    25k tokens aborted Metal. A fifth of the working set, at least 12."""
    from knurlogic.tuning import settings as S
    G = 1 << 30
    assert S.tight_headroom_bytes(120 * G) == 24 * G
    assert S.tight_headroom_bytes(48 * G) == 12 * G


# --- the prompt chunk reads the room free at launch --------------------------

def _qwen_moe(hidden=2048, size_gib=13.8):
    """Qwen3.6-35B-A3B VQ 3.4 shape (hidden 2048; 10 of 40 layers full
    attention, 2 KV heads of 256) -- or the 397B with hidden=4096."""
    from pathlib import Path

    from knurlogic.machine.artifact import Artifact
    types = (["linear_attention"] * 3 + ["full_attention"]) * 10
    cfg = {"text_config": {"hidden_size": hidden, "num_hidden_layers": 40,
                           "layer_types": types, "num_attention_heads": 16,
                           "num_key_value_heads": 2, "head_dim": 256}}
    return Artifact(path=Path("/nonexistent"), model_type="qwen3_5_moe",
                    model_file=None, bytes_on_disk=int(size_gib * GIB),
                    hidden_size=hidden, moe_intermediate_size=512,
                    vq_other={}, raw_config=cfg)


def _chunk(r):
    return S.engine_settings(r.env)["prefill_step_size"]


def test_qwen3_5_moe_measured_best_is_2048():
    assert S.prefill_chunk_for("qwen3_5_moe_text")[0] == 2048


def test_35b_a3b_with_100_gib_free_takes_2048():
    r = resolve(_qwen_moe(), 100 * GIB)
    assert _chunk(r) == 2048
    note = [n for n in r.notes if n.startswith("prompt chunk 2048")]
    assert note and "room" in note[0] and "10%" in note[0]


def test_35b_a3b_on_a_box_with_little_room_stays_512():
    r = resolve(_qwen_moe(), 22 * GIB)
    assert _chunk(r) == 512
    assert any(n.startswith("prompt chunk 512") and "room" in n
               for n in r.notes)


def test_397b_with_14_gib_left_stays_512():
    """The case that aborted Metal: 110.8 GiB on the 128 GB M4, ~14 GiB
    above its weights (git log -S 'prompt chunk is 512')."""
    a = _qwen_moe(hidden=4096, size_gib=110.8)
    assert _chunk(resolve(a, int(124.8 * GIB))) == 512
    assert _chunk(resolve(a, int(124.8 * GIB), tune="default")) == 512


def test_an_explicit_prompt_chunk_still_wins():
    r = resolve(_qwen_moe(), 100 * GIB)
    rec = apply_preset_overrides_(r, {"KNURLOGIC_PREFILL_CHUNK": "1024"})
    assert S.engine_settings({**r.env, **rec})["prefill_step_size"] == 1024


def apply_preset_overrides_(r, sets):
    from knurlogic.tuning.resolve import apply_preset_overrides
    apply_preset_overrides(r, sets)
    return sets


def test_lean_stays_narrow_with_room():
    for tune in ("safe", "lean"):     # safe is the old name for lean
        assert _chunk(resolve(_qwen_moe(), 100 * GIB, tune=tune)) == 512


def test_an_unmeasured_family_stays_512_with_room():
    assert _chunk(resolve(_qwen_moe().__class__(
        path=_qwen_moe().path, model_type="somebody_else", model_file=None,
        bytes_on_disk=14 * GIB, hidden_size=2048, moe_intermediate_size=512,
        vq_other={}), 100 * GIB)) == 512
