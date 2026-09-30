"""Launch presets: the tune axis as named bundles (default and lean);
explicit per-model settings beat a preset's values and the resolution says
which came from where."""

from pathlib import Path

import pytest

from knurlogic.machine.artifact import Artifact
from knurlogic.tuning import settings as S
from knurlogic.tuning.resolve import (apply_preset_overrides, preset_env,
                                      resolve)

GIB = 1 << 30


def _art(model_type="qwen3_5"):
    return Artifact(path=Path("/nonexistent"), model_type=model_type,
                    model_file=None, bytes_on_disk=20 * GIB,
                    hidden_size=4096, moe_intermediate_size=1024,
                    vq_other={})


def test_presets_are_the_tune_axis_and_default_is_the_default():
    assert set(S.PRESETS) == set(S.TUNE_PROFILES) == {"default", "lean"}
    assert S.PRESET_DEFAULT == "default"
    assert S.preset_of("") == "default"
    assert S.preset_of(None, "lean") == "lean"
    with pytest.raises(ValueError):
        S.preset_of("turbo")
    r = resolve(_art(), 96 * GIB)
    assert r.env["KNURLOGIC_PRESET"] == "default"
    assert r.preset["name"] == "default"
    # default sets nothing of its own: the resolver's defaults stand
    assert r.preset["from_preset"] == {}


def test_the_names_presets_once_had_are_the_two_now():
    import argparse
    for old, now in (("balanced", "default"), ("fast", "default"),
                     ("stable", "default"), ("safe", "lean"),
                     ("Default", "default"), ("LEAN", "lean")):
        assert S.preset_of(old) == now
        assert S.preset_arg(old) == now
    r = resolve(_art(), 96 * GIB, tune="safe")
    assert r.env["KNURLOGIC_PRESET"] == "lean"
    assert resolve(_art(), 96 * GIB, tune="fast").env[
        "KNURLOGIC_PRESET"] == "default"
    assert S.check_knob("KNURLOGIC_PRESET", "stable") is None
    with pytest.raises(argparse.ArgumentTypeError):
        S.preset_arg("turbo")


def test_default_takes_the_family_width_and_dynamic_mtp():
    r = resolve(_art(), 96 * GIB, tune="default")
    e = S.engine_settings(r.env)
    assert e["prefill_step_size"] == 2048    # the room rule
    assert e["kv_bits"] is None and e["cross_chip"] == "off"
    assert preset_env(_art(), "default") == {}    # it asks for nothing


def test_lean_is_512_and_keeps_less_cache():
    r = resolve(_art(), 96 * GIB, tune="lean")
    b = resolve(_art(), 96 * GIB)
    e, eb = S.engine_settings(r.env), S.engine_settings(b.env)
    assert e["prefill_step_size"] == 512
    assert e["cross_chip"] == "off"    # per-chip rounding is no preset's
    assert e["cache_limit_gb"] < eb["cache_limit_gb"]


def test_lean_quantizes_kv_where_the_family_takes_it():
    r = resolve(_art(), 96 * GIB, tune="lean")
    e = S.engine_settings(r.env)
    assert e["kv_bits"] == 8
    assert e["prefill_step_size"] == 512
    assert "prompt_concurrency" not in e    # the engine prefills one anyway
    assert preset_env(_art(), "lean")["KNURLOGIC_MTP"] == "off"


def test_lean_on_a_family_that_refuses_kv_quant_stays_bf16_and_says_so(
        monkeypatch):
    """Every served family takes 8 bits now; a family that declared none
    (a new one, before its caches are wired) still falls back."""
    monkeypatch.setattr(S, "kv_quant_for",
                        lambda mt: ([], "no family declares it"))
    r = resolve(_art("glm5_next"), 96 * GIB, tune="lean")
    assert S.engine_settings(r.env)["kv_bits"] is None
    assert any("stays bf16" in n and "preset lean" in n for n in r.notes)
    assert preset_env(_art("glm5_next"), "lean")["KNURLOGIC_KV_BITS"] == "bf16"


@pytest.mark.parametrize("mt", ["qwen3_5", "qwen4_exp_text", "glm5_next",
                                "gemma4_text"])
def test_lean_takes_8_bit_kv_on_every_family(mt):
    r = resolve(_art(mt), 96 * GIB, tune="lean")
    assert S.engine_settings(r.env)["kv_bits"] == 8


def test_an_explicit_setting_beats_the_preset_and_is_reported():
    r = resolve(_art(), 96 * GIB, tune="lean")
    rec = apply_preset_overrides(r, {"KNURLOGIC_KV_BITS": "4",
                                     "KNURLOGIC_PREFILL_CHUNK": "512"})
    assert rec["overridden"] == {
        "KNURLOGIC_KV_BITS": {"preset": "8", "set": "4"}}
    # the same value set explicitly is still the preset's
    assert rec["from_preset"]["KNURLOGIC_PREFILL_CHUNK"] == "512"
    assert "KNURLOGIC_KV_BITS" not in rec["from_preset"]


def test_the_preset_is_a_launch_knob_with_a_native_range():
    assert "KNURLOGIC_PRESET" in S.MODEL_KNOBS
    assert S.KNOB_RANGE["KNURLOGIC_PRESET"][0] == list(S.PRESETS)
    from knurlogic.interfaces.page.server import clean_sets
    ok, bad = clean_sets({"KNURLOGIC_PRESET": "lean"})
    assert ok == {"KNURLOGIC_PRESET": "lean"} and not bad
