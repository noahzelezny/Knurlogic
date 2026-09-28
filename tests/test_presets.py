"""Launch presets: the tune axis as named bundles (balanced default, fast,
stable, lean, safe); explicit per-model settings beat a preset's values and
the resolution says which came from where."""

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


def test_presets_are_the_tune_axis_and_balanced_is_the_default():
    assert set(S.PRESETS) == set(S.TUNE_PROFILES)
    assert S.PRESET_DEFAULT == "balanced"
    assert S.preset_of("") == "balanced"
    assert S.preset_of(None, "fast") == "fast"
    with pytest.raises(ValueError):
        S.preset_of("turbo")
    r = resolve(_art(), 96 * GIB)
    assert r.env["KNURLOGIC_PRESET"] == "balanced"
    assert r.preset["name"] == "balanced"
    # balanced sets nothing of its own: the resolver's defaults stand
    assert r.preset["from_preset"] == {}


def test_fast_takes_the_family_width_and_dynamic_mtp():
    r = resolve(_art(), 96 * GIB, tune="fast")
    e = S.engine_settings(r.env)
    assert e["prefill_step_size"] == 4096
    assert e["kv_bits"] is None and e["cross_chip"] == "off"
    assert preset_env(_art(), "fast") == {
        "KNURLOGIC_MTP": "on", "KNURLOGIC_MTP_DYNAMIC": "on",
        "KNURLOGIC_KV_BITS": "bf16", "KNURLOGIC_CROSS_CHIP": "off"}
    assert "KNURLOGIC_KV_BITS" in r.preset["from_preset"]


def test_stable_is_512_cross_chip_always_draft_conservative_memory():
    r = resolve(_art(), 96 * GIB, tune="stable")
    b = resolve(_art(), 96 * GIB)
    e, eb = S.engine_settings(r.env), S.engine_settings(b.env)
    assert e["prefill_step_size"] == 512
    assert e["cross_chip"] == "on"
    assert e["cache_limit_gb"] < eb["cache_limit_gb"]
    assert preset_env(_art(), "stable")["KNURLOGIC_MTP_DYNAMIC"] == "off"
    assert preset_env(_art(), "stable")["KNURLOGIC_MTP"] == "on"


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
    from knurlogic.interfaces.ui import TUNES, clean_sets
    assert set(TUNES) == set(S.PRESETS)
    ok, bad = clean_sets({"KNURLOGIC_PRESET": "lean"})
    assert ok == {"KNURLOGIC_PRESET": "lean"} and not bad
