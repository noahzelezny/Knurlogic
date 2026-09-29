"""KNURLOGIC_LONG_CONTEXT=yarn: Qwen's documented YaRN (factor 4 over
262,144 -> ~1M tokens), set on the loaded config, never the artifact.

Held here: off is today's rope exactly; on, every rope the Qwen families
apply (mlx-lm's YarnRoPE on qwen3_5/qwen3_5_moe's text path, the MRoPE
image path, qwen4_exp's own RotaryEmbedding) matches an independent YaRN
reference (arXiv 2309.00071 as transformers' _compute_yarn_parameters
computes it); the context cap moves to the YaRN window; a box that cannot
hold that much KV is told so."""
from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from knurlogic.machine.artifact import Artifact  # noqa: E402
from knurlogic.tuning import settings as S  # noqa: E402
from knurlogic.tuning.resolve import (GIB, long_context_room,  # noqa: E402
                                      resolve)

FACTOR, ORIG = 4.0, 262144


def ref_yarn(dim, base, factor=FACTOR, orig=ORIG, beta_fast=32, beta_slow=1):
    """(inv_freq float64, attention_factor): transformers'
    _compute_yarn_parameters (truncate=True), written out independently."""
    pos_freqs = base ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
    extrap, interp = 1.0 / pos_freqs, 1.0 / (factor * pos_freqs)

    def corr(n):
        return dim * math.log(orig / (n * 2 * math.pi)) / (2 * math.log(base))
    low = max(math.floor(corr(beta_fast)), 0)
    high = min(math.ceil(corr(beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = np.clip((np.arange(dim // 2) - low) / (high - low), 0, 1)
    extrap_factor = 1 - ramp
    inv = interp * (1 - extrap_factor) + extrap * extrap_factor
    return inv, 0.1 * math.log(factor) + 1.0


def ref_rotate(x, dims, inv, af, pos):
    """x [..., L, D] rotated on its first `dims` (half-split pairs) at
    positions `pos` [L], cos/sin scaled by `af`; the rest passed through."""
    ang = np.asarray(pos, np.float64)[:, None] * inv[None]
    cos = np.concatenate([np.cos(ang)] * 2, -1) * af
    sin = np.concatenate([np.sin(ang)] * 2, -1) * af
    xr = x[..., :dims]
    h = dims // 2
    rot = np.concatenate([-xr[..., h:], xr[..., :h]], -1)
    return np.concatenate([xr * cos + rot * sin, x[..., dims:]], -1)


# --- the settings: pure, no mlx ----------------------------------------------

REAL_35B = {"model_type": "qwen3_5_moe", "text_config": {
    "model_type": "qwen3_5_moe_text", "max_position_embeddings": 262144,
    "num_hidden_layers": 40, "full_attention_interval": 4,
    "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 256,
    "rope_parameters": {"mrope_interleaved": True,
                        "mrope_section": [11, 11, 10],
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 10000000, "type": "default"}}}


def test_off_is_no_overlay_and_the_window_stays():
    assert S.long_context_config(REAL_35B, "off") == {}
    assert S.long_context_config(REAL_35B, None) == {}
    assert S.model_window(S.with_long_context(REAL_35B, "off"))[0] == 262144


def test_yarn_overlays_qwens_documented_rope_and_keeps_mrope():
    ov = S.long_context_config(REAL_35B, "yarn")
    rp = ov["text_config"]["rope_parameters"]
    assert rp["rope_type"] == "yarn" and "type" not in rp
    assert rp["factor"] == 4.0
    assert rp["original_max_position_embeddings"] == 262144
    assert rp["mrope_section"] == [11, 11, 10]
    assert rp["partial_rotary_factor"] == 0.25
    assert rp["rope_theta"] == 10000000
    # the artifact's dict is not touched
    assert REAL_35B["text_config"]["rope_parameters"]["type"] == "default"
    w, why = S.model_window(S.with_long_context(REAL_35B, "yarn"))
    assert w == 1_048_576 and "YaRN" in why


def test_only_documented_families_take_it():
    assert S.long_context_refusal("qwen4_exp", "yarn") is None
    assert S.long_context_refusal("qwen3_5_text", "yarn") is None
    why = S.long_context_refusal("glm5_next", "yarn")
    assert why and "qwen3_5" in why
    assert S.long_context_refusal("glm5_next", "off") is None
    with pytest.raises(ValueError):
        S.long_context_config({"model_type": "gemma4"}, "yarn")
    assert S.check_knob("KNURLOGIC_LONG_CONTEXT", "yarn") is None
    assert S.check_knob("KNURLOGIC_LONG_CONTEXT", "ntk")


def _art(cfg=REAL_35B, gib=20):
    return Artifact(path=Path("/nonexistent/q"), model_type=cfg["model_type"],
                    model_file=None, bytes_on_disk=int(gib * GIB),
                    raw_config=cfg, hidden_size=2048,
                    moe_intermediate_size=512)


def test_the_resolver_raises_the_cap_only_when_on():
    a = _art()
    off = resolve(a, 96 * GIB)
    assert off.env["KNURLOGIC_CONTEXT_LENGTH"] == "262144"
    assert off.env["KNURLOGIC_LONG_CONTEXT"] == "off"
    assert off.ranges["KNURLOGIC_LONG_CONTEXT"] == ["off", "yarn"]
    on = resolve(a, 96 * GIB, long_context="yarn")
    assert on.env["KNURLOGIC_CONTEXT_LENGTH"] == "1048576"
    assert on.ranges["KNURLOGIC_CONTEXT_LENGTH"][-1] == 1048576
    assert not on.warnings
    # an unsupported family is not offered the knob at all
    glm = _art({"model_type": "glm5_next", "max_position_embeddings": 202752})
    assert "KNURLOGIC_LONG_CONTEXT" not in resolve(glm, 96 * GIB).env


def test_a_context_the_box_cannot_hold_is_refused_with_the_numbers():
    a = _art()
    # 35B: 10 full-attention layers x 2 KV heads x 256 x K,V x bf16
    assert long_context_room(a, 96 * GIB, a.bytes_on_disk, 1_048_576) is None
    why = long_context_room(a, 40 * GIB, a.bytes_on_disk, 1_048_576)
    assert why.startswith("1,048,576 tokens of context need 20.0 GiB of KV")
    assert "20,480 bytes per token" in why
    assert "leaves 16.0 GiB" in why and "8-bit KV" in why
    assert "10.6 GiB" in why          # 8-bit: 1.0625 bytes per element
    # the resolver says it as a warning on a box that small
    r = resolve(a, 40 * GIB, long_context="yarn")
    assert any("20.0 GiB of KV" in w for w in r.warnings)


def test_the_page_accepts_a_million_tokens_only_with_yarn():
    from knurlogic.interfaces.page import documents
    a = _art()
    assert documents.refuse_sets(a, {"KNURLOGIC_CONTEXT_LENGTH": "1000000"})
    assert documents.refuse_sets(a, {"KNURLOGIC_CONTEXT_LENGTH": "1000000",
                               "KNURLOGIC_LONG_CONTEXT": "yarn"}) is None
    glm = _art({"model_type": "glm5_next", "max_position_embeddings": 202752})
    assert "refused" in documents.refuse_sets(glm, {"KNURLOGIC_LONG_CONTEXT": "yarn"})


def test_the_loader_overlays_from_the_launch_env(tmp_path):
    from knurlogic.engine.serve.load import long_context_overlay
    (tmp_path / "config.json").write_text(json.dumps(REAL_35B))
    assert long_context_overlay(tmp_path, {}) == {}
    assert long_context_overlay(tmp_path, {"KNURLOGIC_LONG_CONTEXT": "off"}) == {}
    ov = long_context_overlay(tmp_path, {"KNURLOGIC_LONG_CONTEXT": "yarn"})
    assert ov["text_config"]["rope_parameters"]["rope_type"] == "yarn"
    assert json.loads((tmp_path / "config.json").read_text()) == REAL_35B


# --- the engine: tiny models, numerics against the reference -----------------

mx = pytest.importorskip("mlx.core")


def _tiny(fam, mode):
    import fixtures_vision as fv
    from knurlogic.engine import register
    register.register("qwen3_5", "qwen3_5_moe", "qwen4_exp", override=True)
    arch = importlib.import_module(f"mlx_lm.models.{fam}")
    _, meta = fv.load_golden("qwen_g5_text")
    cfg = S.with_long_context(meta[f"{fam}/config"], mode)
    mx.random.seed(0)
    model = arch.Model(arch.ModelArgs.from_dict(cfg))
    model.set_dtype(mx.float32)
    return model, cfg["text_config"]


def _attns(model):
    return [m for _, m in model.named_modules()
            if type(m).__name__ == "Attention" and hasattr(m, "rotary_dims")]


@pytest.mark.parametrize("fam", ["qwen3_5", "qwen3_5_moe"])
def test_qwen3_5_off_is_plain_rope(fam):
    import mlx.nn as nn
    model, _ = _tiny(fam, "off")
    for a in _attns(model):
        assert type(a.rope) is nn.RoPE


@pytest.mark.parametrize("fam", ["qwen3_5", "qwen3_5_moe"])
def test_qwen3_5_yarn_matches_the_reference(fam):
    from mlx_lm.models.rope_utils import YarnRoPE
    model, tc = _tiny(fam, "yarn")
    attns = _attns(model)
    assert attns
    dims = attns[0].rotary_dims
    inv, af = ref_yarn(dims, tc["rope_parameters"]["rope_theta"])
    x = np.random.default_rng(0).standard_normal((1, 2, 5, 128)).astype(
        np.float32)
    want = ref_rotate(x, dims, inv, af, np.arange(5000, 5005))
    for a in attns:
        assert isinstance(a.rope, YarnRoPE)
        np.testing.assert_allclose(1.0 / np.array(a.rope._freqs), inv,
                                   rtol=1e-5)
        assert a.rope.mscale == pytest.approx(af)
        got = np.array(a.rope(mx.array(x), offset=5000))
        np.testing.assert_allclose(got, want, atol=2e-3, rtol=1e-3)
        # the MRoPE image path, three equal axes, is the same YaRN rope
        apply_mrope = importlib.import_module(
            "mlx_lm.models.qwen3_5").apply_mrope
        pid = mx.broadcast_to(mx.arange(5000, 5005)[None, None], (3, 1, 5))
        got = np.array(apply_mrope(mx.array(x), pid, dims, a.rope_base,
                                   a.mrope_section, a.rope))
        np.testing.assert_allclose(got, want, atol=2e-3, rtol=1e-3)


def _q4_rope(model):
    return model.language_model.model.rope if hasattr(
        model, "language_model") else next(
        m.rope for _, m in model.named_modules() if hasattr(m, "rope")
        and type(m.rope).__name__ == "RotaryEmbedding")


def test_qwen4_exp_off_is_todays_rope():
    model, tc = _tiny("qwen4_exp", "off")
    r = _q4_rope(model)
    dim = r.dim
    base = tc["rope_parameters"]["rope_theta"]
    assert r.attention_factor == 1.0 and r.scaling is None
    np.testing.assert_array_equal(
        np.array(r.inv_freq),
        np.array(base ** (-mx.arange(0, dim, 2, dtype=mx.float32) / dim)))


def test_qwen4_exp_yarn_matches_the_reference():
    model, tc = _tiny("qwen4_exp", "yarn")
    r = _q4_rope(model)
    inv, af = ref_yarn(r.dim, tc["rope_parameters"]["rope_theta"])
    np.testing.assert_allclose(np.array(r.inv_freq), inv, rtol=1e-5)
    assert r.attention_factor == pytest.approx(af)
    pos = np.arange(5000, 5005)
    cos, sin = r(mx.array(pos)[None])
    ang = pos[:, None] * inv[None]
    np.testing.assert_allclose(np.array(cos)[0], np.concatenate(
        [np.cos(ang)] * 2, -1) * af, atol=2e-3)
    np.testing.assert_allclose(np.array(sin)[0], np.concatenate(
        [np.sin(ang)] * 2, -1) * af, atol=2e-3)
    # frequencies above the ramp are extrapolated (kept), below it
    # interpolated by the factor
    base = base_inv = 10000000 ** (-np.arange(0, r.dim, 2) / r.dim)
    assert np.isclose(inv[0], base[0]) and np.isclose(
        inv[-1], base_inv[-1] / FACTOR)
