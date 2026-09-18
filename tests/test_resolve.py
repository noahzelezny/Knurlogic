"""The resolver's job is to be the last word on every knob. Pin that."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knurlogic import settings as S            # noqa: E402
from knurlogic.artifact import Artifact        # noqa: E402
from knurlogic.resolve import decode_chunk_for, resolve  # noqa: E402

GIB = 1 << 30


def _art(**kw):
    base = dict(path=Path("/nonexistent/art"), model_type="qwen4_exp_text",
                model_file="model.py", bytes_on_disk=70 * GIB,
                hidden_size=2560, moe_intermediate_size=640,
                vq_modules={"m": {"d": 4, "K": 2048}})
    base.update(kw)
    return Artifact(**base)


def test_unknown_budget_keeps_defaults():
    r = resolve(_art(), 0)
    assert r.env["VQ_DECODE_CHUNK"] == str(S.DECODE_CHUNK_DEFAULT)
    assert r.env["VQLAB_PREFILL_CHUNK"] == str(S.PREFILL_CHUNK_DEFAULT)


def test_does_not_fit_goes_tightest_not_default():
    """The bug this pins: negative headroom must not fall into the
    'unknown budget' branch and hand back the roomy default."""
    r = resolve(_art(), 48 * GIB)
    assert r.env["VQ_DECODE_CHUNK"] == str(S.DECODE_CHUNK_MIN)
    assert r.warnings, "an artifact that does not fit must say so"


def test_chunk_never_exceeds_default():
    """Smaller is also faster (128 -> 32 is 1.37x), so headroom never buys
    a larger chunk."""
    assert decode_chunk_for(10_000 * GIB) == S.DECODE_CHUNK_DEFAULT


def test_rtile_is_never_64():
    """F25/F33: RTILE=64 measured 0.75-0.97x and never faster."""
    r = resolve(_art(), 96 * GIB)
    assert r.env["VQ_MOE_GEMMSEG_RTILE"] == "32"


def test_profile_selects_numerics_flags():
    assert all(resolve(_art(), 96 * GIB, "v1.5").env[f] == "0"
               for f in S.NUMERICS_FLAGS)
    assert all(resolve(_art(), 96 * GIB, "v2").env[f] == "1"
               for f in S.NUMERICS_FLAGS)


def test_vq_without_model_file_is_a_warning():
    r = resolve(_art(model_file=None), 96 * GIB)
    assert any("model_file" in w for w in r.warnings)


def test_non_vq_artifact_gets_only_the_generic_knobs():
    """The value proposition is 'it runs', not 'it runs VQ'. A stock affine
    artifact still needs the prefill and cache knobs; it has no dense-expert
    decode buffer, so VQ_DECODE_CHUNK would be cargo cult."""
    r = resolve(_art(vq_modules={}), 96 * GIB)
    assert "VQ_DECODE_CHUNK" not in r.env
    assert "VQ_MOE_GEMMSEG_RTILE" not in r.env
    assert r.env["VQLAB_PREFILL_CHUNK"] == str(S.PREFILL_CHUNK_DEFAULT)
    assert r.env["VQLAB_CACHE_LIMIT_GB"] == str(S.CACHE_LIMIT_GB_DEFAULT)


def test_vendored_architecture_wins_over_site_packages():
    """Vendoring is only meaningful if the vendored copy is the one used."""
    from knurlogic import arch
    from knurlogic.register import ARCH_DIR
    for row in arch.check("qwen4_exp_text"):
        if (ARCH_DIR / f"{row.module}.py").is_file():
            assert row.vendored, f"{row.module} should resolve to the vendored copy"


def test_moe_pulls_in_its_base_architecture():
    """One drifted base reaches 11 artifacts through the subclass."""
    from knurlogic import arch
    assert arch.required_modules("qwen3_5_moe_text") == ["qwen3_5_moe", "qwen3_5"]


def test_registering_a_subclass_pulls_its_base_first():
    """A vendored subclass must never land on a site-packages base: that
    silently mixes two versions of the arithmetic, which is the exact failure
    this package exists to end. It passed once only by alphabetical luck."""
    from knurlogic.register import _with_dependencies
    order = _with_dependencies(["qwen3_5_moe"])
    assert order.index("qwen3_5") < order.index("qwen3_5_moe")


def test_pins_are_loaded_and_make_doctor_say_ok():
    """A pin is only written after a model generated a token with clean
    provenance, so an 'ok' from doctor means 'it ran', not 'it imports'."""
    from knurlogic import arch
    if not arch.PINNED_SHA256:
        return  # nothing validated on this checkout yet
    for row in arch.check("qwen3_5_moe_text"):
        if row.module in arch.PINNED_SHA256 and row.vendored:
            assert row.state == "OK", f"{row.module} is {row.state}"


def test_package_architectures_are_found_and_hosted_correctly():
    """glm5_next is an mlx_vlm PACKAGE, not an mlx_lm file. Registered under
    the wrong parent its eight relative sibling imports cannot resolve."""
    from knurlogic import arch
    from knurlogic.register import available, source_for
    if "glm5_next" not in available():
        return
    src, is_pkg = source_for("glm5_next")
    assert is_pkg, "glm5_next must vendor as a package"
    assert arch.host_for("glm5_next") == "mlx_vlm"
    assert arch.host_for("qwen4_exp") == "mlx_lm"
