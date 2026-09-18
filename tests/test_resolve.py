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
