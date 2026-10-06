"""GLM-5.3-Flash's vendored trunk held to the maker's reference IN BF16
(tests/support/goldens/build_glm5_next_bf16.py): HF transformers'
Glm5NextTextModel run in bfloat16 with its _keep_in_fp32_modules_strict
(e_score_correction_bias, conv1d, dt_bias, A_log) in float32, on
bf16-exact weights. knurlogic loads the same weights and casts with its
own cast_predicate, as a conversion does.

TOLERANCE. Two bf16 implementations never agree bit for bit (matmul
accumulation order, Metal vs CPU exp), and on a random tiny model a bf16
rounding can tip a discrete choice (an expert, a pool) on one row, which
moves that row's logits by ~1 while the token stays the same. So the gate
is on the MEDIAN over rows of each row's max |logit diff|, held against how
far bf16 itself moves the reference on the same weights (median 0.046
prefill / 0.049 decode). Measured: prefill 0.055 before glm5_next edits
7-8, 0.051 with edit 7 alone, 0.031 with both; decode 0.063 -> 0.067 (the
4 decode rows sit at bf16 noise either way). Gates: prefill < 0.04 (fails
without edit 8), decode < 0.08, and every greedy token equal."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests" / "support" / "goldens"))

mx = pytest.importorskip("mlx.core")

import build_glm5_next_bf16 as G  # noqa: E402

GOLD = dict(np.load(G.OUT))


def _model():
    from knurlogic.engine.families.glm5.architecture.glm5_next.config import TextConfig
    from knurlogic.engine.families.glm5.architecture.glm5_next.language import (
        LanguageModel,
    )
    cfg = json.loads(str(GOLD["config"]))
    model = LanguageModel(TextConfig.from_dict(cfg))
    w = {k[2:]: mx.array(v) for k, v in GOLD.items() if k.startswith("w/")}
    model.load_weights(list(model.sanitize(w).items()), strict=True)
    # as mlx_lm.convert casts a checkpoint (convert.py, cast_predicate)
    from mlx.utils import tree_map_with_path
    keep = model.cast_predicate
    model.update(tree_map_with_path(
        lambda k, v: v.astype(mx.bfloat16)
        if keep(k) and mx.issubdtype(v.dtype, mx.floating) else v,
        model.parameters()))
    mx.eval(model.parameters())
    return model


def _run(model):
    cache = model.make_cache()
    pre = model(mx.array(GOLD["prompt"])[None], cache=cache).logits[0]
    dec = [model(mx.array([[t]]), cache=cache).logits[0, -1]
           for t in GOLD["decode"].tolist()]
    return (np.array(pre.astype(mx.float32)),
            np.array(mx.stack(dec).astype(mx.float32)))


def test_the_golden_says_what_built_it():
    meta = json.loads(str(GOLD["meta"]))
    assert meta["transformers"] == "5.16.1"
    assert meta["experts_implementation"] == "grouped_mm"
    assert set(meta["keep_in_fp32"]) == {"e_score_correction_bias", "conv1d",
                                         "dt_bias", "A_log"}


def _median_row_max(a, b):
    return float(np.median(np.abs(a - b).max(-1)))


def test_bf16_prefill_and_decode_match_the_bf16_reference():
    pre, dec = _run(_model())
    d_pre = _median_row_max(pre, GOLD["prefill_logits"])
    d_dec = _median_row_max(dec, GOLD["decode_logits"])
    assert d_pre < 0.04 and d_dec < 0.08, (d_pre, d_dec)


def test_bf16_greedy_tokens_match_the_reference():
    pre, dec = _run(_model())
    assert (pre.argmax(-1) == GOLD["prefill_logits"].argmax(-1)).all()
    assert (dec.argmax(-1) == GOLD["decode_logits"].argmax(-1)).all()


def test_the_kda_parameters_stay_float32_through_a_conversion():
    """The reference's _keep_in_fp32_modules_strict: a conversion's
    cast_predicate keeps them (and the router's correction bias)."""
    model = _model()
    la = model.model.layers[0].self_attn
    assert la.conv1d.weight.dtype == mx.float32
    assert la.forget_gate.dt_bias.dtype == mx.float32
    assert la.forget_gate.A_log.dtype == mx.float32
    assert la.q_proj.weight.dtype == mx.bfloat16
