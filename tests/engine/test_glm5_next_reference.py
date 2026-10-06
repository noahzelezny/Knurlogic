"""GLM-5.3-Flash's vendored trunk (families/glm5/architecture/glm5_next)
held to the maker's reference: HF transformers' Glm5NextTextModel, run on a
tiny random float32 model (tests/support/goldens/build_glm5_next.py).

The same weights go through the vendored sanitize; a 21-token prefill and
4 decode steps through the model's own cache must give the reference's
logits. index_topk is 8, so the DSA indexer selects pools (and the tail)
on every attention row past the eighth token, and some SwiGLU inputs leave
+-swiglu_limit (PROVENANCE.md, glm5_next edits 1-5)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests" / "support" / "goldens"))

mx = pytest.importorskip("mlx.core")

import build_glm5_next as G  # noqa: E402

GOLD = dict(np.load(G.OUT))
TOL = 2e-4


def _model():
    from knurlogic.engine.families.glm5.architecture.glm5_next.config import TextConfig
    from knurlogic.engine.families.glm5.architecture.glm5_next.language import (
        LanguageModel,
    )
    cfg = json.loads(str(GOLD["config"]))
    model = LanguageModel(TextConfig.from_dict(cfg))
    w = {k[2:]: mx.array(v.astype(np.float32))
         for k, v in GOLD.items() if k.startswith("w/")}
    model.load_weights(list(model.sanitize(w).items()), strict=True)
    mx.eval(model.parameters())
    return model


def _run(model):
    cache = model.make_cache()
    pre = model(mx.array(GOLD["prompt"])[None], cache=cache).logits[0]
    dec = []
    for t in GOLD["decode"].tolist():
        dec.append(model(mx.array([[t]]), cache=cache).logits[0, -1])
    return np.array(pre), np.array(mx.stack(dec))


def test_prefill_and_decode_match_the_reference():
    pre, dec = _run(_model())
    d_pre = np.abs(pre - GOLD["prefill_logits"]).max()
    d_dec = np.abs(dec - GOLD["decode_logits"]).max()
    assert d_pre < TOL and d_dec < TOL, (d_pre, d_dec)


def test_the_golden_exercises_selection_and_the_clamp(monkeypatch):
    """The fixture is only a test of the indexer and the clamp if both
    act: rows past index_topk select, and some SwiGLU input leaves +-10."""
    from knurlogic.engine.families.glm5.architecture.glm5_next import language
    model = _model()
    for layer in model.model.layers:
        layer.compile_ffn = False
    lim = model.args.swiglu_limit
    seen = []
    orig = language._clamped_swiglu

    def spy(gate, up, limit):
        seen.append(max(float(gate.max()), float(mx.abs(up).max())))
        return orig(gate, up, limit)

    monkeypatch.setattr(language, "_clamped_swiglu", spy)
    _run(model)
    assert max(seen) > lim
    assert len(GOLD["prompt"]) > model.args.index_topk


def test_norm_eps_match_the_reference():
    """Edits 3-4: eps a float32 golden on unit-scale weights cannot see."""
    model = _model()
    attn = model.model.layers[1].self_attn
    assert attn.q_a_layernorm.eps == attn.kv_a_layernorm.eps == 1e-5
    assert attn.indexer.k_norm.eps == 1e-6
