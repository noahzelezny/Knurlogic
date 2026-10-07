"""GLM-5.3's prefill, once the indexer selects, attends in the latent over
each query's own selection (glm5_next edit 10, _gathered_attention)
instead of expanding per-head K/V for the whole context: the same keys and
the same softmax, so the logits match the expanded path's -- chunked
through the cache past index_topk, in float32 to rounding; in bf16 / an
8-bit latent, no worse-rounded than the expanded path against a float32
reference of the same weights."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests" / "support" / "goldens"))

mx = pytest.importorskip("mlx.core")

import build_glm5_next as G  # noqa: E402

CFG = dict(G.CONFIG, kv_lora_rank=64)


def _model(dtype, bits=None):
    from knurlogic.engine.families.glm5.architecture.glm5_next.config import TextConfig
    from knurlogic.engine.families.glm5.architecture.glm5_next.language import (
        LanguageModel,
    )
    from knurlogic.engine.kvquant import install
    mx.random.seed(0)
    model = LanguageModel(TextConfig.from_dict(json.loads(json.dumps(CFG))))
    model.set_dtype(dtype)
    mx.eval(model.parameters())
    if bits:
        assert install(model, bits) == 2
    return model


def _prefill(model, ids, chunk):
    """Logits of every position, the prompt fed `chunk` tokens at a time
    through one cache, as the server's prefill does."""
    cache = model.make_cache()
    out = []
    for c0 in range(0, len(ids), chunk):
        x = mx.array([ids[c0:c0 + chunk]])
        lg = model(x, cache=cache)
        lg = getattr(lg, "logits", lg)
        mx.eval(lg)
        out.append(lg)
    return mx.concatenate(out, axis=1).astype(mx.float32)


def _ids(n):
    return [(7 * i + 3) % CFG["vocab_size"] for i in range(n)]


def _both_paths(monkeypatch, dtype, bits, ids):
    """(expanded, gathered) logits of one model; asserts the gathered path
    was taken in the second run only."""
    from knurlogic.engine.families.glm5.architecture.glm5_next import language as L
    model = _model(getattr(mx, dtype), bits)
    calls = {"n": 0}
    real = L._gathered_attention

    def counted(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(L, "_gathered_attention", counted)
    monkeypatch.setattr(L, "EXPANDED_PREFILL", True)
    try:
        want = _prefill(model, ids, 16)
    except RuntimeError as e:          # mlx's CPU build: bf16 MoE gather
        if "only supports float32" in str(e):
            pytest.skip(f"this mlx backend cannot run it: {e}")
        raise
    assert calls["n"] == 0
    monkeypatch.setattr(L, "EXPANDED_PREFILL", False)
    got = _prefill(model, ids, 16)
    assert calls["n"] > 0
    return want, got


@pytest.mark.parametrize("bits", [None, 8])
def test_in_float32_the_gathered_prefill_is_the_expanded(monkeypatch, bits):
    """Same keys, same softmax: equal to float32 rounding."""
    want, got = _both_paths(monkeypatch, "float32", bits, _ids(72))
    rng = float(mx.max(want) - mx.min(want))
    diff = float(mx.max(mx.abs(got - want)))
    assert diff <= 1e-4 * rng, (diff, rng)


@pytest.mark.parametrize("bits", [None, 8])
def test_in_bf16_the_gathered_prefill_rounds_no_worse_than_the_expanded(
        monkeypatch, bits):
    """In bf16 the two paths round in different orders, so they DISAGREE
    by an amount no fixed threshold can call right or wrong (it was 8% of
    the logit range, then a measured 10%, with float32 equal to 1e-4). A
    disagreement says the roundings differ, not which is worse. The
    question that is a defect: against a float32 reference of the SAME
    weights (the seeded model, before its cast), is the gathered path's
    error any worse than the expanded path's? Allowed: 1.5x, for the
    run-to-run spread of which order happens to round better."""
    ids = _ids(72)                     # past index_topk (8) from chunk 2 on
    ref, _ = _both_paths(monkeypatch, "float32", bits, ids)
    want, got = _both_paths(monkeypatch, "bfloat16", bits, ids)
    rng = float(mx.max(ref) - mx.min(ref))
    e_exp = float(mx.max(mx.abs(want - ref)))
    e_gat = float(mx.max(mx.abs(got - ref)))
    assert e_gat <= 1.5 * e_exp + 1e-4 * rng, (e_gat, e_exp, rng)


def test_gathered_attention_bounds_its_rows():
    """A chunk wider than GATHER_ROWS is attended in blocks with the same
    result as one block."""
    from knurlogic.engine.families.glm5.architecture.glm5_next import language as L
    mx.random.seed(1)
    B, H, Lq, D, Kv, K = 1, 2, 300, 16, 500, 8
    q = mx.random.normal((B, H, Lq, D))
    lat = mx.random.normal((B, 1, Kv, D))
    topk = mx.random.randint(-1, Kv, (B, Lq, K))
    a = L._gathered_attention(q, lat, topk, None, 0.25)
    old = L.GATHER_ROWS
    try:
        L.GATHER_ROWS = 4096
        b = L._gathered_attention(q, lat, topk, None, 0.25)
    finally:
        L.GATHER_ROWS = old
    assert a.shape == (B, H, Lq, D)
    assert float(mx.max(mx.abs(a - b))) < 1e-5
