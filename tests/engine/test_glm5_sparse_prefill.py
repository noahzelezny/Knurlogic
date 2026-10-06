"""GLM-5.3's prefill past the crossover attends in the latent over each
query's own selection (glm5_next edit 10, _gathered_attention) instead of
expanding per-head K/V for the whole context: the same keys and the same
softmax, so the logits match the expanded path's -- chunked through the
cache past index_topk, in float32 (to rounding) and bf16 / an 8-bit
latent (to bf16's), and the setting picks the path."""
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
ENV = "KNURLOGIC_SPARSE_PREFILL_FROM"


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


@pytest.mark.parametrize("dtype,bits,tol", [
    ("float32", None, 1e-4), ("float32", 8, 1e-4),
    ("bfloat16", None, 0.08), ("bfloat16", 8, 0.08)])
def test_the_gathered_prefill_matches_the_expanded(monkeypatch, dtype, bits,
                                                   tol):
    from knurlogic.engine.families.glm5.architecture.glm5_next import language as L
    model = _model(getattr(mx, dtype), bits)
    ids = _ids(72)                     # past index_topk (8) from chunk 2 on
    calls = {"n": 0}
    real = L._gathered_attention

    def counted(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(L, "_gathered_attention", counted)
    monkeypatch.setenv(ENV, "off")
    try:
        want = _prefill(model, ids, 16)
    except RuntimeError as e:          # mlx's CPU build: bf16 MoE gather
        if "only supports float32" in str(e):
            pytest.skip(f"this mlx backend cannot run it: {e}")
        raise
    assert calls["n"] == 0
    monkeypatch.setenv(ENV, "0")
    got = _prefill(model, ids, 16)
    assert calls["n"] > 0
    rng = float(mx.max(want) - mx.min(want))
    diff = float(mx.max(mx.abs(got - want)))
    assert diff <= tol * rng, (diff, rng)


def test_the_crossover_is_tokens_of_context(monkeypatch):
    from knurlogic.engine.families.glm5.architecture.glm5_next import language as L
    model = _model(mx.float32)
    seen = []
    real = L._gathered_attention

    def counted(q, kv_latent, *a, **k):
        seen.append(kv_latent.shape[2])
        return real(q, kv_latent, *a, **k)
    monkeypatch.setattr(L, "_gathered_attention", counted)
    monkeypatch.setenv(ENV, "48")
    _prefill(model, _ids(72), 16)
    assert seen and min(seen) >= 48


def test_the_setting_reads():
    import os

    from knurlogic.engine.families.glm5.architecture.glm5_next.language import (
        sparse_prefill_from,
    )
    old = os.environ.pop(ENV, None)
    try:
        assert sparse_prefill_from() == 0
        for v, want in (("off", None), ("32768", 32768), ("junk", 0),
                        ("-5", 0)):
            os.environ[ENV] = v
            assert sparse_prefill_from() == want
    finally:
        os.environ.pop(ENV, None)
        if old is not None:
            os.environ[ENV] = old


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
