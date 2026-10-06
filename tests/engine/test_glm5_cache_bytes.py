"""GLM-5.3-Flash's (glm5_next) cache: the MLA latent is stored once
(glm5_next edit 6), and the planner's kv_bytes_per_token is what a real
cache holds per token, in bf16 and at 8 bits.

The tiny model is the reference golden's config (build_glm5_next.CONFIG)
with kv_lora_rank 64, so the 8-bit latent groups by 64 as the real model's
512 does, run in bf16 as a release runs."""
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


def _model(bits=None):
    from knurlogic.engine.families.glm5.architecture.glm5_next.config import TextConfig
    from knurlogic.engine.families.glm5.architecture.glm5_next.language import (
        LanguageModel,
    )
    from knurlogic.engine.kvquant import install
    mx.random.seed(0)
    model = LanguageModel(TextConfig.from_dict(json.loads(json.dumps(CFG))))
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    if bits:
        assert install(model, bits) == 2
    return model


def _arrays(x):
    if isinstance(x, mx.array):
        return [x]
    if isinstance(x, (tuple, list)):
        return [a for y in x for a in _arrays(y)]
    return []


def _stored(cache) -> int:
    """Bytes the cache holds for its tokens: its `state` (sliced to the
    offset, so the 256-token allocation steps do not count)."""
    out = 0
    for c in cache:
        for s in getattr(c, "caches", None) or [c]:
            if getattr(s, "keys", None) is not None:
                out += sum(a.nbytes for a in _arrays(s.state))
            elif hasattr(s, "state"):
                out += sum(a.nbytes for a in _arrays(s.state))
    return out


def _bytes_per_token(model, n=48):
    """Growth of the whole cache from n to 2n tokens, per token: the
    deltanet layers' recurrent state is bounded and cancels."""
    def at(m):
        cache = model.make_cache()
        mx.random.seed(1)
        model(mx.random.randint(0, 128, (1, m)), cache=cache)
        mx.eval([a for c in cache for a in _arrays(
            [s.state for s in (getattr(c, "caches", None) or [c])])])
        return _stored(cache)
    return (at(2 * n) - at(n)) / n


def test_the_mla_latent_is_stored_once():
    """K = V = the latent: one copy, a zero-width values beside it (it was
    stored as both K and V, twice the bytes)."""
    model = _model()
    cache = model.make_cache()
    model(mx.array([[1, 2, 3, 4, 5]]), cache=cache)
    lat = [c.caches[0] for c in cache if hasattr(c, "caches")]
    assert len(lat) == 2
    for c in lat:
        k, v = c.state
        assert k.shape[-1] == CFG["kv_lora_rank"] and v.shape[-1] == 0


def test_the_quantized_latent_is_stored_once():
    model = _model(8)
    cache = model.make_cache()
    model(mx.array([[1, 2, 3, 4, 5]]), cache=cache)
    for c in (c.caches[0] for c in cache if hasattr(c, "caches")):
        assert c.keys[0].shape[-1] > 0
        assert all(p.shape[-1] == 0 for p in c.values)


@pytest.mark.parametrize("bits", [None, 8])
def test_kv_bytes_per_token_is_what_the_cache_stores(bits):
    from knurlogic.tuning.resolve import kv_bytes_per_token
    got = _bytes_per_token(_model(bits))
    want, _ = kv_bytes_per_token(CFG, bits)
    assert got == want, (got, want)


@pytest.mark.parametrize("bits", [None, 8])
def test_a_latent_cache_state_round_trips(bits):
    """The prompt cache's path over the once-stored latent: a prefix's
    state set on fresh caches, then the rest, gives the uncached run's
    logits (bf16; 8 bits within its drift)."""
    model = _model(bits)
    ids = mx.array([[(5 * i + 1) % 128 for i in range(30)]])
    ref = model.make_cache()
    full = model(ids, cache=ref).logits[0, -1].astype(mx.float32)
    cache = model.make_cache()
    model(ids[:, :26], cache=cache)
    fresh = model.make_cache()
    for a, b in zip(fresh, cache):
        for x, y in zip(getattr(a, "caches", None) or [a],
                        getattr(b, "caches", None) or [b]):
            x.state = y.state
    got = model(ids[:, 26:], cache=fresh).logits[0, -1].astype(mx.float32)
    assert mx.argmax(got).item() == mx.argmax(full).item()
    assert (mx.abs(got - full).max() / mx.abs(full).max()).item() < 0.02
