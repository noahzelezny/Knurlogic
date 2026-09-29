"""The 8-bit KV decode kernel (engine/kvattn.py) against a float32
reference: shapes, GQA ratios, query lengths, ragged batches; and that
install routes a quantized model's decode through it with the same
tokens. Needs mlx (Metal)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
mx = pytest.importorskip("mlx.core")

from test_batch_drafting import _run  # noqa: E402


def _tiny(vocab=512):
    """test_batch_drafting's tiny qwen3_5, at head dim 128 (the kernel's
    smallest)."""
    from knurlogic.engine import register
    register.register("qwen3_5")
    from mlx_lm.models import qwen3_5 as arch
    mx.random.seed(0)
    tc = dict(model_type="qwen3_5", hidden_size=128, intermediate_size=256,
              num_hidden_layers=4, num_attention_heads=4,
              num_key_value_heads=2, head_dim=128, vocab_size=vocab,
              linear_num_value_heads=4, linear_num_key_heads=2,
              linear_key_head_dim=32, linear_value_head_dim=32,
              full_attention_interval=2, tie_word_embeddings=False)
    model = arch.Model(arch.ModelArgs(model_type="qwen3_5", text_config=tc))
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    prompts = [mx.random.randint(0, vocab, (n,)).tolist() for n in (37, 9, 70)]
    return model, None, prompts


def _case(B, H, Hk, L, N, D, g=None, seed=0, dtype=None):
    from knurlogic.engine.kvquant import group_for
    g = g or group_for(D)
    mx.random.seed(seed)
    dt = dtype or mx.bfloat16
    k = mx.random.normal((B, Hk, N, D)).astype(dt)
    v = mx.random.normal((B, Hk, N, D)).astype(dt)
    q = mx.random.normal((B, H, L, D)).astype(dt)
    K = tuple(mx.quantize(k, group_size=g, bits=8))
    V = tuple(mx.quantize(v, group_size=g, bits=8))
    return q, K, V, g


def _ref(q, K, V, g, mask):
    f = mx.float32
    kd = mx.dequantize(*K, group_size=g, bits=8).astype(f)
    vd = mx.dequantize(*V, group_size=g, bits=8).astype(f)
    D = q.shape[-1]
    return mx.fast.scaled_dot_product_attention(q.astype(f), kd, vd,
                                                scale=D ** -0.5, mask=mask)


def _check(q, K, V, g, mask, tol=None):
    from knurlogic.engine import kvattn
    assert kvattn.supports(q, K, V, 8, g, mask)
    got = kvattn.decode_sdpa(q, K, V, q.shape[-1] ** -0.5, mask, g, 8)
    ref = _ref(q, K, V, g, mask)
    assert got.shape == q.shape and got.dtype == q.dtype
    big = max(1.0, mx.abs(ref).max().item())
    tol = tol or (2e-5 if q.dtype == mx.float32 else 8e-3) * big
    err = mx.abs(got.astype(mx.float32) - ref).max().item()
    assert err < tol, err


@pytest.mark.parametrize("D", [128, 256, 512])
@pytest.mark.parametrize("H,Hk", [(4, 4), (8, 2), (16, 2), (16, 1)])
@pytest.mark.parametrize("N", [1, 7, 256, 1000, 4099])
def test_a_decode_step_matches_float32(D, H, Hk, N):
    q, K, V, g = _case(1, H, Hk, 1, N, D)
    _check(q, K, V, g, None)


@pytest.mark.parametrize("L,H,Hk", [(2, 8, 2), (4, 4, 2), (3, 2, 2),
                                    (4, 16, 2), (8, 8, 1)])
def test_a_short_causal_query_matches_float32(L, H, Hk):
    q, K, V, g = _case(1, H, Hk, L, 700, 128)
    _check(q, K, V, g, "causal")


def test_float32_queries_are_float32_close():
    q, K, V, g = _case(1, 8, 2, 1, 3000, 128, dtype=mx.float32)
    _check(q, K, V, g, None)


def test_group_32():
    q, K, V, g = _case(1, 8, 2, 1, 500, 128, g=32)
    _check(q, K, V, g, None)


@pytest.mark.parametrize("L", [1, 2])
def test_a_ragged_batch_is_each_row_masked_alone(L):
    """Rows of different lengths: the batch's left padding masked out, as
    BatchKVCache.make_mask builds it; one row fully padded but the last
    keys."""
    from mlx_lm.models.cache import create_causal_mask
    B, N = 3, 1200
    q, K, V, g = _case(B, 16, 2, L, N, 256)
    mask = create_causal_mask(L, offset=N - L,
                              left_padding=mx.array([0, 513, N - L - 1]))
    _check(q, K, V, g, mask)


def test_unsupported_shapes_fall_back():
    from knurlogic.engine import kvattn
    q, K, V, g = _case(1, 8, 2, 1, 64, 128)
    assert not kvattn.supports(q, K, V, 4, g)               # not 8-bit
    assert not kvattn.supports(q, K, V, 8, g, sinks=mx.zeros((8,)))
    q6, K6, V6, g6 = _case(1, 8, 2, 1, 64, 64)              # head dim 64
    assert not kvattn.supports(q6, K6, V6, 8, g6)
    add = mx.zeros((1, 1, 1, 64))                           # additive mask
    assert not kvattn.supports(q, K, V, 8, g, add)
    per_head = mx.ones((1, 8, 1, 64), mx.bool_)
    assert not kvattn.supports(q, K, V, 8, g, per_head)
    q9, K9, V9, g9 = _case(1, 32, 1, 3, 64, 128)            # 96 rows / head
    assert not kvattn.supports(q9, K9, V9, 8, g9)


def _greedy(model, prompt, n=12):
    cache = model.make_cache()
    out = model(mx.array(prompt)[None], cache=cache)[:, -1]
    toks, logits = [], [out]
    for _ in range(n):
        t = mx.argmax(out, axis=-1)
        toks.append(t.item())
        out = model(t[None], cache=cache)[:, -1]
        logits.append(out)
    return toks, mx.concatenate(logits)


def test_install_takes_the_kernel_and_matches_the_dequantized_path(
        monkeypatch):
    from knurlogic.engine import kvattn, kvquant
    calls = []
    real = kvattn.decode_sdpa
    monkeypatch.setattr(kvattn, "decode_sdpa",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    model, _, prompts = _tiny(512)
    whole = model.make_cache
    monkeypatch.setenv(kvattn.ENV, "off")
    kvquant.install(model, 8)
    assert kvquant.KERNEL is False
    off_t, off_l = _greedy(model, prompts[0])
    assert not calls
    model.make_cache = whole
    monkeypatch.delenv(kvattn.ENV)
    kvquant.install(model, 8)
    assert kvquant.KERNEL is True
    on_t, on_l = _greedy(model, prompts[0])
    assert calls                        # decode steps took the kernel
    assert on_t == off_t
    scale = mx.abs(off_l).max().item()
    assert mx.abs(on_l - off_l).max().item() < 1e-3 * scale


def test_a_batched_quantized_decode_through_the_kernel_is_each_row_alone(
        monkeypatch):
    """The decode loop's merged, left-padded batch: same tokens with the
    kernel on and off."""
    from knurlogic.engine import kvattn, kvquant
    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    calls = []
    real = kvattn.decode_sdpa
    monkeypatch.setattr(kvattn, "decode_sdpa",
                        lambda *a, **k: calls.append(a[0].shape) or
                        real(*a, **k))
    model, _, prompts = _tiny(512)
    whole = model.make_cache
    monkeypatch.setenv(kvattn.ENV, "off")
    kvquant.install(model, 8)
    off = _run(MTPBatchGenerator(model, None, prefill_step_size=16),
               prompts, 20)
    model.make_cache = whole
    monkeypatch.delenv(kvattn.ENV)
    kvquant.install(model, 8)
    on = _run(MTPBatchGenerator(model, None, prefill_step_size=16),
              prompts, 20)
    assert any(shape[0] > 1 for shape in calls)     # a batch took it
    assert on == off
