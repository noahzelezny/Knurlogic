"""KV-cache precision (engine/kvquant.py): attention K/V stored at 8/6/4
bits, dequantized on fetch. Tiny fixtures only -- nothing here is a
measurement of a real model."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
mx = pytest.importorskip("mlx.core")

from test_batch_drafting import _run, _tiny  # noqa: E402


def _kv(shape, seed=0, dtype=None):
    mx.random.seed(seed)
    x = mx.random.normal(shape)
    return x.astype(dtype or mx.bfloat16)


# --- the caches on their own -------------------------------------------------

@pytest.mark.parametrize("bits,ratio", [(8, 1.0625 / 2), (6, 0.8125 / 2),
                                        (4, 0.5625 / 2)])
def test_stored_bytes_shrink_by_the_bits(bits, ratio):
    from mlx_lm.models.cache import KVCache
    from knurlogic.engine.kvquant import QuantKVCache
    k, v = _kv((1, 4, 512, 128)), _kv((1, 4, 512, 128), 1)
    plain, q = KVCache(), QuantKVCache(bits)
    plain.update_and_fetch(k, v)
    q.update_and_fetch(k, v)
    mx.eval(plain.keys, q.keys)
    assert q.nbytes / plain.nbytes == pytest.approx(ratio, rel=1e-6)


def test_a_fetch_is_close_to_what_went_in_at_8_bits():
    from knurlogic.engine.kvquant import QuantKVCache
    k, v = _kv((1, 2, 40, 64)), _kv((1, 2, 40, 64), 1)
    kk, vv = QuantKVCache(8).update_and_fetch(k, v)
    assert kk.dtype == k.dtype and kk.shape == k.shape
    assert mx.abs(kk.astype(mx.float32) - k.astype(mx.float32)).max().item() < 0.05
    assert mx.abs(vv.astype(mx.float32) - v.astype(mx.float32)).max().item() < 0.05


def test_the_quantized_cache_has_no_bits_attribute():
    """mlx_lm.models.base sends a cache with `bits` down its quantized
    SDPA, which would read the DEQUANTIZED arrays as packed ones."""
    from knurlogic.engine.kvquant import BatchQuantKVCache, QuantKVCache
    assert not hasattr(QuantKVCache(8), "bits")
    assert not hasattr(BatchQuantKVCache([0], 8), "bits")


def test_a_batched_rollback_rewrites_like_a_fresh_cache():
    """The speculative step: trim(2) then replay must leave exactly what
    a cache fed only the committed tokens holds."""
    from knurlogic.engine.kvquant import BatchQuantKVCache
    from knurlogic.engine.mtp import caches
    a, b = BatchQuantKVCache([0, 0], 8), BatchQuantKVCache([0, 0], 8)
    base = _kv((2, 2, 9, 32))
    for c in (a, b):
        c.update_and_fetch(base, base)
    snaps = caches.snapshot([a])
    assert snaps[0][0] == "battn"
    wrong = _kv((2, 2, 2, 32), 5)
    a.update_and_fetch(wrong, wrong)
    caches.restore([a], snaps)
    assert a.size() == 9
    right = _kv((2, 2, 2, 32), 6)
    ka, va = a.update_and_fetch(right, right)
    kb, vb = b.update_and_fetch(right, right)
    assert mx.array_equal(ka, kb) and mx.array_equal(va, vb)


def test_merge_extract_filter_extend_keep_each_row():
    from mlx_lm.models.cache import BatchKVCache, KVCache
    from knurlogic.engine.kvquant import QuantKVCache
    rows = [(_kv((1, 2, n, 32), n), _kv((1, 2, n, 32), n + 50))
            for n in (5, 12, 3)]
    qs, ps = [], []
    for k, v in rows:
        q, p = QuantKVCache(8), KVCache()
        q.update_and_fetch(k, v)
        p.update_and_fetch(k, v)
        qs.append(q)
        ps.append(p)
    qb = QuantKVCache.merge(qs[:2])
    pb = KVCache.merge(ps[:2])
    qb.extend(QuantKVCache.merge(qs[2:]))
    pb.extend(BatchKVCache.merge(ps[2:]))
    step = _kv((3, 2, 1, 32), 99)
    kq, _ = qb.update_and_fetch(step, step)
    kp, _ = pb.update_and_fetch(step, step)
    assert kq.shape == kp.shape
    assert mx.abs(kq.astype(mx.float32)
                  - kp.astype(mx.float32)).max().item() < 0.05
    assert qb.offset.tolist() == pb.offset.tolist()
    qb.filter(mx.array([0, 2]))
    pb.filter(mx.array([0, 2]))
    assert qb.size() == pb.size()
    one = qb.extract(1)
    assert type(one) is QuantKVCache and one.offset == 4
    assert one.nbytes < pb.extract(1).nbytes


def test_bits_are_bf16_8_6_or_4():
    from knurlogic.engine.kvquant import parse_bits
    assert parse_bits("bf16") is None and parse_bits(None) is None
    assert parse_bits("6") == 6
    with pytest.raises(ValueError):
        parse_bits("5")


# --- on a tiny model -----------------------------------------------------------

def _logits(model, prompt, steps=6):
    cache = model.make_cache()
    out = model(mx.array([prompt]), cache=cache)[:, -1]
    got = [out]
    for _ in range(steps):
        t = mx.argmax(out, axis=-1)
        out = model(t[None], cache=cache)[:, -1]
        got.append(out)
    return mx.concatenate(got)


def test_install_quantizes_attention_only_and_greedy_logits_stay_close():
    """qwen3_5 at full_attention_interval 2: two of four layers are
    attention (KVCache); the two deltanet layers keep their state."""
    from mlx_lm.models.cache import ArraysCache
    from knurlogic.engine.kvquant import QuantKVCache, install
    model, _, prompts = _tiny(512)
    ref = _logits(model, prompts[0])
    assert install(model, 8) == 2
    kinds = [type(c) for c in model.make_cache()]
    assert kinds.count(QuantKVCache) == 2 and kinds.count(ArraysCache) == 2
    got = _logits(model, prompts[0])
    assert mx.array_equal(mx.argmax(got, -1), mx.argmax(ref, -1))
    scale = mx.abs(ref).max().item()
    assert mx.abs(got - ref).max().item() < 0.02 * scale


def test_install_refuses_a_model_with_nothing_to_quantize():
    from knurlogic.engine.kvquant import install

    class M:
        def make_cache(self):
            return [object()]
    assert install(M(), 8) == 0
    assert install(M(), None) == 0


@pytest.mark.parametrize("bits", [8, 4])
def test_drafting_on_a_quantized_cache_matches_plain_steps(bits, monkeypatch):
    """Rollback on the quantized cache is exact: drafting every step (a
    random head, rejected nearly always) gives the same tokens as the
    same model decoding without a head. The prompt cache it hands back
    is the quantized kind."""
    from knurlogic.engine.kvquant import QuantKVCache, install
    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    monkeypatch.setenv("KNURLOGIC_MTP_BATCH_MAX_ROWS", "8")
    model, head, prompts = _tiny(512)
    install(model, bits)
    kept = []
    plain = _run(MTPBatchGenerator(model, None, prefill_step_size=16),
                 prompts, 30)
    stats = {}
    draft = _run(MTPBatchGenerator(model, head, stats=stats,
                                   prefill_step_size=16), prompts, 30,
                 on_finish=lambda r: kept.append(r.prompt_cache))
    assert stats["steps"] > 0
    assert draft == plain
    assert any(type(c) is QuantKVCache for c in kept[0])


def test_a_quantized_checkpoint_restores_like_a_fresh_prefill():
    """Segment checkpoints hold the quantized cache; a new turn restored
    from one emits what a fresh prefill of the whole prompt emits, and
    prefills only the new tokens."""
    import copy
    from test_batch_drafting import _drive, _turns
    from knurlogic.engine.kvquant import QuantKVCache, install
    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    model, head, sys_, user, tail_a, next_b = _turns()
    install(model, 8)
    _, ckpts = _drive(MTPBatchGenerator(model, head, prefill_step_size=16),
                      [sys_, user, tail_a], 4)
    key, entry = ckpts[-1]
    assert any(type(c) is QuantKVCache for c in entry)
    gen = MTPBatchGenerator(model, head, prefill_step_size=16)
    restored, _ = _drive(gen, [next_b], 20, cache=copy.deepcopy(entry),
                         prefix=key)
    assert gen._prompt_tokens_counter == len(next_b)
    gen.close()
    fresh, _ = _drive(MTPBatchGenerator(model, head, prefill_step_size=16),
                      [sys_ + user + next_b], 20)
    assert restored == fresh


def test_a_quantized_prefix_trims_back_for_reuse():
    """The prompt cache trims an entry to a shorter shared prefix."""
    from mlx_lm.models.cache import trim_prompt_cache
    from knurlogic.engine.kvquant import QuantKVCache
    a, b = QuantKVCache(8), QuantKVCache(8)
    x = _kv((1, 2, 20, 32))
    a.update_and_fetch(x, x)
    b.update_and_fetch(x[:, :, :12], x[:, :, :12])
    assert trim_prompt_cache([a], 8) == 8 and a.offset == 12
    y = _kv((1, 2, 3, 32), 4)
    ka, _ = a.update_and_fetch(y, y)
    kb, _ = b.update_and_fetch(y, y)
    assert mx.array_equal(ka, kb)


def test_gemma4_quantizes_its_full_attention_and_keeps_the_windows():
    """gemma4: full-attention KVCache quantized, sliding windows kept, and
    the KV-shared layers read what their source layer returned. Teacher-
    forced: this random tiny gemma is so sensitive that merely storing its
    K/V in bf16 moves logits by ~1% of their range, and greedy decode
    flips near-ties either way."""
    from mlx_lm.models.cache import RotatingKVCache
    from fixtures_vision_gemma4 import tiny_gemma4_config, tiny_text_model
    from knurlogic.engine.kvquant import QuantKVCache, install
    tc = dict(tiny_gemma4_config()["text_config"], num_hidden_layers=6,
              num_attention_heads=4, num_key_value_heads=2, head_dim=64,
              global_head_dim=64, intermediate_size=256, sliding_window=16,
              num_kv_shared_layers=2,
              layer_types=["sliding_attention", "full_attention"] * 3)
    model, _ = tiny_text_model(tc)
    mx.random.seed(3)
    prompt = mx.random.randint(0, 512, (40,)).tolist()
    toks = mx.random.randint(0, 512, (8,)).tolist()

    def forced():
        c = model.make_cache()
        out = [model(mx.array([prompt]), cache=c)[:, -1]]
        out += [model(mx.array([[t]]), cache=c)[:, -1] for t in toks]
        return mx.concatenate(out)

    ref = forced()
    assert install(model, 8) >= 1
    kinds = [type(c) for c in model.make_cache()]
    assert QuantKVCache in kinds and RotatingKVCache in kinds
    got = forced()
    assert mx.argmax(got[0]).item() == mx.argmax(ref[0]).item()
    assert mx.abs(got - ref).max().item() < 0.1 * mx.abs(ref).max().item()
