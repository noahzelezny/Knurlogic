"""KV-cache precision (engine/kvquant.py): attention K/V stored at 8/6/4
bits, dequantized on fetch. Tiny fixtures only -- nothing here is a
measurement of a real model."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
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


@pytest.mark.parametrize("bits", [None, 8])
def test_a_quantized_checkpoint_restores_like_a_fresh_prefill(bits,
                                                               monkeypatch):
    """Segment checkpoints hold the quantized cache; a new turn restored
    from one emits what a fresh prefill of the whole prompt emits, and
    prefills only the new tokens.

    The generator that stored the checkpoint closes AFTER the restoring one
    has opened, as a server's collected executor does: its hidden-state
    capture used to put back the module it found, cutting the live
    generator's capture out of the trunk, so the first drafting step read
    the last prefill chunk's hidden state ((1, 16, D) against two ids).
    Drafting every step (KNURLOGIC_MTP_BATCH_MAX_ROWS) so that step runs."""
    import copy

    from test_batch_drafting import _drive, _turns

    from knurlogic.engine.kvquant import QuantKVCache, install
    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    monkeypatch.setenv("KNURLOGIC_MTP_BATCH_MAX_ROWS", "8")
    model, head, sys_, user, tail_a, next_b = _turns()
    install(model, bits)
    first = MTPBatchGenerator(model, head, prefill_step_size=16)
    _, ckpts = _drive(first, [sys_, user, tail_a], 4)
    key, entry = ckpts[-1]
    if bits:
        assert any(type(c) is QuantKVCache for c in entry)
    gen = MTPBatchGenerator(model, head, prefill_step_size=16)
    first.close()
    stats = {}
    gen._stats = stats if getattr(gen, "_stats", None) is None else gen._stats
    restored, _ = _drive(gen, [next_b], 20, cache=copy.deepcopy(entry),
                         prefix=key)
    assert gen._counters.prompt_tokens == len(next_b)
    gen.close()
    fresh, _ = _drive(MTPBatchGenerator(model, head, prefill_step_size=16),
                      [sys_ + user + next_b], 20)
    assert len(restored) == 20 and restored == fresh


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
    from fixtures_vision_gemma4 import tiny_gemma4_config, tiny_text_model
    from mlx_lm.models.cache import RotatingKVCache

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


# --- qwen4_exp (Flash-Next) and glm5_next ---------------------------------------
#
# TOLERANCE. 8-bit affine in groups of 64 rounds each K/V element to within
# half a step of its group's range / 255 -- about 0.2% of that range. The
# tiny float32 fixtures move their logits by 0.2-0.3% of the largest |logit|
# at 8 bits (qwen3_5 0.26%, qwen4_exp 0.31%, glm5_next 0.21%), ~1% at 6 and
# 4-6% at 4. The gate is 1% of the largest |logit| at 8 bits -- 3x the
# measured drift, and below what 6 bits already costs, so a regression that
# halves the precision fails -- plus the same greedy token at every position.

def _family(name):
    import pipeline_ring_worker as W
    return W.build(name)


def _forced(model, prompt, toks, widths=(1,)):
    """Teacher-forced logits: the prompt, then `toks` fed `widths[i]` at a
    time (a width of 2-4 is GLM's SMALL_L verify path)."""
    c = model.make_cache()

    def call(x):
        y = model(mx.array([x]), cache=c)
        return (y if isinstance(y, mx.array) else y.logits)[0]
    out = [call(prompt)[-1:]]
    i, w = 0, 0
    while i < len(toks):
        n = widths[w % len(widths)]
        out.append(call(toks[i:i + n]))
        i, w = i + n, w + 1
    return mx.concatenate(out), c


def _close(got, ref, frac):
    assert mx.array_equal(mx.argmax(got, -1), mx.argmax(ref, -1))
    drift = mx.abs(got - ref).max().item() / mx.abs(ref).max().item()
    assert drift < frac, drift


def _prompt(n=90, m=8, seed=3):
    mx.random.seed(seed)
    return (mx.random.randint(0, 200, (n,)).tolist(),
            mx.random.randint(0, 200, (m,)).tolist())


def test_flash_next_quantizes_its_attention_and_keeps_the_indexer():
    import importlib

    from knurlogic.engine.kvquant import install
    model = _family("qwen4_exp")
    Q = importlib.import_module(type(model).__module__)
    prompt, toks = _prompt()
    ref, _ = _forced(model, prompt, toks)
    assert install(model, 8) == 1
    got, cache = _forced(model, prompt, toks)
    _close(got, ref, 0.01)
    attn = [c for c in cache if hasattr(c, "indexer")]
    assert [type(c).__name__ for c in attn] == ["QuantAttnCache"]
    a = attn[0]
    assert type(a.keys) is tuple and a.keys[0].dtype == mx.uint32
    assert a.indexer.keys.dtype != mx.uint32          # exact, as before
    assert a.indexer.keys.shape[1] == a.offset == len(prompt) + len(toks)
    assert isinstance(a, Q._AttnCache) and not hasattr(a, "bits")


def test_flash_next_quantized_cache_trims_and_restores_its_indexer():
    import copy

    from mlx_lm.models.cache import trim_prompt_cache

    from knurlogic.engine.kvquant import install
    model = _family("qwen4_exp")
    install(model, 8)
    prompt, toks = _prompt(40, 4)
    _, cache = _forced(model, prompt, toks)
    a = [c for c in cache if hasattr(c, "indexer")][0]
    b = copy.deepcopy(a)
    b.state = a.state
    assert b.offset == a.offset and b.indexer.keys.shape == a.indexer.keys.shape
    assert trim_prompt_cache([a], 4) == 4
    assert a.offset == a.indexer.keys.shape[1] == len(prompt)


def test_glm_quantizes_the_mla_latent_and_keeps_the_indexer():
    """90 tokens: past index_topk (64), so the DSA picks a sparse set; the
    prefill takes the expanded path, single steps and widths 2-4 the
    absorbed SMALL_L one -- both read the dequantized latent. Seed 4: on
    seed 3 the reference-exact trunk (glm5 PROVENANCE edits 1-5) has a
    near tie (top two logits 4e-4 apart) that 8-bit drift flips; seed 4's
    closest top two are 0.036 apart, 20x the drift."""
    from knurlogic.engine.kvquant import QuantKVCache, install
    model = _family("glm5_next")
    prompt, toks = _prompt(90, 9, seed=4)
    for widths in ((1,), (2, 3, 4)):
        ref, _ = _forced(model, prompt, toks, widths)
        m = _family("glm5_next")
        assert install(m, 8) == 1
        got, cache = _forced(m, prompt, toks, widths)
        _close(got, ref, 0.01)
    fa = [c for c in cache if hasattr(c, "caches")][0]
    latent, indexer = fa.caches
    assert type(latent) is QuantKVCache
    assert type(indexer).__name__ == "KVCache"
    assert indexer.keys.dtype != mx.uint32


@pytest.mark.parametrize("family", ["qwen4_exp", "glm5_next"])
def test_a_quantized_batch_is_each_row_alone(family):
    """merge (admission), filter (a row finishing), extract (the prompt
    cache it hands back) on the family's quantized cache: three rows in one
    batch emit what each emits alone."""
    from knurlogic.engine.kvquant import install
    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    model = _family(family)
    install(model, 8)
    mx.random.seed(1)
    prompts = [mx.random.randint(0, 200, (n,)).tolist() for n in (37, 9, 90)]
    kept = []
    both = _run(MTPBatchGenerator(model, None, prefill_step_size=16),
                prompts, 20, on_finish=lambda r: kept.append(r.prompt_cache))
    alone = [_run(MTPBatchGenerator(model, None, prefill_step_size=16),
                  [p], 20)[0] for p in prompts]
    assert both == alone
    held = [c.caches[0] if hasattr(c, "caches") else c for c in kept[0]]
    assert any(getattr(c, "kv_bits", None) == 8 for c in held)


def _head(family, model):
    import importlib
    arch = importlib.import_module(type(model).__module__)
    if family == "qwen4_exp":
        from knurlogic.engine.families.qwen.heads.qwen4_exp import MTPHead
        h = MTPHead(model, arch)
        D, hc = h.D, h.hc
        h.norm_e = h._norm(D, mx.ones((D,)))
        h.norm_h = h._norm(hc * D, mx.ones((hc * D,)), group_size=D)
        h.fc = 0.02 * mx.random.normal((D, 2 * D))
        return h
    from knurlogic.engine.families.glm5.heads.glm5 import MTPHeadGlm5
    h = MTPHeadGlm5(model, arch)
    for m in h._modules().values():
        m.set_dtype(mx.float32)
        mx.eval(m.parameters())
    return h


@pytest.mark.parametrize("family", ["qwen4_exp", "glm5_next"])
def test_drafting_over_a_quantized_family_cache_matches_plain_steps(
        family, monkeypatch):
    """A random head drafting every step: its rejections roll the quantized
    trunk cache (and the exact indexer beside it) back, GLM's verify runs
    the SMALL_L path over the dequantized latent -- and the tokens are
    the plain steps'. The head's own draft cache stays bf16."""
    from knurlogic.engine.kvquant import install
    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    monkeypatch.setenv("KNURLOGIC_MTP_BATCH_MAX_ROWS", "8")
    model = _family(family)
    install(model, 8)
    mx.random.seed(1)
    prompts = [mx.random.randint(0, 200, (n,)).tolist() for n in (37, 9, 90)]
    head = _head(family, model)
    plain = _run(MTPBatchGenerator(model, None, prefill_step_size=16),
                 prompts, 30)
    stats = {}
    draft = _run(MTPBatchGenerator(model, head, stats=stats,
                                   prefill_step_size=16), prompts, 30)
    assert stats["steps"] > 0
    assert draft == plain


def test_the_glm_head_builds_the_trunks_moe():
    """The head's layer (45) has a trunk MoE layer's weight names and
    shapes, and is built as one: Glm5NextMoE, with the clamped SwiGLU and
    float32 router of glm5_next edits 1-2 (it was mlx-vlm's unclamped
    DeepseekV32MoE)."""
    import importlib

    from knurlogic.engine.families.glm5.heads.glm5 import MTPHeadGlm5
    model = _family("glm5_next")
    arch = importlib.import_module(type(model).__module__)
    h = MTPHeadGlm5(model, arch)
    assert type(h.mlp).__name__ == "Glm5NextMoE"
    assert type(h.mlp) is type(next(
        l.mlp for l in model.model.layers
        if type(l.mlp).__name__ == "Glm5NextMoE"))
