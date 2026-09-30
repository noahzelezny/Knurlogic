"""Batch drafting: the gate is token identity against mlx-lm's own
BatchGenerator, greedy, on a tiny random qwen3_5 with a random head.

A random head is rejected almost every step, so this is the rollback path at
its hardest; the small-vocab case forces drafting every step
(KNURLOGIC_MTP_BATCH_MAX_ROWS) so accepts happen too. Three prompts of different
lengths, admitted one per call, so rows join a batch already decoding.

WHY NOT VOCAB 4. Measured: the verify forward is 2 tokens wide and the plain
one is 1, and through the recurrent kernels that alone moves logprobs by up
to 2e-2 in float32. At vocab 4 a row reached a top-2 margin of 8.6e-4 and
flipped -- a near-tie, not a logic fault (which would diverge at once, at
every vocab). A test that fails on numerics trains people to ignore it.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

mx = pytest.importorskip("mlx.core")


def _tiny(vocab):
    from knurlogic.engine import register
    register.register("qwen3_5")
    from mlx_lm.models import qwen3_5 as arch
    from knurlogic.engine.families.qwen.heads.qwen35 import MTPHeadQwen35

    mx.random.seed(0)
    tc = dict(model_type="qwen3_5", hidden_size=128, intermediate_size=256,
              num_hidden_layers=4, num_attention_heads=4,
              num_key_value_heads=2, head_dim=32, vocab_size=vocab,
              linear_num_value_heads=4, linear_num_key_heads=2,
              linear_key_head_dim=32, linear_value_head_dim=32,
              full_attention_interval=2, tie_word_embeddings=False)
    model = arch.Model(arch.ModelArgs(model_type="qwen3_5", text_config=tc))
    model.set_dtype(mx.float32)
    head = MTPHeadQwen35(model, arch, norm_shift=0.0)
    D = tc["hidden_size"]
    for n in ("norm_e", "norm_h", "norm_out"):
        setattr(head, n, head._norm(mx.ones((D,)), 0.0))
    head.fc = mx.random.normal((D, 2 * D)) * 0.05
    head.block.set_dtype(mx.float32)
    mx.eval(model.parameters(), head.block.parameters(), head.fc)
    prompts = [mx.random.randint(0, vocab, (n,)).tolist() for n in (37, 9, 70)]
    return model, head, prompts


def _run(gen, prompts, max_tokens, on_finish=None):
    uids = gen.insert(prompts, max_tokens=[max_tokens] * len(prompts))
    out, done = {u: [] for u in uids}, set()
    for _ in range(10_000):
        _, responses = gen.next()
        for r in responses:
            assert r.uid not in done, "a response after its finish"
            out[r.uid].append(r.token)
            if r.finish_reason:
                done.add(r.uid)
                if on_finish:
                    on_finish(r)
        if len(done) == len(uids):
            break
    gen.close()
    return [out[u] for u in uids]


@pytest.mark.parametrize("vocab,always", [(512, False), (8, True)])
def test_drafting_batch_is_token_identical_to_mlx_lm(vocab, always,
                                                     monkeypatch):
    from mlx_lm.generate import BatchGenerator
    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator, trunk_offset

    if always:
        monkeypatch.setenv("KNURLOGIC_MTP_BATCH_MAX_ROWS", "8")
    model, head, prompts = _tiny(vocab)
    n_trunk = len(model.make_cache())

    def finished(r):
        # The prompt cache is keyed by all_tokens: it must be exactly what
        # the returned cache holds, and the head must ride beside the trunk
        # or a restore can never draft.
        assert trunk_offset(r.prompt_cache) == len(r.all_tokens)
        assert len(r.prompt_cache) == n_trunk + 1

    plain = _run(BatchGenerator(model, prefill_step_size=16), prompts, 60)
    stats = {}
    draft = _run(MTPBatchGenerator(model, head, stats=stats,
                                   prefill_step_size=16), prompts, 60, finished)
    assert stats["requests"] == 3          # the channel: it went through MTP
    assert stats["steps"] > 0
    if always:
        assert stats["accepted"] > 0       # the accept branch ran, not just reject
    assert draft == plain


def test_a_restored_prefix_keeps_drafting():
    """The entry handed back to the prompt cache, restored at its own length,
    admits WITH its head -- not the silent fresh prefill a misaligned head
    would cause."""
    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator, split_pool_entry

    model, head, prompts = _tiny(512)
    got = {}
    _run(MTPBatchGenerator(model, head, prefill_step_size=16), prompts[:1], 12,
         on_finish=lambda r: got.update(entry=r.prompt_cache, toks=r.all_tokens))
    trunk, hc, hit = split_pool_entry(got["entry"], len(model.make_cache()),
                                      drafts=True, hit_len=len(got["toks"]))
    assert hit == len(got["toks"]) and hc is not None


def _cachelist_at(n):
    """GLM's head cache shape: CacheList(main KV, indexer KV), at n."""
    from mlx_lm.models.cache import CacheList, KVCache
    cl = CacheList(KVCache(), KVCache())
    for c in cl.caches:
        c.update_and_fetch(mx.zeros((1, 1, n, 4)), mx.zeros((1, 1, n, 4)))
    return cl


def test_a_composite_cache_has_its_members_position():
    from knurlogic.engine.mtp.caches import position
    from mlx_lm.models.cache import ArraysCache
    assert position(_cachelist_at(7)) == 7
    assert position(ArraysCache(size=2)) is None


def test_a_glm_style_head_restores_at_a_prefix_and_at_a_checkpoint():
    """GLM's head cache is a CacheList, which has no `offset`: read as -1,
    every entry -- whole prompt or segment checkpoint -- was discarded as
    'no aligned head cache' (a 472-token system prompt)."""
    from knurlogic.engine.mtp.batch_generator import HeadCarry, split_pool_entry
    trunk = [object()]
    _, hc, hit = split_pool_entry(trunk + [_cachelist_at(10)], 1,
                                  drafts=True, hit_len=10)
    assert hit == 10 and hc is not None
    replayed = []
    _, hc, hit = split_pool_entry(
        trunk + [_cachelist_at(9), HeadCarry(mx.zeros((1, 1, 4)))], 1,
        drafts=True, hit_len=10, replay=lambda c, h: replayed.append(c))
    assert hit == 10 and hc is not None and replayed == [hc]


def test_sampling_params_are_read_as_given():
    """The executor passes a row's sampling as a dict; anything else is
    greedy -- rejection sampling needs the temperature itself."""
    from knurlogic.engine.mtp.batch_generator import sampling_of
    assert sampling_of({"temp": 0.7, "top_p": 0.9}) == {"temp": 0.7,
                                                         "top_p": 0.9}
    assert sampling_of(lambda x: x) is None and sampling_of(None) is None


# --- segment checkpoints -------------------------------------------------------

def _drive(gen, segments, max_tokens, cache=None, prefix=()):
    """One request through insert_segments/next, the way mlx-lm's server
    drives it: end-of-segment responses are answered with extract_cache.
    Returns (tokens, [(key, entry)] checkpoints)."""
    (uid,) = gen.insert_segments(segments=[segments], max_tokens=[max_tokens],
                                 caches=[cache], all_tokens=[list(prefix)])
    toks, ckpts = [], []
    for _ in range(10_000):
        prs, grs = gen.next()
        eos = [r.uid for r in prs if r.end_of_segment and not r.end_of_prompt]
        for u, (entry, key) in gen.extract_cache(eos).items():
            ckpts.append((list(key), entry))
        done = False
        for r in grs:
            toks.append(r.token)
            done = done or r.finish_reason is not None
        if done:
            break
    return toks, ckpts


def _turns(vocab=512):
    model, head, _ = _tiny(vocab)
    mx.random.seed(1)
    sys_, user = (mx.random.randint(0, vocab, (n,)).tolist() for n in (20, 30))
    tail_a = mx.random.randint(0, vocab, (3,)).tolist()
    next_b = mx.random.randint(0, vocab, (15,)).tolist()
    return model, head, sys_, user, tail_a, next_b


def test_prefill_stores_a_checkpoint_at_each_segment_end():
    import copy
    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    model, head, sys_, user, tail_a, _ = _turns()
    _, ckpts = _drive(MTPBatchGenerator(model, head, prefill_step_size=16),
                      [sys_, user, tail_a], 4)
    assert [len(k) for k, _ in ckpts] == [20, 50]
    assert ckpts[1][0] == sys_ + user


def test_a_new_turn_restored_from_the_checkpoint_matches_a_fresh_prefill():
    """The case the checkpoint exists for: the next turn agrees with the
    stored prompt only up to the user segment's end (Qwen3.6 re-renders the
    assistant tail), on a model whose linear-attention caches cannot be
    trimmed. Restored there -- head replayed one step with the NEW token --
    it must emit exactly what a from-scratch prefill emits, and prefill only
    the new tokens (the channel: without the replay the head is misaligned
    and the row falls back to a full prefill)."""
    import copy
    from mlx_lm.generate import BatchGenerator
    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    model, head, sys_, user, tail_a, next_b = _turns()
    _, ckpts = _drive(MTPBatchGenerator(model, head, prefill_step_size=16),
                      [sys_, user, tail_a], 4)
    key, entry = ckpts[-1]
    prompt_b = sys_ + user + next_b

    gen = MTPBatchGenerator(model, head, prefill_step_size=16)
    restored, _ = _drive(gen, [next_b], 30, cache=copy.deepcopy(entry),
                         prefix=key)
    assert gen._prompt_tokens_counter == len(next_b)
    gen.close()

    fresh, _ = _drive(MTPBatchGenerator(model, head, prefill_step_size=16),
                      [prompt_b], 30)
    plain, _ = _drive(BatchGenerator(model, prefill_step_size=16),
                      [prompt_b], 30)
    assert restored == fresh == plain


def test_checkpoint_restore_can_fail(monkeypatch):
    """Drop the carried h and the same restore can no longer draft from the
    checkpoint: the row prefills everything again."""
    import copy
    from knurlogic.engine.mtp import batch_generator as bg
    model, head, sys_, user, tail_a, next_b = _turns()
    _, ckpts = _drive(bg.MTPBatchGenerator(model, head, prefill_step_size=16),
                      [sys_, user, tail_a], 4)
    key, entry = ckpts[-1]
    entry = [e for e in entry if not isinstance(e, bg.HeadCarry)]
    gen = bg.MTPBatchGenerator(model, head, prefill_step_size=16)
    _drive(gen, [next_b], 5, cache=copy.deepcopy(entry), prefix=key)
    assert gen._prompt_tokens_counter == len(sys_ + user + next_b)


# --- a failed admission fails its request, not the server ----------------------

def test_a_failed_admission_fails_only_its_request(monkeypatch):
    """An exception while admitting one row used to escape next() and end
    mlx-lm's generation thread: that request and every later one hung with
    no error. It must reach that request as an exception in its progress,
    while the other row still finishes."""
    from knurlogic.engine.mtp import batch_generator as bg
    model, head, prompts = _tiny(512)
    gen = bg.MTPBatchGenerator(model, head, prefill_step_size=16)
    real = bg.admit
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return real(*a, **k)

    monkeypatch.setattr(bg, "admit", flaky)
    bad, good = gen.insert(prompts[:2], max_tokens=[5, 5])
    errors, done = [], set()
    for _ in range(200):
        prs, grs = gen.next()                       # must not raise
        errors += [r.progress for r in prs if r.uid == bad
                   and isinstance(r.progress, Exception)]
        done |= {r.uid for r in grs if r.finish_reason}
        if good in done and errors:
            break
    assert [str(e) for e in errors] == ["boom"]
    assert good in done and bad not in done
    gen.remove([bad])                               # the server's cleanup
    assert not any(r.uid == bad for r in gen.next()[0])
    gen.close()


def test_failed_admission_can_fail(monkeypatch):
    """Without the guard the same exception escapes next()."""
    from knurlogic.engine.mtp import batch_generator as bg
    model, head, prompts = _tiny(512)
    gen = bg.MTPBatchGenerator(model, head, prefill_step_size=16)
    monkeypatch.setattr(bg, "admit", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("boom")))
    monkeypatch.setattr(gen, "_failed_responses", lambda: [])
    gen.insert(prompts[:1], max_tokens=[5])
    prs, _ = gen.next()
    assert not any(isinstance(r.progress, Exception) for r in prs)


def test_a_failed_decode_step_fails_its_rows_not_the_server(monkeypatch):
    """An exception in a decode step used to end the generation thread too.
    The rows in the step get it as progress; the next request still runs."""
    from knurlogic.engine.mtp import batch_generator as bg
    model, head, prompts = _tiny(512)
    gen = bg.MTPBatchGenerator(model, head, prefill_step_size=16)
    (uid,) = gen.insert(prompts[:1], max_tokens=[5])
    real = gen._batch.step
    monkeypatch.setattr(gen._batch, "step", lambda: (_ for _ in ()).throw(
        RuntimeError("decode boom")))
    errs = []
    for _ in range(5):
        prs, _ = gen.next()                         # must not raise
        errs += [r.progress for r in prs if isinstance(r.progress, Exception)]
    assert [str(e) for e in errs] == ["decode boom"]
    gen.remove([uid])
    monkeypatch.setattr(gen._batch, "step", real)
    (u2,) = gen.insert(prompts[1:2], max_tokens=[3])
    done = False
    for _ in range(50):
        _, grs = gen.next()
        done = done or any(r.uid == u2 and r.finish_reason for r in grs)
        if done:
            break
    assert done
    gen.close()


# --- NaN guard: a row whose logits go non-finite fails, and says where ---------

def test_nan_logits_fail_that_row_with_the_position_not_bangs(monkeypatch):
    """Sampled, NaN logits are token 0 forever ('!!!!!' in Qwen). The row is
    failed before any token of the bad step leaves; the other row finishes."""
    from knurlogic.engine.mtp import batch_generator as bg
    from knurlogic.engine.mtp.sampling import NonFiniteLogits
    model, head, prompts = _tiny(512)
    gen = bg.MTPBatchGenerator(model, head, prefill_step_size=16)
    bad, good = gen.insert(prompts[:2], max_tokens=[12, 12])
    real = gen._batch.step
    calls = {"n": 0}

    def poisoned():
        # The trunk row the next token is emitted from goes NaN for one
        # row, as a bad forward would leave it; the step's own eval must
        # notice (the flag is computed there, not after).
        calls["n"] += 1
        b = gen._batch
        if calls["n"] == 2 and bad in b.uids:
            i = b.uids.index(bad)
            mask = (mx.arange(b.row_t1.shape[0]) == i)[:, None]
            b.row_t1 = mx.where(mask, mx.array(float("nan")), b.row_t1)
        return real()
    monkeypatch.setattr(gen._batch, "step", poisoned)
    errors, toks, done = [], {bad: 0, good: 0}, set()
    for _ in range(200):
        prs, grs = gen.next()                       # must not raise
        errors += [r.progress for r in prs if r.uid == bad
                   and isinstance(r.progress, Exception)]
        for r in grs:
            toks[r.uid] += 1
            if r.finish_reason:
                done.add(r.uid)
        if good in done and errors:
            break
    [err] = errors
    assert isinstance(err, NonFiniteLogits) and "non-finite" in str(err)
    assert "generated token" in str(err)
    assert good in done and bad not in done
    assert bad not in gen._batch.uids
    gen.remove([bad])
    gen.close()


def test_a_headless_model_reuses_a_shared_prefix():
    """No head, nothing to align: a stored prefix is used, not discarded.
    gemma re-prefilled a 509-token shared system prompt on every request
    because admission asked for a head cache the model cannot have."""
    import copy
    from knurlogic.engine.mtp import batch_generator as bg
    model, _, sys_, user, tail_a, next_b = _turns()
    gen = bg.MTPBatchGenerator(model, None, prefill_step_size=16)
    _, ckpts = _drive(gen, [sys_, user + tail_a], 3)
    key, entry = next((k, e) for k, e in ckpts if k == sys_)
    gen2 = bg.MTPBatchGenerator(model, None, prefill_step_size=16)
    _drive(gen2, [next_b], 3, cache=copy.deepcopy(entry), prefix=key)
    assert gen2._prompt_tokens_counter == len(next_b)


def test_every_prefill_chunk_is_reported_to_the_job_marker(monkeypatch):
    """A long prompt is admitted in ONE engine step; the cluster watcher
    judges a stall by progress, so each prefill chunk says it finished
    (cluster/jobs.chunk_done). 70 tokens at 16 a chunk: 5 chunks."""
    from knurlogic.engine.mtp import batch_loop as bl
    from knurlogic.engine.mtp import batch_generator as bg
    model, head, prompts = _tiny(512)
    n = {"c": 0}
    monkeypatch.setattr(bl, "chunk_done", lambda: n.__setitem__("c", n["c"] + 1))
    _run(bg.MTPBatchGenerator(model, head, prefill_step_size=16),
         [prompts[2]], 2)
    assert n["c"] == 5
