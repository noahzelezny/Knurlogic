"""Batch drafting: the gate is token identity against mlx-lm's own
BatchGenerator, greedy, on a tiny random qwen3_5 with a random head.

A random head is rejected almost every step, so this is the rollback path at
its hardest; the small-vocab case forces drafting every step
(EXO_MTP_BATCH_MAX_ROWS) so accepts happen too. Three prompts of different
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
        monkeypatch.setenv("EXO_MTP_BATCH_MAX_ROWS", "8")
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


def test_the_server_gets_the_drafting_generator_only_for_the_headed_model():
    """Installation is one name swap in the server module. It must decide per
    construction: the head's own model drafts; anything else (a switch to an
    artifact without a head) gets mlx-lm's generator untouched."""
    import types
    from mlx_lm.generate import BatchGenerator
    from knurlogic.engine.serve import drafting, state
    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator

    model, head, _ = _tiny(512)
    other, _, _ = _tiny(512)
    srv = types.SimpleNamespace(BatchGenerator=BatchGenerator,
                                _make_sampler=lambda args, tok: (lambda x: x))
    saved = dict(state.DRAFT), dict(state.SERVED)
    try:
        state.DRAFT.update(head=head, on=True, batch_installed=False)
        state.SERVED["provider"] = types.SimpleNamespace(model=model)
        drafting.install_batch(srv)
        g = srv.BatchGenerator(model, prefill_step_size=16)
        assert isinstance(g, MTPBatchGenerator)
        g.close()
        g = srv.BatchGenerator(other, prefill_step_size=16)
        assert type(g) is BatchGenerator
        g.close()
        state.DRAFT["on"] = False                      # --no-draft
        g = srv.BatchGenerator(model, prefill_step_size=16)
        assert type(g) is BatchGenerator
        g.close()
    finally:
        state.DRAFT.clear(); state.DRAFT.update(saved[0])
        state.SERVED.clear(); state.SERVED.update(saved[1])


def test_a_built_sampler_carries_the_parameters_verification_needs():
    """insert_segments only ever sees a BUILT sampler. Rejection sampling
    needs the temperature itself, so the sampler has to carry it -- or a
    temp-0.7 request would be verified as greedy."""
    import types
    from knurlogic.engine.mtp.batch_generator import sampling_of, tag_samplers

    srv = types.SimpleNamespace(_make_sampler=lambda args, tok: (lambda x: x))
    tag_samplers(srv)
    s = types.SimpleNamespace(temperature=0.7, top_p=0.9, top_k=20, min_p=0.0,
                              xtc_probability=0.0, xtc_threshold=0.0)
    tok = types.SimpleNamespace(eos_token_id=2, encode=lambda t: [10])
    fn = srv._make_sampler(types.SimpleNamespace(sampling=s), tok)
    got = sampling_of(fn)
    assert got["temp"] == 0.7 and got["top_p"] == 0.9 and got["top_k"] == 20
    assert got["xtc_special_tokens"] == [2, 10]


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
        out = real()
        calls["n"] += 1
        if calls["n"] == 2:
            for rs in out:
                if rs.uid == bad:
                    for em in rs.tokens:
                        em.logits = em.logits * mx.array(float("nan"))
        return out
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


def test_finite_checks_rows_in_one_pass():
    from knurlogic.engine.mtp.sampling import finite
    ok = mx.zeros((1, 8))
    assert finite([ok, ok / mx.array(0.0), ok + mx.array(float("inf"))]) \
        == [True, False, False]
    assert finite([]) == []
