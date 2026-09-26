"""The executor protocol over the local batch engine (docs/SERVER.md step 1):
same tokens as the engine driven directly, typed events, failures as
RowFailure, checkpoints and finished caches as events, the cache report
delivered to the object admission names."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

mx = pytest.importorskip("mlx.core")

from test_batch_drafting import _run, _tiny  # noqa: E402


def _drain(ex, uids, limit=10_000):
    toks, events = {u: [] for u in uids}, []
    done = set()
    for _ in range(limit):
        evs = ex.step()
        events += evs
        for e in evs:
            if type(e).__name__ == "Token":
                assert e.uid not in done, "a token after its finish"
                toks[e.uid].append(e.token)
            if type(e).__name__ in ("Finished", "RowFailure"):
                done.add(e.uid)
        if done >= set(uids):
            break
    return toks, events


def _executor(model, head, **kw):
    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    from knurlogic.engine.runtime.executor import LocalExecutor
    return LocalExecutor(MTPBatchGenerator(model, head, prefill_step_size=16,
                                           **kw))


def _assert_key_holds_stream(f, prompt, streamed):
    """Finished.tokens keys the cache: the prompt, every streamed token, and
    at most ONE more -- a drafting step commits two tokens to the cache, and
    when the row finishes on the first the second is in the cache (and so in
    the key) but never streamed. Whether the last step drafted depends on the
    timed draft/plain choice (drafting_pays), so the key's tail is not the
    stream's tail; the cache's length is the key's, always."""
    from knurlogic.engine.mtp.batch_generator import trunk_offset
    n = len(prompt)
    assert f.tokens[:n] == prompt
    assert f.tokens[n:n + len(streamed)] == streamed
    assert len(f.tokens) - n - len(streamed) in (0, 1)
    assert f.cache and trunk_offset(f.cache) == len(f.tokens)


def test_the_executor_emits_the_engines_own_tokens():
    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    from knurlogic.engine.runtime.executor import Admission, Finished, Token
    model, head, prompts = _tiny(512)
    direct = _run(MTPBatchGenerator(model, head, prefill_step_size=16),
                  prompts, 24)
    ex = _executor(model, head)
    uids = [ex.insert(Admission(segments=[p], max_tokens=24)) for p in prompts]
    toks, events = _drain(ex, uids)
    assert [toks[u] for u in uids] == direct
    fin = [e for e in events if isinstance(e, Finished)]
    assert sorted(f.uid for f in fin) == sorted(uids)
    for f in fin:
        _assert_key_holds_stream(f, prompts[uids.index(f.uid)], toks[f.uid])
    lps = [e.logprob for e in events if isinstance(e, Token)]
    assert all(isinstance(x, float) and x <= 0 for x in lps)
    ex.close()


@pytest.mark.parametrize("always", [False, True])
def test_a_row_finishing_mid_draft_streams_exactly_max_tokens(always,
                                                              monkeypatch):
    """Drafting forced (every step commits two) and max_tokens=1: the row
    finishes on the first, so the second is in its cache and key but must
    not be streamed. Without the force the timed choice decides; the
    stream is the same either way."""
    from knurlogic.engine.runtime.executor import Admission, Finished
    if always:
        monkeypatch.setenv("KNURLOGIC_MTP_BATCH_MAX_ROWS", "8")
    model, head, prompts = _tiny(512)
    ex = _executor(model, head)
    uids = [ex.insert(Admission(segments=[p], max_tokens=1)) for p in prompts]
    toks, events = _drain(ex, uids)
    for f in (e for e in events if isinstance(e, Finished)):
        p = prompts[uids.index(f.uid)]
        assert len(toks[f.uid]) == 1
        _assert_key_holds_stream(f, p, toks[f.uid])
        if always:
            assert len(f.tokens) == len(p) + 2
    ex.close()


def test_top_logprobs_only_when_asked_and_sorted():
    from knurlogic.engine.runtime.executor import Admission, Token
    model, head, prompts = _tiny(512)
    ex = _executor(model, head)
    a = ex.insert(Admission(segments=[prompts[0]], max_tokens=4,
                            top_logprobs=3))
    b = ex.insert(Admission(segments=[prompts[1]], max_tokens=4))
    _, events = _drain(ex, [a, b])
    for e in (e for e in events if isinstance(e, Token)):
        if e.uid == b:
            assert e.top_logprobs is None
        else:
            vals = [v for _, v in e.top_logprobs]
            assert len(vals) == 3 and vals == sorted(vals, reverse=True)
    ex.close()


def test_a_failing_admission_is_a_row_failure_beside_a_working_row(
        monkeypatch):
    from knurlogic.engine.runtime.executor import (Admission, Progress,
                                                   RowFailure, Token)
    model, head, prompts = _tiny(512)
    ex = _executor(model, head)
    real = ex.gen._admit_one
    boom = {"n": 0}

    def admit():
        boom["n"] += 1
        if boom["n"] == 1:
            ex.gen._unprocessed_sequences.popleft()
            raise RuntimeError("bad admission")
        return real()
    monkeypatch.setattr(ex.gen, "_admit_one", admit)
    bad = ex.insert(Admission(segments=[prompts[0]], max_tokens=6))
    good = ex.insert(Admission(segments=[prompts[1]], max_tokens=6))
    toks, events = _drain(ex, [bad, good])
    [f] = [e for e in events if isinstance(e, RowFailure)]
    assert f.uid == bad and "bad admission" in str(f.error)
    # nothing about the failed row after its failure
    after = events[events.index(f) + 1:]
    assert not [e for e in after if getattr(e, "uid", None) == bad]
    assert not [e for e in events
                if isinstance(e, (Progress, Token)) and e.uid == bad]
    assert len(toks[good]) == 6
    ex.close()


def test_a_segment_end_is_a_checkpoint_event_and_the_report_arrives():
    from knurlogic.engine.runtime.executor import Admission, Checkpoint

    class Req:
        pass
    model, head, prompts = _tiny(512)
    ex = _executor(model, head)
    req = Req()
    sys_, user = prompts[2][:40], prompts[2][40:]
    uid = ex.insert(Admission(segments=[sys_, user], max_tokens=4, report=req))
    _, events = _drain(ex, [uid])
    # The system segment's end, and the prompt less its last token (the
    # engine feeds that token as its own segment, as mlx-lm does).
    cps = [e for e in events if isinstance(e, Checkpoint)]
    assert [c.tokens for c in cps] == [sys_, prompts[2][:-1]]
    assert all(c.uid == uid and c.cache for c in cps)
    from knurlogic.engine.serve import cache_report
    rep = cache_report.of(req)
    assert rep["checkpoints_stored"] == 2 and rep["prefilled"] == len(prompts[2])
    ex.close()


def test_sampling_params_reach_the_engine_as_given():
    """Temperature goes in as a dict, not a sampler tagged after the fact:
    draws at high temperature differ from greedy, and both rows finish (a
    wrong key would fail the row, and differ for the wrong reason)."""
    from knurlogic.engine.runtime.executor import Admission
    model, head, prompts = _tiny(512)
    ex = _executor(model, head)
    g = ex.insert(Admission(segments=[prompts[0]], max_tokens=16))
    t = ex.insert(Admission(segments=[prompts[0]], max_tokens=16,
                            sampling={"temp": 5.0}))
    toks, events = _drain(ex, [g, t])
    assert not [e for e in events if type(e).__name__ == "RowFailure"]
    assert len(toks[g]) == len(toks[t]) == 16 and toks[g] != toks[t]
    ex.close()


@pytest.mark.parametrize("vocab,always", [(512, False), (8, True)])
def test_a_seeded_row_draws_the_same_alone_and_in_a_batch(vocab, always,
                                                           monkeypatch):
    """A seed is the row's own key: the same tokens whether it runs alone
    or beside other sampling rows (which draw from their own streams), on
    the drafting path too -- rejection sampling takes the row's keys."""
    from knurlogic.engine.runtime.executor import Admission
    if always:
        monkeypatch.setenv("KNURLOGIC_MTP_BATCH_MAX_ROWS", "8")
    model, head, prompts = _tiny(vocab)
    s = {"temp": 1.0, "seed": 7}

    ex = _executor(model, head)
    a = ex.insert(Admission(segments=[prompts[0]], max_tokens=20, sampling=s))
    alone, _ = _drain(ex, [a])
    ex.close()

    ex = _executor(model, head)
    others = [ex.insert(Admission(segments=[p], max_tokens=20,
                                  sampling={"temp": 1.0}))
              for p in prompts[1:]]
    b = ex.insert(Admission(segments=[prompts[0]], max_tokens=20, sampling=s))
    c = ex.insert(Admission(segments=[prompts[0]], max_tokens=20,
                            sampling={"temp": 1.0, "seed": 8}))
    batched, _ = _drain(ex, others + [b, c])
    ex.close()
    assert batched[b] == alone[a]
    assert batched[c] != alone[a]


def test_seeded_draws_are_addressed_by_position_and_couple_draft_to_target():
    """key(seed, n) alone decides the token at n: the same key gives the
    same draw, another position or seed another; and a draft distribution
    equal to the target is accepted every time (shared Gumbel noise)."""
    from knurlogic.engine.mtp.sampling import Keys, make_distribution
    dist = make_distribution(temp=1.0)
    logits = mx.random.normal((1, 64))
    d = dist(logits)
    k = Keys(3)
    draws = [d.sample(k.at(n)).item() for n in range(40)]
    assert draws == [d.sample(Keys(3).at(n)).item() for n in range(40)]
    assert draws != [d.sample(Keys(4).at(n)).item() for n in range(40)]
    assert len(set(draws)) > 5
    assert all(d.sample(k.at(n)).item() == dist(logits).sample(k.at(n)).item()
               for n in range(40))


def test_the_draft_step_hands_processors_the_same_history_as_a_plain_one(
        monkeypatch):
    """Position n+1's processors see t1 at the end of the history, drafting
    or not (build review item 1). The probe boosts token id 10 + len(history),
    so the output counts up 10, 11, 12, ... only if every position's
    history has exactly the tokens before it."""
    from knurlogic.engine.runtime.executor import Admission
    model, head, prompts = _tiny(512)

    def count(hist, row):
        return row + 1e4 * (mx.arange(row.shape[-1]) == 10 + hist.size)

    def run(max_rows):
        monkeypatch.setenv("KNURLOGIC_MTP_BATCH_MAX_ROWS", str(max_rows))
        ex = _executor(model, head)
        u = ex.insert(Admission(segments=[prompts[0]], max_tokens=12,
                                processors=[count]))
        toks, _ = _drain(ex, [u])
        ex.close()
        return toks[u]
    plain = run(0)                 # never drafting
    assert plain == list(range(10, 22))
    assert run(8) == plain         # always drafting


def test_removing_a_row_frees_its_memory_now_not_at_the_next_step():
    """Filtering the batch is lazy in MLX: without an eval the old
    full-width arrays stay alive, and the scheduler's memory guard --
    stopping one row to get back under the limit -- saw no drop and
    stopped every row (Fable 5.1)."""
    import gc
    import mlx.core as mx
    from knurlogic.engine.runtime.executor import Admission
    model, head, prompts = _tiny(512)
    ex = _executor(model, head)
    uids = [ex.insert(Admission(segments=[p * 8], max_tokens=64))
            for p in prompts[:3]]
    for _ in range(6):
        ex.step()
    gc.collect()
    mx.clear_cache()
    before = mx.get_active_memory()
    ex.remove([uids[-1]])
    gc.collect()
    mx.clear_cache()
    assert mx.get_active_memory() < before
    ex.close()
