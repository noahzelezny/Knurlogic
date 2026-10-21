"""engine/probe/stop: at a paragraph break inside the thinking, a score at
or over the threshold forces the think-end tokens; nowhere else."""
import mlx.core as mx
import numpy as np
import pytest

from knurlogic.engine.probe import stop as S

# id -> text: 0 "a", 1 ".\n\n", 2 "\n", 3 "<think>", 4 "</think>", 5 "b"
TEXT = ["a", ".\n\n", "\n", "<think>", "</think>", "b"]
V = len(TEXT)


class Tok:
    vocab_size = V
    think_start_tokens = (3,)
    think_end_tokens = (4,)

    def decode(self, ids):
        return "".join(TEXT[i] for i in ids)


def _call(st, hist):
    return st(mx.array(hist), mx.zeros((1, V)))


def _forced(logits):
    row = np.array(logits)[0]
    return int(np.argmax(row)) if np.isinf(row).sum() == V - 1 else None


def test_breaks_are_one_token_or_two_meeting_newlines():
    both, ends, starts = S.break_ids(Tok())
    assert both == {1} and ends == {1, 2} and starts == {2}


def test_a_confident_break_closes_the_think_block():
    rep = {}
    st = S.AutoStop(Tok(), reasoning=True, threshold=0.9, report=rep)
    st.feed(2, mx.array(0.95), None)
    out = _call(st, [0, 1])               # "a.\n\n": a break, read at 0.95
    assert _forced(out) == 4
    assert rep["stopped"] and rep["confidence"] == 0.95
    assert rep["stopped_at_token"] == 2
    # the end token went out; after it nothing is forced
    assert _forced(_call(st, [0, 1, 4])) is None


def test_an_unsure_break_thinks_on():
    rep = {}
    st = S.AutoStop(Tok(), reasoning=True, threshold=0.9, report=rep)
    st.feed(2, mx.array(0.5), None)
    assert _forced(_call(st, [0, 1])) is None
    assert not rep["stopped"] and rep["breaks_read"] == 1
    assert rep["max_confidence"] == 0.5


def test_no_read_for_this_token_means_no_stop():
    st = S.AutoStop(Tok(), reasoning=True, threshold=0.1)
    st.feed(5, mx.array(1.0), None)        # a read for another position
    assert _forced(_call(st, [0, 1])) is None


def test_outside_the_thinking_a_break_is_text():
    st = S.AutoStop(Tok(), reasoning=True, threshold=0.1)
    st.feed(4, mx.array(1.0), None)
    assert _forced(_call(st, [0, 4, 0, 1])) is None   # after </think>
    st2 = S.AutoStop(Tok(), reasoning=False, threshold=0.1)
    st2.feed(2, mx.array(1.0), None)
    assert _forced(_call(st2, [0, 1])) is None


def test_the_model_opening_its_think_block_counts():
    st = S.AutoStop(Tok(), reasoning=False, threshold=0.5)
    st.feed(3, mx.array(0.9), None)
    assert _forced(_call(st, [3, 0, 1])) == 4


def test_two_newline_tokens_meeting_are_a_break():
    st = S.AutoStop(Tok(), reasoning=True, threshold=0.5)
    st.feed(3, mx.array(0.9), None)
    assert _forced(_call(st, [0, 2, 2])) == 4


def test_a_break_is_read_once_though_asked_twice():
    rep = {}
    st = S.AutoStop(Tok(), reasoning=True, threshold=0.99, report=rep)
    st.feed(2, mx.array(0.5), None)
    _call(st, [0, 1])
    _call(st, [0, 1])                      # the draft's and the verify's ask
    assert rep["breaks_read"] == 1


def test_onboarding_keeps_states_and_never_stops():
    st = S.AutoStop(Tok(), reasoning=True, threshold=None, keep_states=True)
    st.feed(2, mx.array(0.99), mx.ones((3, 4)))
    assert _forced(_call(st, [0, 1])) is None
    assert len(st.states) == 1 and st.states[0][0] == 2
    assert st.states[0][1].shape == (3, 4)
    assert st.tokens == [0, 1]


# ---- in the batch engine: the loop feeds the stop, the stop closes ----

class _LoopTok:
    """Vocab 8: every token is a paragraph break but 6 (<think>) and 7
    (</think>)."""
    vocab_size = 8
    think_start_tokens = (6,)
    think_end_tokens = (7,)

    def decode(self, ids):
        return "".join({6: "<think>", 7: "</think>"}.get(i, "x\n\n")
                       for i in ids)


def _generate(always, monkeypatch, bias=None):
    """Three tokens of one row through MTPBatchGenerator, with a probe that
    always (bias +30) or never (-30) reads the answer as ready."""
    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    from knurlogic.engine.probe import store
    from knurlogic.engine.probe.tap import Tap
    test_batch_drafting = pytest.importorskip("test_batch_drafting")
    if always:
        monkeypatch.setenv("KNURLOGIC_MTP_BATCH_MAX_ROWS", "8")
    model, head, prompts = test_batch_drafting._tiny(8)
    procs, stop, tap = [], None, None
    if bias is not None:
        D = 128
        probe = store.Probe(base="toy", layer=2, mean=np.zeros(D, np.float32),
                            scale=np.ones(D, np.float32),
                            w=np.zeros(D, np.float32), b=bias, threshold=0.9)
        tap = Tap(model, [2])
        monkeypatch.setitem(store.ACTIVE, "tap", tap)
        monkeypatch.setitem(store.ACTIVE, "probe", probe)
        stop = S.AutoStop(_LoopTok(), reasoning=True, threshold=0.9)
        procs = [stop]
    gen = MTPBatchGenerator(model, head, prefill_step_size=16)
    (uid,) = gen.insert_segments([[prompts[0]]], max_tokens=[6],
                                 logits_processors=[procs])
    out = []
    while len(out) < 6:
        _, rs = gen.next()
        out += [r.token for r in rs if r.uid == uid]
    gen.close()
    if tap is not None:
        tap.remove()
    return out, stop


@pytest.mark.parametrize("always", [False, True])
def test_the_engine_closes_the_think_block_at_a_ready_break(always,
                                                            monkeypatch):
    out, stop = _generate(always, monkeypatch, bias=30.0)
    first = next(i for i, t in enumerate(out) if t not in (6, 7))
    assert out[first + 1] == 7, out
    assert stop.report["stopped"]


@pytest.mark.parametrize("always", [False, True])
def test_an_unready_probe_leaves_every_token_as_it_was(always, monkeypatch):
    plain, _ = _generate(always, monkeypatch)
    out, stop = _generate(always, monkeypatch, bias=-30.0)
    assert out == plain
    assert not stop.report["stopped"] and stop.report["breaks_read"] > 0
