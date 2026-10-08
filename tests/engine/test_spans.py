"""usage.knurlogic.timing.spans_s: a request's wall time, partitioned.

Pure host clocks, so these run without mlx: the cursor itself, and the
scheduler's step loop driven by a fake executor."""
import time
from types import SimpleNamespace as NS

from knurlogic.engine.runtime import prompt as P
from knurlogic.engine.runtime import scheduler as S
from knurlogic.engine.runtime.executor import Finished, Token
from knurlogic.engine.runtime.spans import Spans, step_bucket


def test_the_buckets_sum_to_the_whole_by_construction():
    sp = Spans(10.0)
    sp.to("queue", 10.5)
    sp.to("tokenize", 10.7)
    sp.to("prefill_forward", 12.0)
    r = sp.report(12.0)
    assert r["spans_s"] == {"prefill_forward": 1.3, "queue": 0.5,
                            "tokenize": 0.2}
    assert r["spans_whole_s"] == 2.0 and r["spans_unaccounted_s"] == 0.0


def test_a_skipped_mark_shows_as_unaccounted_not_as_a_bucket():
    sp = Spans(0.0)
    sp.to("queue", 1.0)
    assert sp.report(3.0)["spans_unaccounted_s"] == 2.0


def test_a_backwards_clock_charges_nothing():
    sp = Spans(5.0)
    assert sp.to("queue", 4.0) == 0.0 and sp.last == 5.0


def test_bucket_names():
    assert step_bucket(True, "forward") == "prefill_forward"
    assert step_bucket(False, "gap") == "decode_gap"
    assert step_bucket(False, "forward", shared=True) == "decode_forward_shared"
    # a row not yet admitted, in a step that admitted another: in line
    assert step_bucket(True, "forward", shared=True) == "prefill_waiting"
    assert step_bucket(True, "gap", shared=True) == "prefill_gap"


def _bare(**host):
    """A Scheduler without __init__ (which builds an mlx prompt cache):
    only the attributes submit, _step and _done read."""
    import queue
    import threading
    s = S.Scheduler.__new__(S.Scheduler)
    s.host = NS(state="ready", **host)
    s._aborted = None
    s._jobs = queue.Queue()
    s._wake = threading.Event()
    s._rows, s._chunk_of = {}, {}
    s.tensor = None
    s.prefill_step_size = 2048
    return s


def test_off_switch(monkeypatch):
    monkeypatch.setenv("KNURLOGIC_TIMING_SPANS", "off")
    s = _bare()
    j = s.submit(S.Job(P.ChatRequest(), P.PromptArgs()))
    assert j.spans is None


def test_http_build_is_charged_from_the_received_stamp():
    s = _bare()
    j = S.Job(P.ChatRequest(), P.PromptArgs())
    j.received = time.perf_counter() - 0.25
    s.submit(j)
    assert j.spans.buckets["http_build"] >= 0.25


class _Text:
    """The request's text stage, reduced to what _step and _done touch."""
    finished = False

    def feed(self, e):
        return None

    def finish(self, why):
        return None

    def usage(self, report):
        return {"completion_tokens": 2}


class _Ex:
    """Two steps: the first prefills and emits the first token, the second
    emits the last and finishes. Each sleeps so the forward is visible."""

    def __init__(self):
        self.n = 0

    def step(self):
        time.sleep(0.02)
        self.n += 1
        if self.n == 1:
            return [Token(1, 5, 0.0)]
        return [Token(1, 6, 0.0, finish="stop"), Finished(1, [5, 6], [])]

    def remove(self, uids):
        pass


def test_the_step_loop_partitions_a_request():
    s = _bare(model_key="m")
    s._reset_peak = lambda: 0
    s._measure = lambda *a, **k: None
    s._learn = lambda *a, **k: None
    s.cache = NS(insert=lambda *a, **k: None)
    s._ex = _Ex()
    job = S.Job(P.ChatRequest(), P.PromptArgs())
    job.request_id = "client-1"
    job = s.submit(job)
    job.spans.to("admit_other")
    s._rows = {1: S._Row(job, _Text(), [], admitted=time.perf_counter())}
    s._step()
    s._step()
    kind, usage = job.outbox.get_nowait()
    while kind != "done":
        kind, usage = job.outbox.get_nowait()
    assert usage["knurlogic"]["request_id"] == "client-1"
    t = usage["knurlogic"]["timing"]
    spans = t["spans_s"]
    assert spans["prefill_forward"] >= 0.02
    assert spans["decode_forward"] >= 0.02
    assert abs(t["spans_unaccounted_s"]) < 1e-3
    assert abs(sum(spans.values()) - t["spans_whole_s"]) < 1e-3
    # the telemetry contract's names (docs/design/telemetry.md)
    assert {"queue_ms", "prefill_ms", "decode_ms"} <= set(t)
    assert t["queue_ms"] == round(t["queue_s"] * 1000, 1)
    assert t["prefill_ms"] >= 20 and t["decode_ms"] >= 20
    assert t.get("decode_tps") == t.get("decode_tok_s")


def test_a_bare_job_like_object_is_still_accepted():
    """A ring's control jobs and test doubles carry no HTTP stamp, spans or
    request id: submit and the step loop must not ask them for one."""
    import queue
    s = _bare()
    job = type("J", (), {})()
    job.outbox = queue.Queue()
    s.submit(job)
    assert job.spans.buckets == {}
