"""A cluster job's SIGTERM stops the ring between steps: killed mid-step, a
rank left its peer's GPU waiting on a collective forever (100%, the job's
memory pinned past every exit until a reboot; M3 + M4, tensor over TCP and
RDMA, 4 requests in flight)."""
import os
import signal
import threading
import time
from types import SimpleNamespace

from knurlogic.interfaces import http as H


class Sched:
    busy = False

    def __init__(self, ends=True):
        self.calls = []
        self.ends = ends

    def stop_ring(self, timeout):
        self.calls.append(("stop_ring", timeout))
        return self.ends

    def abort(self, err):
        self.calls.append(("abort",))


def _term(monkeypatch, state, ends=True, within=15.0):
    exited = threading.Event()
    monkeypatch.setattr(os, "_exit", lambda code: exited.set())
    old = signal.getsignal(signal.SIGTERM)
    s = Sched(ends)
    try:
        H.watch_ring(s, SimpleNamespace(state=state), exit_after=0.01,
                     stop_within=within)
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        assert exited.wait(2)
    finally:
        signal.signal(signal.SIGTERM, old)
    return s.calls


def test_a_ready_ring_stops_between_steps_not_by_abort(monkeypatch):
    assert _term(monkeypatch, "ready") == [("stop_ring", 15.0)]


def test_stop_ring_waits_for_the_step_then_says_whether_it_ended():
    from knurlogic.engine.runtime.scheduler import Scheduler
    s = Scheduler.__new__(Scheduler)
    s._wake = threading.Event()
    s._stop = False
    s._aborted = None
    s._rows, s._waiting = {}, []
    import queue
    s._jobs = queue.Queue()

    def loop():                      # a step that ends when told to stop
        while not s._stop:
            time.sleep(0.01)
    s._thread = threading.Thread(target=loop)
    s._thread.start()
    assert s.stop_ring(timeout=2) is True

    s._stop = False
    s._thread = threading.Thread(target=lambda: time.sleep(1))  # stuck step
    s._thread.start()
    assert s.stop_ring(timeout=0.05) is False


def test_a_step_that_never_ends_fails_the_requests_then_leaves(monkeypatch):
    # a peer is gone: the old path, so in-flight requests get their 503
    assert _term(monkeypatch, "ready", ends=False) == [("stop_ring", 15.0),
                                                       ("abort",)]


def test_a_warming_ring_also_stops_between_steps(monkeypatch):
    # the warm-up runs real forwards with the ring's collectives: a rank
    # killed in one pinned its peer's GPU like one killed mid-generation
    assert _term(monkeypatch, "warming") == [("stop_ring", 15.0)]


def test_a_loading_rank_0_stops_its_read_between_batches(monkeypatch):
    from knurlogic.engine.runtime import model_host as Hst
    Hst.LOAD_STOP.clear()
    try:
        # the read never stops here (no load runs): the wait runs out and
        # the requests are failed as before
        assert _term(monkeypatch, "loading", within=0.1) == [("abort",)]
        assert Hst.LOAD_STOP.is_set()
    finally:
        Hst.LOAD_STOP.clear()


def test_the_weight_read_stops_at_a_batch_boundary(monkeypatch):
    import mlx.core as mx
    import pytest

    from knurlogic.engine.runtime import model_host as Hst
    m = {"a": [mx.zeros((256,)) + i for i in range(8)]}
    monkeypatch.setattr(Hst, "LOAD_BATCH_BYTES", 2048)    # two per batch
    evals = []
    real = mx.eval

    def counting(x):
        evals.append(len(x))
        if len(evals) == 2:
            Hst.LOAD_STOP.set()          # a SIGTERM mid-load
        return real(x)
    monkeypatch.setattr(mx, "eval", counting)
    try:
        with pytest.raises(Hst.LoadCancelled):
            Hst.evaluate_everything(m)
        assert evals == [2, 2]           # whole batches, then the stop
    finally:
        Hst.LOAD_STOP.clear()
    assert Hst.evaluate_everything(m) == 8
