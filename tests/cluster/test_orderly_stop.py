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


def _term(monkeypatch, state, ends=True):
    exited = threading.Event()
    monkeypatch.setattr(os, "_exit", lambda code: exited.set())
    old = signal.getsignal(signal.SIGTERM)
    s = Sched(ends)
    try:
        H.watch_ring(s, SimpleNamespace(state=state), exit_after=0.01)
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        assert exited.wait(2)
    finally:
        signal.signal(signal.SIGTERM, old)
    return s.calls


def test_a_ready_ring_stops_between_steps_not_by_abort(monkeypatch):
    assert _term(monkeypatch, "ready") == [("stop_ring", 6.0)]


def test_a_ring_still_loading_aborts_as_before(monkeypatch):
    assert _term(monkeypatch, "loading") == [("abort",)]


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
    assert _term(monkeypatch, "ready", ends=False) == [("stop_ring", 6.0),
                                                       ("abort",)]
