"""A failed collective on a split model's rank 0 ends the job: every
request is answered with RingFailed, the scheduler stops at the step
boundary without the `stop` collective a gone peer would never answer, and
the process leaves non-zero so the page's recovery relaunches the job.

Live (job 2fa0e08d065c3b31, DeepSeek Flash tensor over RDMA): rank 1's
send failed with jaccl error -12 at the first admission and it left; rank
0 failed that step, then raised Desync in every idle park() -- thousands
of "scheduler tick failed" -- answered 503s and hung, and recovery never
started."""
import threading

import pytest

from knurlogic.engine.runtime.scheduler import RingFailed, Scheduler, ring_error
from knurlogic.engine.split import link as split_link
from knurlogic.engine.split import ring as split_ring


class _Host:
    state, path, error, model = "ready", "/m/a", "", object()
    unloaded = False

    def unload(self):
        self.unloaded = True


class _Ring:
    world = 2

    def __init__(self, park_raises=None):
        self.park_raises = park_raises
        self.stopped = 0
        self.journal = split_ring.Journal()

    def park(self):
        if self.park_raises is not None:
            raise self.park_raises

    def stop(self):
        self.stopped += 1


def test_what_is_a_ring_error():
    assert ring_error(split_link.Desync("ranks at different steps: [1, 0]"))
    assert ring_error(RuntimeError("[jaccl] Recv failed with error code -12"))
    assert ring_error(RuntimeError("[ring] Send failed"))

    class ForwardFailed(RuntimeError):
        pass
    assert ring_error(ForwardFailed("[jaccl] Send failed with error code -12"))
    assert ring_error(BrokenPipeError("bell"))       # the bell broke
    assert ring_error(ConnectionError("a rank did not line up"))
    assert not ring_error(FileNotFoundError("an image path"))
    assert not ring_error(RuntimeError("the prompt is too long"))
    assert not ring_error(ValueError("[jaccl] in a value error"))


def _sched(ring):
    s = Scheduler(_Host(), tensor=ring)
    s._journal_sets = lambda: None
    return s


def test_a_desync_in_an_idle_tick_ends_the_job_once():
    ring = _Ring(park_raises=split_link.Desync("ranks at different steps: [1, 0]"))
    s = _sched(ring)
    told = []
    s.on_ring_failed = told.append
    s._loop_once()
    s._loop_once()                       # a second tick tells no one again
    assert isinstance(s.ring_failed, split_link.Desync) and s._stop
    assert len(told) == 1 and told[0] is s.ring_failed
    job = type("J", (), {})()
    import queue
    job.outbox = queue.Queue()
    job.submitted = 0
    s.submit(job)                        # later requests: RingFailed at once
    kind, err = job.outbox.get_nowait()
    assert kind == "error" and isinstance(err, RingFailed)
    assert "link between them failed" in str(err)


def test_a_failed_collective_in_a_step_ends_the_job():
    s = _sched(_Ring())

    class Ex:
        def step(self):
            raise RuntimeError("[jaccl] Send failed with error code -12")

        def close(self):
            pass
    s._ex = Ex()
    told = []
    s.on_ring_failed = told.append
    s._step()
    assert told and "[jaccl]" in str(s.ring_failed) and s._stop


def test_a_request_error_on_a_ring_is_not_fatal():
    s = _sched(_Ring())

    class Ex:
        def step(self):
            raise RuntimeError("one request's tokenizer blew up")

        def close(self):
            pass
    s._ex = Ex()
    s.on_ring_failed = lambda e: pytest.fail("not a ring failure")
    s._step()
    assert s.ring_failed is None and not s._stop


def test_the_stopped_scheduler_sends_no_stop_over_a_broken_ring():
    ring = _Ring(park_raises=split_link.Desync("ranks at different steps: [1, 0]"))
    s = _sched(ring)
    done = threading.Event()
    s.on_ring_failed = lambda e: done.set()
    s.start()
    assert done.wait(10)
    assert s.wait_stopped(10)
    assert ring.stopped == 0 and s.host.unloaded


def test_a_request_error_in_a_tick_on_a_ring_is_not_fatal():
    s = _sched(_Ring())
    s._tick = lambda: (_ for _ in ()).throw(ValueError("bad image"))
    s.on_ring_failed = lambda e: pytest.fail("not a ring failure")
    s._loop_once()
    assert s.ring_failed is None and not s._stop


def test_the_executors_closing_reset_does_not_wait_on_a_dead_ring():
    """RingExecutor.close() sends the others a `reset` (an exchange): after
    the ring failed it must raise at once, or the cleanup hangs in a
    collective and the process leaves with a thread stuck in it."""
    class Link:
        dead = False
        sent = 0

        def exchange(self, over, payload):
            if self.dead:
                raise split_link.Desync("the ring between the ranks failed")
            self.sent += 1
            raise AssertionError("waited on the dead ring")
    ring = _Ring(park_raises=split_link.Desync("ranks at different steps: [1, 0]"))
    ring.link = Link()
    s = _sched(ring)

    class Ex:
        closed = False

        def close(self):
            try:
                ring.link.exchange(0, b"reset")
            finally:
                self.closed = True
    ex = s._ex = Ex()
    done = threading.Event()
    s.on_ring_failed = lambda e: done.set()
    s.start()
    assert done.wait(10) and s.wait_stopped(10)
    assert ring.link.dead and ring.link.sent == 0 and ex.closed


def test_without_a_ring_a_bad_tick_is_logged_and_survived():
    s = Scheduler(_Host())
    s._tick = lambda: (_ for _ in ()).throw(RuntimeError("[jaccl] x"))
    s._loop_once()
    assert s.ring_failed is None and not s._stop


def test_rank_0_leaves_non_zero_after_the_ring_fails(monkeypatch):
    """watch_ring's callback: waits for the scheduler's cleanup, then
    os._exit(1) so the page counts the rank as down and relaunches."""
    import os

    from knurlogic.interfaces import http as H
    exited = threading.Event()
    code = []

    def fake_exit(c):
        code.append(c)
        exited.set()
    monkeypatch.setattr(os, "_exit", fake_exit)
    monkeypatch.setattr("signal.signal", lambda *a: None)

    class S:
        busy = False
        waited = []
        on_ring_failed = None

        def wait_stopped(self, t):
            self.waited.append(t)
            return True

        def abort(self, e):
            pass
    s = S()
    H.watch_ring(s, _Host(), exit_after=0.0, stop_within=3.0)
    s.on_ring_failed(split_link.Desync("x"))
    assert exited.wait(5) and code == [1] and s.waited == [3.0]
