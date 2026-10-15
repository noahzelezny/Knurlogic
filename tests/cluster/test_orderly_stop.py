"""A cluster job's SIGTERM stops the ring between steps: killed mid-step, a
rank left its peer's GPU waiting on a collective forever (100%, the job's
memory pinned past every exit until a reboot; M3 + M4, tensor over TCP and
RDMA, 4 requests in flight)."""
import os
import signal
import threading
import time
from types import SimpleNamespace

from knurlogic.interfaces.http import lifecycle as H


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

    def wait_stopped(self, timeout):
        return True


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


def test_a_scheduler_still_inside_a_step_is_not_exited(monkeypatch):
    """Rank 0 whose step never ends (its peer's GPU hung): leaving
    mid-step wedged the Mac until a reboot, so it stays, says stuck and
    keeps answering 503."""
    from knurlogic.engine.split import marker
    exited = []
    monkeypatch.setattr(os, "_exit", exited.append)
    said = []
    monkeypatch.setattr(marker, "progress", lambda **f: said.append(f))
    s = Sched(ends=False)
    s.busy = True
    s.wait_stopped = lambda timeout: False
    old = signal.getsignal(signal.SIGTERM)
    try:
        H.watch_ring(s, SimpleNamespace(state="ready"), exit_after=0.01,
                     stop_within=0.01)
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        end = time.monotonic() + 2
        while {"phase": "stuck", "busy": True} not in said \
                and time.monotonic() < end:
            time.sleep(0.01)
    finally:
        signal.signal(signal.SIGTERM, old)
    assert {"phase": "stuck", "busy": True} in said
    assert ("abort",) in s.calls and exited == []


def test_a_ring_failure_whose_cleanup_never_ends_is_not_exited(monkeypatch):
    """The ring failed and the scheduler's cleanup is stuck behind a
    collective: the same stay-stuck path as a stop, not os._exit."""
    from knurlogic.engine.split import marker
    exited = []
    monkeypatch.setattr(os, "_exit", exited.append)
    said = []
    monkeypatch.setattr(marker, "progress", lambda **f: said.append(f))
    s = Sched()
    s.wait_stopped = lambda timeout: False
    H.watch_ring(s, SimpleNamespace(state="ready"), exit_after=0.01,
                 stop_within=0.01)
    s.on_ring_failed(RuntimeError("collective failed"))
    end = time.monotonic() + 2
    while {"phase": "stuck", "busy": True} not in said \
            and time.monotonic() < end:
        time.sleep(0.01)
    assert {"phase": "stuck", "busy": True} in said and exited == []


def test_a_ring_failure_whose_cleanup_ends_leaves_non_zero(monkeypatch):
    exited = threading.Event()
    codes = []
    monkeypatch.setattr(os, "_exit", lambda c: (codes.append(c), exited.set()))
    s = Sched()
    H.watch_ring(s, SimpleNamespace(state="ready"), exit_after=0.01,
                 stop_within=0.01)
    s.on_ring_failed(RuntimeError("collective failed"))
    assert exited.wait(2) and codes == [1]


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


class _Link:
    entered = None


class _Ring:
    def __init__(self):
        self.link = _Link()


class _Stuck:
    """stop_ring timed out; the scheduler thread ends after `ends_in`."""

    def __init__(self, entered_ago, ends_in):
        self.tensor = _Ring()
        self.tensor.link.entered = time.monotonic() - entered_ago
        self.end = time.monotonic() + ends_in

    def stop_ring(self, timeout):
        return False

    def wait_stopped(self, timeout):
        time.sleep(timeout)
        return time.monotonic() >= self.end


def test_a_stopping_rank_0_lets_a_young_exchange_finish():
    # rank 0 left 1.5 s after its stop timed out, inside an exchange the
    # follower had half done: the follower died in its plan bytes
    assert H.ring_stopped(_Stuck(entered_ago=0.0, ends_in=0.2), 15.0)


def test_an_exchange_a_gone_peer_holds_is_not_waited_for():
    t = time.monotonic()
    assert not H.ring_stopped(_Stuck(entered_ago=H.EXCHANGE_S, ends_in=60),
                              15.0)
    assert time.monotonic() - t < 1


def test_outside_an_exchange_a_timed_out_stop_leaves_at_once():
    s = _Stuck(entered_ago=0.0, ends_in=60)
    s.tensor.link.entered = None
    assert not H.ring_stopped(s, 15.0)


_LEAVES = """
import os, sys
import mlx.core as mx
from knurlogic.engine.split.link import Link
link = Link(mx.distributed.init(backend="ring", strict=True))
if link.rank == 0:
    # rank 0's control row says plan bytes follow; it leaves before them
    from knurlogic.engine.split import plan as P
    ctl = mx.array([P.control(0, 0, 16, 0, 0)], dtype=mx.int64)
    mx.distributed.all_gather(ctl, group=link.group, stream=mx.cpu).tolist()
    os._exit(0)
try:
    link.exchange(0, None)
except Exception as e:
    from knurlogic.engine.runtime.scheduler import ring_error
    print("raised", ring_error(e), flush=True)
"""


def test_a_follower_whose_rank_0_leaves_mid_exchange_raises_not_aborts(
        tmp_path):
    # it aborted (libc++abi, uncaught C++ exception, a crash report) when
    # the plan's bytes were read through numpy
    import json
    import socket
    import subprocess
    import sys
    from pathlib import Path

    def free_port():
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        p = s.getsockname()[1]
        s.close()
        return p
    hosts = tmp_path / "hosts.json"
    hosts.write_text(json.dumps([[f"127.0.0.1:{free_port()}"],
                                 [f"127.0.0.1:{free_port()}"]]))
    src = Path(__file__).resolve().parents[2] / "src"
    env = dict(os.environ, MLX_HOSTFILE=str(hosts),
               PYTHONPATH=os.pathsep.join([str(src)] + sys.path))
    procs = [subprocess.Popen([sys.executable, "-c", _LEAVES],
                              env=dict(env, MLX_RANK=str(r)),
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT) for r in range(2)]
    try:
        logs = [p.communicate(timeout=60)[0].decode() for p in procs]
    finally:
        for p in procs:
            p.kill()
    assert procs[1].returncode == 0, logs[1]
    assert "raised True" in logs[1], logs[1]
