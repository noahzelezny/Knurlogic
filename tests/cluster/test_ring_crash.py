"""A rank that fails says so, and a survivor never waits forever.

Live: a 2-rank tensor split over jaccl, rank 1 raised Desync and exited
telling no one; rank 0 went on into a collective (the executor's reset
exchange) on a peer that no longer existed -- jaccl's have no timeout --
and was later killed inside it, leaving the M4's GPU pinned until a reboot.

Two real processes on a TCP ring (mlx's ring backend) on the serving path
(tests/support/ring_crash_worker.py); one rank fails at a chosen point and
the other must finish on its own, by a normal exit, within a bound."""
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("mlx.core")

#: a survivor's exit, after the fault (seconds); TCP's own "peer lost" takes
#: ~1 s of retries, a bell abort milliseconds
BOUND = 15.0


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _ring(tmp_path, owned_procs, split, fault, linger_rank=None):
    """-> ({rank: record}, {rank: returncode}, survivor's end vs the other's
    process: whether the other was still running when the survivor ended)."""
    hosts = tmp_path / "hosts.json"
    hosts.write_text(json.dumps([[f"127.0.0.1:{_free_port()}"],
                                 [f"127.0.0.1:{_free_port()}"]]))
    support = Path(__file__).resolve().parents[1] / "support"
    engine = support.parent / "engine"
    env = dict(os.environ, MLX_HOSTFILE=str(hosts),
               PYTHONPATH=os.pathsep.join(
                   [str(support.parents[1] / "src"), str(support),
                    str(engine)] + sys.path))
    logs = [open(tmp_path / f"log{r}.txt", "wb") for r in range(2)]
    procs = [owned_procs(subprocess.Popen(
        [sys.executable, str(support / "ring_crash_worker.py"),
         str(tmp_path), split, fault],
        env=dict(env, MLX_RANK=str(r)), stdout=logs[r],
        stderr=subprocess.STDOUT)) for r in range(2)]
    other_alive = None
    try:
        end = time.monotonic() + 150
        while time.monotonic() < end:
            codes = [p.poll() for p in procs]
            if linger_rank is not None:
                surv = 1 - linger_rank
                if codes[surv] is not None:
                    other_alive = codes[linger_rank] is None
                    break
            elif all(c is not None for c in codes):
                break
            time.sleep(0.05)
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
        for p in procs:
            p.wait(10)
        for f in logs:
            f.close()
    text = "\n".join(f"--- rank {r}\n" + (tmp_path / f"log{r}.txt")
                     .read_text(errors="replace") for r in range(2))
    recs = {}
    for r in range(2):
        f = tmp_path / f"rank{r}.json"
        recs[r] = json.loads(f.read_text()) if f.exists() else {}
    return recs, [p.returncode for p in procs], other_alive, text


def _survived(rec, code, faulty, text, surv):
    assert code == 0, f"rank {surv} ended {code}\n{text}"
    assert "injected" in faulty, text
    took = rec["done"] - faulty["injected"]
    assert 0 <= took < BOUND, f"rank {surv} took {took:.1f}s\n{text}"
    return took


@pytest.mark.parametrize("split", ["tensor", "pipeline"])
def test_a_desynced_follower_that_lingers_does_not_hold_rank_0(
        tmp_path, owned_procs, split):
    """The live failure: both ranks see Desync in the same exchange; the
    follower stays alive (as a jaccl peer in no collective would never fail
    one). Rank 0 must not enter its reset or stop exchange: it ends while
    the follower is still running -- before this change it waited in that
    exchange for as long as the follower lived."""
    recs, codes, other_alive, text = _ring(tmp_path, owned_procs, split,
                                           "f_desync", linger_rank=1)
    _survived(recs[0], codes[0], recs[1], text, 0)
    assert other_alive, "rank 0 ended only once the follower was gone\n" + \
        text
    assert recs[0]["outcome"] == "raised" and "Desync" in recs[0]["error"]
    assert recs[0]["stopped"], text      # stop skipped the collective
    assert recs[1]["outcome"] in ("raised", "returned"), text


@pytest.mark.parametrize("split,fault", [
    ("tensor", "f_raise_step"), ("pipeline", "f_raise_step"),
    ("tensor", "f_kill_step"), ("pipeline", "f_kill_step"),
    ("pipeline", "f_raise_coord")])
def test_rank_0_leaves_when_its_follower_fails_mid_step(
        tmp_path, owned_procs, split, fault):
    """Rank 0 is inside a step's collectives (or a drafting broadcast) when
    the follower raises or is SIGKILLed: the follower's exit fails that
    collective (TCP), rank 0's step raises, and it closes and stops with no
    further collective -- a normal exit within the bound."""
    recs, codes, _, text = _ring(tmp_path, owned_procs, split, fault)
    _survived(recs[0], codes[0], recs[1], text, 0)
    assert recs[0]["outcome"] == "raised", text
    assert recs[0]["stopped"], text
    if fault == "f_kill_step":
        assert codes[1] == -signal.SIGKILL
    else:
        assert codes[1] == 3, text           # it raised, and said so
        assert recs[1]["outcome"] == "raised", text


@pytest.mark.parametrize("split", ["tensor", "pipeline"])
@pytest.mark.parametrize("who", ["follower", "rank 0"])
def test_a_rank_whose_admission_fails_mid_prefill_enters_no_b0(
        tmp_path, owned_procs, split, who):
    """A rank's trunk raises in an admission's prefill forward (ForwardFailed
    out of _admit_one) while the other is in that forward's collectives.
    It must mark the ring down and skip the admission's b0 (an all_gather
    the other rank never joins: both waited forever, and on jaccl the GPU
    stayed pinned). Both ranks exit on their own within the bound."""
    fault = "f_raise_admit" if who == "follower" else "r0_raise_admit"
    recs, codes, _, text = _ring(tmp_path, owned_procs, split, fault)
    bad, good = (1, 0) if who == "follower" else (0, 1)
    _survived(recs[good], codes[good], recs[bad], text, good)
    assert -signal.SIGKILL not in codes, text
    assert recs[bad]["what"] == "raise mid-prefill", text
    assert "failed mid-forward" in recs[bad]["down"], text
    assert recs[0]["outcome"] == "raised", text
    # close and stop enter no collective; a forward that failed on a lost
    # peer can leave a Metal error for close's sync to surface (not a hang)
    assert recs[0].get("stopped") or "METAL" in recs[0].get(
        "stop_error", ""), f"{recs}\n{text}"
    if who == "follower":
        assert codes[1] == 3 and recs[1]["outcome"] == "raised", text
    else:
        assert codes[0] == 0 and recs[1]["outcome"] == "returned", text
        assert "rank 0" in recs[1]["down"], text


@pytest.mark.parametrize("split", ["tensor", "pipeline"])
def test_an_idle_rank_0_hears_a_parked_follower_die(tmp_path, owned_procs,
                                                    split):
    """Rank 0 idle (in no collective) and its follower SIGKILLed while
    parked: the bell connection closing marks the ring down at once."""
    recs, codes, _, text = _ring(tmp_path, owned_procs, split,
                                 "f_kill_parked")
    _survived(recs[0], codes[0], recs[1], text, 0)
    assert recs[0]["outcome"] == "down", text
    assert "bell connection closed" in recs[0]["down"], text
    assert recs[0]["stopped"], text


@pytest.mark.parametrize("split", ["tensor", "pipeline"])
def test_a_follower_leaves_when_rank_0_fails_mid_step(tmp_path, owned_procs,
                                                      split):
    """Rank 0 raises inside a step after the exchange: it says so on the
    bell before it leaves; the follower, in that step's collectives, leaves
    when they fail and returns from follow() -- not killed."""
    recs, codes, _, text = _ring(tmp_path, owned_procs, split,
                                 "r0_raise_step")
    took = _survived(recs[1], codes[1], recs[0], text, 1)
    assert recs[1]["outcome"] == "returned", text
    assert "rank 0" in recs[1]["down"], text
    assert codes[0] == 0 and recs[0]["stopped"], text
    assert took < BOUND


@pytest.mark.parametrize("split", ["tensor", "pipeline"])
def test_a_parked_follower_leaves_when_rank_0_dies(tmp_path, owned_procs,
                                                   split):
    """Rank 0 SIGKILLed with its follower asleep on the bell: the follower
    wakes on the closed connection and returns from follow()."""
    recs, codes, _, text = _ring(tmp_path, owned_procs, split,
                                 "r0_kill_parked")
    _survived(recs[1], codes[1], recs[0], text, 1)
    assert codes[0] == -signal.SIGKILL
    assert recs[1]["outcome"] == "returned", text
    assert "bell connection closed" in recs[1]["down"], text


@pytest.mark.parametrize("split", ["tensor", "pipeline"])
def test_an_orderly_stop_is_never_read_as_a_failure(tmp_path, owned_procs,
                                                    split):
    """Rank 0 stops the ring and exits the moment its stop exchange is
    done: its bell may close before the follower applies `stop`. Rank 0
    rings STOP first, so the follower never logs the ring down and returns
    from follow() normally."""
    for _ in range(3):
        recs, codes, _, text = _ring(tmp_path, owned_procs, split, "none")
        assert codes == [0, 0], text
        assert "ring down" not in text, text
        assert recs[0]["stopped"] and recs[0]["tokens"] > 0, text
        assert recs[1]["outcome"] == "returned", text
        assert recs[1]["down"] == "", text


# ----------------------------------------------------------- one process

class _Group:
    def rank(self):
        return 0

    def size(self):
        return 2


def _pair():
    """A Link with a bell to a fake peer (a socketpair), watched."""
    from knurlogic.engine.runtime import tensor as T
    a, b = socket.socketpair()
    link = T.Link(_Group())
    link.socks = [a]
    link.watch()
    return link, b


def test_a_down_ring_enters_no_collective(monkeypatch):
    """Once the bell says ABORT, the exchange, Ring.stop / park, the
    executor's reset and every Coord broadcast refuse before mlx is asked."""
    import mlx.core as mx

    from knurlogic.engine.runtime import pipeline as PL
    from knurlogic.engine.runtime import tensor as T

    def never(*a, **k):
        raise AssertionError("a collective on a down ring")
    monkeypatch.setattr(mx.distributed, "all_gather", never)
    monkeypatch.setattr(mx.distributed, "all_sum", never)
    link, peer = _pair()
    try:
        peer.sendall(T.Link.ABORT)
        assert link.down.wait(5)
        assert "rank 1 failed" in link.why
        with pytest.raises(T.PeerGone):
            link.exchange(0, None)
        ring = T.Ring(link)
        ring.park()
        ring.stop()                       # no exchange, no raise
        coord = PL.Coord.__new__(PL.Coord)
        coord.group, coord.leader, coord.head = None, True, True
        coord.calls = {"b0": 0, "b1": 0, "b2": 0, "ba": 0, "img": 0}
        for call in (lambda: coord._bcast([1]), lambda: coord.b0(None),
                     lambda: coord.ba(True, 0, False)):
            with pytest.raises(T.PeerGone):
                call()
    finally:
        T._LINK = None
        peer.close()


def test_a_failing_rank_says_so_and_a_parked_one_wakes():
    from knurlogic.engine.runtime import tensor as T
    link, peer = _pair()
    try:
        hooks = []
        link.on_down.append(lambda: hooks.append(1))
        peer.sendall(T.Link.WAKE)
        link.sleep()                      # woken, not down
        assert not link.down.is_set()
        link.fail("rank 0 failed: test")
        assert peer.recv(1) == T.Link.ABORT
        assert hooks == [1]
        link.fail("again")                # once only
        assert hooks == [1] and link.why == "rank 0 failed: test"
        with pytest.raises(T.PeerGone):
            link.sleep()
    finally:
        T._LINK = None
        peer.close()


def test_a_stop_bell_marks_the_close_that_follows_orderly():
    """A follower's bell: STOP from rank 0, then the connection closes
    before the stop op is applied -- not a failure."""
    from knurlogic.engine.runtime import tensor as T

    class G1(_Group):
        def rank(self):
            return 1
    a, b = socket.socketpair()
    link = T.Link(G1())
    link.socks = [b]
    link.watch()
    try:
        a.sendall(T.Link.STOP)
        a.close()
        time.sleep(0.2)
        assert link.closing and not link.down.is_set()
    finally:
        T._LINK = None
        b.close()


def test_ring_stop_rings_stop_before_its_exchange(monkeypatch):
    from knurlogic.engine.runtime import tensor as T
    link, peer = _pair()
    try:
        seen = []
        monkeypatch.setattr(link, "exchange",
                            lambda over, payload=None: seen.append(
                                peer.recv(1)))
        T.Ring(link).stop()
        assert seen == [T.Link.STOP]
    finally:
        T._LINK = None
        peer.close()


def test_an_orderly_stop_is_not_a_failure():
    from knurlogic.engine.runtime import tensor as T
    link, peer = _pair()
    try:
        link.closing = True
        peer.close()                      # the follower exits after `stop`
        time.sleep(0.2)
        assert not link.down.is_set()
    finally:
        T._LINK = None


def test_the_scheduler_ends_the_ring_without_a_collective():
    """A down ring: every request is answered RingFailed and the scheduler
    thread stops; its exit's stop sends nothing."""
    import queue
    import threading

    from knurlogic.engine.runtime import tensor as T
    from knurlogic.engine.runtime.scheduler import RingFailed, Scheduler
    link, peer = _pair()
    try:
        s = Scheduler.__new__(Scheduler)
        s.tensor = T.Ring(link)
        s._wake = threading.Event()
        s._stop = False
        s._aborted = None
        s._rows, s._waiting, s._ex = {}, [], None
        s._jobs = queue.Queue()
        link.on_down.append(s._wake.set)
        peer.close()                      # the follower vanished
        assert s._wake.wait(5)
        s._tick()
        assert s._stop and isinstance(s._aborted, RingFailed)
        assert "bell connection closed" in str(s._aborted)
    finally:
        T._LINK = None


def test_rank_0_waits_past_an_armed_jaccl_deadline(monkeypatch):
    from knurlogic.interfaces.http import in_flight_s
    monkeypatch.delenv("JACCL_COLLECTIVE_TIMEOUT_MS", raising=False)
    assert in_flight_s(15.0) == 15.0
    monkeypatch.setenv("JACCL_COLLECTIVE_TIMEOUT_MS", "60000")
    assert in_flight_s(15.0) == 65.0


def _ring_scheduler(monkeypatch, link):
    """A Scheduler on rank 0 of `link` holding a TensorExecutor whose batch
    engine is a stand-in; the collectives recorded (or failed: `fail`)."""
    import queue
    import threading

    import mlx.core as mx

    from knurlogic.engine.runtime import tensor as T
    from knurlogic.engine.runtime.scheduler import Scheduler
    calls = {"collectives": 0, "fail": False, "closed": 0}

    def all_gather(x, group=None, stream=None):
        calls["collectives"] += 1
        if calls["fail"]:
            raise RuntimeError("[ring] connection to a peer was lost")
        return mx.concatenate([x, x])

    def all_sum(x, group=None, stream=None):
        calls["collectives"] += 1
        return x

    monkeypatch.setattr(mx.distributed, "all_gather", all_gather)
    monkeypatch.setattr(mx.distributed, "all_sum", all_sum)

    class Gen:
        def close(self):
            calls["closed"] += 1
    ex = T.TensorExecutor.__new__(T.TensorExecutor)
    ex.gen, ex._top, ex.ring = Gen(), {}, T.Ring(link)
    s = Scheduler.__new__(Scheduler)
    s.tensor = ex.ring
    s._wake = threading.Event()
    s._stop = False
    s._aborted = None
    s._rows, s._waiting, s._ex = {}, [], ex
    s._jobs = queue.Queue()
    link.on_down.append(s._wake.set)
    return s, calls


def test_a_tick_fault_outside_a_collective_keeps_the_ring(monkeypatch):
    """A tick that raises with the ranks in step (a command, the journal,
    the memory guard, events after a step): the rows fail and the executor
    resets over the ring as before the bell -- the ring is not ended."""
    from knurlogic.engine.runtime import tensor as T
    link, peer = _pair()
    try:
        s, calls = _ring_scheduler(monkeypatch, link)

        def tick():
            raise RuntimeError("a bad journal entry")
        s._tick = tick
        s._loop_once()
        assert not link.down.is_set() and not s._stop
        assert calls["collectives"] == 2      # the reset exchange
        assert calls["closed"] == 1 and s._ex is None
        peer.setblocking(False)
        with pytest.raises(BlockingIOError):
            peer.recv(1)                      # nothing on the bell
    finally:
        T._LINK = None
        peer.close()


def test_a_tick_fault_in_a_collective_ends_the_ring(monkeypatch):
    """A tick whose control exchange fails (a peer lost mid-collective):
    the ring is marked down where it failed, the others are told on the
    bell, the executor's reset enters no collective and the next tick
    ends the ring."""
    from knurlogic.engine.runtime import tensor as T
    from knurlogic.engine.runtime.scheduler import RingFailed, Scheduler
    link, peer = _pair()
    try:
        s, calls = _ring_scheduler(monkeypatch, link)
        calls["fail"] = True

        def tick():
            s._ex.ring.exchange(0, {"ops": [{"op": "pop", "n": 1}]})
        s._tick = tick
        s._loop_once()
        assert link.down.is_set() and "control exchange failed" in link.why
        assert peer.recv(1) == T.Link.ABORT
        assert calls["collectives"] == 1      # no reset exchange after it
        assert calls["closed"] == 1 and s._ex is None
        del s._tick
        Scheduler._tick(s)
        assert s._stop and isinstance(s._aborted, RingFailed)
    finally:
        T._LINK = None
        peer.close()
