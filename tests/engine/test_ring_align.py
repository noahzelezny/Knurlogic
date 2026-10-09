"""The ranks line up over the bell (TCP) before the collectives that follow
an uneven stretch -- the join, a load, a wake-up -- so no rank's first
RDMA message reaches a rank that is not there yet.

Live (M4 + M3 over Thunderbolt 5 RDMA, stock mlx 0.32.3): rank 1 entered
the join's [port, nonce] exchange 10 s after rank 0, never got rank 0's
half, and the job sat until rank 0's bell timed out; another job's first
admission failed with "[jaccl] Send failed with error code -12"."""
import socket
import threading
import time

import pytest

from knurlogic.cluster import launch
from knurlogic.engine.split import link as split_link


def test_the_launcher_says_where_the_bell_is():
    spec = {"job": "1001be732c9d0cde", "world": 2, "bell_nonce": 4242,
            "hosts": ["198.51.100.2:47460", "198.51.100.1:47461"]}
    assert launch.bell_address(spec) == "198.51.100.2:47478:4242:2"
    # the nonce is random per job, not read off the job id (shown on the
    # page); a spec without one (an older page) keeps the old way
    assert "bell_nonce" in launch.SPEC_KEYS
    assert launch.bell_address({**spec, "bell_nonce": None}) == ""
    env = launch.rank_env({**spec, "rank": 1, "link": "ring"},
                          {"hostfile": "/x"}, False)
    assert env["KNURLOGIC_BELL"] == launch.bell_address(spec)
    # no room in the slot, or nothing to ring: the old way
    assert launch.bell_address({**spec, "world": 19,
                                "hosts": spec["hosts"] * 10}) == ""
    assert launch.bell_address({"job": "ab" * 8, "hosts": []}) == ""


class _G:
    def __init__(self, rank, size):
        self._r, self._s = rank, size

    def rank(self):
        return self._r

    def size(self):
        return self._s


def _pair():
    srv = socket.create_server(("127.0.0.1", 0))
    port = srv.getsockname()[1]
    got = {}
    t = threading.Thread(target=lambda: got.setdefault(
        "c", split_link.bell_dial("127.0.0.1", port, 7, 1, 10.0)))
    t.start()
    socks = split_link.bell_answer(srv, "127.0.0.1", 7, 2, 10.0)
    t.join(5)
    a, b = split_link.Link(_G(0, 2)), split_link.Link(_G(1, 2))
    a.socks, b.socks = socks, [got["c"]]
    return a, b


def test_rank_0_waits_for_a_late_rank_before_going_on():
    r0, r1 = _pair()
    done = {}

    def late():
        time.sleep(1.0)                      # the slower Mac
        r1.align()
        done["r1"] = time.monotonic()
    t = threading.Thread(target=late)
    t0 = time.monotonic()
    t.start()
    r0.align()
    went = time.monotonic() - t0
    t.join(5)
    assert went >= 0.9 and "r1" in done
    assert not r0.need_align and not r1.need_align


def test_a_rank_gone_while_lining_up_is_a_connection_error():
    r0, r1 = _pair()
    r1.socks[0].close()
    with pytest.raises(ConnectionError):
        r0.align()


def test_a_woken_rank_lines_up_again():
    r0, r1 = _pair()
    r0.need_align = r1.need_align = False
    r0.socks[0].sendall(b"w")                # rank 0 rings
    r1.sleep()
    assert r1.need_align


def test_without_a_bell_lining_up_is_a_no_op():
    lk = split_link.Link(_G(0, 2))
    lk.align()
    assert not lk.socks


def test_the_early_bell_connects_every_rank_before_the_ring(monkeypatch):
    srv = socket.create_server(("127.0.0.1", 0))
    port = srv.getsockname()[1]
    srv.close()
    spec = f"127.0.0.1:{port}:99:2"
    got = {}

    def rank1():
        time.sleep(0.3)
        got["socks"] = split_link.bell_dial("127.0.0.1", port, 99, 1, 10.0)
    t = threading.Thread(target=rank1)
    t.start()
    monkeypatch.setenv("KNURLOGIC_BELL", spec)
    monkeypatch.setenv("MLX_RANK", "0")
    socks = split_link.bell_early()
    t.join(5)
    assert len(socks) == 1 and got["socks"] is not None
    for c in socks + [got["socks"]]:
        c.close()
    monkeypatch.delenv("KNURLOGIC_BELL")
    assert split_link.bell_early() is None


def test_a_lining_up_rank_that_never_comes_is_a_connection_error():
    r0, r1 = _pair()
    t0 = time.monotonic()
    with pytest.raises(ConnectionError, match="did not line up"):
        r0.align(timeout=0.5)
    assert time.monotonic() - t0 < 3


def test_a_dead_link_raises_at_once():
    r0, _ = _pair()
    r0.dead = True
    with pytest.raises(split_link.Desync):
        r0.align()
    with pytest.raises(split_link.Desync):
        r0.exchange(0, None)
