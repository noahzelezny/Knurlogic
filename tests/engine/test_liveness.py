"""A ring's peer machines, by evidence (engine/split/liveness.py,
docs/design/liveness.md): the bell's TCP state read without consuming
anything, mlx's ring sockets found in this process, and the watcher that
shuts them when a peer's kernel stops answering -- never on a reset or a
hang-up, which is a peer process that ended (mlx's own to see).

Measured on loopback rigs (mlx 0.32.3): on macOS `poll(events=0)` reports
nothing when keepalive drops a connection, while its TCP state reads
closed and stays so; `shutdown` on mlx's ring sockets makes a blocked
collective raise "[ring] connection to a peer was lost"."""
import errno
import json
import os
import socket
import struct
import subprocess
import sys
import textwrap

import pytest

from knurlogic.engine.split import link as split_link
from knurlogic.engine.split import liveness


def _pair():
    srv = socket.create_server(("127.0.0.1", 0))
    a = socket.create_connection(srv.getsockname())
    b, _ = srv.accept()
    srv.close()
    return a, b


def test_keepalive_is_set_on_the_bell():
    a, b = _pair()
    try:
        liveness.keepalive(a)
        assert a.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE)
        assert a.getsockopt(socket.IPPROTO_TCP,
                            socket.TCP_KEEPCNT) == liveness.KEEP_COUNT
    finally:
        a.close()
        b.close()


def test_closed_reads_the_kernel_state_and_consumes_nothing():
    a, b = _pair()
    try:
        b.sendall(b"w")
        assert not liveness.closed(a)
        assert a.recv(1) == b"w"           # the byte is still there to read
        # a hang-up (FIN) leaves this end open (CLOSE_WAIT): not dropped
        b.close()
        assert not liveness.closed(a)
    finally:
        a.close()
    c, d = _pair()
    try:
        # a reset drops the connection: closed, and stays closed
        d.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                     struct.pack("ii", 1, 0))
        d.close()
        c.settimeout(2)
        with pytest.raises(OSError):
            c.recv(1)
        assert liveness.closed(c) and liveness.closed(c)
    finally:
        c.close()
    assert not liveness.closed(c)          # a closed socket object: unknown


class _Sock:
    """A bell the kernel dropped with `err` (the state and SO_ERROR a
    keepalive failure or a reset leave), or a ring socket shut by the
    watcher."""

    def __init__(self, err=0, dropped=True):
        self.err, self.dropped, self.shut = err, dropped, []

    def getsockopt(self, level, opt, size=None):
        if size is not None:               # the TCP state query
            return bytes([liveness._TCP_STATE[1] if self.dropped else 4])
        return self.err

    def shutdown(self, how):
        self.shut.append(how)


def test_the_watcher_acts_only_when_a_peers_kernel_went_silent():
    ring = [_Sock(), _Sock()]
    gone = []
    w = liveness.Watcher({1: _Sock(errno.ECONNRESET)}, ring, gone.append)
    # a reset: a peer process that ended -- mlx and the pages see it; the
    # bell is no longer watched, and nothing is shut
    assert w.check() == "" and w.bells == {}
    w.bells = {1: _Sock(dropped=False), 2: _Sock(errno.ETIMEDOUT)}
    why = w.check()
    assert "rank 2 stopped answering" in why and "ETIMEDOUT" in why
    assert list(w.bells) == [1]
    w._gone(why)
    assert all(s.shut == [socket.SHUT_RDWR] for s in ring) and gone == [why]


def test_the_watcher_thread_shuts_the_ring_once():
    ring = [_Sock()]
    gone = []
    w = liveness.Watcher({0: _Sock(errno.EHOSTUNREACH)}, ring, gone.append,
                         sample_s=0.01).start()
    import time
    deadline = time.monotonic() + 5
    while not gone and time.monotonic() < deadline:
        time.sleep(0.01)
    w.stop.set()
    assert len(gone) == 1 and ring[0].shut == [socket.SHUT_RDWR]


def test_ring_sockets_finds_each_connection_of_the_ring_once():
    srv = socket.create_server(("127.0.0.1", 0))
    port = srv.getsockname()[1]
    out = socket.create_connection(("127.0.0.1", port))
    inn, _ = srv.accept()
    other = socket.socketpair()                # not the ring's
    try:
        found = liveness.ring_sockets([f"127.0.0.1:{port}"])
        pairs = sorted((s.getsockname(), s.getpeername()) for s in found)
        assert pairs == sorted([(out.getsockname(), out.getpeername()),
                                (inn.getsockname(), inn.getpeername())])
        for s in found:
            s.close()
        # the dups were the finder's own: the originals still work
        out.sendall(b"x")
        assert inn.recv(1) == b"x"
    finally:
        for s in (srv, out, inn, *other):
            s.close()


class _G:
    def __init__(self, rank, size):
        self._r, self._s = rank, size

    def rank(self):
        return self._r

    def size(self):
        return self._s


class _Bell:
    def __init__(self, exc):
        self.exc = exc

    def recv(self, n):
        raise self.exc

    def sendall(self, b):
        raise self.exc

    def settimeout(self, t):
        pass


@pytest.mark.parametrize("exc", [
    TimeoutError(errno.ETIMEDOUT, "Operation timed out"),
    OSError(errno.EHOSTUNREACH, "No route to host")])
def test_every_bell_operation_fails_as_a_ring_error(exc):
    """With keepalive on the bell, a dead peer machine arrives as
    TimeoutError or a plain OSError -- neither a ConnectionError, so
    neither a ring failure (scheduler.ring_error) -- in each bell op."""
    follower = split_link.Link(_G(1, 2))
    follower.socks = [_Bell(exc)]
    with pytest.raises(ConnectionError):
        follower.sleep()
    with pytest.raises(ConnectionError):
        follower.align()
    rank0 = split_link.Link(_G(0, 2))
    rank0.socks = [_Bell(exc)]
    with pytest.raises(ConnectionError) as e:
        rank0.align()                      # timeout=None: no "None" in it
    assert "None" not in str(e.value)
    rank0.parked = True
    with pytest.raises(ConnectionError):
        rank0._exchange(0, None)


def test_a_gone_peer_marks_the_link_and_wakes_its_scheduler():
    link = split_link.Link(_G(0, 2))
    woken = []
    link.on_dead = lambda: woken.append(1)
    link.gone("the machine holding rank 1 stopped answering")
    assert link.dead and woken == [1]
    with pytest.raises(split_link.Desync, match="rank 1 stopped answering"):
        link.align()


_RANK = textwrap.dedent("""
    import json, os, socket, sys, threading, time
    import mlx.core as mx
    from knurlogic.engine.split import liveness
    g = mx.distributed.init(backend="ring", strict=True)
    hosts = [h for row in json.load(open(os.environ["MLX_HOSTFILE"]))
             for h in row]
    ring = liveness.ring_sockets(hosts)
    mx.eval(mx.distributed.all_sum(mx.array(1), stream=mx.cpu))
    print("found", len(ring), flush=True)
    if g.rank() == 1:
        time.sleep(30)
        sys.exit(0)
    threading.Timer(1.0, lambda: [s.shutdown(socket.SHUT_RDWR)
                                  for s in ring]).start()
    try:
        mx.eval(mx.distributed.all_sum(mx.array(1), stream=mx.cpu))
        print("not raised", flush=True)
    except RuntimeError as e:
        print("raised", e, flush=True)
    os._exit(0)
""")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_against_the_installed_mlx_the_ring_sockets_are_found_and_shut(
        tmp_path):
    """The test that fails first on an mlx that opens its ring sockets
    differently: two ranks on loopback each find exactly their two ring
    sockets, and shutting them makes rank 0's blocked all_sum raise."""
    pytest.importorskip("mlx.core")
    hosts = tmp_path / "hosts.json"
    hosts.write_text(json.dumps([[f"127.0.0.1:{_free_port()}"],
                                 [f"127.0.0.1:{_free_port()}"]]))
    env = {**os.environ, "MLX_HOSTFILE": str(hosts)}
    procs = [subprocess.Popen([sys.executable, "-c", _RANK],
                              env={**env, "MLX_RANK": str(r)},
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True)
             for r in (0, 1)]
    try:
        out0, _ = procs[0].communicate(timeout=60)
    finally:
        for p in procs:
            p.kill()
            p.wait()
    assert "found 2" in out0, out0
    assert "raised" in out0 and "connection to a peer was lost" in out0, out0
