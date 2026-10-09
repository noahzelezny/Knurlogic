"""How the ranks of a split model reach each other: joining the ring
(`init`), the per-step control exchange (`Link`), and the bell -- a plain
TCP side channel a parked rank sleeps on instead of spinning in a
collective (`bell_answer`, `bell_dial`, `bell_early`); `Desync` when the ranks
no longer agree."""

from __future__ import annotations

import logging
import os
import socket

import mlx.core as mx

from . import plan as P
from .marker import progress

logger = logging.getLogger(__name__)


class Desync(RuntimeError):
    """The ranks no longer agree on what they are doing."""


# ------------------------------------------------------------------- link

class Link:
    """The control exchange between ranks: one all_gather of a fixed-size
    vector per step, then the plan's bytes (all_sum; only rank 0's are
    non-zero) when rank 0 has any."""

    def __init__(self, group):
        self.group = group
        self.rank = group.rank()
        self.size = group.size()
        self.step = 0
        #: the side channel a parked rank sleeps on (bell()); None until set
        self.socks: list = []
        self.parked = False
        #: line the ranks up over the bell before the next exchange (align):
        #: after joining, and after a parked rank wakes
        self.need_align = True
        #: the ring failed (scheduler._ring_fatal): every later exchange or
        #: align raises at once instead of waiting on a gone peer
        self.dead = False

    def barrier(self) -> None:
        mx.eval(mx.distributed.all_sum(mx.array(1), group=self.group,
                                       stream=mx.cpu))

    def bell(self) -> None:
        """A plain TCP connection from each rank to rank 0, beside the ring.
        jaccl's collectives busy-poll the Thunderbolt completion queue: a
        rank waiting in one for an idle rank 0 burns a whole core (and an
        M3 Ultra shows ~50% GPU) for as long as nothing is asked. So an idle
        rank 0 parks the others (the `park` op) and they sleep in a recv
        here until it rings. Rank 0 listens on the address the ring
        already uses; a connection must present the nonce shared over the
        ring, so nothing else can take a rank's place."""
        import secrets
        import socket
        host = _rank0_host()
        if self.rank == 0:
            srv = socket.create_server((host, 0))
            port = srv.getsockname()[1]
            nonce = secrets.randbits(62)
            logger.info("bell: rank 0 listening on %s:%d for ranks 1..%d",
                        host, port, self.size - 1)
        else:
            srv, port, nonce = None, 0, 0
            logger.info("bell: rank %d waiting for rank 0's port over the "
                        "ring", self.rank)
        got = mx.distributed.all_sum(mx.array([port, nonce], dtype=mx.int64),
                                     group=self.group, stream=mx.cpu).tolist()
        port, nonce = int(got[0]), int(got[1])
        if self.rank == 0:
            self.socks = bell_answer(srv, host, nonce, self.size, BELL_S)
        else:
            logger.info("bell: rank %d dialing rank 0 at %s:%d", self.rank,
                        host, port)
            self.socks = [bell_dial(host, port, nonce, self.rank, BELL_S)]
        for c in self.socks:
            c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def sleep(self) -> None:
        """A parked rank >= 1: wait, without spinning, for rank 0's bell."""
        if not self.socks:
            return
        if self.socks[0].recv(1) != b"w":
            raise ConnectionError("rank 0 left while this rank was parked")
        self.need_align = True

    def align(self, timeout: float | None = None) -> None:
        """Every rank reaches the next collective together, over the bell.

        Ranks finish a load, a warm-up or a wake-up at different times (an
        M3 and an M4 at different clocks: seconds apart). On jaccl, rank
        0's first message of a collective reaching a rank that has not yet
        posted its receive can be dropped -- stock mlx sets no
        receive-not-ready retries and does not check the completion -- and
        that rank then waits forever (live: rank 1 entered the join's
        exchange 10 s late and never got rank 0's half). So: each rank >= 1
        says it is here; rank 0, once all have, says go. A dead peer is a
        ConnectionError (its socket closes)."""
        if self.dead:
            raise Desync("the ring between the ranks failed")
        if not self.socks:
            return
        if self.rank == 0:
            for c in self.socks:
                c.settimeout(timeout)
                try:
                    got = _recv_exact(c, 1)
                except TimeoutError as e:
                    raise ConnectionError(
                        f"a rank did not line up within {timeout:.0f} s") from e
                finally:
                    c.settimeout(None)
                if got != b"r":
                    raise ConnectionError("a rank left while the ranks were "
                                          "lining up")
            for c in self.socks:
                c.sendall(b"g")
        else:
            c = self.socks[0]
            c.sendall(b"r")
            if _recv_exact(c, 1) != b"g":
                raise ConnectionError("rank 0 left while the ranks were "
                                      "lining up")
        self.need_align = False

    def exchange(self, over: int, payload: bytes | None = None):
        """-> (control rows, one per rank; the plan bytes or None)."""
        import numpy as np
        if self.dead:
            raise Desync("the ring between the ranks failed")
        if self.parked:
            for c in self.socks:
                c.sendall(b"w")
            self.parked = False
            self.need_align = True
        if self.need_align:
            # a wake-up or the first step: the others are milliseconds
            # away, or a load apart (rank 0's own load is done by now)
            self.align(timeout=ALIGN_S)
        n = len(payload) if (self.rank == 0 and payload) else 0
        ctl = mx.array([P.control(over, self.step, n,
                                  int(mx.get_active_memory()),
                                  int(mx.get_peak_memory()))],
                       dtype=mx.int64)
        rows = mx.distributed.all_gather(ctl, group=self.group,
                                         stream=mx.cpu).tolist()
        steps = {r[P.STEP] for r in rows}
        if len(steps) != 1:
            raise Desync(f"ranks at different steps: {[r[P.STEP] for r in rows]}")
        self.step += 1
        progress(step=self.step)
        length = rows[0][P.LENGTH]
        if not length:
            return rows, None
        if self.rank == 0:
            assert payload is not None      # rank 0 is the one that sends
            buf = mx.array(np.frombuffer(payload, dtype=np.uint8))
        else:
            buf = mx.zeros((length,), dtype=mx.uint8)
        out = mx.distributed.all_sum(buf, group=self.group, stream=mx.cpu)
        return rows, np.array(out).tobytes()


#: how long rank 0 waits for every rank's bell connection, and a rank
#: keeps dialing rank 0 for it
BELL_S = 60.0


#: a rank's bell hello: the ring's nonce, then its rank
_HELLO = "<qq"


def _recv_exact(c, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        part = c.recv(n - len(buf))
        if not part:
            break
        buf += part
    return buf


def bell_answer(srv, host: str, nonce: int, size: int, wait_s: float,
                stop=None):
    """Rank 0: accept one bell connection per rank 1..size-1 on `srv`
    (closed on return), each presenting the ring's nonce and its rank, in
    order by rank. Past `wait_s`, a TimeoutError naming the address and the
    ranks that never connected."""
    import struct
    import time
    port = srv.getsockname()[1]
    want = set(range(1, size))
    got: dict = {}
    deadline = time.monotonic() + wait_s
    try:
        while want - set(got):
            if stop is not None and stop():
                raise ConnectionError("bell: stopped while waiting for the "
                                      "other ranks")
            left = deadline - time.monotonic()
            if left <= 0:
                missing = sorted(want - set(got))
                raise TimeoutError(
                    f"bell: rank(s) {missing} never connected to rank 0 at "
                    f"{host}:{port} within {wait_s:.0f}s (connected: "
                    f"{sorted(got) or 'none'}); see those ranks' logs -- a "
                    f"rank that never logged 'dialing rank 0' is stuck "
                    f"before it, in the ring's port exchange")
            srv.settimeout(min(left, 1.0))
            try:
                c, addr = srv.accept()
            except TimeoutError:
                continue
            c.settimeout(min(10.0, max(left, 0.1)))
            try:
                hello = _recv_exact(c, struct.calcsize(_HELLO))
                n, r = struct.unpack(_HELLO, hello) if len(hello) == \
                    struct.calcsize(_HELLO) else (None, None)
            except OSError:
                n = r = None
            if n != nonce or r not in want or r in got:
                logger.warning("bell: refused a connection from %s:%d "
                               "(%s)", addr[0], addr[1],
                               "wrong nonce" if n != nonce else
                               f"rank {r} not expected")
                c.close()
                continue
            c.settimeout(None)
            got[r] = c
            logger.info("bell: rank %d connected from %s:%d", r, addr[0],
                        addr[1])
    except BaseException:
        for c in got.values():
            c.close()
        raise
    finally:
        srv.close()
    return [got[r] for r in sorted(got)]


def bell_dial(host: str, port: int, nonce: int, rank: int, wait_s: float,
              pause_s: float = 0.5, stop=None):
    """Rank >= 1: connect to rank 0's bell and say who it is, retrying a
    refused or unanswered connect until `wait_s`; then a ConnectionError
    naming the address and the last error."""
    import socket
    import struct
    import time
    deadline = time.monotonic() + wait_s
    tries, last = 0, None
    while True:
        if stop is not None and stop():
            raise ConnectionError(f"bell: rank {rank} stopped while dialing "
                                  "rank 0")
        left = deadline - time.monotonic()
        if left <= 0:
            raise ConnectionError(
                f"bell: rank {rank} could not reach rank 0 at {host}:{port} "
                f"in {wait_s:.0f}s ({tries} tries; last: "
                f"{type(last).__name__}: {last})")
        tries += 1
        try:
            c = socket.create_connection((host, port),
                                         timeout=min(10.0, left))
        except OSError as e:
            last = e
            time.sleep(min(pause_s, max(deadline - time.monotonic(), 0)))
            continue
        try:
            c.sendall(struct.pack(_HELLO, nonce, rank))
        except OSError as e:
            c.close()
            last = e
            continue
        c.settimeout(None)
        c.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        logger.info("bell: rank %d connected to rank 0 at %s:%d (try %d)",
                    rank, host, port, tries)
        return c


# ------------------------------------------------------------ bring-up

def init(link_kind: str) -> Link:
    """Join the ring (MLX_RANK and MLX_HOSTFILE, or the jaccl variables,
    already set) and wait for every rank before anything loads.

    JACCL_COLLECTIVE_TIMEOUT_MS is 0 (no timeout) around the load: a cold
    read of a 400 GB artifact is not a hang. But that same 0 covers the
    join below (mx.distributed.init's handshake, the barrier, the
    [port, nonce] exchange in bell()) -- so a peer stuck in one of those,
    on a collective this rank has already passed, hangs forever instead
    of failing. Read live by libjaccl on every collective, so it can be
    armed here for the join alone and dropped back to 0 once the ring is
    up, before the model's own cold read."""
    backend = {"ring": "ring", "jaccl": "jaccl"}[link_kind]
    join_ms = os.environ.get("KNURLOGIC_JACCL_TIMEOUT_MS")
    armed = bool(join_ms and join_ms.isdigit())
    if armed and join_ms:
        os.environ["JACCL_COLLECTIVE_TIMEOUT_MS"] = join_ms
    try:
        # the bell first, over TCP (the launcher said where): the ranks then
        # join and run the first collectives together (Link.align)
        early = bell_early()
        group = mx.distributed.init(backend=backend, strict=True)
        link = Link(group)
        logger.info("rank %d of %d joined the %s ring", link.rank,
                    link.size, backend)
        if early is not None:
            link.socks = early
            for c in link.socks:
                c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            link.align()
            link.need_align = True
            link.barrier()
        else:
            link.barrier()
            link.bell()
    finally:
        if armed:
            os.environ["JACCL_COLLECTIVE_TIMEOUT_MS"] = "0"
    progress(phase="loading")
    return link


#: how long ranks wait for each other on the bell before joining the ring
#: (process starts differ by seconds; a cold Python import on a slow disk
#: by more); under the page's own JOIN_S (cluster/jobs) so the bell's error
#: is the one reported
BELL_EARLY_S = 240.0


#: how long rank 0 waits for the others to line up at an exchange: a wake
#: is milliseconds, the first step after a load as long as the slowest
#: rank's remaining load (the page's stall watch covers longer)
ALIGN_S = 600.0


def bell_early():
    """KNURLOGIC_BELL ("host:port:nonce:world", cluster/launch.bell_address)
    set: connect the bell before the ring exists -- rank 0 listens on
    host:port, the others dial it -- and return this rank's sockets. None
    without it (the port then comes over the ring: Link.bell)."""
    spec = os.environ.get("KNURLOGIC_BELL")
    if not spec:
        return None
    host, port, nonce, world = spec.rsplit(":", 3)
    port, nonce, world = int(port), int(nonce), int(world)
    rank = int(os.environ.get("MLX_RANK", "0"))
    from knurlogic.engine.runtime.model_host import LOAD_STOP
    if rank == 0:
        srv = socket.create_server((host, port))
        logger.info("bell: rank 0 listening on %s:%d for ranks 1..%d before "
                    "joining the ring", host, port, world - 1)
        return bell_answer(srv, host, nonce, world, BELL_EARLY_S,
                           stop=LOAD_STOP.is_set)
    return [bell_dial(host, port, nonce, rank, BELL_EARLY_S,
                      stop=LOAD_STOP.is_set)]


def _rank0_host() -> str:
    """Rank 0's address on the ring: the jaccl coordinator's host, or the
    ring hostfile's first entry."""
    import json
    coord = os.environ.get("MLX_JACCL_COORDINATOR")
    if coord:
        return coord.rsplit(":", 1)[0]
    with open(os.environ["MLX_HOSTFILE"]) as f:
        return json.load(f)[0][0].rsplit(":", 1)[0]
