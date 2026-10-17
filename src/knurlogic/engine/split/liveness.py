"""Whether the other machines of a ring are still there, by evidence, never
by a timer on the work (docs/design/liveness.md).

The bell (link.py) is an idle TCP connection from rank 0 to each other
rank, on the ring's own link. With keepalive on it, the peer's kernel
answers probes whatever its rank is doing -- busy, stuck in a long step,
not reading -- so the connection is dropped only when the machine or the
link has gone silent. `Watcher` samples each bell's kernel TCP state and,
on such a drop, shuts mlx's ring sockets (`ring_sockets`): a rank blocked
in a collective then fails as it does when a peer process dies, on its own
thread, through the existing ring-failure paths.

A peer process that ended is not this module's: mlx sees its closed ring
sockets itself, and the pages see the rank exit.
"""
from __future__ import annotations

import contextlib
import errno
import logging
import os
import socket
import stat
import sys
import threading
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

#: keepalive on the bell: idle, probe interval, probes -- about 30 s from a
#: machine's last acknowledgement to the drop. The only number here that
#: can end a job whose ranks are alive (a link silent this long looks like
#: a dead machine); Noah's choice, 2026-10-15
KEEP_IDLE_S = 10
KEEP_INTVL_S = 5
KEEP_COUNT = 4

#: what keepalive allows a silent machine: from its last acknowledgement
#: to the drop. A page probing a machine (`kernel_answers`) waits as long
SILENT_S = KEEP_IDLE_S + KEEP_INTVL_S * KEEP_COUNT

#: one probe's connect: a live kernel on the same cable answers in
#: milliseconds (refused in 0.2 s, measured over Thunderbolt)
PROBE_S = 1.0

#: how often the watcher samples each bell's TCP state; a dropped
#: connection stays closed, so this is a sampling interval, not a deadline
SAMPLE_S = 1.0

#: SO_ERROR values that mean the peer's kernel stopped answering. A reset
#: or a hang-up is a peer process that ended, and is left to mlx and the
#: pages: acting on it could discard a `stop` op still unread (a normal
#: stop closes the bell too)
SILENCE = frozenset({errno.ETIMEDOUT, errno.EHOSTUNREACH, errno.EHOSTDOWN,
                     errno.ENETUNREACH, errno.ENETDOWN})

# (getsockopt level option, the first byte's value for a closed connection)
# darwin <netinet/tcp.h>: TCP_CONNECTION_INFO, tcpi_state TCPS_CLOSED;
# Linux <netinet/tcp.h>: TCP_INFO, tcpi_state TCP_CLOSE
_TCP_STATE = ((0x106, 0) if sys.platform == "darwin"
              else (getattr(socket, "TCP_INFO", 11), 7))


def keepalive(sock: socket.socket) -> None:
    """Probe `sock` while it is idle (KEEP_*), so a peer machine that goes
    silent drops it."""
    idle = getattr(socket, "TCP_KEEPIDLE", None) or socket.TCP_KEEPALIVE
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    sock.setsockopt(socket.IPPROTO_TCP, idle, KEEP_IDLE_S)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, KEEP_INTVL_S)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, KEEP_COUNT)


def closed(sock: socket.socket) -> bool:
    """Whether the kernel has dropped `sock`'s connection. Reads its TCP
    state, which consumes nothing and races no reader; False when the
    state cannot be read (a closed socket object included)."""
    level, gone = _TCP_STATE
    try:
        return sock.getsockopt(socket.IPPROTO_TCP, level, 64)[0] == gone
    except (OSError, IndexError):
        return False


def kernel_answers(host: str, port: int,
                   timeout: float = PROBE_S) -> bool | None:
    """Whether the machine at `host` answers a TCP connect to `port`: True
    when it accepts or refuses (its kernel is there, whatever runs on it),
    False when nothing answers (a timeout, or an errno in SILENCE: no
    route, host down), None for anything else (not evidence either way).
    Connects and closes; sends nothing."""
    try:
        socket.create_connection((host, port), timeout=timeout).close()
    except (ConnectionRefusedError, ConnectionResetError):
        return True
    except TimeoutError:
        return False
    except OSError as e:
        return False if e.errno in SILENCE else None
    return True


def ring_sockets(hosts: list[str]) -> list[socket.socket]:
    """mlx's ring sockets in this process: each open TCP socket whose peer
    or local address is one of the ring's "host:port" entries, once per
    connection. Returned as dups, held by the caller for the job's life,
    so a descriptor number reused later is never touched; socket options
    and shutdown act on the connection, not the descriptor."""
    want = set()
    for h in hosts:
        host, _, port = h.rpartition(":")
        want.add((host.strip("[]"), int(port)))
    out: list[socket.socket] = []
    seen: set = set()
    for entry in Path("/dev/fd").iterdir():
        fd = int(entry.name)
        try:
            if not stat.S_ISSOCK(os.fstat(fd).st_mode):
                continue
            s = socket.socket(fileno=os.dup(fd))
        except OSError:
            continue
        try:
            local, peer = s.getsockname()[:2], s.getpeername()[:2]
        except (OSError, TypeError):
            s.close()
            continue
        key = (local, peer)
        if key in seen or (local not in want and peer not in want):
            s.close()
            continue
        seen.add(key)
        out.append(s)
    return out


class Watcher:
    """Samples the bell sockets' TCP state every SAMPLE_S on a daemon
    thread. On a drop whose SO_ERROR is in SILENCE: shuts every ring
    socket (a blocked collective then raises in mlx) and calls
    `on_gone(why)` once. A bell dropped otherwise (reset, hang-up) is
    no longer watched."""

    def __init__(self, bells: dict[int, socket.socket],
                 ring: list[socket.socket], on_gone: Callable[[str], None],
                 sample_s: float = SAMPLE_S):
        self.bells = dict(bells)          # peer rank -> bell socket
        self.ring = list(ring)
        self.on_gone = on_gone
        self.sample_s = sample_s
        self.stop = threading.Event()

    def start(self) -> Watcher:
        threading.Thread(target=self._run, daemon=True,
                         name="knurlogic-bell-watch").start()
        return self

    def check(self) -> str:
        """One sample: why a peer is gone, or ""."""
        for rank, s in list(self.bells.items()):
            if not closed(s):
                continue
            del self.bells[rank]
            try:
                err = s.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
            except OSError:
                err = 0
            if err in SILENCE:
                return (f"the machine holding rank {rank} stopped answering "
                        f"on the ring's link ({errno.errorcode.get(err, err)}"
                        f" after {KEEP_IDLE_S + KEEP_INTVL_S * KEEP_COUNT} s"
                        f" of silence)")
        return ""

    def _run(self) -> None:
        while not self.stop.wait(self.sample_s):
            why = self.check()
            if why:
                self._gone(why)
                return
            if not self.bells:
                return

    def _gone(self, why: str) -> None:
        logger.error("%s; closing the ring so this rank leaves", why)
        for s in self.ring:
            with contextlib.suppress(OSError):    # already closed by mlx
                s.shutdown(socket.SHUT_RDWR)
        try:
            self.on_gone(why)
        except Exception:  # the ring is shut either way (logged)
            logger.exception("bell watcher: on_gone")
