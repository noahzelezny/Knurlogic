"""The model-load lock: one real model loading on this box at a time.

WHY. A Mac may be shared by several servers and agents loading models. MCP
`ready()` is a CHECK, not a mutex: two agents can both see `ready: true`
and both load, and the second load lands on a budget measured before the
first one's weights arrived. The lock makes the check-then-act atomic.

HOW. `fcntl.flock(LOCK_EX | LOCK_NB)` on ~/.cache/knurlogic/load.lock
(next to servers.json). The KERNEL releases a flock when the holding process
dies, however it dies -- a `kill -9`'d gate never leaves a stale lock, which
a pidfile would (tests/test_loadlock.py kills a holder with SIGKILL and
takes the lock after). The JSON record written inside is for display only
(`holder()`, `ready()`'s blocker); whether the lock is held is always asked
of the kernel, never read from the record.

WHO TAKES IT. Real-model loads: serve.load/switch (P4), MCP load and
`knurlogic serve` (P5), tools/*.py gates. Tiny-fixture tests do not -- they
use < 1 GB and must run in parallel. A long-lived serve holds it only until
its phase is `serving`.

Stdlib only (machine/): asking who holds the lock must not import an engine.
"""
from __future__ import annotations

import errno
import fcntl
import json
import os
import socket
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

#: Exit code for a gate tool that found the lock busy (EX_TEMPFAIL): "try
#: again later", distinct from a failed gate.
EXIT_BUSY = 75


class Busy(RuntimeError):
    """The lock is held by someone else. `.holder` is their record (or {}
    if they had not written it yet)."""

    def __init__(self, holder: Dict[str, Any]):
        self.holder = holder or {}
        who = ", ".join(f"{k}={v}" for k, v in self.holder.items()
                        if k in ("pid", "artifact", "purpose", "agent"))
        super().__init__(f"model-load lock held ({who or 'no record yet'})")


def lock_path() -> Path:
    """~/.cache/knurlogic/load.lock (XDG_CACHE_HOME honoured, as servers.py
    does), or KNURLOGIC_LOADLOCK when set -- how tests isolate it."""
    env = os.environ.get("KNURLOGIC_LOADLOCK")
    if env:
        return Path(env)
    root = Path(os.environ.get("XDG_CACHE_HOME",
                               Path.home() / ".cache")) / "knurlogic"
    return root / "load.lock"


def _open(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    return os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)


def _read_record(fd: int) -> Dict[str, Any]:
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        raw = os.read(fd, 65536)
        return json.loads(raw.decode() or "{}")
    except (OSError, ValueError):
        return {}


def _try_lock(fd: int) -> bool:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError as e:
        if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
            return False
        raise


def holder(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """The holder's record if the lock is held (by anyone, this process
    included), else None. For `ready()`: a held lock is a blocker.

    Asked of the kernel with a non-blocking attempt on a fresh descriptor:
    flock belongs to the open file description, so even this process's own
    hold makes the attempt fail and is reported."""
    p = path or lock_path()
    if not p.exists():
        return None
    fd = _open(p)
    try:
        if _try_lock(fd):
            fcntl.flock(fd, fcntl.LOCK_UN)
            return None
        return _read_record(fd)
    finally:
        os.close(fd)


@contextmanager
def model_load(artifact: str, purpose: str, wait_s: float = 0.0,
               agent: Optional[str] = None,
               path: Optional[Path] = None) -> Iterator[Dict[str, Any]]:
    """Hold the load lock for the block; yields the record written.

    wait_s=0 (the default, and what gate tools use) fails at once with Busy
    -- an agent that finds the box busy stops and reports, it does not
    poll-load. wait_s>0 retries every 0.1 s until then."""
    p = path or lock_path()
    fd = _open(p)
    try:
        deadline = time.monotonic() + max(0.0, wait_s)
        while not _try_lock(fd):
            if time.monotonic() >= deadline:
                raise Busy(_read_record(fd))
            time.sleep(0.1)
        record = {"pid": os.getpid(), "host": socket.gethostname(),
                  "agent": agent or os.environ.get("KNURLOGIC_AGENT", ""),
                  "artifact": artifact, "purpose": purpose,
                  "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, json.dumps(record).encode())
        os.fsync(fd)
        try:
            yield record
        finally:
            # Clear the record before releasing, so a reader never sees a
            # released lock with a live-looking record; holder() asks the
            # kernel regardless.
            try:
                os.ftruncate(fd, 0)
            except OSError:
                pass
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
