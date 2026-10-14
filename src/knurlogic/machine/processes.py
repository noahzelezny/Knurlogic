"""This machine's processes: command lines, resident size, environment, and
who listens on which TCP port.

The one home for asking, on psutil (macOS, Linux and Windows alike). Each
answer is empty -- "", 0, {}, [] -- for a process that is gone or that this
user may not read, never an exception.
"""
from __future__ import annotations

import os
import re

import psutil

_GONE = (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess,
         OSError, ValueError)


def command(pid: int) -> str:
    """`pid`'s command line, its arguments joined by spaces."""
    try:
        return " ".join(psutil.Process(int(pid)).cmdline())
    except _GONE:
        return ""


def running(pid: int) -> bool:
    """`pid` exists and has not exited (a zombie, exited and not yet
    reaped, has)."""
    try:
        return psutil.Process(int(pid)).status() != psutil.STATUS_ZOMBIE
    except _GONE:
        return False


def rss(pid: int) -> int:
    """`pid`'s resident size in bytes (on Apple silicon its Metal buffers
    are in it)."""
    try:
        return int(psutil.Process(int(pid)).memory_info().rss)
    except _GONE:
        return 0


def rss_all() -> dict:
    """{pid: resident bytes} of every process this user can read."""
    return {p.pid: p.info["memory_info"].rss
            for p in psutil.process_iter(["memory_info"])
            if p.info.get("memory_info")}


def environ(pid: int) -> dict:
    try:
        return dict(psutil.Process(int(pid)).environ())
    except _GONE:
        return {}


def commands() -> dict:
    """{pid: command line} of every process this user can read."""
    out = {}
    for p in psutil.process_iter(["cmdline"]):
        cmd = p.info.get("cmdline")
        if cmd:
            out[p.pid] = " ".join(cmd)
    return out


def matching(pattern: str) -> list:
    """Pids whose command line matches the regular expression `pattern`
    (searched, as pgrep -f), this process left out."""
    rx = re.compile(pattern)
    me = os.getpid()
    return sorted(pid for pid, cmd in commands().items()
                  if pid != me and rx.search(cmd))


def listening(pids=None) -> dict:
    """{port: pid} of the TCP ports `pids` (every readable process when
    None) listen on."""
    found: dict = {}
    for p in _processes(pids):
        try:
            conns = p.net_connections(kind="tcp")
        except _GONE:
            continue
        for c in conns:
            if c.status == psutil.CONN_LISTEN and c.laddr:
                found.setdefault(c.laddr.port, p.pid)
    return found


def _processes(pids):
    if pids is None:
        yield from psutil.process_iter()
        return
    for pid in pids:
        try:
            yield psutil.Process(int(pid))
        except _GONE:
            continue


def listener(port: int) -> int | None:
    """The pid listening on TCP `port`, or None."""
    return listening().get(int(port))
