"""This machine's processes: whether one is running, command lines, resident
size, environment, and who listens on which TCP port.

The one home for asking, on psutil (macOS, Linux and Windows alike). Each
answer is empty -- "", 0, {}, [] -- for a process that is gone or that this
user may not read, never an exception.
"""
from __future__ import annotations

import os
import re
import subprocess

import psutil

#: What starting or talking to a system tool can raise. Callers catch these
#: (plus whatever their own parsing adds), not Exception.
PROC_ERRORS = (OSError, subprocess.SubprocessError)

#: what reading one process can raise when it is gone or not ours to read.
#: SystemError too: on macOS psutil's proc_cmdline raises it (around a
#: KERN_PROCARGS2 PermissionError) for a process exiting mid-read --
#: seen under load, where it failed a page's whole JobState answer
_GONE = (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess,
         OSError, ValueError, SystemError)


def command(pid: int) -> str:
    """`pid`'s command line, its arguments joined by spaces."""
    try:
        return " ".join(psutil.Process(int(pid)).cmdline())
    except _GONE:
        return ""


def running(pid: int, reap: bool = False) -> bool:
    """`pid` exists and has not exited (a zombie, exited and not yet
    reaped, has). A process this user may not inspect is running.

    `reap`: `pid` may be a child of this process; one that has exited is
    reaped (waitpid) here, so it does not stay a zombie."""
    pid = int(pid)
    if reap:
        try:
            done, _ = os.waitpid(pid, os.WNOHANG)
            if done == pid:
                return False
        except ChildProcessError:
            pass
        except OSError:
            return False
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.AccessDenied:
        return True
    except _GONE:
        return False


def rss(pid: int) -> int:
    """`pid`'s resident size in bytes (on Apple silicon its Metal buffers
    are in it)."""
    try:
        return int(psutil.Process(int(pid)).memory_info().rss)
    except _GONE:
        return 0


def _each(read) -> dict:
    """{pid: read(process)} of every process this user can read. One that
    cannot be read -- gone, denied, or exiting mid-read -- is left out,
    never the whole table."""
    def one(p):
        try:
            return read(p)
        except _GONE:
            return None
    return {p.pid: v for p in psutil.process_iter()
            if (v := one(p)) is not None}


def rss_all() -> dict:
    """{pid: resident bytes} of every process this user can read."""
    return _each(lambda p: int(p.memory_info().rss))


def environ(pid: int) -> dict:
    try:
        return dict(psutil.Process(int(pid)).environ())
    except _GONE:
        return {}


def commands() -> dict:
    """{pid: command line} of every process this user can read (`_each`)."""
    return _each(lambda p: " ".join(p.cmdline()) or None)


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
