"""Reaping what a test spawned (not a test module): every process a test
starts is killed AND waited on however the test ends.

`Owned` is the per-test owner (conftest's `owned_procs` fixture): register
a Popen as it is made, and the fixture's finalizer kills and waits on it
even when setup failed before a fixture's yield. `reap_registries` kills
every rank a page's job registry recorded (fake ranks run in their own
session, so killing the page does not take them along)."""
import json
import os
import signal
import subprocess
from pathlib import Path


class Owned:
    def __init__(self):
        self.procs: list = []

    def __call__(self, proc: subprocess.Popen) -> subprocess.Popen:
        self.procs.append(proc)
        return proc

    def reap(self) -> None:
        for p in self.procs:
            if p.poll() is None:
                try:
                    p.kill()
                except OSError:
                    pass
        for p in self.procs:
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                pass
        self.procs.clear()


def reap_registries(*caches) -> list:
    """SIGKILL every rank pid in each cache's knurlogic/jobs/jobs.json;
    -> the pids signalled."""
    pids = []
    for c in caches:
        f = Path(c) / "knurlogic" / "jobs" / "jobs.json"
        try:
            reg = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        for r in (reg.values() if isinstance(reg, dict) else []):
            try:
                pid = int(r["pid"])
            except (KeyError, TypeError, ValueError):
                continue
            pids.append(pid)
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
            try:                         # our child: reap it, no zombie
                os.waitpid(pid, 0)
            except (ChildProcessError, OSError):
                pass
    return pids
