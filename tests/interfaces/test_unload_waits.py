"""Unload answers when the server process is gone, not when it was
signalled; one that does not leave is reported stuck and never killed."""

import signal

from knurlogic.interfaces import spawn
from knurlogic.interfaces.http import lifecycle
from knurlogic.machine import servers


class FakeProc:
    def __init__(self, exits_after):
        self.polls, self.exits_after = 0, exits_after

    def poll(self):
        self.polls += 1
        return 0 if self.polls > self.exits_after else None

    def wait(self, timeout=None):
        return 0


def _setup(monkeypatch, proc, sent=None):
    reg = {8080: {"pid": 4242, "artifact": "m", "log": "l"}}
    monkeypatch.setattr(spawn, "registry", lambda: reg)
    monkeypatch.setattr(spawn, "save_registry", lambda r: None)
    monkeypatch.setattr(spawn, "is_our_server", lambda pid: True)
    monkeypatch.setattr("os.kill",
                        lambda pid, sig: (sent if sent is not None else []).append(sig))
    monkeypatch.setattr(spawn.time, "sleep", lambda s: None)
    # a short ceiling: the wait reads the wall clock, which the patched
    # sleep does not advance
    monkeypatch.setattr(lifecycle, "STOP_SAVE_S", 0.0)
    monkeypatch.setattr(spawn, "STOP_MARGIN_S", 0.05)
    monkeypatch.setitem(spawn.CHILDREN, 8080, (proc,))
    return reg


def test_unload_waits_for_the_process_to_exit(monkeypatch):
    proc = FakeProc(exits_after=5)
    reg = _setup(monkeypatch, proc)
    out = spawn.stop(8080)
    assert out.get("stopped") == "m" and "exiting" not in out
    assert proc.polls > 5 and 8080 not in reg


def test_a_server_that_does_not_leave_is_stuck_not_killed(monkeypatch):
    proc, sent = FakeProc(exits_after=10 ** 9), []
    reg = _setup(monkeypatch, proc, sent)
    out = spawn.stop(8080)
    assert sent == [signal.SIGTERM]
    assert out["stuck"] == [4242]
    assert out["note"].endswith(servers.stuck_reason())
    assert reg[8080]["stuck"] is True             # still listed, as stuck
    spawn.CHILDREN.pop(8080, None)
