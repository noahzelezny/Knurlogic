"""Unload answers when the server process is gone, not when it was signalled."""

from knurlogic.interfaces import spawn


class FakeProc:
    def __init__(self, exits_after):
        self.polls, self.exits_after = 0, exits_after

    def poll(self):
        self.polls += 1
        return 0 if self.polls > self.exits_after else None

    def wait(self, timeout=None):
        return 0


def _setup(monkeypatch, proc):
    reg = {8080: {"pid": 4242, "artifact": "m", "log": "l"}}
    monkeypatch.setattr(spawn, "registry", lambda: reg)
    monkeypatch.setattr(spawn, "save_registry", lambda r: None)
    monkeypatch.setattr(spawn, "is_our_server", lambda pid: True)
    monkeypatch.setattr("os.kill", lambda pid, sig: None)
    monkeypatch.setattr(spawn.time, "sleep", lambda s: None)
    monkeypatch.setitem(spawn.CHILDREN, 8080, (proc,))
    return reg


def test_unload_waits_for_the_process_to_exit(monkeypatch):
    proc = FakeProc(exits_after=5)
    reg = _setup(monkeypatch, proc)
    out = spawn.stop(8080)
    assert out.get("stopped") == "m" and "exiting" not in out
    assert proc.polls > 5 and 8080 not in reg


def test_unload_reports_still_exiting_instead_of_hanging(monkeypatch):
    proc = FakeProc(exits_after=10 ** 9)
    reg = _setup(monkeypatch, proc)
    monkeypatch.setattr(spawn, "EXIT_WAIT_S", 0.0)
    out = spawn.stop(8080)
    assert out["exiting"] == [4242] and "still exiting" in out["note"]
    assert 8080 in reg
    spawn.CHILDREN.pop(8080, None)
