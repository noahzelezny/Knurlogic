"""machine/processes: liveness answered without raising."""
import os

from knurlogic.machine import processes


def test_this_process_exists_and_a_free_pid_does_not(monkeypatch):
    assert processes.exists(os.getpid())
    monkeypatch.setattr(processes.psutil, "pid_exists", lambda pid: False)
    assert not processes.exists(os.getpid())


def test_a_pid_that_cannot_be_read_still_exists(monkeypatch):
    """AccessDenied on its command line: command() says "", exists() still
    says it is there."""
    def denied(pid):
        raise processes.psutil.AccessDenied(pid)
    monkeypatch.setattr(processes.psutil, "Process", denied)
    assert processes.command(os.getpid()) == ""
    assert processes.exists(os.getpid())
