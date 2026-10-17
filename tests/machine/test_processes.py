"""machine/processes.py: one process that cannot be read -- gone, denied,
or exiting mid-read -- is left out, never the whole table."""
from types import SimpleNamespace

import psutil

from knurlogic.machine import processes


class _Proc:
    def __init__(self, pid, cmd=None, rss=0, exc=None):
        self.pid, self._cmd, self._rss, self._exc = pid, cmd, rss, exc

    def cmdline(self):
        if self._exc:
            raise self._exc
        return self._cmd

    def memory_info(self):
        if self._exc:
            raise self._exc
        return type("M", (), {"rss": self._rss})()


def test_a_process_exiting_mid_read_is_left_out(monkeypatch):
    """On macOS psutil raises SystemError (around a KERN_PROCARGS2
    PermissionError) for a process that exits while its command line is
    read. Under load that failed a page's JobState, which a peer then
    read as "no rank here" and stopped a healthy job."""
    exiting = SystemError("<built-in function proc_cmdline> returned a "
                          "result with an exception set")
    table = [_Proc(1, ["python", "-m", "knurlogic", "serve", "--job", "ab"],
                   rss=10),
             _Proc(2, exc=exiting),
             _Proc(3, exc=psutil.AccessDenied(3)),
             _Proc(4, exc=psutil.NoSuchProcess(4))]
    # processes' own psutil only: the suite's leak sweep lists the real one
    monkeypatch.setattr(processes, "psutil", SimpleNamespace(
        process_iter=lambda *a, **k: iter(table)))
    assert processes.commands() == {1: "python -m knurlogic serve --job ab"}
    assert processes.rss_all() == {1: 10}
    assert processes.matching(r"--job ab") == [1]


def test_one_process_read_that_fails_mid_exit_reads_as_gone(monkeypatch):
    def raises(pid):
        raise SystemError("exiting")
    monkeypatch.setattr(processes, "psutil", SimpleNamespace(Process=raises))
    assert processes.command(7) == ""
    assert processes.rss(7) == 0
