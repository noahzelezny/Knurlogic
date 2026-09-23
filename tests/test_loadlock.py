"""The model-load lock (machine/loadlock.py): report-test-plan.md section 4.

The holder is a separate PROCESS, because flock's whole value is what the
kernel does across processes -- including releasing the lock of one that
was killed with SIGKILL, which a pidfile would leave stale.
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from knurlogic.machine import loadlock  # noqa: E402

HOLD = """
import sys, time
sys.path.insert(0, {src!r})
from pathlib import Path
from knurlogic.machine import loadlock
with loadlock.model_load("tiny-artifact", "test hold", agent="holder",
                         path=Path({path!r})):
    print("held", flush=True)
    time.sleep(120)
"""


@pytest.fixture
def lock(tmp_path, monkeypatch):
    p = tmp_path / "load.lock"
    monkeypatch.setenv("KNURLOGIC_LOADLOCK", str(p))
    return p


def _holder_proc(path):
    proc = subprocess.Popen(
        [sys.executable, "-c", HOLD.format(src=str(ROOT / "src"),
                                           path=str(path))],
        stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "held"
    return proc


def test_second_taker_gets_busy_with_the_holder_record(lock):
    proc = _holder_proc(lock)
    try:
        with pytest.raises(loadlock.Busy) as e:
            with loadlock.model_load("other", "gate"):
                pass
        assert e.value.holder["pid"] == proc.pid
        assert e.value.holder["artifact"] == "tiny-artifact"
        assert loadlock.holder()["agent"] == "holder"
        t0 = time.monotonic()
        with pytest.raises(loadlock.Busy):
            with loadlock.model_load("other", "gate", wait_s=0.3):
                pass
        assert time.monotonic() - t0 >= 0.3
    finally:
        proc.kill()
        proc.wait()


def test_kill_9_releases_the_lock(lock):
    proc = _holder_proc(lock)
    assert loadlock.holder() is not None
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait()
    assert loadlock.holder() is None, "the kernel releases a dead holder"
    with loadlock.model_load("next", "gate") as rec:
        assert rec["pid"] == os.getpid()


def test_holder_sees_this_process_too_and_clears_after(lock):
    assert loadlock.holder() is None
    with loadlock.model_load("a", "p", agent="me"):
        h = loadlock.holder()
        assert h["agent"] == "me" and h["purpose"] == "p"
        with pytest.raises(loadlock.Busy):         # not reentrant: one load
            with loadlock.model_load("b", "p"):
                pass
    assert loadlock.holder() is None
    assert lock.read_text() == "", "record cleared on release"


def test_wait_acquires_once_the_holder_leaves(lock):
    proc = _holder_proc(lock)
    try:
        subprocess.Popen(["/bin/sh", "-c",
                          f"sleep 0.4; kill -9 {proc.pid}"])
        with loadlock.model_load("w", "p", wait_s=10) as rec:
            assert json.loads(lock.read_text())["pid"] == rec["pid"]
    finally:
        proc.kill()
        proc.wait()


def test_default_path_is_beside_servers_json(monkeypatch, tmp_path):
    monkeypatch.delenv("KNURLOGIC_LOADLOCK", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert loadlock.lock_path() == tmp_path / "knurlogic" / "load.lock"
    assert loadlock.EXIT_BUSY == 75


def test_no_engine_import():
    """machine/ is stdlib: asking who holds the lock (ready()) must not
    load mlx."""
    out = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r});"
         "import knurlogic.machine.loadlock as l; l.holder();"
         "print(any(k == 'mlx' or k.startswith(('mlx.', 'mlx_')) "
         "for k in sys.modules))"],
        capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"
