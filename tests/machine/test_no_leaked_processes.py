"""A process a test starts never outlives it (conftest's guard), and a
test's fake rank is never something a real page lists as a model."""
import subprocess
import sys

import conftest
from knurlogic.machine import loaded, servers

FAKE = ("/opt/python /Users/x/knurlogic/tests/support/cluster_fake_rank.py "
        "/fake/artifact knurlogic serve --rank 0 --job ab --port 8123 "
        "--cable  --knurlogic-test")


def test_a_detached_child_is_found_and_killed():
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                         start_new_session=True)
    try:
        assert p.pid in {pid for pid, _ in conftest._ours()}
        conftest._kill([(p.pid, "")])
        assert p.wait(timeout=5) is not None
        assert p.pid not in {pid for pid, _ in conftest._ours()}
    finally:
        p.kill()


def test_a_fake_rank_is_not_a_listening_serve(monkeypatch):
    class R:
        stdout = f"4242 {FAKE}\n"
    calls = []

    def run(cmd, **k):
        calls.append(cmd)
        return R
    monkeypatch.setattr(servers.subprocess, "run", run)
    assert servers.listening_serves() == {}
    assert len(calls) == 1               # lsof never asked about it


def test_a_fake_rank_is_no_runtime():
    assert servers.is_test_process(FAKE)
    assert servers.is_test_process("python -m knurlogic serve "
                                   "/private/var/folders/x/pytest-of-n/m")
    assert not servers.is_test_process("python -m knurlogic serve /m --port 1")
    assert loaded._runtime_of(FAKE) == ""
    assert loaded._runtime_of(
        "/usr/bin/python3 -m knurlogic serve /m --port 8080") == "knurlogic"


def test_fake_ranks_carry_the_mark():
    import cluster_fake_page
    argv = cluster_fake_page.fake_argv("/fake", {"rank": 0, "job": "ab"}, {})
    assert servers.is_test_process(" ".join(argv))
