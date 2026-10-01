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


INNER = '''
import os, subprocess, sys
import pytest

RANK = [sys.executable, os.path.join(os.environ["SUPPORT"],
        "cluster_fake_rank.py"), "/fake/artifact", "knurlogic", "serve",
        "--rank", "0", "--job", "leak", "--port", "0", "--knurlogic-test"]


def spawn(tag):
    p = subprocess.Popen(RANK, start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(os.path.join(os.environ["PIDS"], tag), "w") as f:
        f.write(str(p.pid))
    assert p.poll() is None
    return p


def test_fails_with_a_bare_rank_running():
    spawn("bare")
    assert False, "a failing test"


@pytest.fixture
def broken(owned_procs):
    owned_procs(spawn("owned"))
    raise RuntimeError("setup fails before the yield")
    yield


def test_fixture_setup_fails(broken):
    pass
'''


def test_a_failing_test_leaves_no_fake_rank_behind(tmp_path):
    """A test that fails with a fake rank 0 running, and a fixture whose
    setup raises after spawning one, both end with the rank killed and
    reaped -- in an inner pytest session with its own token."""
    import os
    import shutil
    from pathlib import Path
    tests = Path(conftest.__file__).resolve().parent
    inner, pids = tmp_path / "inner", tmp_path / "pids"
    inner.mkdir()
    pids.mkdir()
    shutil.copy(tests / "conftest.py", inner / "conftest.py")
    (inner / "test_leak.py").write_text(INNER)
    (inner / "pytest.ini").write_text("[pytest]\n")
    env = {k: v for k, v in os.environ.items() if k != conftest.TEST_ENV}
    env.update(PIDS=str(pids), SUPPORT=str(tests / "support"),
               PYTHONPATH=os.pathsep.join(
                   [str(tests.parent / "src"), str(tests / "support"),
                    os.environ.get("PYTHONPATH", "")]))
    spawned = []
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
             "-c", str(inner / "pytest.ini"), "--rootdir", str(inner),
             str(inner / "test_leak.py")],
            cwd=inner, env=env, capture_output=True, text=True, timeout=120)
        spawned = [int((pids / t).read_text()) for t in ("bare", "owned")
                   if (pids / t).exists()]
        assert len(spawned) == 2, r.stdout + r.stderr
        assert "1 failed" in r.stdout and "1 error" in r.stdout, r.stdout
        assert "outlived" not in r.stdout, r.stdout
        for pid in spawned:
            assert not _alive(pid), f"fake rank {pid} outlived its test"
    finally:
        import signal
        for pid in spawned:                  # only what this test spawned
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass


def _alive(pid: int) -> bool:
    import os
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    st = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                        capture_output=True, text=True).stdout.strip()
    return bool(st) and not st.startswith("Z")
