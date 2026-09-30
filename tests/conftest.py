"""Suite-wide guards."""

import pytest


@pytest.fixture(autouse=True)
def _no_cluster_watcher_thread(monkeypatch):
    """A test that starts a rank (launch.start) would start the
    page's watcher thread, which outlives the test and its XDG_CACHE_HOME:
    would read the REAL ~/.cache/knurlogic/jobs registry, find a real
    job's peer page unknown for 20 s, and stop that job. Tests call
    watch_once by hand; no thread is started."""
    from knurlogic.cluster import launch
    monkeypatch.setattr(launch, "_WATCHER", [1])


@pytest.fixture(autouse=True)
def _no_real_page(monkeypatch):
    """The MCP's cross-machine tools ask the page on this Mac (127.0.0.1:8899
    by default). A test must never reach the REAL page -- an `unload` there
    stops a real job -- so it points at a port nothing listens on, and a
    test that wants a page starts one and says where."""
    monkeypatch.setenv("KNURLOGIC_PAGE", "127.0.0.1:9")


@pytest.fixture(autouse=True)
def _no_recovery_thread(monkeypatch, tmp_path):
    """Auto-recovery (cluster/recovery.py) tracks what a launch starts
    and relaunches it from a thread: a test's killed rank must not come back
    after the test, nor a test write the REAL recovery.json. Tests call
    recovery.tick by hand, on their own records and file."""
    from knurlogic.cluster import recovery
    monkeypatch.setattr(recovery, "_THREAD", [1])
    monkeypatch.setattr(recovery, "MODELS", {})
    monkeypatch.setattr(recovery, "_SAVED", {})
    path = tmp_path / "recovery.json"
    monkeypatch.setattr(recovery, "_path", lambda: path)


@pytest.fixture(autouse=True)
def _no_real_cache(monkeypatch, tmp_path_factory):
    """Every test gets its own ~/.cache/knurlogic: the server registry, the
    job registry, the load lock and recovery.json all live there. With the
    real one, a test's `unload(port=8080)` would find the REAL server on
    :8080 in the registry and stop it. A test that wants a cache sets its
    own XDG_CACHE_HOME after this."""
    monkeypatch.setenv("XDG_CACHE_HOME",
                       str(tmp_path_factory.mktemp("xdg-cache")))
    # and its own ~/.config/knurlogic: the knurlogic-wide settings
    # (machine/preferences) are read per request and at launch, so the
    # person's real ones must not steer a test
    monkeypatch.setenv("XDG_CONFIG_HOME",
                       str(tmp_path_factory.mktemp("xdg-config")))


# --- no process a test starts outlives it ------------------------------------
# Fake ranks and fake pages (cluster_fake_rank.py, cluster_fake_page.py),
# ring workers and holders are real processes, some in their own session
# (start_new_session), so a failed test -- or a fixture whose setup failed
# after it started one -- left them running: orphans that answered on real
# ports and showed on the maintainer's page as models named "fake". Every
# process of this session carries TEST_ENV=<token> in its environment
# (inherited through every Popen), so it is found however it was started,
# reparented or not; each test's teardown kills what is left, and the
# session fails if anything survives to its end.

import os as _os
import secrets as _secrets
import signal as _signal
import subprocess as _subprocess
import time as _time

TEST_ENV = "KNURLOGIC_TEST_SESSION"


def pytest_configure(config):
    _os.environ.setdefault(TEST_ENV, _secrets.token_hex(8))


def _ours() -> list:
    """[(pid, command)] of live processes carrying this session's token,
    this pytest process excluded."""
    mark = f"{TEST_ENV}={_os.environ.get(TEST_ENV, '')}"
    try:
        out = _subprocess.run(["ps", "-E", "-ww", "-ax", "-o",
                               "pid=,stat=,command="],
                              capture_output=True, text=True,
                              timeout=10).stdout
    except Exception:
        return []
    me, got = _os.getpid(), []
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        pid, stat, cmd = int(parts[0]), parts[1], parts[2]
        if pid == me or stat.startswith("Z") or mark not in cmd \
                or cmd.startswith("ps -E -ww -ax "):
            continue
        got.append((pid, cmd[:160]))
    return got


def _kill(procs: list) -> None:
    for pid, _ in procs:
        for sig in (_signal.SIGKILL,):
            try:
                _os.kill(pid, sig)
            except OSError:
                pass
    end = _time.time() + 5
    while _time.time() < end and _ours():
        _time.sleep(0.05)


@pytest.fixture(autouse=True)
def _no_process_outlives_its_test():
    """Teardown, pass or fail: kill every process this test left."""
    yield
    left = _ours()
    if left:
        _kill(left)


LEFT: list = []


def pytest_sessionfinish(session, exitstatus):
    left = _ours()
    if left:
        LEFT.extend(left)
        _kill(left)
        session.exitstatus = 1


def pytest_terminal_summary(terminalreporter):
    if LEFT:
        terminalreporter.write_line(
            f"FAILED: {len(LEFT)} process(es) outlived the test session "
            f"(killed now):", red=True)
        for pid, cmd in LEFT:
            terminalreporter.write_line(f"  pid {pid}: {cmd}")
