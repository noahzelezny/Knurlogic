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
    # (tuning/preferences) are read per request and at launch, so the
    # person's real ones must not steer a test
    monkeypatch.setenv("XDG_CONFIG_HOME",
                       str(tmp_path_factory.mktemp("xdg-config")))
    # and its own KNURLOGIC_HOME: the request ledger (machine/ledger.py)
    # is written there by every served request
    monkeypatch.setenv("KNURLOGIC_HOME",
                       str(tmp_path_factory.mktemp("knurlogic-home")))
    # and no port lookup: an unload finds a server the page did not start
    # by the port it listens on (interfaces/spawn.stop), which would reach
    # the REAL server on :8080 past the registry above. A test of that
    # lookup patches it itself.
    from knurlogic.machine import servers
    monkeypatch.setattr(servers, "listener_pid", lambda port: None)


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
    # an xdist worker gets a token of its own: inherited, every worker's
    # sweep found the other workers (and the controller) carrying it and
    # killed them after its first test. A worker's own ps environment still
    # shows the controller's token, so no worker's sweep matches another.
    if _os.environ.get("PYTEST_XDIST_WORKER"):
        _os.environ[TEST_ENV] = _secrets.token_hex(8)
    else:
        _os.environ.setdefault(TEST_ENV, _secrets.token_hex(8))


def _ours() -> list:
    """[(pid, command)] of live processes carrying this session's token,
    this pytest process excluded."""
    mark = f"{TEST_ENV}={_os.environ.get(TEST_ENV, '')}"
    out = _ps()
    if out is None:
        return []
    me, got = _os.getpid(), []
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        pid, stat, cmd = int(parts[0]), parts[1], parts[2]
        if pid == me or stat.startswith("Z") or mark not in cmd \
                or cmd.startswith(("ps -E -ww -ax ", "/bin/ps -E -ww -ax ")):
            continue
        got.append((pid, cmd[:160]))
    return got


PS_FAILED: list = []
# the real Popen, bound now: a test may monkeypatch subprocess.run or
# subprocess.Popen (the module is shared, and run looks Popen up when
# called), and the sweep after it must still look
_POPEN = _subprocess.Popen


def _ps():
    """`ps` with every process's environment, or None when it failed three
    times (a loaded box can time it out). A failure is recorded: it used to
    read as "nothing left", so a sweep that could not look passed."""
    why = ""
    for _ in range(3):
        try:
            p = _POPEN(["/bin/ps", "-E", "-ww", "-ax", "-o",
                        "pid=,stat=,command="],
                       stdout=_subprocess.PIPE, stderr=_subprocess.PIPE,
                       text=True, errors="replace")
            try:
                out, err = p.communicate(timeout=30)
            except _subprocess.TimeoutExpired:
                p.kill()
                p.communicate()
                raise
            # a process that exits mid-scan makes ps exit non-zero with
            # the rest of the list printed: that list still answers
            if out.strip():
                return out
            why = f"rc {p.returncode}, no output: {err.strip()[:200]}"
        except Exception as e:
            why = f"{type(e).__name__}: {e}"
        _time.sleep(0.5)
    PS_FAILED.append(why)
    return None


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


@pytest.fixture
def owned_procs():
    """Register a Popen as it is made -- `p = owned_procs(Popen(...))` --
    and it is killed and waited on however the test or fixture ends,
    setup failure included (this finalizer runs even when the requesting
    fixture raised before its yield)."""
    from procs import Owned
    owner = Owned()
    yield owner
    owner.reap()


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
    if PS_FAILED:
        # the sweep could not look: say so, rather than pass blind
        session.exitstatus = 1


def pytest_terminal_summary(terminalreporter):
    _log_run(terminalreporter)
    if PS_FAILED:
        terminalreporter.write_line(
            f"FAILED: the leftover-process sweep could not run ps "
            f"{len(PS_FAILED)} time(s); a leaked process may be running "
            f"({PS_FAILED[-1]})",
            red=True)
    if LEFT:
        terminalreporter.write_line(
            f"FAILED: {len(LEFT)} process(es) outlived the test session "
            f"(killed now):", red=True)
        for pid, cmd in LEFT:
            terminalreporter.write_line(f"  pid {pid}: {cmd}")


_STARTED = __import__("time").time()


def _log_run(tr) -> None:
    """One record per run in ~/.cache/knurlogic/test-runs.log: when, the
    commit, what ran, the counts, and every failing test by name, so a break
    shows against the last run that passed."""
    import subprocess
    import sys
    import time
    from pathlib import Path
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              cwd=Path(__file__).parent).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "-uno"],
                               capture_output=True, text=True,
                               cwd=Path(__file__).parent).stdout.strip()
    except OSError:
        head, dirty = "?", ""
    counts = {k: len(tr.stats.get(k, [])) for k in
              ("passed", "failed", "error", "skipped")}
    took = time.time() - _STARTED
    args = " ".join(sys.argv[1:]) or "(all)"
    lines = [f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {head}"
             f"{'+dirty' if dirty else ''}  {took:.0f}s  "
             + "  ".join(f"{k} {v}" for k, v in counts.items() if v)
             + f"  [{args}]"]
    for k in ("failed", "error"):
        for r in tr.stats.get(k, []):
            lines.append(f"    {k.upper()} {r.nodeid}")
    try:
        log = Path.home() / ".cache" / "knurlogic" / "test-runs.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as f:
            f.write("\n".join(lines) + "\n")
    except OSError:
        pass
