"""Suite-wide guards."""

import threading

import pytest


def _started() -> threading.Event:
    """A loop's started flag, already set: the loop is never begun."""
    ev = threading.Event()
    ev.set()
    return ev


@pytest.fixture(autouse=True)
def _no_cluster_watcher_thread(monkeypatch):
    """A test that starts a rank (launch.start) would start the
    page's watcher thread, which outlives the test and its XDG_CACHE_HOME:
    would read the REAL ~/.cache/knurlogic/jobs registry, find a real
    job's peer page unknown for 20 s, and stop that job. Tests call
    watch_once by hand; no thread is started."""
    from knurlogic.cluster import launch
    monkeypatch.setattr(launch, "_WATCHER", _started())


@pytest.fixture(autouse=True)
def _own_job_state(monkeypatch):
    """Every test gets its own copy of launch's in-memory job state
    (prepared ranks, ended jobs, the peer watcher's clocks): what one test
    leaves behind -- an assertion failing before its cleanup -- is
    otherwise seen by whichever test the same worker runs next."""
    from knurlogic.cluster import launch
    for name in ("PREPARED", "ENDED", "SPECS", "_PROCS", "_PEER_OK",
                 "_PEER_BOOT"):
        monkeypatch.setattr(launch, name, {})
    for name in ("_STOPPING", "FOLLOWING", "_PEER_SLOW"):
        monkeypatch.setattr(launch, name, set())


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
    monkeypatch.setattr(recovery, "_THREAD", _started())
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
    from knurlogic.machine import processes
    monkeypatch.setattr(processes, "listener", lambda port: None)
    # and no prompt-cache fetcher: a page started in this process sets the
    # port its servers ask for peers; a restore must not ask a real page
    monkeypatch.delenv("KNURLOGIC_PAGE_PORT", raising=False)
    from knurlogic.engine.prompt_cache import transfer
    monkeypatch.setattr(transfer, "_FETCHER", [None])


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
    """[(pid, command)] of live processes carrying this session's token in
    their environment, this pytest process excluded. A listing that fails
    is recorded: it used to read as "nothing left", so a sweep that could
    not look passed."""
    import psutil
    token, me, got = _os.environ.get(TEST_ENV, ""), _os.getpid(), []
    try:
        procs = list(psutil.process_iter(["status", "cmdline", "environ"]))
    except Exception as e:
        PS_FAILED.append(f"{type(e).__name__}: {e}")
        return []
    for p in procs:
        info = p.info
        if p.pid == me or info.get("status") == psutil.STATUS_ZOMBIE \
                or (info.get("environ") or {}).get(TEST_ENV) != token:
            continue
        got.append((p.pid, " ".join(info.get("cmdline") or [])[:160]))
    return got


PS_FAILED: list = []


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
            f"FAILED: the leftover-process sweep could not list processes "
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
