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
