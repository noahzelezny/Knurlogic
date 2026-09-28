"""Suite-wide guards."""

import pytest


@pytest.fixture(autouse=True)
def _no_cluster_watcher_thread(monkeypatch):
    """A test that starts a rank (cluster_jobs.start) would start the
    page's watcher thread, which outlives the test and its XDG_CACHE_HOME:
    it then read the REAL ~/.cache/knurlogic/jobs registry, found a real
    job's peer page unknown for 20 s, and stopped that job -- a pytest run
    on the M3 killed the 397B cluster job's rank 1 mid-review (2026-09-27).
    Tests call watch_once by hand; no thread is started."""
    from knurlogic.interfaces import cluster_jobs
    monkeypatch.setattr(cluster_jobs, "_WATCHER", [1])


@pytest.fixture(autouse=True)
def _no_real_page(monkeypatch):
    """The MCP's cross-machine tools ask the page on this Mac (127.0.0.1:8899
    by default). A test must never reach the REAL page -- an `unload` there
    stops a real job -- so it points at a port nothing listens on, and a
    test that wants a page starts one and says where."""
    monkeypatch.setenv("KNURLOGIC_PAGE", "127.0.0.1:9")
