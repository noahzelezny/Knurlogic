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
