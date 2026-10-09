"""The page's load indicator counts every rank this machine started."""
import time

from knurlogic.cluster import jobs as J
from knurlogic.interfaces.page import loads as page_loads
from knurlogic.machine import servers


def test_follower_ranks_counted_once(monkeypatch, tmp_path):
    """A follower binds no port and lives only in jobs.json; rank 0 is in
    both registries. Every rank shows once, with its own process bytes."""
    art = tmp_path / "Flash-Next-8bit"
    art.mkdir()
    log = tmp_path / "r.log"
    log.write_text("loading\n")
    now = time.time()
    rank0 = {"pid": 100, "artifact": str(art), "log": str(log), "t": now,
             "job": "j1"}
    jobs = {"j1/0": {"job": "j1", "rank": 0, "pid": 100, "port": 8000,
                     "artifact": str(art), "log": str(log), "t": now},
            "j1/1": {"job": "j1", "rank": 1, "pid": 101,
                     "artifact": str(art), "log": str(log), "t": now}}
    monkeypatch.setattr(page_loads, "registry", lambda: {8000: rank0})
    monkeypatch.setattr(J, "registry", lambda: jobs)
    monkeypatch.setattr(servers, "is_our_server", lambda pid: True)
    doc = {"memory": {"processes": [{"pid": 100, "bytes": 10},
                                    {"pid": 101, "bytes": 600}]}}
    loads = page_loads.load_progress(doc)
    assert sorted((e["job"], e["rank"], e["bytes"]) for e in loads) == [
        ("j1", 0, 10), ("j1", 1, 600)]
    assert [e["port"] for e in loads if e["rank"] == 1] == [0]


def test_each_rank_is_measured_against_its_own_share(monkeypatch, tmp_path):
    # a tensor job's rank 0 holds its half plus the 5.4 GiB MTP head: against
    # the artifact the two ranks summed past 100% while rank 0 still loaded
    art = tmp_path / "397B"
    art.mkdir()
    log = tmp_path / "r.log"
    log.write_text("loading\n")
    now = time.time()
    rank0 = {"pid": 100, "artifact": str(art), "log": str(log), "t": now,
             "job": "j1"}
    jobs = {"j1/0": {"job": "j1", "rank": 0, "pid": 100, "port": 8000,
                     "artifact": str(art), "log": str(log), "t": now},
            "j1/1": {"job": "j1", "rank": 1, "pid": 101,
                     "artifact": str(art), "log": str(log), "t": now}}
    # each rank writes its share to its own marker (marker.progress), not to
    # the registry: the test that put it there passed while no page saw it
    monkeypatch.setattr(J, "read_marker", lambda job, rank: {
        ("j1", 0): {"share_bytes": 560}, ("j1", 1): {"share_bytes": 500}}
        .get((job, rank)))
    monkeypatch.setattr(page_loads, "registry", lambda: {8000: rank0})
    monkeypatch.setattr(J, "registry", lambda: jobs)
    monkeypatch.setattr(servers, "is_our_server", lambda pid: True)
    doc = {"memory": {"processes": [{"pid": 100, "bytes": 300},
                                    {"pid": 101, "bytes": 500}]}}
    got = {e["rank"]: e["total_bytes"] for e in page_loads.load_progress(doc)}
    assert got == {0: 560, 1: 500}
