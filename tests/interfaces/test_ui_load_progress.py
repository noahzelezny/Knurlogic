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


def _follower(monkeypatch, tmp_path, quiet_s, held, job_phase):
    """Rank 1 of job j1 (no port, no server row), its log quiet for
    `quiet_s`, holding `held` of its 500-byte share."""
    import os
    from knurlogic.interfaces import spawn
    art = tmp_path / "Flash"
    art.mkdir(parents=True)
    log = tmp_path / "r1.log"
    log.write_text("loaded\n")
    now = time.time()
    os.utime(log, (now - quiet_s, now - quiet_s))
    jobs = {"j1/1": {"job": "j1", "rank": 1, "pid": 101, "artifact": str(art),
                     "log": str(log), "t": now - quiet_s - 40}}
    monkeypatch.setattr(J, "read_marker",
                        lambda job, rank: {"share_bytes": 500})
    monkeypatch.setattr(page_loads, "registry", lambda: {})
    monkeypatch.setattr(J, "registry", lambda: jobs)
    monkeypatch.setattr(servers, "is_our_server", lambda pid: True)
    monkeypatch.setattr(spawn, "_GROWTH", {})
    doc = {"memory": {"processes": [{"pid": 101, "bytes": held}]},
           "jobs": [{"job": "j1", "phase": job_phase}]}
    [e] = page_loads.load_progress(doc)
    return e["phase"]


def test_a_follower_with_its_share_in_is_waiting_not_stalled(
        monkeypatch, tmp_path):
    # live 2026-10-10: the M3's rank of a pipeline split had its 42 GiB in
    # after 40 s, then read "stalled" while the M4 read its 68 GiB share
    assert _follower(monkeypatch, tmp_path, 200, 500, "loading") == "waiting"


def test_a_follower_of_a_serving_job_is_ready(monkeypatch, tmp_path):
    # it has no server row, so it read "loading" after the job served
    assert _follower(monkeypatch, tmp_path, 200, 500, "ready") == "ready"


def test_a_follower_short_of_its_share_can_still_stall(monkeypatch, tmp_path):
    assert _follower(monkeypatch, tmp_path, 200, 100, "loading") == "stalled"
    assert _follower(monkeypatch, tmp_path / "b", 5, 100, "loading") == \
        "loading"
