"""Auto-recovery (interfaces/recovery.py): a model that dies unasked is
relaunched by the page that launched it, bounded; a requested stop or an
out-of-memory one never is.

The cluster cases run the real two pages and fake ranks of
tests/test_cluster_jobs.py (its `two_pages` fixture: this process is page A,
the coordinator; page B is a process). The limits are tested on the loop
itself with the launch faked."""

import os
import time

import pytest

from knurlogic.cluster import jobs as J
from knurlogic.interfaces import cluster_jobs as C
from knurlogic.interfaces import recovery as R
from knurlogic.interfaces import ui

from test_cluster_jobs import alive, two_pages, wait  # noqa: F401


def ticks_until(pred, t=40.0):
    """tick() until one of its reports satisfies `pred`; -> that report."""
    end = time.time() + t
    while time.time() < end:
        for key, what in R.tick():
            if pred(what):
                return what
        time.sleep(0.2)
    return None


def test_a_killed_rank_comes_back(two_pages, monkeypatch):
    p = two_pages
    monkeypatch.setattr(R, "BACKOFF_S", (0.0, 0.0, 0.0))
    assert R.for_job(p.job) is None       # nothing to report yet
    os.kill(p.rank1["pid"], 9)            # B's page stops the job, tells A
    assert wait(lambda: p.job in C.ENDED, 30)
    new = None
    try:
        what = ticks_until(lambda w: w.startswith("relaunch 1"))
        assert what, R.MODELS
        new = next(iter(R.MODELS.values()))["job"]
        assert new != p.job
        v = R.for_job(new)
        assert v["state"] == "recovering" and v["attempts"] == 1
        assert "rank 1 on B" in v["last_reason"] and "exited" in \
            v["last_reason"]
        # the same launch: machines, order, split, link and port
        recs = J.by_job()[new]
        assert recs[0]["machines"] == ["A", "B"]
        assert recs[0]["split"] == "tensor" and recs[0]["link"] == "ring"
        assert recs[0].get("port") == p.port
        assert ticks_until(lambda w: w.startswith("recovered"))
        v = R.for_job(new)
        assert v["state"] == "recovered" and v["attempts"] == 1
        assert v["next_at"] is None
        # reported: /loaded.json's job and rank 0's row, and the server's
        # own /v1/residency row (this machine's file, by port)
        doc = ui.with_jobs({"resident": [
            {"runtime": "knurlogic",
             "where": f"http://127.0.0.1:{p.port}"}]})
        assert doc["resident"][0]["recovery"]["state"] == "recovered"
        job = next(j for j in doc["jobs"] if j["job"] == new)
        assert job["recovery"]["attempts"] == 1
        assert R.served_view(p.port, True)["state"] == "recovered"
        # the MCP's models: one entry, carrying it
        from knurlogic.interfaces import mcp
        ms = mcp.models_across(doc, "A")
        assert [m["recovery"]["state"] for m in ms] == ["recovered"]
    finally:
        if new:
            C.stop(new, grace=1)


def test_an_unload_is_not_recovered(two_pages, monkeypatch):
    p = two_pages
    monkeypatch.setattr(R, "BACKOFF_S", (0.0, 0.0, 0.0))
    assert R.MODELS
    ui._stop(p.port)                      # the Unload button
    assert wait(lambda: not alive(p.rank1["pid"]), 20)
    assert not R.MODELS
    assert R.tick() == []
    assert R.for_job(p.job) is None and R.not_serving() == []


def test_an_unload_from_the_other_page_is_not_recovered(two_pages,
                                                        monkeypatch):
    p = two_pages
    monkeypatch.setattr(R, "BACKOFF_S", (0.0, 0.0, 0.0))
    # B's Unload: B stops its rank and tells A, reason "unloaded"
    C._post(f"http://127.0.0.1:{p.ui_b}{C.STOP_PATH}",
            {"job": p.job, "reason": "unloaded"})
    # A's watcher asks B, and stops the job for B's reason
    assert wait(lambda: C.watch_once() or p.job in C.ENDED, 30)
    assert "unloaded" in C.ENDED[p.job]["reason"]
    assert R.tick() == [] and not R.MODELS


def test_out_of_memory_is_failed_not_relaunched(two_pages, monkeypatch):
    p = two_pages
    monkeypatch.setattr(R, "BACKOFF_S", (0.0, 0.0, 0.0))
    launched = []
    monkeypatch.setattr(C, "launch", lambda *a, **k: launched.append(k)
                        or {"job": "f" * 16})
    why = ("rank 1 on B (pid 1) exited: out of memory: [METAL] Command "
           "buffer execution failed: Insufficient Memory")
    C.stop(p.job, reason=why)
    assert wait(lambda: not alive(p.rank1["pid"]), 20)
    what = ticks_until(lambda w: w.startswith("failed"), 10)
    assert what and "Insufficient Memory" in what
    for _ in range(3):
        R.tick()
    assert launched == []
    v = R.for_job(p.job)
    assert v["state"] == "failed" and v["attempts"] == 0
    assert "Insufficient Memory" in v["last_reason"]
    # surfaced while nothing serves: /loaded.json's `recovery`, and the
    # MCP's models
    down = ui.with_jobs({"resident": []})["recovery"]
    assert [d["state"] for d in down] == ["failed"]
    from knurlogic.interfaces import mcp
    ms = mcp.models_across({"resident": [], "recovery": down}, "A")
    assert ms[0]["state"] == "failed" and ms[0]["machines"] == ["A", "B"]


def test_a_rank_log_saying_out_of_memory_names_it_in_the_reason(
        two_pages, monkeypatch):
    p = two_pages
    with open(p.rank0["log"], "a") as fh:
        fh.write("libc++abi: terminating: [metal::malloc] Unable to "
                 "allocate 8589934592 bytes\n")
    os.kill(p.rank0["pid"], 9)
    assert wait(lambda: not alive(p.rank0["pid"]), 5)
    stopped = C.watch_once()
    assert stopped and "out of memory" in stopped[0][1]
    assert R.kind(stopped[0][1]) == "memory"


# --- the loop's limits, with the launch faked ----------------------------

@pytest.fixture
def faked(monkeypatch):
    """One tracked cluster model whose job ends whenever `ended` says, and
    whose relaunches are recorded."""
    state = {"ended": None, "launches": [], "n": 0, "down": ""}
    monkeypatch.setattr(C, "_job_end", lambda job, order, post:
                        state["ended"])
    monkeypatch.setattr(R, "_leftovers", lambda rec, job: "")
    monkeypatch.setattr(R, "_machines_down", lambda rec: state["down"])
    monkeypatch.setattr(R, "_cluster_phase", lambda rec: "ready")

    def launch(req, **k):
        state["n"] += 1
        state["launches"].append((req, k))
        state["ended"] = None
        return {"job": f"{state['n']:016x}"}
    monkeypatch.setattr(C, "launch", launch)
    req = {"action": "load", "identity": "abc", "nodes": ["aaaa", "bbbb"],
           "split": "pipeline", "link": "jaccl", "order": ["A", "B"],
           "port": 8080, "sets": {"kv_bits": "8"}, "tune": "fast"}
    order = [{"name": "A", "page": None}, {"name": "B", "page": "b:1"}]
    R.track_cluster("a" * 16, req=req, args={"me": {}, "peers": []},
                    order=order, port=8080, leader_here=True)
    return state


def test_backoff_then_three_relaunches_then_failed(faked):
    t = 1000.0
    for n, wait_s in enumerate(R.BACKOFF_S, 1):
        faked["ended"] = f"rank 1 on B (pid {n}) exited"
        what = R.tick(t)[0][1]
        assert f"relaunch {n} in {wait_s:.0f} s" in what
        v = next(iter(R.MODELS.values()))
        assert v["next_at"] == t + wait_s and v["state"] == "recovering"
        assert R.tick(t + wait_s - 1) == []          # not before its time
        t += wait_s
        assert R.tick(t)[0][1].startswith(f"relaunch {n}:")
        t += 1
    # every relaunch is the same launch
    assert all(req["order"] == ["A", "B"] and req["split"] == "pipeline"
               and req["sets"] == {"kv_bits": "8"} and req["port"] == 8080
               for req, _ in faked["launches"])
    assert [k["recovering"]["attempts"] for _, k in faked["launches"]] \
        == [1, 2, 3]
    faked["ended"] = "rank 0 on A (pid 9) exited"
    what = R.tick(t)[0][1]
    assert what.startswith("failed") and "3 relaunches in 15 min" in what
    v = R.view(next(iter(R.MODELS.values())), t)
    assert v["state"] == "failed" and v["attempts"] == 3
    assert "rank 0 on A" in v["last_reason"]
    assert R.tick(t + 1000) == [] and len(faked["launches"]) == 3
    # until someone loads it again: a launch by somebody starts afresh
    R.track_cluster("e" * 16, req={"identity": "abc",
                                   "nodes": ["aaaa", "bbbb"]},
                    args={}, order=[], port=8080, leader_here=True)
    assert R.for_job("e" * 16) is None


def test_attempts_outside_the_window_do_not_count(faked):
    t = 0.0
    for n in range(3):
        faked["ended"] = "rank 1 exited"
        R.tick(t)
        t += 100
        R.tick(t)
        t += R.WINDOW_S + 1                 # each well apart
    faked["ended"] = "rank 1 exited"
    assert "relaunch 1 in" in R.tick(t)[0][1]


def test_a_machine_that_went_away_waits_until_it_answers(faked):
    faked["ended"] = "B's page has not answered (URLError) for 20 s; " \
                     "the job cannot run"
    R.tick(0)
    faked["down"] = "B is not answering (URLError)"
    assert R.tick(100) == [] and faked["launches"] == []
    faked["down"] = ""
    assert R.tick(102)[0][1].startswith("relaunch 1:")


def test_a_machine_gone_past_the_window_is_failed(faked):
    faked["ended"] = "B's page has not answered (URLError) for 20 s"
    R.tick(0)
    faked["down"] = "B is not answering (URLError)"
    what = R.tick(R.WINDOW_S + 20)[0][1]
    assert what.startswith("failed") and "not answering" in what
    assert faked["launches"] == []


def test_a_relaunch_refused_for_memory_is_failed(faked, monkeypatch):
    faked["ended"] = "rank 1 exited"
    R.tick(0)
    monkeypatch.setattr(C, "launch", lambda req, **k: {
        "refused": "nothing started: B refuses rank 1: rank 1's share is "
                   "40.0 GiB and B's working set ... 30.0 GiB of it held "
                   "now by the server on port 8081"})
    what = R.tick(20)[0][1]
    assert what.startswith("failed") and "held now by" in what


def test_the_switch_turns_it_off(faked, monkeypatch):
    monkeypatch.setenv("KNURLOGIC_RECOVER", "off")
    faked["ended"] = "rank 1 exited"
    assert R.tick(0) == [] and R.tick(100) == []
    assert faked["launches"] == []
    monkeypatch.setenv("KNURLOGIC_RECOVER", "on")
    assert R.tick(200)


def test_reasons_are_sorted_into_kinds():
    assert R.kind("unloaded") == "requested"
    assert R.kind("the page that started it closed") == "requested"
    assert R.kind("B stopped the job: unloaded") == "requested"
    assert R.kind("rank 1 on B (pid 3) exited") == "failure"
    assert R.kind("rank 0 on A has had work in flight and not taken a "
                  "step in 130 s (step 4); the ring is stalled") == "failure"
    assert R.kind("will not fit") == "memory"
    assert R.kind("x exited: out of memory: [metal::malloc] Unable to "
                  "allocate") == "memory"
    assert R.kind("B's page has not answered (URLError) for 20 s") \
        == "machine"
    assert R.kind("B no longer runs its rank of the job for 21 s") \
        == "machine"


# --- one Mac --------------------------------------------------------------

def test_a_one_mac_server_that_dies_is_relaunched_the_same_way(
        monkeypatch, tmp_path):
    from knurlogic.interfaces import mcp
    from knurlogic.machine import servers
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(R, "BACKOFF_S", (0.0, 0.0, 0.0))
    monkeypatch.setattr(R, "_serve_pids", lambda port: [])
    loads = []

    def load(**kw):
        loads.append(kw)
        reg = servers.registry()
        reg[8093] = {"pid": 424242, "artifact": "/m/qwen", "log": "",
                     "t": time.time()}
        servers.save_registry(reg)
        return {"starting": "/m/qwen", "port": 8093, "pid": 424242}
    monkeypatch.setattr(mcp, "load", load)
    out = ui.tracked_load(artifact="/m/qwen", port=8093, tune="fast",
                          sets={"kv_bits": "8"})
    assert out["pid"] == 424242 and R.MODELS
    monkeypatch.setattr(servers, "is_our_server", lambda pid: False)
    what = R.tick(0)[0][1]
    assert "port 8093 (pid 424242) exited" in what
    assert R.tick(1)[0][1].startswith("relaunch 1")
    assert loads[-1] == {"artifact": "/m/qwen", "port": 8093,
                         "tune": "fast", "sets": {"kv_bits": "8"},
                         "force": False, "draft": True}
    assert R.for_port(8093)["state"] == "recovering"
    assert R.read_file()["8093"]["state"] == "recovering"
    # its Unload: gone from recovery
    ui._stop(8093)
    assert not R.MODELS and "8093" not in R.read_file()
