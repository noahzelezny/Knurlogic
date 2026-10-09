"""Auto-recovery (cluster/recovery.py): a model that dies unasked is
relaunched by the page that launched it, bounded; a requested stop or an
out-of-memory one never is.

The cluster cases run the real two pages and fake ranks of
tests/test_cluster_jobs.py (its `two_pages` fixture: this process is page A,
the coordinator; page B is a process). The limits are tested on the loop
itself with the launch faked."""

import os
import time

import pytest
from test_cluster_jobs import alive, two_pages, wait  # noqa: F401

from knurlogic.cluster import jobs as J
from knurlogic.cluster import launch as C
from knurlogic.cluster import recovery as R
from knurlogic.interfaces import spawn
from knurlogic.interfaces.page import loads as page_loads


def ticks_until(pred, t=40.0):
    """tick() until one of its reports satisfies `pred`; -> that report."""
    end = time.time() + t
    while time.time() < end:
        for _key, what in R.tick():
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
        doc = page_loads.with_jobs({"resident": [
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
    spawn.stop(p.port)                      # the Unload button
    assert wait(lambda: not alive(p.rank1["pid"]), 20)
    assert not R.MODELS
    assert R.tick() == []
    assert R.for_job(p.job) is None and R.not_serving() == []


def test_an_unload_from_the_other_page_is_not_recovered(two_pages,
                                                        monkeypatch):
    p = two_pages
    monkeypatch.setattr(R, "BACKOFF_S", (0.0, 0.0, 0.0))
    # B's Unload: B stops its rank and tells A, reason "unloaded"
    from knurlogic.cluster import transport
    transport.send(f"127.0.0.1:{p.ui_b}", "Stop",
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
    down = page_loads.with_jobs({"resident": []})["recovery"]
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
           "port": 8080, "sets": {"kv_bits": "8"}, "tune": "lean"}
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
    for _n in range(3):
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


def test_a_relaunch_refused_while_the_old_rank_holds_the_port_is_not_an_attempt(
        faked, monkeypatch):
    """Live (2026-10-04): every first relaunch, 10 s after a failure, was
    refused "port 8080 on A is in use" -- the old rank 0 was still exiting
    -- and spent one of the three attempts. It waits instead."""
    faked["ended"] = "rank 1 exited"
    R.tick(0)
    tries = []

    def launch(req, **k):
        tries.append(1)
        if len(tries) < 3:
            return {"refused": "nothing started: A refuses rank 0: port "
                               "8080 on A is in use"}
        return {"job": "c0ffee0000000001"}
    monkeypatch.setattr(C, "launch", launch)
    t = R.BACKOFF_S[0] + 1
    for _ in range(3):
        R.tick(t)
        t += R.TICK_S + 1
    rec = next(iter(R.MODELS.values()))
    assert len(tries) == 3 and rec["job"] == "c0ffee0000000001"
    assert len(rec["attempts"]) == 1      # only the one that started


def test_a_port_held_past_the_wait_is_a_refused_attempt(faked, monkeypatch):
    """Another program on the port (not an exiting rank): after
    LEAVING_WAIT_S it counts as an attempt rather than waiting 15 min."""
    faked["ended"] = "rank 1 exited"
    R.tick(0)
    monkeypatch.setattr(C, "launch", lambda req, **k: {
        "refused": "nothing started: A refuses rank 0: port 8080 on A is "
                   "in use"})
    rec = next(iter(R.MODELS.values()))
    R.tick(R.BACKOFF_S[0] + 1)
    assert rec["attempts"] == []          # deferred, not spent
    what = R.tick(R.LEAVING_WAIT_S + R.BACKOFF_S[0] + 5)
    assert rec["attempts"] and "relaunch 2 in" in what[0][1]


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
    out = page_loads.tracked_load(artifact="/m/qwen", port=8093, tune="lean",
                          sets={"kv_bits": "8"})
    assert out["pid"] == 424242 and R.MODELS
    monkeypatch.setattr(servers, "is_our_server", lambda pid: False)
    what = R.tick(0)[0][1]
    assert "port 8093 (pid 424242) exited" in what
    assert R.tick(1)[0][1].startswith("relaunch 1")
    assert loads[-1] == {"artifact": "/m/qwen", "port": 8093,
                         "tune": "lean", "sets": {"kv_bits": "8"},
                         "force": False, "draft": True}
    assert R.for_port(8093)["state"] == "recovering"
    assert R.read_file()["8093"]["state"] == "recovering"
    # its Unload: gone from recovery
    spawn.stop(8093)
    assert not R.MODELS and "8093" not in R.read_file()


# --- a page restart -------------------------------------------------------

def _restart_page():
    """What a page restart leaves: the files, not this process's memory."""
    R.MODELS.clear()
    R._SAVED.clear()
    C.ENDED.clear()
    C.SPECS.clear()


def test_a_page_restarted_mid_recovery_still_relaunches(two_pages,
                                                        monkeypatch):
    """Rank 1 dies, the page schedules the relaunch, then the page
    restarts: the new page process reads what to relaunch back and
    relaunches the same launch."""
    p = two_pages
    monkeypatch.setattr(R, "BACKOFF_S", (0.5, 0.5, 0.5))
    os.kill(p.rank1["pid"], 9)
    assert wait(lambda: p.job in C.ENDED, 30)
    what = ticks_until(lambda w: w.startswith("down"))
    assert what and "relaunch 1 in" in what
    before = next(iter(R.MODELS.values()))
    _restart_page()
    assert R.for_job(p.job) is None             # nothing in memory
    C.start_watching_existing()                 # the page's start
    assert list(R.MODELS) == [before["key"]]
    assert R.for_job(p.job)["state"] == "recovering"
    new = None
    try:
        what = ticks_until(lambda w: w.startswith("relaunch 1"))
        assert what, R.MODELS
        new = next(iter(R.MODELS.values()))["job"]
        assert new != p.job
        recs = J.by_job()[new]
        assert recs[0]["machines"] == ["A", "B"]
        assert recs[0]["split"] == "tensor" and recs[0].get("port") == p.port
        assert ticks_until(lambda w: w.startswith("recovered"))
        assert R.for_job(new)["attempts"] == 1
    finally:
        if new:
            C.stop(new, grace=1)


def test_a_restored_model_keeps_its_attempts_and_limits(faked):
    t = 1000.0
    faked["ended"] = "rank 1 on B (pid 1) exited"
    R.tick(t)
    t += R.BACKOFF_S[0]
    assert R.tick(t)[0][1].startswith("relaunch 1:")
    _restart_page()
    assert R.restore() and R.restore() == []    # once, not twice
    faked["ended"] = "rank 1 on B (pid 2) exited"
    what = R.tick(t + 1)[0][1]
    assert f"relaunch 2 in {R.BACKOFF_S[1]:.0f} s" in what
    t += 1 + R.BACKOFF_S[1]
    assert R.tick(t)[0][1].startswith("relaunch 2:")
    req, k = faked["launches"][-1]
    assert req["order"] == ["A", "B"] and req["sets"] == {"kv_bits": "8"}
    assert k["recovering"]["attempts"] == 2


def test_a_one_mac_server_is_restored_too(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    R.track_single(8093, {"artifact": "/m/qwen", "tune": "lean"}, pid=7)
    _restart_page()
    assert R.restore() == ["single:8093"]
    assert R.MODELS["single:8093"]["load"]["tune"] == "lean"
    R.cancel_port(8093)                          # an unload after it
    _restart_page()
    assert R.restore() == []


# --- a refusal at startup is deterministic: failed at once, never relaunched

REFUSAL = ("REFUSING: KNURLOGIC_PRESET='bogus': preset 'bogus': one of "
           "['default']")


def test_a_rank_that_refused_to_start_fails_the_job_with_its_words(
        two_pages, monkeypatch):
    p = two_pages
    monkeypatch.setattr(R, "BACKOFF_S", (0.0, 0.0, 0.0))
    launched = []
    monkeypatch.setattr(C, "launch", lambda *a, **k: launched.append(k)
                        or {"job": "f" * 16})
    with open(p.rank0["log"], "a") as fh:
        fh.write("artifact  x\n" + REFUSAL + "\n")
    os.kill(p.rank0["pid"], 9)
    assert wait(lambda: not alive(p.rank0["pid"]), 5)
    stopped = C.watch_once()
    assert stopped and REFUSAL in stopped[0][1]
    assert R.kind(stopped[0][1]) == "refusal"
    what = ticks_until(lambda w: w.startswith("failed"), 10)
    assert what and REFUSAL in what
    for _ in range(3):
        R.tick()
    assert launched == []
    v = R.for_job(p.job)
    assert v["state"] == "failed" and v["attempts"] == 0
    assert REFUSAL in v["last_reason"]


def test_a_refusal_is_its_own_kind_and_never_relaunched(faked):
    faked["ended"] = "rank 0 on A (pid 9) exited: " + REFUSAL
    what = R.tick(0)[0][1]
    assert what.startswith("failed") and "REFUSING" in what
    assert R.tick(100) == [] and faked["launches"] == []


def test_a_relaunch_whose_settings_are_refused_is_failed(faked, monkeypatch):
    faked["ended"] = "rank 1 exited"
    R.tick(0)
    monkeypatch.setattr(C, "launch", lambda req, **k: {
        "refused": "nothing started: its launch settings are refused: x"})
    what = R.tick(20)[0][1]
    assert what.startswith("failed") and "launch settings are refused" in what


def test_refusal_line_reads_the_reason_and_its_bullets():
    log = ("artifact  m\nREFUSING a 2-rank tensor split of m:\n"
           "  - 8 KV heads do not split 3 ways\n  - and another\n"
           "Traceback (most recent call last):\n")
    assert R.refusal_line(log) == ("REFUSING a 2-rank tensor split of m: "
                                   "- 8 KV heads do not split 3 ways "
                                   "- and another")
    assert R.refusal_line("all fine\n") == ""
    assert R.kind("refusing to guess") != "refusal"    # REFUSING is serve's


def test_a_missing_module_is_a_refusal_never_relaunched():
    log = ('Traceback (most recent call last):\n  File "model.py", line 4084\n'
           "ModuleNotFoundError: No module named 'mlx_vlm'\n")
    line = R.refusal_line(log)
    assert line == "ModuleNotFoundError: No module named 'mlx_vlm'"
    assert R.kind(f"rank 1 exited with code 1: {line}") == "refusal"


def test_a_rank_that_cannot_build_the_model_is_a_refusal():
    from knurlogic.cluster import recovery
    tail = ("RuntimeError: rank 1 could not load /m/GLM: AttributeError: "
            "module 'mlx_lm.models.glm5_next' has no attribute 'ModelArgs'")
    assert recovery.refusal_line(tail).startswith("rank 1 could not load")
    assert recovery.kind("rank 1 exited: " + tail) == "refusal"


def test_a_relaunched_job_is_one_record_found_by_the_id_its_load_answered(
        faked):
    """Each relaunch is a new job id; the record keeps the first (the id
    the load answered, the one a caller polls) and every one after it, and
    an unload by any of them clears it -- the page showed one FAILED card
    per relaunch, and mcp.unload(job=<the first id>) found nothing."""
    t = 1000.0
    for n, wait_s in enumerate(R.BACKOFF_S, 1):
        faked["ended"] = f"rank 0 on A (pid {n}) exited"
        R.tick(t)
        t += wait_s
        R.tick(t)
        t += 1
    faked["ended"] = "rank 0 on A (pid 48653) exited"
    R.tick(t)
    first = "a" * 16
    ids = [first] + [f"{n:016x}" for n in (1, 2, 3)]
    for job in ids:
        v = R.for_job(job)
        assert v["state"] == "failed" and v["jobs"] == ids
        assert "pid 48653" in v["last_reason"]
    down = R.not_serving()
    assert len(down) == 1 and down[0]["job"] == first
    assert R.cancel_job(ids[-1]) == [first]
    assert R.MODELS == {} and R.not_serving() == []
    assert R.cancel_job(first) == []


def test_an_unload_of_a_job_with_nothing_here_tells_every_peer(monkeypatch):
    """A failed job's ranks are gone and recovery's record may be on the
    other Mac: the stop goes to every peer page, so it is cleared there."""
    told = []
    monkeypatch.setattr(C, "_peer_pages", lambda: {"bbbb": "b:1"})
    out = C.stop("f" * 16, reason="unloaded",
                 post=lambda page, kind, doc: told.append((page, kind, doc)))
    assert told == [("b:1", "Stop", {"job": "f" * 16, "reason": "unloaded"})]
    assert out["told"] == ["bbbb"]


def test_an_unload_says_what_a_peer_cleared(monkeypatch):
    """The record of a failed job lives on the page that started it: the
    answer carries what the peer's stop cleared, not only this Mac's."""
    monkeypatch.setattr(C, "_peer_pages", lambda: {"bbbb": "b:1"})
    out = C.stop("f" * 16, reason="unloaded",
                 post=lambda page, kind, doc: {"stopped": doc["job"],
                                               "cleared": ["e" * 16]})
    assert out["cleared"] == ["e" * 16]


def test_an_unload_during_a_relaunch_stops_the_new_job(faked, monkeypatch):
    """The relaunch runs without the lock: an unload landing meanwhile
    drops the record, so the job the relaunch then answers is stopped,
    not left running untracked."""
    stopped = []
    monkeypatch.setattr(C, "stop", lambda job, reason="unloaded", **k:
                        stopped.append((job, reason)) or {})

    def launch(req, **k):
        R.cancel_job("a" * 16)            # the user unloads it mid-launch
        faked["launches"].append((req, k))
        return {"job": "f" * 16}
    monkeypatch.setattr(C, "launch", launch)
    faked["ended"] = "rank 1 on B (pid 1) exited"
    R.tick(1000.0)
    R.tick(1000.0 + R.BACKOFF_S[0])
    assert len(faked["launches"]) == 1
    assert stopped == [("f" * 16, "unloaded")]
    assert R.MODELS == {} and R.for_job("f" * 16) is None
    assert R.tick(1e6) == [] and len(faked["launches"]) == 1
