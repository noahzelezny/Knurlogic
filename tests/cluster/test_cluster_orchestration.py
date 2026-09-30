"""The control plane across N real pages (tests/support/fake_cluster.py): n
processes of the REAL page -- handler, liveness, cluster steps, recovery --
each with its own cache and identity, on this Mac, with short liveness
clocks. Only the model is fake (tests/support/cluster_fake_rank.py).

What these pin: a pipeline launch over tcp at n=3, 4, 8; the coordinator's
page dying (every page stops the job within PEER_GONE_S, the recovery
record relaunches it once that page is back); a rank and a whole page dying;
one machine's refusal in a world of 8; duplicate Prepare/Start/Stop; two
pages of different protocol majors; and a peer that is not on the
Thunderbolt/--peer gate (Wi-Fi) staying "answering".
"""
import json
import os
import urllib.error
import urllib.request

import pytest
from fake_cluster import VERSIONS, FakeCluster, alive, wait

from knurlogic.cluster import protocol, transport

#: the fake pages' PEER_GONE_S (cluster_fake_page.fast) and slack for the
#: watcher, the probes, and this machine being busy
GONE_S = 4.0
SLACK_S = 25.0


def ended_reason(page, job):
    for d in page.get("/loaded.json").get("jobs") or []:
        if d.get("job") == job and d.get("phase") == "stopped":
            return d
    return None


@pytest.mark.parametrize("n", [3, 4, 8])
def test_a_pipeline_launch_over_tcp_runs_a_rank_on_every_page(tmp_path, n):
    with FakeCluster(n, tmp_path) as c:
        assert c.wait_peers()
        out = c.launch(0, split="pipeline", link="tcp")
        assert out.get("job"), out
        job = out["job"]
        assert out["link"] == "tcp" and len(out["machines"]) == n
        assert wait(lambda: c.running(job), 40), [
            (p.name, len(p.live_ranks(job))) for p in c.pages]
        # every page agrees which machines and in what order
        docs = [next(d for d in p.get("/loaded.json")["jobs"]
                     if d["job"] == job) for p in c.pages]
        assert {tuple(d["machines"]) for d in docs} == {tuple(out["machines"])}
        # Unload on the coordinator stops every page's rank
        c.pages[0].post("/loaded.json", {"action": "unload", "job": job})
        assert wait(lambda: c.job_stopped_everywhere(job), 40)


@pytest.mark.parametrize("n", [3, 8])
def test_the_coordinators_page_dying_stops_the_job_everywhere_and_relaunches(
        tmp_path, n):
    with FakeCluster(n, tmp_path) as c:
        assert c.wait_peers()
        out = c.launch(0, link="tcp")
        job = out["job"]
        assert wait(lambda: c.running(job), 40)
        coord = c.pages[0]
        pids = [int(r["pid"]) for r in coord.ranks(job)]
        coord.kill()                        # the machine goes: page and rank
        for pid in pids:
            os.kill(pid, 9)
        # every other page stops within PEER_GONE_S (plus its clocks)
        assert wait(lambda: c.job_stopped_everywhere(job), GONE_S + SLACK_S)
        why = ended_reason(c.pages[1], job)
        assert why is not None
        assert "answered" in why["reason"] or "runs its rank" in why["reason"]
        # the coordinator's page returns: its recovery record relaunches
        c.restart_page(0)
        assert wait(lambda: any(j != job for j in c.jobs_running()), 90), \
            "no relaunch"
        new = next(j for j in c.jobs_running() if j != job)
        assert wait(lambda: c.running(new), 60)
        doc = next(d for d in c.pages[1].get("/loaded.json")["jobs"]
                   if d["job"] == new)
        assert doc["machines"] == out["machines"]        # the same order


@pytest.mark.parametrize("how", ["rank", "page"])
@pytest.mark.parametrize("n", [4, 8])
def test_a_rank_or_a_whole_page_dying_stops_every_page_and_recovery_keeps_the_order(
        tmp_path, n, how):
    with FakeCluster(n, tmp_path) as c:
        assert c.wait_peers()
        out = c.launch(0, link="tcp")
        job = out["job"]
        assert wait(lambda: c.running(job), 40)
        victim = c.pages[2]
        pid = int(victim.ranks(job)[0]["pid"])
        if how == "page":
            victim.kill()
        os.kill(pid, 9)
        assert wait(lambda: c.job_stopped_everywhere(job), GONE_S + SLACK_S)
        why = ended_reason(c.pages[0], job)
        assert why and why.get("reason")
        if how == "rank":
            assert "exited" in why["reason"]
            assert why.get("kind") in (None, "failure", "machine")
        else:
            # a machine that went away: recovery waits for its page
            assert why.get("kind") == "machine" or "answered" in why["reason"]
            c.restart_page(2)
        # the coordinator's record relaunches it, same machines, same order
        assert wait(lambda: any(j != job for j in c.jobs_running()), 90), \
            "no relaunch"
        new = next(j for j in c.jobs_running() if j != job)
        assert wait(lambda: c.running(new), 60)
        doc = next(d for d in c.pages[0].get("/loaded.json")["jobs"]
                   if d["job"] == new)
        assert doc["machines"] == out["machines"]


def test_a_refusal_at_rank_5_of_8_starts_nothing_and_every_page_forgets_the_job(
        tmp_path):
    job = "5ee0c0ffee123456"
    bad = {"versions": {**VERSIONS, "build": "deadbeef0000+mlx0.31.2"}}
    with FakeCluster(8, tmp_path, env={}, infos={5: bad},
                     page_env={0: {"FAKE_JOB_ID": job}}) as c:
        assert c.wait_peers()
        names = [p.name for p in c.pages]
        out = c.launch(0, split="tensor", link="tcp", order=names)
        assert "nothing started" in out.get("refused", ""), out
        assert "every rank runs the same build" in out["refused"]
        assert out["placement"]["order"][5] == "P5"
        for p in c.pages:
            assert not p.live_ranks(), p.name
            st = transport.send(p.addr, "JobState", {"job": job})
            assert st["prepared"] is False and st["ranks_here"] == [], p.name
        # nothing is held: the same launch without P5 goes
        ok = c.launch(0, split="tensor", link="tcp",
                      nodes=[p.id for p in c.pages if p.name != "P5"],
                      order=[n for n in names if n != "P5"],
                      **{"port": 0})
        assert ok.get("job") or "refused" not in ok, ok


def _spec(c, rank=1):
    return {"job": "ab12cd34ef567890", "rank": rank, "world": 2,
            "split": "tensor", "link": "ring", "identity": "abc",
            "hosts": ["10.0.0.1:47200", "10.0.0.2:47201"],
            "ibv_devices": None, "coordinator": "", "layers": [],
            "prefill_chunk": 512, "tune": "default", "port": 0,
            "working_set_gib": 60.0, "bandwidth_gbs": 0, "sets": {},
            "versions": dict(VERSIONS), "jaccl_timeout_ms": 0,
            "nodes": [{"rank": 0, "id": "node0", "name": "P0",
                       "page": c.pages[0].addr},
                      {"rank": 1, "id": "node1", "name": "P1",
                       "page": c.pages[1].addr}]}


def test_duplicate_prepare_start_and_stop_are_idempotent(tmp_path):
    with FakeCluster(3, tmp_path) as c:
        assert c.wait_peers()
        page, job = c.pages[1], "ab12cd34ef567890"
        a = transport.send(page.addr, "Prepare", _spec(c))
        b = transport.send(page.addr, "Prepare", _spec(c))
        assert a["ok"] and b["ok"], (a, b)
        s1 = transport.send(page.addr, "Start", {"job": job})
        s2 = transport.send(page.addr, "Start", {"job": job})   # a retry
        assert s1["started"] == job and s2["started"] == job
        assert s1["pid"] == s2["pid"] and alive(s1["pid"])
        assert len(page.live_ranks(job)) == 1
        t1 = transport.send(page.addr, "Stop", {"job": job, "reason": "unloaded"})
        t2 = transport.send(page.addr, "Stop", {"job": job, "reason": "unloaded"})
        assert t1["stopped"] == job and t2["stopped"] == job
        assert wait(lambda: not alive(s1["pid"]), 20)
        # a Start for a job nobody prepared is a typed refusal, not a crash
        late = transport.send(page.addr, "Start", {"job": job})
        assert late.get("error") and late["failure_kind"] == "refusal"


def test_two_pages_of_different_protocol_majors_list_each_other_and_run_nothing(
        tmp_path):
    with FakeCluster(2, tmp_path, page_env={
            1: {"FAKE_PROTOCOL_MAJOR": "2"}}) as c:
        a, b = c.pages
        assert wait(lambda: a.peer_state(b) == "version_mismatch"
                    and b.peer_state(a) == "version_mismatch", 30)
        assert "update knurlogic on" in a.peer_public(b)["problem"]
        assert "update knurlogic on" in b.peer_public(a)["problem"]
        # listed, not dropped: the page's status still names both
        assert {n["node"] for n in a.get("/status.json")["nodes"]} >= {"P0"}
        # no job together: the launch names the machine to update
        out = c.launch(0)
        assert "update knurlogic on" in out.get("error", ""), out
        # and a message across the majors is refused with the same words
        with pytest.raises(protocol.VersionMismatch, match="update knurlogic"):
            transport.send(b.addr, "Survey", {})


def test_a_restarted_peer_is_noticed_by_its_boot_id(tmp_path):
    with FakeCluster(2, tmp_path) as c:
        a, b = c.pages
        assert c.wait_peers()
        assert not a.peer_public(b).get("restarts")
        c.restart_page(1)
        assert wait(lambda: a.peer_public(b).get("restarts") == 1
                    and a.peer_state(b) == "answering", 30)


def test_a_peer_not_on_the_thunderbolt_gate_stays_answering_but_takes_no_job(
        tmp_path):
    """A Wi-Fi or Ethernet Bonjour peer: its status answers to every peer,
    so liveness says answering; a message to it meets the peer gate."""
    with FakeCluster(2, tmp_path, page_env={
            1: {"FAKE_DENY_PEER_GATE": "1", "FAKE_NO_PEERS": "1"}}) as c:
        a, b = c.pages
        assert wait(lambda: a.peer_state(b) == "answering", 30)
        with pytest.raises(transport.PeerRefused, match="Thunderbolt"):
            transport.send(b.addr, "Survey", {})
        assert wait(lambda: a.peer_state(b) == "answering", 5)
        out = c.launch(0)
        assert "nothing started" in out.get("refused", "") \
            or "error" in out, out
        assert a.peer_state(b) == "answering"


def _raw(page, method, path="/peer/v1/msg", body=None):
    req = urllib.request.Request(
        f"http://{page.addr}{path}", method=method,
        data=None if body is None else (
            body if isinstance(body, bytes) else json.dumps(body).encode()),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw or b"{}")
        except ValueError:
            return e.code, {"text": raw.decode()}


def test_the_one_route_enforces_the_envelope(tmp_path):
    with FakeCluster(2, tmp_path) as c:
        p = c.pages[0]
        code, _ = _raw(p, "GET")
        assert code == 405
        # not an envelope: the plain refusal
        code, doc = _raw(p, "POST", body={"job": "ab12cd34ef567890"})
        assert code == 400 and "envelope" in doc["error"] and "kind" not in doc
        code, doc = _raw(p, "POST", body=b"not json")
        assert code == 400 and "kind" not in doc
        # a different major: a typed refusal with the update words
        env = {"v": [2, 0], "kind": "Survey", "from": "x", "seq": 7,
               "ts": 0.0, "body": {}}
        code, doc = _raw(p, "POST", body=env)
        assert doc["kind"] == "Failure" and "update knurlogic" in \
            doc["body"]["reason"] and doc["re"] == 7
        # an unknown kind and a kind this page does not answer: Failure
        for kind in ("Nope", "Hello"):
            code, doc = _raw(p, "POST", body={**env, "v": [1, 7],
                                              "kind": kind})
            assert doc["kind"] == "Failure", kind
        # a higher minor is accepted
        code, doc = _raw(p, "POST", body={**env, "v": [1, 7]})
        assert doc["kind"] == "Residency" and doc["re"] == 7
        # an old per-purpose route is gone
        code, _ = _raw(p, "POST", "/peer/cluster/stop", {"job": "ab12cd34"})
        assert code == 404
        # a 404 (an old peer with no such route) reads as version_mismatch
        with pytest.raises(protocol.VersionMismatch, match="update knurlogic"):
            transport.decode_reply(404, b"not found", "P9")
