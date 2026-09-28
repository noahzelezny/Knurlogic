"""The MCP across machines: `load` with machines, `unload` by model or job,
`state` listing a cluster job once. Everything past this Mac goes through
the page on this Mac (POST /loaded.json, the Launch and Unload buttons'
request), so the unit tests fake that page's answers and the end-to-end one
runs the real page handler over HTTP with a fake peer page and fake ranks
(tests/cluster_fake_page.py, tests/cluster_fake_rank.py)."""
import json
import os
import subprocess
import sys
import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from knurlogic.interfaces import cluster_jobs as C
from knurlogic.interfaces import mcp, ui, web
from knurlogic.machine import identity

import test_cluster_jobs as T


STATUS = {"me": {"id": "aaaa", "name": "A"},
          "peers": [{"id": "bbbb", "name": "B", "state": "answering"},
                    {"id": "cccc", "name": "C", "state": "unreachable"}]}
REQS = {"in_flight": 1, "pending": 2, "capacity": 4,
        "oldest_pending_s": 0.5, "holding": "batch_full"}
JOB = {"job": "j1", "split": "pipeline", "link": "jaccl",
       "machines": ["B", "A"], "leader": "B", "world": 2,
       "artifact": "M", "phase": "ready", "cable": "198.51.100",
       "cable_note": "cable 198.51.100: Thunderbolt 5"}


def residency(local_jobs=True, rank0_row=True):
    """A's /loaded.json?peers=1: rank 1 here, rank 0 (and its row) on B."""
    row = {"name": "M", "runtime": "knurlogic", "where":
           "http://198.51.100.2:8080", "state": "loaded", "requests": REQS,
           "cluster": {k: JOB[k] for k in ("job", "split", "link",
                                          "machines", "leader", "phase")}}
    return {"resident": [{"name": "small", "runtime": "knurlogic",
                          "where": "http://127.0.0.1:8081",
                          "requests": None}],
            "jobs": [dict(JOB, ranks_here=[1], port=None)]
            if local_jobs else [],
            "peers": [{"machine": "B", "resident": [row] if rank0_row
                       else [],
                       "jobs": [dict(JOB, ranks_here=[0], port=8080)]}]}


@pytest.fixture
def page(monkeypatch):
    """A fake page on this Mac: GETs answered from `docs`, POSTs recorded
    and answered with `answer`."""
    p = SimpleNamespace(posts=[], answer={}, docs={
        "/status.json": STATUS, "/loaded.json?peers=1": residency()})

    def call(path, doc=None, timeout=0):
        if doc is None:
            return p.docs[path]
        p.posts.append(doc)
        return p.answer
    monkeypatch.setattr(mcp, "_page_call", call)
    monkeypatch.setitem(identity._ID, "id", "aaaa")
    monkeypatch.setitem(identity._ID, "name", "A")
    monkeypatch.setattr(mcp, "_identity_of", lambda a: ("abc", None))
    return p


def test_a_cluster_job_is_one_model_with_its_requests_and_link():
    ms = mcp.models_across(residency(), "A")
    job = [m for m in ms if m["job"] == "j1"]
    assert len(job) == 1, ms
    j = job[0]
    assert j["machine"] == "B" and j["port"] == 8080
    assert j["machines"] == ["B", "A"] and j["split"] == "pipeline"
    assert j["link"] == "rdma" and j["requests"] == REQS
    assert j["cable"] == "198.51.100"
    small = next(m for m in ms if m["name"] == "small")
    assert small["machine"] == "A" and small["job"] is None
    assert small["machines"] == ["A"]


def test_a_job_whose_rank_0_has_no_row_yet_is_listed_once_from_the_job():
    ms = mcp.models_across(residency(rank0_row=False), "A")
    job = [m for m in ms if m["job"] == "j1"]
    assert len(job) == 1 and job[0]["port"] == 8080
    assert job[0]["phase"] == "ready" and job[0]["requests"] is None


def test_a_stopped_job_is_not_resident():
    doc = residency(rank0_row=False)
    doc["jobs"][0]["phase"] = "stopped"
    doc["peers"][0]["jobs"][0]["phase"] = "stopped"
    assert not [m for m in mcp.models_across(doc, "A") if m["job"]]


def test_load_on_two_machines_sends_the_pages_launch(page):
    page.answer = {"job": "j2", "starting": "abc", "port": 8090,
                   "leader": "B", "machines": ["B", "A"],
                   "placement": {"order": ["B", "A"], "leader": "B",
                                 "layers": [4, 4]},
                   "cable": "198.51.100", "cable_note": "fastest", "note": "x"}
    out = mcp.load(artifact="M", port=8090, machines=["A", "b"],
                   split="pipeline", link="rdma", tune="fast")
    assert page.posts == [{"action": "load", "identity": "abc",
                           "tune": "fast", "sets": {}, "port": 8090,
                           "nodes": ["aaaa", "bbbb"], "split": "pipeline",
                           "link": "rdma"}]
    assert out["job"] == "j2" and out["port"] == 8090
    assert out["placement"] == {"order": ["B", "A"], "leader": "B",
                                "layers": [4, 4], "cable": "198.51.100",
                                "cable_note": "fastest"}


def test_load_on_one_peer_is_the_pages_single_peer_load(page):
    page.answer = {"starting": "/x/M", "port": 8080, "machine": "B"}
    out = mcp.load(artifact="M", machines=["B"])
    assert page.posts[0]["node"] == "bbbb" and "nodes" not in page.posts[0]
    assert out["starting"] == "/x/M"


@pytest.mark.parametrize("answer", [
    {"refused": "nothing started: B refuses rank 1: rank 1's share is "
                "40.0 GiB and 12.0 GiB fits", "placement": {"order": []}},
    {"refused": "nothing started: B refuses rank 1: another job's rank "
                "is still loading here"},
    {"refused": "A and B share no Thunderbolt 5 cable. RDMA needs "
                "Thunderbolt 5."},
    {"error": "'bbbb' is not a machine answering this page"},
])
def test_the_pages_refusal_is_a_refusal(page, answer):
    page.answer = answer
    out = mcp.load(artifact="M", machines=["A", "B"], split="tensor",
                   link="tcp")
    assert out["loaded"] is False
    assert out["refused"] == answer.get("refused") or answer.get("error")
    assert "error" not in out


def test_load_refuses_what_it_cannot_send(page):
    for kw, why in (({"machines": ["A", "Z"], "split": "tensor",
                      "link": "tcp"}, "not a machine"),
                    ({"machines": ["A", "C"], "split": "tensor",
                      "link": "tcp"}, "not answering"),
                    ({"machines": ["A", "B"], "split": "rows",
                      "link": "tcp"}, "tensor | pipeline"),
                    ({"machines": ["A", "B"], "split": "tensor",
                      "link": "jaccl"}, "tcp | rdma")):
        out = mcp.load(artifact="M", **kw)
        assert out["loaded"] is False and why in out["refused"], out
    assert page.posts == []


def test_no_page_is_an_error_that_says_to_start_it(monkeypatch):
    monkeypatch.setattr(mcp, "_identity_of", lambda a: ("abc", None))
    out = mcp.load(artifact="M", machines=["A", "B"], split="tensor",
                   link="tcp")
    assert "knurlogic ui" in out["error"]


def test_unload_a_job_with_a_rank_here_stops_it_from_this_page(page):
    page.answer = {"stopped": "j1", "told": ["B"]}
    out = mcp.unload(job="j1")
    assert page.posts == [{"action": "unload", "job": "j1"}]
    assert out["machines"] == ["B", "A"]


def test_unload_a_peers_job_by_name_goes_to_its_leader(page):
    page.docs["/loaded.json?peers=1"] = residency(local_jobs=False)
    page.answer = {"stopped": "M", "port": 8080}
    mcp.unload(model="M")
    assert page.posts == [{"action": "unload", "node": "bbbb",
                           "port": 8080}]


def test_unload_by_port_is_this_macs(page):
    page.answer = {"stopped": "small"}
    mcp.unload(port=8081)
    assert page.posts == [{"action": "unload", "target": "8081"}]
    assert "error" in mcp.unload(port=8080)       # B's port, not named
    assert "error" in mcp.unload(model="nothing")


# --- end to end: the real page handler, a fake peer page, fake ranks -------

@pytest.fixture
def page_a(tmp_path, monkeypatch):
    """Page A served over HTTP with its real /loaded.json routes, page B a
    process; nothing launched yet."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "A"))
    monkeypatch.setenv("PYTHONPATH", T.SRC + os.pathsep
                       + os.environ.get("PYTHONPATH", ""))
    monkeypatch.setitem(identity._ID, "id", "aaaa")
    monkeypatch.setitem(identity._ID, "name", "A")
    monkeypatch.setattr(C, "_resolve",
                        lambda i: "/fake/artifact" if i == "abc" else None)
    monkeypatch.setattr(C, "shape_of", lambda p, w, s: T.SHAPE)
    info_a = T.info("Apple M4 Max", "127.0.0.1")
    info_b = T.info("Apple M3 Ultra", "127.0.0.1")
    monkeypatch.setattr(C, "_local_info", lambda: info_a)
    import cluster_fake_page
    monkeypatch.setattr(C, "RANK_ARGV", [cluster_fake_page.fake_argv])
    ui_a, ui_b = T.free_port(), T.free_port()
    env = {**os.environ, "XDG_CACHE_HOME": str(tmp_path / "B")}
    page_b = subprocess.Popen(
        [sys.executable, str(T.HERE / "cluster_fake_page.py"), str(ui_b),
         "bbbb", "B", json.dumps(info_b), "aaaa", f"127.0.0.1:{ui_a}"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True)
    assert "up" in page_b.stdout.readline()
    peer = SimpleNamespace(id="bbbb", name="B", host="127.0.0.1",
                           key=f"127.0.0.1:{ui_b}", state="answering",
                           link="thunderbolt", node={"cluster": info_b},
                           found_by=set())
    monkeypatch.setattr(ui, "PEERS", SimpleNamespace(
        all=lambda: [peer], introduce=lambda *a, **k: None))
    monkeypatch.setattr(ui, "_status_fn", lambda _n=0: ({
        "nodes": [{"role": "local", "cluster": info_a}],
        "me": {"id": "aaaa", "name": "A"},
        "peers": [{"id": "bbbb", "name": "B", "state": "answering"}]}, ""))
    monkeypatch.setitem(ui._SERVE_PORT, "ui", ui_a)
    from knurlogic.machine import servers

    def local_residency():
        def doc(_q):
            return {"resident": [
                {"name": "fake", "runtime": "knurlogic", "state": "loaded",
                 "where": f"http://127.0.0.1:{port}", "requests": REQS}
                for port in servers.registry()]}
        return doc
    monkeypatch.setattr(web, "loaded_document", local_residency)
    monkeypatch.setattr("knurlogic.machine.loaded.survey", lambda: {})
    serve_port = T.free_port()
    routes = web.routes(status_fn=ui._status_fn, loaded_fn=ui._loaded_fn(),
                        load_fn=ui._load_fn(serve_port))
    srv = ThreadingHTTPServer(("127.0.0.1", ui_a), ui.make_handler(routes))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("KNURLOGIC_PAGE", f"127.0.0.1:{ui_a}")
    ctx = SimpleNamespace(port=serve_port, page_b=page_b, pids=[])
    yield ctx
    for job in list(C.SPECS):
        C.stop(job, propagate=False, grace=1)
    # every rank either page started, whatever the test reached
    for d in ("A", "B"):
        f = tmp_path / d / "knurlogic" / "jobs" / "jobs.json"
        if f.exists():
            ctx.pids += [int(r["pid"]) for r in
                         json.loads(f.read_text()).values()]
    for pid in ctx.pids:
        try:
            os.kill(pid, 9)
        except OSError:
            pass
    page_b.kill()
    page_b.wait(5)
    srv.shutdown()
    C.ENDED.clear()


def test_mcp_load_state_unload_across_two_pages(page_a, monkeypatch):
    from knurlogic.cluster import jobs as J
    monkeypatch.setattr(mcp, "_identity_of", lambda a: ("abc", None))
    out = mcp.load(artifact="M", port=page_a.port, machines=["A", "B"],
                   split="tensor", link="tcp")
    assert out.get("job"), out
    job = out["job"]
    assert out["leader"] == "A" and out["port"] == page_a.port
    assert out["placement"]["order"] == ["A", "B"]
    assert "cable" in out["placement"]
    b_jobs = page_a.pids
    rank0 = J.registry()[f"{job}/0"]
    b_dir = T.Path(os.environ["XDG_CACHE_HOME"]).parent / "B" / \
        "knurlogic" / "jobs" / "jobs.json"
    rank1 = T.wait(lambda: json.loads(b_dir.read_text()).get(f"{job}/1")
                   if b_dir.exists() else None)
    b_jobs += [rank0["pid"], rank1["pid"]]
    T.wait(lambda: (J.read_marker(job, 0) or {}).get("phase") == "ready")

    st = mcp.state()
    mine = [m for m in st["models"] if m.get("job") == job]
    assert len(mine) == 1, st["models"]
    assert mine[0]["machines"] == ["A", "B"] and mine[0]["link"] == "tcp"
    assert mine[0]["requests"] == REQS
    assert any(r.get("machine") == "A" for r in st["requests"])

    gone = mcp.unload(job=job)
    assert gone.get("told") == ["B"], gone
    assert not T.alive(rank0["pid"])
    assert T.wait(lambda: not T.alive(rank1["pid"]), 20)
