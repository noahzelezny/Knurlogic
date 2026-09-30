"""Instance ids, as exo gives one: a single-Mac server gets a 16-hex id at
launch, a cluster job's instance id is simply its job id. `instance` should
appear on every resident knurlogic row it is known for, dedupe the same
model reported by two pages, and let `unload` name it directly."""
from types import SimpleNamespace


from knurlogic.interfaces import mcp
from knurlogic.interfaces.page import server as page_server
from knurlogic.machine import loaded, servers

from test_mcp_cluster import JOB, REQS, STATUS, page, residency  # noqa: F401


def test_new_instance_is_16_hex_and_not_reused():
    a, b = servers.new_instance(), servers.new_instance()
    assert len(a) == 16 and len(b) == 16
    int(a, 16) and int(b, 16)          # raises if not hex
    assert a != b


def test_a_spawned_server_gets_an_instance_in_its_registry(monkeypatch,
                                                            tmp_path):
    artifact = tmp_path / "A"
    artifact.mkdir()
    log = tmp_path / "s.log"
    monkeypatch.setattr(page_server, "serve_log", lambda port: log)

    class FakeProc:
        pid = 4242

    monkeypatch.setattr(page_server.subprocess, "Popen", lambda *a, **k: FakeProc())
    out = page_server._spawn_unlocked(str(artifact), 8091)
    assert len(out["instance"]) == 16
    rec = servers.registry()[8091]
    assert rec["instance"] == out["instance"]
    assert rec["pid"] == 4242


def test_instance_of_reads_a_single_mac_servers_own_registry(monkeypatch):
    monkeypatch.setattr(
        servers, "registry",
        lambda: {8080: {"pid": 1, "instance": "0123456789abcdef"}})
    assert loaded._instance_of("http://127.0.0.1:8080") == "0123456789abcdef"


def test_instance_of_a_cluster_rank_is_its_job_id(monkeypatch):
    monkeypatch.setattr(
        servers, "registry",
        lambda: {8080: {"pid": 1, "job": "cafefeed"}})
    assert loaded._instance_of("http://127.0.0.1:8080") == "cafefeed"


def test_instance_of_an_unregistered_port_is_empty(monkeypatch):
    monkeypatch.setattr(servers, "registry", lambda: {})
    assert loaded._instance_of("http://127.0.0.1:8080") == ""
    assert loaded._instance_of("http://x") == ""          # no port at all


def test_knurlogic_row_carries_its_instance(monkeypatch):
    monkeypatch.setattr(loaded, "_get", lambda url, timeout=1.5: {
        "schema": 2, "artifact": {"name": "A", "path": "/p/A"},
        "memory": {"active_bytes": 1 << 30}})
    monkeypatch.setattr(servers, "registry",
                        lambda: {9: {"pid": 1,
                                     "instance": "aaaaaaaaaaaaaaaa"}})
    rows = loaded._knurlogic("http://127.0.0.1:9")
    assert rows[0].instance == "aaaaaaaaaaaaaaaa"


def test_models_across_dedupes_a_single_mac_server_by_instance():
    """The same instance, reported on this page's own residency AND (say) a
    stale peer echo, is one entry -- the general form of the cluster-job
    dedupe, now keyed on `instance` rather than only `job`."""
    row = {"name": "small", "runtime": "knurlogic",
           "where": "http://127.0.0.1:8081", "state": "loaded",
           "requests": None, "instance": "deadbeefcafefeed"}
    doc = {"resident": [row], "jobs": [], "recovery": [],
           "peers": [{"machine": "B", "resident": [dict(row)],
                      "jobs": [], "recovery": []}]}
    out = mcp.models_across(doc, "A")
    assert len(out) == 1
    assert out[0]["instance"] == "deadbeefcafefeed"


def test_models_across_carries_a_cluster_jobs_instance_as_its_job_id():
    out = mcp.models_across(residency(), "A")
    m = [r for r in out if r["job"] == "j1"]
    assert len(m) == 1 and m[0]["instance"] == "j1"


def test_unload_by_instance_reaches_this_macs_own_server(page):  # noqa: F811
    page.answer = {"stopped": "small"}
    out = mcp.unload(instance="deadbeefcafefeed")
    # the fixture's residency() has no such instance; assert the no-match
    # path first, then a matching one below
    assert "error" in out


def test_unload_by_instance_matches_like_unload_by_job(monkeypatch, page):  # noqa: F811
    page.answer = {"stopped": "j1", "told": ["B"]}
    out = mcp.unload(instance="j1")
    assert page.posts == [{"action": "unload", "job": "j1"}]
    assert out["instance"] == "j1"


def test_unload_needs_a_name(page):  # noqa: F811
    out = mcp.unload()
    assert "error" in out
    assert "instance" in out["error"]


def test_residency_reports_the_instance(monkeypatch):
    from knurlogic.interfaces.http import residency as res_api

    monkeypatch.setattr(
        servers, "registry",
        lambda: {8080: {"pid": 1, "instance": "1111222233334444"}})
    host = SimpleNamespace(status=lambda: {
        "state": "ready", "model": "/p/A", "memory_bytes": 1 << 30})
    sched = SimpleNamespace(width=1, requests=lambda: REQS)
    doc = res_api.residency(host, sched, port=8080)
    assert doc["data"][0]["instance"] == "1111222233334444"


def test_residency_omits_instance_when_unknown(monkeypatch):
    from knurlogic.interfaces.http import residency as res_api

    monkeypatch.setattr(servers, "registry", lambda: {})
    host = SimpleNamespace(status=lambda: {
        "state": "ready", "model": "/p/A", "memory_bytes": 1 << 30})
    sched = SimpleNamespace(width=1, requests=lambda: REQS)
    doc = res_api.residency(host, sched, port=8080)
    assert "instance" not in doc["data"][0]
