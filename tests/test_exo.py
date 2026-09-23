"""Placing on exo, and knowing what every instance is doing.

Every fixture here is in exo's own shape -- tagged unions, camelCase,
runners keyed by id and linked through shardAssignments -- because the
`ready` bug that hid behind a made-up download shape is the lesson. Each test
says which live finding it pins.
"""
import time

import pytest

from knurlogic.machine import exo

GIB = 1 << 30


@pytest.fixture(autouse=True)
def _watch(tmp_path, monkeypatch):
    """The stall clock lives on disk; give every test its own."""
    monkeypatch.setattr(exo, "_watch_path", lambda: tmp_path / "watch.json")


def _instance(iid, model, runners):
    """{iid: {MlxRingInstance: {...}}} with runners {rid: node_id}."""
    return {iid: {"MlxRingInstance": {"instanceId": iid, "shardAssignments": {
        "modelId": model,
        "nodeToRunner": {n: r for r, n in runners.items()},
        "runnerToShard": {r: {"PipelineShardMetadata": {
            "modelCard": {"modelId": model, "nLayers": 64,
                          "storageSize": {"inBytes": 16 * GIB}},
            "startLayer": 0, "endLayer": 64}} for r in runners}}}}}


def _state(instances=None, runners=None, downloads=None):
    return {"instances": instances or {}, "runners": runners or {},
            "downloads": downloads or {},
            "nodeIdentities": {"n1": {"friendlyName": "Studio"}},
            "nodeMemory": {"n1": {"ramAvailable": {"inBytes": 20 * GIB}}}}


@pytest.mark.parametrize("runner,want", [
    ({"RunnerLoading": {"layersLoaded": 25, "totalLayers": 64}}, "loading"),
    ({"RunnerWarmingUp": {}}, "warming"),
    ({"RunnerReady": {}}, "serving"),
    ({"RunnerRunning": {}}, "serving"),
    ({"RunnerFailed": {"errorMessage": "Metal OOM"}}, "failed"),
    ({"RunnerShuttingDown": {}}, "shutting down"),
])
def test_every_instance_has_a_phase_read_off_exos_evidence(runner, want):
    """Measured live: 27B loading 25 -> 63 of 64 layers, warming, serving,
    all visible at one-second polls."""
    st = _state(_instance("i1", "org/m", {"r1": "n1"}), {"r1": runner})
    (row,) = exo.phases(st=st)
    assert row["phase"] == want
    if want == "loading":
        assert row["runners"][0]["layers_loaded"] == 25
    if want == "failed":
        assert row["runners"][0]["error"] == "Metal OOM"


def test_a_download_in_progress_is_its_own_phase_with_bytes():
    st = _state(_instance("i1", "org/m", {"r1": "n1"}),
                {"r1": {"RunnerIdle": {}}},
                {"n1": [{"DownloadOngoing": {
                    "shardMetadata": {"PipelineShardMetadata": {
                        "modelCard": {"modelId": "org/m"}}},
                    "downloadProgress": {"downloadedBytes": {"inBytes": 4 * GIB},
                                         "totalBytes": {"inBytes": 16 * GIB}}}}]})
    (row,) = exo.phases(st=st)
    assert row["phase"] == "downloading"
    assert row["downloads"][0]["progress"] == "25%"


def test_a_phase_that_stops_moving_is_named_stalled(monkeypatch):
    """exo sits in loading forever when a rank never connects. The clock is
    how long the evidence has not changed -- not how long since placing."""
    st = _state(_instance("i1", "org/m", {"r1": "n1"}),
                {"r1": {"RunnerConnecting": {}}})
    exo.phases(st=st)                               # first sight
    real = time.time
    monkeypatch.setattr(exo.time, "time", lambda: real() + exo.STALL_S + 5)
    (row,) = exo.phases(st=st)
    assert row["phase"] == "stalled" and row["stalled_in"] == "loading"
    assert "Stop waiting" in row["advice"]


def test_progress_resets_the_stall_clock(monkeypatch):
    real = time.time
    st = _state(_instance("i1", "org/m", {"r1": "n1"}),
                {"r1": {"RunnerLoading": {"layersLoaded": 1, "totalLayers": 64}}})
    exo.phases(st=st)
    monkeypatch.setattr(exo.time, "time", lambda: real() + exo.STALL_S + 5)
    st["runners"]["r1"] = {"RunnerLoading": {"layersLoaded": 2, "totalLayers": 64}}
    (row,) = exo.phases(st=st)
    assert row["phase"] == "loading"


def test_a_placement_exo_has_not_published_yet_is_memory_in_motion():
    """Measured live: a second placement posted one call after the first was
    accepted, because exo had not yet published the first -- no instance, no
    runners in /state, memory still reading free."""
    w = {"new-iid": {"placed_at": time.time(), "model": "org/m"}}
    exo._watch_path().write_text(__import__("json").dumps(w))
    blockers = exo.moving(st=_state())
    assert blockers[0]["what"] == "a placement exo has not published yet"
    assert blockers[0]["detail"]["model"] == "org/m"


def test_a_serving_instance_does_not_block():
    st = _state(_instance("i1", "org/m", {"r1": "n1"}),
                {"r1": {"RunnerReady": {}}})
    assert exo.moving(st=st) == []


def test_an_orphaned_shutdown_record_stops_blocking(monkeypatch):
    """Measured live: two removed instances left runners in
    RunnerShuttingDown indefinitely, referenced by no instance, with the
    node's memory back to exactly what it was. A check that counted them
    would itself wait forever."""
    st = _state(runners={"ghost": {"RunnerShuttingDown": {}}})
    transit, ghosts = exo.runner_activity(st)
    assert transit and not ghosts                   # just seen: maybe real
    real = time.time
    # A fixed minute, not GHOST_S + 1: a test that reads the constant it is
    # testing cannot disagree with it (a mutation of GHOST_S survived that).
    monkeypatch.setattr(exo.time, "time", lambda: real() + 60)
    transit, ghosts = exo.runner_activity(st)
    assert not transit and ghosts[0]["state"] == "RunnerShuttingDown"


def test_a_shutdown_runner_an_instance_still_references_always_blocks(
        monkeypatch):
    st = _state(_instance("i1", "org/m", {"r1": "n1"}),
                {"r1": {"RunnerShuttingDown": {}}})
    real = time.time
    monkeypatch.setattr(exo.time, "time", lambda: real() + 10_000)
    transit, ghosts = exo.runner_activity(st)
    assert transit == {"RunnerShuttingDown": 1} and not ghosts


def test_a_path_from_exos_store_becomes_a_model_id():
    assert exo.model_id_for("/x/.exo/models/TheDrainFlorist--Qwen3.8-27B") \
        == "TheDrainFlorist/Qwen3.8-27B"
    assert exo.model_id_for("org/name") == "org/name"


# --- place, through the MCP ------------------------------------------------

@pytest.fixture
def mcp_world(monkeypatch):
    from knurlogic.interfaces import mcp
    posted = []
    monkeypatch.setattr(mcp, "_holders", lambda: [{"what": "big model",
                                                   "phase": "serving"}])
    monkeypatch.setattr(exo, "place", lambda p, base=None: posted.append(p) or
                        {"instance_id": "new"})
    return mcp, posted


def test_place_refuses_an_unsettled_ring_without_posting(mcp_world,
                                                         monkeypatch):
    mcp, posted = mcp_world
    monkeypatch.setattr(mcp, "ready", lambda: {
        "ready_for_exo_placement": False, "exo_placement_blockers": [],
        "blockers": [{"what": "an exo instance is still coming up"}]})
    r = mcp.place(model="org/m")
    assert r["placed"] is False and r["refused"] == "the cluster is not settled"
    assert not posted


def test_place_refuses_a_shard_that_does_not_fit_its_node(mcp_world,
                                                          monkeypatch):
    """No override: it is arithmetic. And it says what could be unloaded."""
    mcp, posted = mcp_world
    monkeypatch.setattr(mcp, "ready", lambda: {"ready_for_exo_placement": True})
    monkeypatch.setattr(exo, "plan", lambda *a, **k: {
        "model_id": "org/m", "fits": False, "instance": {},
        "nodes": [{"node": "Studio", "gib": 30.0, "available_gib": 20.0,
                   "headroom_gib": -10.0, "fits": False}]})
    r = mcp.place(model="org/m", force=True)
    assert r["refused"] == "a shard does not fit its node"
    assert r["holding_memory"][0]["what"] == "big model"
    assert not posted


def test_place_posts_exos_own_plan_when_everything_holds(mcp_world,
                                                         monkeypatch):
    mcp, posted = mcp_world
    monkeypatch.setattr(mcp, "ready", lambda: {"ready_for_exo_placement": True})
    plan = {"model_id": "org/m", "fits": True, "instance": {"x": 1},
            "nodes": [{"node": "Studio", "gib": 10.0, "available_gib": 60.0,
                       "headroom_gib": 50.0, "fits": True}]}
    monkeypatch.setattr(exo, "plan", lambda *a, **k: plan)
    r = mcp.place(model="org/m")
    assert r["placed"] is True and posted == [plan]
    assert "never silence" in r["next"]


def test_exos_refusal_is_read_in_exos_own_error_format(monkeypatch):
    """exo answers 400 with {"error": {"message": ...}}, not {"detail": ...};
    reading the wrong key turned a real reason into 'None'."""
    import io
    import urllib.error

    def refuse(*a, **k):
        raise urllib.error.HTTPError(
            "u", 400, "Bad Request", {}, io.BytesIO(
                b'{"error":{"message":"No cycles found with sufficient memory"}}'))
    monkeypatch.setattr(exo.urllib.request, "urlopen", refuse)
    monkeypatch.setattr(exo, "_memory_arithmetic", lambda mid, base: {
        "short_by_gib": 63.9})
    p = exo.plan("org/m")
    assert p["refused_by_exo"] == "No cycles found with sufficient memory"
    assert p["short_by_gib"] == 63.9
