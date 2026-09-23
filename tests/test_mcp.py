"""The agent-facing tools, and the two refusals that are the point of them.

Managing local models by hand through exo means guessing: whether the ring
has settled, whether a model fits. An agent guesses worse than a person and
faster, so these tools refuse rather than gamble -- and a refusal that can be
argued out of is not one.
"""
import json

import pytest

from knurlogic import mcp


def _artifact(d, gib=4, **cfg):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(
        {"model_type": "qwen3_5", "hidden_size": 2048,
         "moe_intermediate_size": 768, **cfg}))
    (d / "model.safetensors").write_bytes(b"\0" * (gib << 20))
    return d


def test_every_tool_is_in_the_table_with_a_schema():
    for t in mcp.tool_list():
        assert t["description"] and t["inputSchema"]["type"] == "object"
        assert t["name"] in mcp.TOOLS


def _dl(kind, model, **body):
    """One entry in exo's own shape: /state downloads[node] is a LIST of
    {DownloadKind: {..., shardMetadata: {PipelineShardMetadata: {modelCard}}}}.
    The old fixture was {"m": {"pct": 12}}, a shape exo never sends -- which
    is how counting every entry as 'in flight' went unnoticed."""
    body["shardMetadata"] = {"PipelineShardMetadata": {
        "modelCard": {"modelId": model}}}
    return {kind: body}


def test_ready_names_every_reason_it_is_not(monkeypatch):
    """Runners, downloads and unseen nodes are reported SEPARATELY by exo.
    Runners moving memory block a local load; the rest block ring placement."""
    monkeypatch.setattr(mcp, "_exo_state", lambda: {
        "runners": {"a": {"RunnerShuttingDown": {}},
                    "b": {"RunnerWarmingUp": {}}},
        "downloads": {"node1": [
            _dl("DownloadOngoing", "org/big", downloadProgress={
                "downloadedBytes": {"inBytes": 25}, "totalBytes": {"inBytes": 100}}),
            _dl("DownloadPending", "org/idle")]},
        "topology": {"nodes": ["n1", "n2"]},
        "lastSeen": {"n1": 1},
    })
    r = mcp.ready()
    assert r["ready"] is False and r["ready_for_exo_placement"] is False
    assert {b["what"] for b in r["blockers"]} == {
        "exo runners loading or unloading"}
    assert {b["what"] for b in r["exo_placement_blockers"]} == {
        "downloads in progress", "nodes in the topology not seen recently"}
    ongoing = r["exo_placement_blockers"][0]["detail"]
    assert ongoing == [{"model": "org/big", "node": "node1", "progress": "25%"}]


def test_pending_downloads_are_not_in_flight(monkeypatch):
    """exo lists every card it knows as DownloadPending on every node. That
    is a catalogue, not activity -- and a check that is false forever
    teaches an agent to pass force=true every time."""
    monkeypatch.setattr(mcp, "_exo_state", lambda: {
        "runners": {"a": {"RunnerReady": {}}},
        "downloads": {"n1": [_dl("DownloadPending", "x"),
                             _dl("DownloadCompleted", "y")],
                      "n2": [_dl("DownloadPending", "x"), 7, {}]},
        "topology": {"nodes": ["n1"]}, "lastSeen": {"n1": 1}})
    r = mcp.ready()
    assert r["ready"] is True and r["ready_for_exo_placement"] is True
    assert r["downloads"]["pending_not_started"] == 2


def test_a_settled_ring_is_ready(monkeypatch):
    monkeypatch.setattr(mcp, "_exo_state", lambda: {
        "runners": {"a": {"RunnerReady": {}}},
        "downloads": {}, "topology": {"nodes": ["n1"]}, "lastSeen": {"n1": 1}})
    assert mcp.ready()["ready"] is True


def test_no_exo_is_ready_not_broken(monkeypatch):
    """knurlogic does not need exo. Absent is not a blocker."""
    monkeypatch.setattr(mcp, "_exo_state", lambda: {})
    r = mcp.ready()
    assert r["ready"] is True and r["exo"] is False


def test_load_refuses_a_model_that_does_not_fit(tmp_path, monkeypatch):
    """And no flag overrides it: force is for the ring, not for arithmetic."""
    d = _artifact(tmp_path / "big", gib=8)
    monkeypatch.setattr(mcp, "_exo_state", lambda: {})
    monkeypatch.setattr("knurlogic.loaded.available_memory",
                        lambda: {"available_bytes": 1 << 20,
                                 "free_bytes": 1 << 20, "cached_bytes": 0})
    monkeypatch.setattr("knurlogic.ui._spawn",
                        lambda *a, **k: pytest.fail("spawned anyway"))
    r = mcp.load(artifact=str(d), force=True)
    assert r["loaded"] is False and r["refused"] == "will not fit"
    assert "arithmetic" in r["note"]


def test_load_refuses_an_unsettled_ring_but_force_overrides(tmp_path,
                                                            monkeypatch):
    d = _artifact(tmp_path / "small", gib=1)
    monkeypatch.setattr(mcp, "_exo_state", lambda: {
        "runners": {"a": {"RunnerLoading": {}}}, "downloads": {},
        "topology": {"nodes": []}, "lastSeen": {}})
    monkeypatch.setattr("knurlogic.loaded.available_memory",
                        lambda: {"available_bytes": 64 << 30,
                                 "free_bytes": 64 << 30, "cached_bytes": 0})
    spawned = []
    monkeypatch.setattr("knurlogic.ui._spawn",
                        lambda *a, **k: spawned.append(a) or {"starting": a[0]})

    r = mcp.load(artifact=str(d))
    assert r["loaded"] is False and r["refused"] == "memory is about to move"
    assert not spawned

    r = mcp.load(artifact=str(d), force=True)
    assert spawned and "refused" not in r


def test_settings_carries_the_measurement_not_just_the_value(tmp_path):
    """A number without its provenance is one an agent changes for no
    reason, and these were expensive to establish."""
    d = _artifact(tmp_path / "m", model_type="qwen4_exp",
                  vq_modules={"a": {"d": 2, "K": 256}})
    doc = mcp.settings(artifact=str(d))
    assert doc["knobs"]
    for k in doc["knobs"]:
        assert k["change_at"] in ("runtime", "launch only")
        assert "reach_why" in k
    assert any(k["change_at"] == "launch only" for k in doc["knobs"])


def test_an_unknown_tool_says_what_there_is():
    out = mcp._call("nope", {})
    assert "unknown tool" in out["error"] and "ready" in out["available"]


def test_a_failing_tool_reports_rather_than_raises():
    out = mcp._call("fit", {"artifact": "/does/not/exist"})
    assert "error" in out and out["tool"] == "fit"


def test_deps_reads_the_fix_not_the_version(tmp_path):
    """A version names a build; it does not say what is in it. The jaccl
    verdict comes from the fix's own env read compiled into libjaccl."""
    import json, subprocess, sys as _s
    from knurlogic import deps
    pkg = tmp_path / "site" / "mlx"
    (pkg / "lib").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "lib" / "libjaccl.dylib").write_bytes(b"\0JACCL_COLLECTIVE_TIMEOUT_MS\0")
    (tmp_path / "site" / "mlx-9.9.9.dist-info").mkdir()
    (tmp_path / "site" / "mlx-9.9.9.dist-info" / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: mlx\nVersion: 9.9.9\n")
    r = subprocess.run([_s.executable, "-c", deps._PROBE, "[]"],
                       capture_output=True, text=True,
                       env={"PYTHONPATH": str(tmp_path / "site"),
                            "PYTHONNOUSERSITE": "1"})
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert got["mlx"]["jaccl_selfheal"] is True


def test_glm5_siblings_come_from_the_vendored_source():
    from knurlogic.deps import glm5_siblings
    got = glm5_siblings()
    assert "sparse_attention" in got and "linear" in got
