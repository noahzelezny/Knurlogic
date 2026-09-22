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


def test_ready_names_every_reason_it_is_not(monkeypatch):
    """Runners, downloads and unseen nodes are reported SEPARATELY by exo,
    and a caller that checks one of them loads into a ring still moving."""
    monkeypatch.setattr(mcp, "_exo_state", lambda: {
        "runners": {"a": {"RunnerShuttingDown": {}},
                    "b": {"RunnerWarmingUp": {}}},
        "downloads": {"m": {"pct": 12}},
        "topology": {"nodes": ["n1", "n2"]},
        "lastSeen": {"n1": 1},
    })
    r = mcp.ready()
    assert r["ready"] is False
    what = {b["what"] for b in r["blockers"]}
    assert what == {"runners in transition", "downloads in flight",
                    "nodes in the topology not seen recently"}


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
    assert r["loaded"] is False and r["refused"] == "cluster is not settled"
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
