"""The agent-facing tools, and the two refusals that are the point of them.

Managing local models by hand through exo means guessing: whether the ring
has settled, whether a model fits. An agent guesses worse than a person and
faster, so these tools refuse rather than gamble -- and a refusal that can be
argued out of is not one.
"""
import json

import pytest

from knurlogic.interfaces import mcp


@pytest.fixture(autouse=True)
def _no_real_loadlock(monkeypatch):
    """`ready()` now also asks `machine/loadlock.holder()` (P5, this file's
    own addition below). That reads the REAL default lock path when no test
    sets one, which would make every `ready()` test here depend on whether
    something else on this shared machine happens to hold it right now.
    Default it to "nothing held" so a test that cares about the lock says
    so explicitly (see `test_ready_blocks_on_another_process_holding_the_load_lock`)."""
    monkeypatch.setattr("knurlogic.machine.loadlock.holder", lambda *a, **k: None)


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
    monkeypatch.setattr("knurlogic.machine.loaded.available_memory",
                        lambda: {"available_bytes": 1 << 20,
                                 "free_bytes": 1 << 20, "cached_bytes": 0})
    monkeypatch.setattr("knurlogic.interfaces.ui._spawn",
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
    monkeypatch.setattr("knurlogic.machine.loaded.available_memory",
                        lambda: {"available_bytes": 64 << 30,
                                 "free_bytes": 64 << 30, "cached_bytes": 0})
    spawned = []
    monkeypatch.setattr("knurlogic.interfaces.ui._spawn",
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
    from knurlogic.machine import deps
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


def test_glm5_needs_nothing_from_mlx_vlm_any_more():
    """It used to import nine mlx-vlm modules, which is why GLM-5.3 could
    not load from a plain install. They are vendored now (engine/vision/glm5)
    and the derived list is empty -- if an import from mlx_vlm creeps back
    into the vendored architecture, this names it."""
    from knurlogic.machine.deps import glm5_siblings
    assert glm5_siblings() == []


def _phase_world(monkeypatch, tmp_path, *, alive, answers, held, size,
                 quiet_s=0):
    """One registered server, with every fact the phase is read from faked."""
    import os, time
    from knurlogic.interfaces import ui
    log = tmp_path / "serve.log"
    log.write_text("artifact  x\nloading weights\n")
    t = time.time() - quiet_s
    os.utime(log, (t, t))
    monkeypatch.setattr(ui, "registry", lambda: {
        9001: {"pid": 4242, "artifact": "/m/x", "log": str(log), "t": 0}})
    monkeypatch.setattr(ui, "is_our_server", lambda pid: alive)
    monkeypatch.setattr(ui, "_answers", lambda port: answers)
    monkeypatch.setattr(ui, "_artifact_bytes", lambda p: size)
    monkeypatch.setattr(ui.loaded, "memory_map",
                        lambda: {"processes": [{"pid": 4242, "bytes": held}]})
    return ui


@pytest.mark.parametrize("alive,answers,held,quiet,want", [
    (True, False, 0, 5, "loading"),
    (True, False, 0, 10_000, "stalled"),
    (True, True, 3 << 30, 0, "warming"),        # measured: answered at 3 of 15.5
    (True, True, 15 << 30, 0, "serving"),
    (False, False, 0, 0, "exited"),
])
def test_every_server_says_what_phase_it_is_in(monkeypatch, tmp_path, alive,
                                               answers, held, quiet, want):
    """`alive: true` for loading, serving and hung alike is how an agent ends
    up waiting forever. Each phase is read off evidence: the port, the
    weights actually resident, and how long the log has been quiet."""
    ui = _phase_world(monkeypatch, tmp_path, alive=alive, answers=answers,
                      held=held, size=int(15.5 * (1 << 30)), quiet_s=quiet)
    (c,) = ui.children()
    assert c["phase"] == want
    if want == "stalled":
        assert "Stop waiting" in c["advice"]
    if want == "exited":
        assert c["log_tail"][-1] == "loading weights"


def test_ready_waits_for_knurlogics_own_loads(monkeypatch, tmp_path):
    """Reproduced live before this existed: with one model mid-load, ready
    said true and fit used a budget that did not count it yet."""
    _phase_world(monkeypatch, tmp_path, alive=True, answers=True,
                 held=1 << 30, size=15 << 30)
    monkeypatch.setattr(mcp, "_exo_state", lambda: {})
    r = mcp.ready()
    assert r["ready"] is False
    assert r["blockers"][0]["detail"]["phase"] == "warming"


def test_ready_blocks_on_another_process_holding_the_load_lock(monkeypatch):
    """The lock (`machine/loadlock.py`, P0) is a second source of the same
    blocker `ready()` already reports for knurlogic's own children -- a
    load started by a DIFFERENT process (another agent, a hand-run `serve`)
    holds it, and would not show up in `ui.loading()`."""
    monkeypatch.setattr(mcp, "_exo_state", lambda: {})
    monkeypatch.setattr(
        "knurlogic.machine.loadlock.holder",
        lambda *a, **k: {"pid": 999, "artifact": "other-model",
                         "agent": "someone-else", "started": 0})
    r = mcp.ready()
    assert r["ready"] is False
    assert any("load lock" in b["what"] for b in r["blockers"])


def test_fit_reports_vision_capability(tmp_path, monkeypatch):
    """`registry.registered` is a REGISTERED-model_type-and-package-present
    check (P0); the family packages (qwen, gemma4, glm5) are P1-P3's, not
    yet on disk here, so this drives the registry directly rather than
    asserting True for a real model_type that may resolve False today and
    True once P1-P3 land -- `fit`'s job is only to pass the answer through."""
    d = _artifact(tmp_path / "vqwen", model_type="qwen3_5")
    monkeypatch.setattr(
        "knurlogic.machine.loaded.available_memory",
        lambda: {"available_bytes": 64 << 30, "free_bytes": 64 << 30,
                 "cached_bytes": 0})
    monkeypatch.setattr(
        "knurlogic.engine.vision.registry.registered", lambda mt: True)
    assert mcp.fit(artifact=str(d))["vision_capable"] is True
    monkeypatch.setattr(
        "knurlogic.engine.vision.registry.registered", lambda mt: False)
    assert mcp.fit(artifact=str(d))["vision_capable"] is False


def test_models_lists_vision_capability(tmp_path, monkeypatch):
    from knurlogic.machine.discover import Found

    d = _artifact(tmp_path / "vqwen", model_type="qwen3_5")
    monkeypatch.setattr(
        "knurlogic.machine.discover.find",
        lambda *a, **k: [Found(name="vqwen", path=d, store="given",
                               format="mlx", bytes_on_disk=4 << 20,
                               model_type="qwen3_5", servable=True)])
    monkeypatch.setattr(
        "knurlogic.engine.vision.registry.registered", lambda mt: True)
    r = mcp.models()
    assert r["models"][0]["vision_capable"] is True


def test_state_carries_the_served_vision_spec(monkeypatch):
    from knurlogic.engine.vision import VisionSpec

    spec = VisionSpec(family="qwen3_5", image_token_id=5, patch=14,
                      merge=2, min_pixels=100, max_pixels=1000,
                      fixed_tokens=None, proc_hash="deadbeef" * 2)
    monkeypatch.setattr("knurlogic.engine.vision.served_vision",
                        lambda: spec)
    monkeypatch.setattr("knurlogic.machine.loaded.survey",
                        lambda: {"resident": [], "runtimes": []})
    r = mcp.state()
    assert r["vision"]["family"] == "qwen3_5"


def test_state_vision_is_none_when_nothing_served(monkeypatch):
    monkeypatch.setattr("knurlogic.engine.vision.served_vision",
                        lambda: None)
    monkeypatch.setattr("knurlogic.machine.loaded.survey",
                        lambda: {"resident": [], "runtimes": []})
    r = mcp.state()
    assert r["vision"] is None
