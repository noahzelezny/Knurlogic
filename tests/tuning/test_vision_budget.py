"""A vision model is budgeted BEFORE a load: the tower's weights, the image
store's bound and an allowance for image-span KV, each a named term with
its note, in `resolve` and in the MCP's `fit` and `settings`. Stdlib-built
artifacts; nothing loads."""
from __future__ import annotations

import json
import struct

import pytest

from knurlogic.engine.vision.store import DEFAULT_MAX_BYTES
from knurlogic.machine.artifact import Artifact
from knurlogic.tuning import checks, fit, knobs, measured, pipeline_split
from knurlogic.tuning import resolve as R

GIB = 1 << 30
TOWER = 3 << 20                                  # 3 MiB of tower tensors
TEXT = 5 << 20

TC = {"model_type": "qwen3_5", "hidden_size": 256, "num_hidden_layers": 8,
      "full_attention_interval": 4, "num_attention_heads": 4,
      "num_key_value_heads": 2, "head_dim": 64}


def _safetensors(path, tensors):
    """tensors: {name: nbytes}. A real header; the data is zeros."""
    header, off = {}, 0
    for k, n in tensors.items():
        header[k] = {"dtype": "U8", "shape": [n], "data_offsets": [off, off + n]}
        off += n
    h = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"\0" * off)


def _rung(d, vision=True, sidecar=False):
    d.mkdir(parents=True, exist_ok=True)
    cfg = {"model_type": "qwen3_5", "text_config": TC}
    if vision:
        cfg["vision_config"] = {"depth": 2}
    (d / "config.json").write_text(json.dumps(cfg))
    t = {"model.language_model.embed_tokens.weight": TEXT}
    if vision and not sidecar:
        t["model.visual.blocks.0.attn.qkv.weight"] = TOWER
    _safetensors(d / "model.safetensors", t)
    if vision and sidecar:
        _safetensors(d / "model-vision-graft.safetensors",
                     {"visual.patch_embed.proj.weight": TOWER})
    return d


def _kv_expected():
    # 8 layers / every 4th full = 2 layers x 2 heads x 64 dims x K,V x bf16
    per = 2 * 2 * 2 * 64 * 2
    return per * measured.VISION_KV_IMAGES * measured.VISION_KV_TOKENS_PER_IMAGE


@pytest.mark.parametrize("sidecar", [False, True])
def test_vision_budget_has_three_terms(tmp_path, sidecar):
    a = Artifact.load(_rung(tmp_path / "r", sidecar=sidecar))
    vb = fit.vision_budget(a)
    assert vb["tower_bytes"] == TOWER            # read from the headers
    assert vb["tower_outside_bytes"] == 0        # inside bytes_on_disk
    assert vb["store_bytes"] == DEFAULT_MAX_BYTES and not vb["store_is_live"]
    assert vb["kv_allowance_bytes"] == _kv_expected()
    assert vb["extra_bytes"] == DEFAULT_MAX_BYTES + _kv_expected()
    assert len(vb["notes"]) == 3
    assert fit.vision_budget(a, store_bytes=7 << 20)["store_bytes"] == 7 << 20


def test_text_only_artifact_has_no_vision_budget(tmp_path):
    a = Artifact.load(_rung(tmp_path / "t", vision=False))
    assert fit.vision_budget(a) is None
    assert R.resolve(a, 64 * GIB).vision is None


def test_resolve_counts_vision_before_the_headroom(tmp_path):
    """The fit gate: a budget that holds the weights but not the weights +
    store + KV allowance does not fit a vision rung, and the same budget fits
    the same rung without vision."""
    v = Artifact.load(_rung(tmp_path / "v"))
    t = Artifact.load(_rung(tmp_path / "t", vision=False))
    need = v.bytes_on_disk + DEFAULT_MAX_BYTES + _kv_expected()
    budget = need - (1 << 20)
    rv = R.resolve(v, budget)
    assert rv.vision["extra_bytes"] == DEFAULT_MAX_BYTES + _kv_expected()
    assert any("does not fit" in w for w in rv.warnings)
    assert any("vision tower" in n for n in rv.notes)
    assert any("image store" in n for n in rv.notes)
    assert any("image KV allowance" in n for n in rv.notes)
    assert not any("does not fit" in w for w in R.resolve(t, budget).warnings)
    assert not R.resolve(v, need + GIB).warnings


def test_tower_outside_the_counted_files_is_added(tmp_path):
    d = _rung(tmp_path / "o", vision=True)
    (d / "vision").mkdir()
    _safetensors(d / "vision" / "tower.safetensors",
                 {"vision_tower.encoder.w": 2 << 20})
    vb = fit.vision_budget(Artifact.load(d))
    assert vb["tower_outside_bytes"] == 2 << 20
    assert vb["extra_bytes"] == (2 << 20) + DEFAULT_MAX_BYTES + _kv_expected()


def test_mcp_fit_and_settings_show_the_terms(tmp_path, monkeypatch):
    from knurlogic.interfaces import mcp
    d = _rung(tmp_path / "m")
    a = Artifact.load(d)
    need = a.bytes_on_disk + DEFAULT_MAX_BYTES + _kv_expected()
    monkeypatch.setattr(
        "knurlogic.machine.memory.wired.load_budget",
        lambda: {"bytes": need - (1 << 20), "working_set_bytes": need,
                 "available_bytes": need, "limited_by": "test"})
    f = mcp.fit(artifact=str(d))
    assert f["fits"] is False                    # would fit on weights alone
    vb = f["vision_budget"]
    assert vb["image_store_gib"] == round(DEFAULT_MAX_BYTES / GIB, 2)
    assert "tower_gib" in vb and "image_kv_allowance_gib" in vb
    assert len(vb["notes"]) == 3
    s = mcp.settings(artifact=str(d))
    assert s["vision_budget"]["notes"] == vb["notes"]


def test_tuning_and_interfaces_stay_free_of_mlx():
    import subprocess
    import sys
    code = ("import sys, knurlogic.tuning.resolve, knurlogic.interfaces.mcp, "
            "knurlogic.engine.vision.store;"
            "print([m for m in sys.modules if m.split('.')[0]=='mlx'])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True).stdout.strip()
    assert out == "[]"


# --- KNURLOGIC_VISION=off: nothing of vision is held -------------------------

def _freed():
    return TOWER + DEFAULT_MAX_BYTES + _kv_expected()


def _with_head(d, nbytes):
    with open(d / "mtp-head.safetensors", "wb") as fh:   # sparse: size only
        fh.truncate(nbytes)
    return d


def test_vision_off_frees_the_tower_store_and_image_kv(tmp_path):
    v = Artifact.load(_rung(tmp_path / "v"))
    assert fit.vision_freed_bytes(v) == _freed()
    assert fit.vision_freed_bytes(
        Artifact.load(_rung(tmp_path / "t", vision=False))) == 0
    on, off = R.resolve(v, 64 * GIB), R.resolve(v, 64 * GIB, vision=False)
    assert off.vision is None and on.vision is not None
    assert on.env["KNURLOGIC_VISION"] == "on"
    assert off.env["KNURLOGIC_VISION"] == "off"
    assert any("vision off" in n for n in off.notes)
    # the same budget: short with vision on, room to spare with it off
    need = v.bytes_on_disk + DEFAULT_MAX_BYTES + _kv_expected() - (1 << 20)
    assert any("does not fit" in w for w in R.resolve(v, need).warnings)
    assert not any("does not fit" in w
                   for w in R.resolve(v, need, vision=False).warnings)


def test_single_fit_check_subtracts_vision_when_off_and_says_so(tmp_path):
    v = Artifact.load(_rung(tmp_path / "v"))
    on_need = v.bytes_on_disk + DEFAULT_MAX_BYTES + _kv_expected()
    margin = fit.step_margin(8 * GIB)             # the 4 GiB floor
    budget = on_need + margin - (1 << 20)       # just short with vision on
    c = fit.single_fit_check(v, budget)
    assert c["state"] == "cannot" and c["vision_bytes"] == _freed()
    assert "turn vision off" in c["why"] and "MTP" not in c["why"]
    assert fit.single_fit_check(v, budget, vision=False)["state"] == "fits"
    # off, the tower inside the artifact's size is out of the weights too
    off_line = on_need - _freed() + margin
    assert fit.single_fit_check(v, off_line, vision=False)["state"] == "fits"
    assert fit.single_fit_check(v, off_line - 1,
                                vision=False)["state"] == "cannot"


def test_mtp_and_vision_off_together_is_named_when_only_both_fit(tmp_path):
    head = 2 << 20
    v = Artifact.load(_with_head(_rung(tmp_path / "v"), head))
    assert fit.mtp_head_bytes(v) == head
    need = v.bytes_on_disk + DEFAULT_MAX_BYTES + _kv_expected()
    budget = need - head - _freed() + fit.step_margin(8 * GIB)
    c = fit.single_fit_check(v, budget)
    assert c["state"] == "cannot"
    assert "turn MTP off (Settings → Presets) and vision off" in c["why"]
    assert fit.single_fit_check(v, budget, draft=False,
                                vision=False)["state"] == "fits"
    # either alone fits: both are offered
    roomy = need - min(head, _freed()) + fit.step_margin(8 * GIB)
    assert "or vision off" in fit.single_fit_check(v, roomy)["why"]


def test_the_launch_setting_is_parsed_and_applied(tmp_path, monkeypatch):
    assert knobs.engine_settings({"KNURLOGIC_VISION": "off"}) == {"vision": False}
    assert knobs.engine_settings({"KNURLOGIC_VISION": "on"}) == {"vision": True}
    assert knobs.vision_of({}) is True
    assert knobs.vision_of({"KNURLOGIC_VISION": "off"}) is False
    assert "KNURLOGIC_VISION" in knobs.MODEL_KNOBS
    assert "KNURLOGIC_VISION" in checks.launch_knobs()
    assert checks.check_knob("KNURLOGIC_VISION", "off") is None
    assert checks.check_knob("KNURLOGIC_VISION", "maybe")
    from knurlogic.tuning import preferences
    from knurlogic.tuning.checks import launch_fit
    monkeypatch.setattr(preferences, "launch_sets", lambda s: dict(s))
    v = Artifact.load(_rung(tmp_path / "v"))
    budget = (v.bytes_on_disk + DEFAULT_MAX_BYTES + _kv_expected()
              + fit.step_margin(8 * GIB) - (1 << 20))
    assert launch_fit(v, {}, "default", True, budget)["state"] == "cannot"
    assert launch_fit(v, {"KNURLOGIC_VISION": "off"}, "default", True,
                      budget)["state"] == "fits"


def test_mcp_fit_with_vision_off(tmp_path, monkeypatch):
    from knurlogic.interfaces import mcp
    from knurlogic.tuning import preferences
    monkeypatch.setattr(preferences, "launch_sets", lambda s: dict(s))
    d = _rung(tmp_path / "m")
    a = Artifact.load(d)
    need = a.bytes_on_disk + DEFAULT_MAX_BYTES + _kv_expected()
    b = need + fit.step_margin(8 * GIB) - (1 << 20)
    monkeypatch.setattr(
        "knurlogic.machine.memory.wired.load_budget",
        lambda: {"bytes": b, "working_set_bytes": b,
                 "available_bytes": b, "limited_by": "test"})
    assert mcp.fit(artifact=str(d))["fits"] is False
    f = mcp.fit(artifact=str(d), vision=False)
    assert f["fits"] is True and f["vision"] is False
    assert f["vision_budget"] is None
    assert f["headroom_gib"] == round((b - need + _freed()) / GIB, 1)


def test_the_picked_models_preview_carries_the_part_sizes(tmp_path):
    from knurlogic.interfaces.page import documents
    d = _with_head(_rung(tmp_path / "v"), 3 << 20)
    p = documents._preview(str(d), "default", 64)
    assert (p["mtp_bytes"], p["vision_bytes"]) == (3 << 20, _freed())
    t = documents._preview(str(_rung(tmp_path / "t", vision=False)),
                           "default", 64)
    assert (t["mtp_bytes"], t["vision_bytes"]) == (0, 0)


def test_the_models_listing_loads_no_artifact(tmp_path, monkeypatch):
    # /models.json lists ~80 models off a network store: counting part sizes
    # there (Artifact.load per row) took it from ~7 s to ~40 s cold
    import time
    from types import SimpleNamespace

    from knurlogic.interfaces.page import documents
    d = _with_head(_rung(tmp_path / "v"), 3 << 20)
    rows = [SimpleNamespace(name=f"m{i}", path=d, store="t", bytes_on_disk=1,
                            model_type="qwen3_5", is_vq=False, servable=True,
                            why="", extra={"mtp_head": True})
            for i in range(80)]

    def boom(*a, **k):
        raise AssertionError("the listing loaded an artifact")
    monkeypatch.setattr("knurlogic.machine.artifact.Artifact.load", boom)
    monkeypatch.setattr("knurlogic.machine.discover.find", lambda: rows)
    monkeypatch.setattr("knurlogic.interfaces.page.updates.flagged",
                        lambda paths: set())
    documents.forget_models()
    try:
        t0 = time.perf_counter()
        out = documents.models_document()({"rescan": ["1"]})["models"]
        took = time.perf_counter() - t0
    finally:
        documents.forget_models()
    assert len(out) == 80 and out[0]["mtp"] is True
    assert "mtp_bytes" not in out[0] and "vision_bytes" not in out[0]
    assert took < 5, took


def test_an_image_with_vision_off_is_a_clear_400(monkeypatch):
    from knurlogic.engine.model import state
    from knurlogic.engine.model import vision as SV
    from knurlogic.interfaces.http import openai as O
    body = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;"
                                            "base64,AAAA"}},
        {"type": "text", "text": "hi"}]}]}
    monkeypatch.setitem(state.VISION, "error", SV.VISION_OFF)
    with pytest.raises(O.ApiError) as e:
        O.build_job(body, chat=True, has_vision=lambda: False)
    assert e.value.status == 400
    assert "vision is off for this launch" in str(e.value)
    monkeypatch.setitem(state.VISION, "error", "")
    with pytest.raises(O.ApiError) as e:
        O.build_job(body, chat=True, has_vision=lambda: False)
    assert e.value.status == 400 and "has no vision" in str(e.value)


def test_pipeline_rank_0_holds_no_tower_with_vision_off(tmp_path):
    v = Artifact.load(_rung(tmp_path / "v"))
    assert pipeline_split.leader_bytes(v) - pipeline_split.leader_bytes(
        v, vision=False) == TOWER


def test_rank_0_holds_no_head_with_mtp_off(tmp_path):
    # the head counted against rank 0 with MTP off made a split refuse, or
    # take fewer layers on rank 0, for memory it never uses
    d = _rung(tmp_path / "m", vision=False)
    _safetensors(d / "mtp-head-q6.safetensors", {"mtp.fc.weight": TEXT})
    a = Artifact.load(d)
    on, off = pipeline_split.leader_bytes(a, vision=False), pipeline_split.leader_bytes(
        a, vision=False, mtp=False)
    assert on > 0 and off == 0
    assert knobs.mtp_of({"KNURLOGIC_MTP": "off"}) is False
    assert knobs.mtp_of({}) is True


def test_a_cluster_launchs_mtp_off_reaches_its_shape(tmp_path, monkeypatch):
    from knurlogic.cluster import launch as C
    d = _rung(tmp_path / "m", vision=False)
    _safetensors(d / "mtp-head-q6.safetensors", {"mtp.fc.weight": TEXT})
    for split in ("pipeline", "tensor"):
        assert C.shape_of(str(d), 2, split, vision=False,
                          mtp=False)["leader_bytes"] == 0
        assert C.shape_of(str(d), 2, split, vision=False)["leader_bytes"] > 0


def test_the_picked_models_preview_carries_its_tensor_bytes(tmp_path):
    # the picker fits a tensor split from the picked model's preview (the
    # listing loads no artifact, so it no longer carries them)
    from knurlogic.interfaces.page import documents
    t = documents._preview(str(_with_head(_rung(tmp_path / "t"), 3 << 20)),
                           "default", 64)
    tb = t["tensor_bytes"]
    assert {"sharded", "replicated", "head", "tower"} <= set(tb)


def test_deepseek_vision_exp_is_budgeted_only_with_its_tower(tmp_path):
    """Vision-Exp's vision fields are flat in config.json and its tower is
    `vision.*` / `aligner.*`: counted from the headers. A text-only
    conversion that kept the fields but not the tower has no budget."""
    cfg = {"model_type": "deepseek_v4", "vision_n_layers": 32,
           "num_hidden_layers": 2, "compress_ratios": [0, 0],
           "head_dim": 64, "sliding_window": 128}
    for name, tower in (("vision", True), ("teacher", False)):
        d = tmp_path / name
        d.mkdir()
        (d / "config.json").write_text(json.dumps(cfg))
        t = {"model.norm.weight": TEXT}
        if tower:
            t.update({"vision.norm.weight": TOWER, "aligner.w1.weight": TOWER,
                      "model.layers.0.ffn.gate.bias_vl": 1024,
                      **{f"model.image_{r}": 1024 for r in
                         ("start", "end", "newline", "pad")}})
        _safetensors(d / "model.safetensors", t)
    vb = fit.vision_budget(Artifact.load(tmp_path / "vision"))
    assert vb["tower_bytes"] == 2 * TOWER and vb["tower_tensors"] == 2
    assert vb["store_bytes"] == DEFAULT_MAX_BYTES
    assert fit.vision_budget(Artifact.load(tmp_path / "teacher")) is None


def test_the_previews_room_is_what_this_machine_has_free_now(tmp_path,
                                                            monkeypatch):
    """A 128 GiB Mac already holding a 108 GiB model: the room line said
    "leaves 87 GiB" -- measured against the whole working set. It is the
    load budget's: memory available now."""
    from knurlogic.interfaces.page import documents
    ws, free = 120 * GIB, 10 * GIB
    monkeypatch.setattr(
        "knurlogic.machine.memory.wired.load_budget",
        lambda: {"bytes": free, "working_set_bytes": ws,
                 "available_bytes": free, "allowance_bytes": 0,
                 "limited_by": "memory available now"})
    p = documents._preview(str(_rung(tmp_path / "r")), "default")
    assert p["room"]["working_set_bytes"] == free
