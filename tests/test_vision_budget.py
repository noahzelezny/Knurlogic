"""A vision rung is budgeted BEFORE a load (Flash-Next review point 4, P0's
open issue): the tower's weights, the image store's bound and an allowance
for image-span KV, each a named term with its note, in `resolve` and in
the MCP's `fit` and `settings`. Stdlib-built artifacts; nothing loads."""
from __future__ import annotations

import json
import struct

import pytest

from knurlogic.engine.vision.store import DEFAULT_MAX_BYTES
from knurlogic.machine.artifact import Artifact
from knurlogic.tuning import resolve as R
from knurlogic.tuning import settings as S

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
    return per * S.VISION_KV_IMAGES * S.VISION_KV_TOKENS_PER_IMAGE


@pytest.mark.parametrize("sidecar", [False, True])
def test_vision_budget_has_three_terms(tmp_path, sidecar):
    a = Artifact.load(_rung(tmp_path / "r", sidecar=sidecar))
    vb = R.vision_budget(a)
    assert vb["tower_bytes"] == TOWER            # read from the headers
    assert vb["tower_outside_bytes"] == 0        # inside bytes_on_disk
    assert vb["store_bytes"] == DEFAULT_MAX_BYTES and not vb["store_is_live"]
    assert vb["kv_allowance_bytes"] == _kv_expected()
    assert vb["extra_bytes"] == DEFAULT_MAX_BYTES + _kv_expected()
    assert len(vb["notes"]) == 3
    assert R.vision_budget(a, store_bytes=7 << 20)["store_bytes"] == 7 << 20


def test_text_only_artifact_has_no_vision_budget(tmp_path):
    a = Artifact.load(_rung(tmp_path / "t", vision=False))
    assert R.vision_budget(a) is None
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
    vb = R.vision_budget(Artifact.load(d))
    assert vb["tower_outside_bytes"] == 2 << 20
    assert vb["extra_bytes"] == (2 << 20) + DEFAULT_MAX_BYTES + _kv_expected()


def test_mcp_fit_and_settings_show_the_terms(tmp_path, monkeypatch):
    from knurlogic.interfaces import mcp
    d = _rung(tmp_path / "m")
    a = Artifact.load(d)
    need = a.bytes_on_disk + DEFAULT_MAX_BYTES + _kv_expected()
    monkeypatch.setattr(
        "knurlogic.machine.wired.load_budget",
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
