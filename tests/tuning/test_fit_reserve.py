"""The fit leaves room to run: every rank keeps the first request's transient
and a minimum context's KV free beyond its weights, and a placement that
cannot is refused ("more space required")."""

import pytest

from knurlogic.cluster import launch as L
from knurlogic.tuning import resolve as R

GIB = 1 << 30


def _ranks(*ws):
    return [{"name": f"m{i}", "working_set_bytes": int(w * GIB),
             "memory_bandwidth_gbs": None} for i, w in enumerate(ws)]


def test_fit_reserve_is_the_measured_transient_and_a_minimum_contexts_kv():
    cfg = {"text_config": {"hidden_size": 2560, "num_hidden_layers": 48,
                           "num_attention_heads": 24, "num_key_value_heads": 2,
                           "head_dim": 256, "full_attention_interval": 4}}
    r = R.fit_reserve(cfg)
    # 512 tokens x 2560 x 480 B = 0.59 GiB, under the measured ~1 GiB floor
    assert r["transient_bytes"] == GIB
    # 12 full-attention layers x 2 heads x 256 x K,V x bf16 = 24 KiB/token
    assert r["kv_bytes"] == 8192 * 24 * 1024
    assert R.fit_reserve(cfg, kv_bits=4)["kv_bytes"] < r["kv_bytes"]
    wide = R.fit_reserve({"hidden_size": 16384})
    assert wide["transient_bytes"] > GIB        # the hidden-size line wins


def test_rank_margin_takes_the_larger_of_step_margin_and_transient_then_kv():
    assert R.rank_margin(84 * GIB) == R.step_margin(84 * GIB)
    res = {"transient_bytes": 8 * GIB, "kv_bytes": GIB}
    assert R.rank_margin(84 * GIB, res) == 10 * GIB + GIB   # 8 x 1.25 + 1
    small = {"transient_bytes": GIB, "kv_bytes": GIB}
    assert R.rank_margin(84 * GIB, small) == R.step_margin(84 * GIB) + GIB


def test_an_uneven_split_respects_the_per_rank_reserve():
    res = {"transient_bytes": 10 * GIB, "kv_bytes": 2 * GIB}     # 14.5 GiB
    s = R.pipeline_shares([GIB] * 100, _ranks(120, 84), reserve=res)
    assert s["bytes"][0] / GIB <= 120 - 14.5
    assert s["bytes"][1] / GIB <= 84 - 14.5
    with pytest.raises(ValueError):
        R.pipeline_shares([GIB] * 190, _ranks(120, 84), reserve=res)
    # ... which the step margin alone (6 + 4.2 GiB) would have accepted
    assert sum(R.pipeline_shares([GIB] * 190,
                                 _ranks(120, 84))["layers"]) == 190


def test_placement_refuses_more_space_required_when_a_rank_cannot_keep_it():
    res = {"transient_bytes": 10 * GIB, "kv_bytes": 2 * GIB}
    shape = {"layer_bytes": [GIB] * 196, "other_bytes": 0, "leader_bytes": 0,
             "tensor_per_rank_bytes": 0, "refusals": [], "reserve": res}
    ms = [{"name": "a", "working_set_bytes": 120 * GIB},
          {"name": "b", "working_set_bytes": 84 * GIB}]
    with pytest.raises(ValueError):
        L.placement(ms, shape, "pipeline")
    roomy = dict(shape, layer_bytes=[GIB] * 150)
    L.placement(ms, roomy, "pipeline")
    # memory available now lowers a machine's budget below its working set
    busy = [dict(ms[0], available_bytes=60 * GIB), ms[1]]
    with pytest.raises(ValueError):
        L.placement(busy, roomy, "pipeline")
    tensor = {"layer_bytes": [], "other_bytes": 0, "refusals": [],
              "tensor_per_rank_bytes": 80 * GIB, "reserve": res}
    with pytest.raises(ValueError, match="more space required"):
        L.placement(ms, tensor, "tensor")


def _single(tmp_path, weights, head=0):
    import json

    from knurlogic.machine.artifact import Artifact
    (tmp_path / "config.json").write_text(json.dumps(
        {"model_type": "qwen3_5", "text_config": {"hidden_size": 4096}}))
    with open(tmp_path / "model.safetensors", "wb") as f:
        f.truncate(weights)
    if head:
        with open(tmp_path / "mtp-head.safetensors", "wb") as f:
            f.truncate(head)
    return Artifact.load(tmp_path)


def test_single_machine_keeps_the_cluster_reserve(tmp_path):
    from knurlogic.tuning import resolve as R
    gib = 1 << 30
    a = _single(tmp_path, 108 * gib)
    # 108 into 109.5 left "0.0 room after weights" and swapped
    assert "plus" in R.single_fit(a, int(109.5 * gib))
    assert R.single_fit(a, 130 * gib) == ""


def _res(monkeypatch):
    monkeypatch.setattr(R, "fit_reserve", lambda cfg, kv_bits=None: {
        "transient_bytes": 8 * GIB, "kv_bytes": GIB})     # 11 GiB margin


def test_single_machine_counts_the_head_and_says_to_turn_mtp_off(
        tmp_path, monkeypatch):
    from knurlogic.tuning import resolve as R
    _res(monkeypatch)
    gib = 1 << 30
    a = _single(tmp_path, 92 * gib, head=6 * gib)   # disk counts both
    ws = 106 * gib
    # the weights fit with the step margin: short of the full reserve, it
    # fits all the same (the old tight band)
    assert R.single_fit(a, ws, draft=True) == ""
    assert R.single_fit_check(a, ws, draft=True)["state"] == "fits"
    # past the step margin with the head, inside it without: say so
    c = R.single_fit_check(a, 98 * gib, draft=True)
    assert c["state"] == "cannot" and c["head_bytes"] == 6 * gib
    assert "turn MTP off" in c["why"]
    assert R.single_fit_check(a, 98 * gib, draft=False)["state"] == "fits"


def test_single_machine_former_tight_band_fits_cannot_is_a_refusal(
        tmp_path, monkeypatch):
    from knurlogic.tuning import resolve as R
    _res(monkeypatch)
    gib = 1 << 30
    a = _single(tmp_path, 108 * gib)
    c = R.single_fit_check(a, int(115 * gib))        # 5.7 step margin fits
    assert c == {"state": "fits", "why": "", "head_bytes": 0}
    assert R.single_fit(a, int(115 * gib)) == ""
    c = R.single_fit_check(a, 110 * gib)             # weights + step margin
    assert c["state"] == "cannot" and R.single_fit(a, 110 * gib)


def test_placement_short_of_the_reserve_places_it():
    res = {"transient_bytes": 10 * GIB, "kv_bytes": 2 * GIB}
    shape = {"layer_bytes": [GIB] * 190, "other_bytes": 0, "leader_bytes": 0,
             "tensor_per_rank_bytes": 0, "refusals": [], "reserve": res}
    ms = [{"name": "a", "working_set_bytes": 120 * GIB},
          {"name": "b", "working_set_bytes": 84 * GIB}]
    plan = L.placement(ms, shape, "pipeline")
    assert sum(plan["layers"]) == 190
    tensor = {"layer_bytes": [], "other_bytes": 0, "refusals": [],
              "tensor_per_rank_bytes": 75 * GIB, "reserve": res}
    assert L.placement(ms, tensor, "tensor")["split"] == "tensor"
    # weights past the step margin stay refused
    big = dict(tensor, tensor_per_rank_bytes=80 * GIB)
    with pytest.raises(ValueError, match="more space required"):
        L.placement(ms, big, "tensor")


def _chunk(weights_gib, box_gib, hidden=4096):
    from pathlib import Path

    from knurlogic.machine.artifact import Artifact
    from knurlogic.tuning import settings as S
    a = Artifact(path=Path("/nonexistent"), model_type="qwen3_5",
                 model_file=None, bytes_on_disk=int(weights_gib * GIB),
                 hidden_size=hidden, moe_intermediate_size=1024, vq_other={})
    r = R.resolve(a, int(box_gib * GIB))
    return int(r.env["KNURLOGIC_PREFILL_CHUNK"]), r.notes, S


def test_a_low_headroom_box_gets_the_family_best_the_reserve_covers():
    """118 GiB of weights on 128: 0% of the leftover room, but the step
    margin (6.4 GiB) is reserved for transients and 2048's ~3.75 GiB fits it;
    4096 (~7.5) does not, and is not taken."""
    width, notes, _ = _chunk(118, 128)
    assert width == 2048
    assert any("reserved for transients" in n for n in notes)


def test_a_box_whose_reserve_cannot_cover_the_chunk_steps_down():
    width, _, S = _chunk(126.5, 128)       # ~1.5 GiB of headroom at all
    assert width == S.PREFILL_CHUNK_DEFAULT
    mid, _, _ = _chunk(124, 128)           # ~4 GiB: 1024 (1.9) yes, 4096 no
    assert 512 <= mid <= 2048


def test_the_transient_line_is_not_below_what_was_measured():
    """GLM-5.3 Flash VQ (hidden 6144): 1.39 GiB measured at chunk 512."""
    from knurlogic.tuning import settings as S
    predicted = 512 * 6144 * S.PREFILL_TRANSIENT_BYTES_PER_TOKEN_HIDDEN
    assert predicted / GIB >= 1.39
