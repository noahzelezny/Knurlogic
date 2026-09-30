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
    shape = {"layer_bytes": [GIB] * 190, "other_bytes": 0, "leader_bytes": 0,
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
