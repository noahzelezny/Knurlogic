"""Pipeline split: the layer-share arithmetic, the family refusals, and two
processes on 127.0.0.1 -- the tiny models split by layer runs compute the
unsplit logits, and MTP drafting on the pipeline emits the unsplit engine's
tokens with a fixed number of broadcasts per step."""

import pytest

from knurlogic.tuning import resolve as R

GIB = 1 << 30


# ------------------------------------------------------------- the shares

def _ranks(*ws, bw=None):
    bw = bw or [None] * len(ws)
    return [{"name": f"m{i}", "working_set_bytes": int(w * GIB),
             "memory_bandwidth_gbs": b} for i, (w, b) in enumerate(zip(ws, bw))]


def test_shares_follow_capacity_and_rank_0_holds_the_last_layers():
    per = [GIB] * 40
    s = R.pipeline_shares(per, _ranks(120 + 4, 60 + 4), other_bytes=4 * GIB)
    # capacity 120 : 60 -> 26.67 : 13.33 -> 27 : 13 (largest remainder)
    assert s["layers"] == [27, 13]
    assert s["bounds"] == [(13, 40), (0, 13)]          # rank 1 embeds
    assert s["bytes"] == [27 * GIB, 13 * GIB]
    assert "capacity only (memory bandwidth unknown on m0, m1)" in s["reason"]
    assert "rank 0 m0 holds 27 (layers 13..39" in s["reason"]


def test_shares_weigh_bandwidth_only_when_every_rank_says_it():
    per = [GIB] * 40
    both = R.pipeline_shares(per, _ranks(100, 100, bw=[800, 400]))
    assert both["layers"] == [27, 13]                   # 2 : 1 by bandwidth
    assert "capacity x memory bandwidth" in both["reason"]
    one = R.pipeline_shares(per, _ranks(100, 100, bw=[800, None]))
    assert one["layers"] == [20, 20]
    assert "unknown on m1" in one["reason"]


def test_shares_are_capped_by_what_fits_and_every_rank_gets_one():
    per = [GIB] * 40
    # bandwidth wants rank 0 to take ~36; it holds 30 layers at most
    # (working sets here are what the layers get plus the 4+ GiB margin)
    s = R.pipeline_shares(per, _ranks(34.5, 104, bw=[8000, 100]))
    assert s["layers"] == [30, 10]
    tiny = R.pipeline_shares([GIB] * 4, _ranks(104, 5.001, 104,
                                              bw=[1, 1, 1]))
    assert min(tiny["layers"]) >= 1 and sum(tiny["layers"]) == 4
    with pytest.raises(ValueError, match="cannot give each of 3 ranks"):
        R.pipeline_shares([GIB] * 2, _ranks(10, 10, 10))
    with pytest.raises(ValueError, match="the ranks hold"):
        R.pipeline_shares([GIB] * 40, _ranks(10, 10))
    with pytest.raises(ValueError, match="holds none of the layers"):
        R.pipeline_shares([GIB] * 4, _ranks(10, 3), other_bytes=4 * GIB)


def test_shares_are_deterministic_and_ties_go_to_the_lower_rank():
    per = [3, 1, 4, 1, 5, 9, 2, 6, 5]
    a = R.pipeline_shares(per, _ranks(14, 14))
    assert a == R.pipeline_shares(list(per), _ranks(14, 14))
    assert a["layers"] == [5, 4]                        # 4.5 : 4.5
    three = R.pipeline_shares([1] * 10, _ranks(5, 5, 5))
    assert three["layers"] == [4, 3, 3]
    assert three["bounds"] == [(6, 10), (3, 6), (0, 3)]


def test_uneven_layers_are_checked_exactly():
    # the average says 2 + 2; the real last two layers (6 GiB) do not fit
    # a 5 GiB rank, so the cut goes by bytes: 1+1+3 and 3
    per = [1 * GIB, 1 * GIB, 3 * GIB, 3 * GIB]
    s = R.pipeline_shares(per, _ranks(9, 9))
    assert s["bounds"] == [(3, 4), (0, 3)] and s["layers"] == [1, 3]
    assert s["bytes"] == [3 * GIB, 5 * GIB]
    # and when no contiguous cut fits, the count split's arithmetic is said
    with pytest.raises(ValueError, match="layers 2..3 are 6.0 GiB"):
        R.pipeline_shares(per, _ranks(8, 8))


def test_one_huge_layer_is_placed_by_bytes_not_by_count():
    # Qwen3.8-Flash-Next-6bit: 48 layers of 1.96 GiB, and layer 1 also
    # holds a 41.7 GiB n-gram embedding. By count the M3 (rank 1, first
    # layers) got 18 layers = 79 GiB against 72.8 it holds, and the model
    # was refused though it fits the pair with room
    per = [int(1.96 * GIB)] * 48
    per[1] += int(41.75 * GIB)
    s = R.pipeline_shares(per, _ranks(120, 84), other_bytes=GIB)
    assert sum(s["layers"]) == 48 and s["bounds"][1][0] == 0
    caps = [120 - 1 - R.step_margin(int(120 * GIB)) / GIB,
            84 - 1 - R.step_margin(int(84 * GIB)) / GIB]
    assert all(b / GIB <= c for b, c in zip(s["bytes"], caps))
    assert s["bounds"][0][1] == 48


def test_layer_bytes_keep_the_head_and_the_rest_out_of_the_layers():
    per, other = R.layer_bytes_of({
        "language_model.model.layers.0.mlp.gate.weight": 10,
        "language_model.model.layers.1.self_attn.q_proj.weight": 7,
        "language_model.model.layers.1.input_layernorm.weight": 1,
        "language_model.model.embed_tokens.weight": 100,
        "mtp.layers.0.mlp.gate.weight": 50,             # a grafted head
        "language_model.model.layers.2.mlp.gate.weight": 5,   # past L=2
    }, 2)
    assert per == [10, 8] and other == 155


# ------------------------------------------------------------ refusals

def test_pipeline_families():
    for mt in ("qwen3_5_moe", "qwen3_5", "glm5_next", "qwen4_exp"):
        assert R.pipeline_refusals({"model_type": mt, "text_config": {
            "num_hidden_layers": 40}}, 2) == []
    why = R.pipeline_refusals({"model_type": "gemma4", "text_config": {
        "model_type": "gemma4_text"}}, 2)
    assert "gemma4" in why[0] and "shares KV across layers" in why[0]
    why = R.pipeline_refusals({"model_type": "llama"}, 2)
    assert "'llama'" in why[0]
    assert "fewer than 3 ranks" in R.pipeline_refusals(
        {"model_type": "qwen3_5", "num_hidden_layers": 2}, 3)[0]
    assert R.pipeline_refusals({"model_type": "llama"}, 1) == []


def test_chip_bandwidth_is_known_only_for_unbinned_chips():
    assert R.chip_bandwidth_gbs("Apple M3 Ultra") == 819.0
    assert R.chip_bandwidth_gbs("Apple M4 Max") is None      # 410 or 546
    assert R.chip_bandwidth_gbs(None) is None


# ------------------------------------------------------ two processes

mx = pytest.importorskip("mlx.core")


def test_bounds_of_puts_the_first_layers_on_the_last_rank():
    from knurlogic.engine.runtime.pipeline import bounds_of
    assert bounds_of([1, 3]) == [(3, 4), (0, 3)]
    assert bounds_of([2, 1, 1]) == [(2, 4), (1, 2), (0, 1)]


def _ring(tmp_path, *args, timeout=180):
    import json
    import os
    import socket
    import subprocess
    import sys
    from pathlib import Path

    def free_port():
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        p = s.getsockname()[1]
        s.close()
        return p
    hosts = tmp_path / "hosts.json"
    hosts.write_text(json.dumps([[f"127.0.0.1:{free_port()}"],
                                 [f"127.0.0.1:{free_port()}"]]))
    here = Path(__file__).parent
    out = tmp_path / "out.json"
    env = dict(os.environ, MLX_HOSTFILE=str(hosts),
               PYTHONPATH=os.pathsep.join(
                   [str(here.parent / "src"), str(here)] + sys.path))
    env.pop("KNURLOGIC_MTP_BATCH_MAX_ROWS", None)
    procs = [subprocess.Popen(
        [sys.executable, str(here / "pipeline_ring_worker.py"), args[0],
         str(out), *args[1:]],
        env=dict(env, MLX_RANK=str(r)), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT) for r in range(2)]
    try:
        logs = [p.communicate(timeout=timeout)[0].decode() for p in procs]
    finally:
        for p in procs:
            p.kill()
    assert all(p.returncode == 0 for p in procs), "\n".join(logs)
    return json.loads(out.read_text())


@pytest.mark.parametrize("family,counts", [("qwen3_5_moe", "1,3"),
                                           ("qwen3_5_moe", "3,1"),
                                           ("qwen4_exp", "3,1"),
                                           ("glm5_next", "3,1"),
                                           ("glm5_next", "1,3")])
def test_a_two_rank_pipeline_computes_the_whole_models_logits(
        tmp_path, family, counts):
    """float32, so rounding cannot hide a wrong cut: the split model's
    logits (rank 0 holds the last layers) over a prefill and six decode
    steps are the unsplit model's -- the same arithmetic in the same order,
    only moved between processes."""
    import numpy as np
    d = _ring(tmp_path, "logits", family, counts)
    whole, split = np.array(d["whole"]), np.array(d["split"])
    assert whole.shape == split.shape == (7, whole.shape[1])
    assert np.abs(whole - split).max() < 1e-4, np.abs(whole - split).max()
    assert (whole.argmax(-1) == split.argmax(-1)).all()
    n0 = int(counts.split(",")[0])
    assert (d["info"]["start"], d["info"]["end"]) == (4 - n0, 4)
    assert d["info"]["dtypes"] == ["mlx.core.float32"] * 2


@pytest.mark.parametrize("always", [False, True])
def test_mtp_drafting_on_a_pipeline_is_the_unsplit_engine(tmp_path, always):
    """The tiny qwen3_5 with a random head (rejected almost every step:
    the replay path at its hardest; `always` drafts every step on a vocab of
    8, so accepts happen too): rank 0 drafts and samples, the follower's
    logits are zeros -- and every row's tokens are the unsplit engine's.
    Every rank made the same broadcasts: B1 once per decode step, B2 once
    per drafting step, B0 once per admission."""
    d = _ring(tmp_path, "mtp", "1" if always else "0")
    assert d["split"] == d["whole"]
    (b0, b1, b2, steps), follower = d["calls"]
    assert follower == [b0, b1, b2, steps]
    assert b0 == 3                                  # three admissions
    assert 0 < b2 <= b1 <= steps
    assert d["drafted"] > 0
    if always:
        assert b2 == b1 and d["accepted"] > 0


def test_the_serving_path_follows_rank_0s_plan_on_a_pipeline(tmp_path):
    """Rank 0's TensorExecutor (the step plan) and the follower's
    tensor.follow(split="pipeline"), drafting, two segments per prompt (a
    checkpoint each): rank 0 streams exactly the unsplit executor's tokens,
    and the follower stops when told."""
    d = _ring(tmp_path, "engine")
    assert d["split"] == d["whole"] and all(len(t) == 30 for t in d["whole"])


@pytest.mark.parametrize("split,fail", [("pipeline", "nan"),
                                        ("pipeline", "admit"),
                                        ("tensor", "nan"),
                                        ("tensor", "admit")])
def test_a_row_failing_on_rank_0_only_fails_that_row(tmp_path, split, fail):
    """Rank 0 alone fails one row (non-finite logits mid-decode, or an
    admission that raised after its forward): that row ends, the others
    stream every token, and the follower -- which still held the row --
    drops it from the next plan's remove instead of raising Desync (both
    processes exit 0; _ring asserts it)."""
    d = _ring(tmp_path, "engine", fail, split)
    lens = [len(t) for t in d["split"]]
    assert lens[0] == lens[2] == 30, lens
    assert lens[1] < 30, lens
