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
    # by bytes (36, equal ranks): rank 1 takes 3+1+4+1+5 = 14 (the 9 would
    # make it 23, further from 18); rank 0 the other 22
    assert a["layers"] == [4, 5]
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
    # holds a 41.7 GiB n-gram embedding. By count the M3 Ultra (rank 1,
    # first layers) would get 18 layers = 79 GiB against 72.8 it holds, and
    # the model would be refused though it fits the pair with room
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


def test_what_rank_0_alone_holds_counts_on_rank_0_alone():
    """The MTP head and the vision tower live on rank 0 only: their bytes
    come off rank 0's room for layers, not every rank's."""
    per = [GIB] * 40
    even = R.pipeline_shares(per, _ranks(104, 104), other_bytes=4 * GIB)
    lead = R.pipeline_shares(per, _ranks(104, 104), other_bytes=4 * GIB,
                             leader_bytes=10 * GIB)
    assert even["layers"] == [20, 20]
    assert lead["layers"][0] < 20 < lead["layers"][1]
    assert "leaves" in lead["reason"]
    with pytest.raises(ValueError, match="MTP head"):
        R.pipeline_shares([GIB] * 4, _ranks(10, 104), other_bytes=GIB,
                          leader_bytes=5 * GIB)


def test_the_head_and_tower_are_not_in_the_replicated_bytes(tmp_path):
    import json
    import struct

    def shard(path, tensors):
        header = {}
        at = 0
        for k, n in tensors.items():
            header[k] = {"dtype": "U8", "shape": [n],
                         "data_offsets": [at, at + n]}
            at += n
        h = json.dumps(header).encode()
        path.write_bytes(struct.pack("<Q", len(h)) + h + b"\0" * at)
    shard(tmp_path / "model.safetensors", {
        "language_model.model.layers.0.mlp.weight": 10,
        "language_model.model.layers.1.mlp.weight": 10,
        "language_model.model.embed_tokens.weight": 100,
        "vision_tower.blocks.0.attn.qkv.weight": 30})
    shard(tmp_path / "mtp.safetensors", {"mtp.layers.0.mlp.weight": 50})

    class A:
        path = tmp_path
        raw_config = {"text_config": {"num_hidden_layers": 2}}
    per, other = R.pipeline_layer_bytes(A)
    assert per == [10, 10] and other == 100
    assert R.pipeline_leader_bytes(A) == 50 + 30     # head + tower


# ------------------------------------------------------------ refusals

def test_pipeline_families():
    for mt in ("qwen3_5_moe", "qwen3_5", "glm5_next", "qwen4_exp",
               "deepseek_v4"):
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
    here = Path(__file__).resolve().parents[1] / "support"
    out = tmp_path / "out.json"
    env = dict(os.environ, MLX_HOSTFILE=str(hosts),
               PYTHONPATH=os.pathsep.join(
                   [str(here.parents[1] / "src"), str(here)] + sys.path))
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
                                           ("glm5_next", "1,3"),
                                           ("deepseek_v4", "3,1"),
                                           ("deepseek_v4", "1,3")])
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


@pytest.mark.parametrize("family,counts", [("qwen3_5_moe", "2,2"),
                                           ("qwen4_exp", "3,1"),
                                           ("glm5_next", "3,1")])
def test_a_pipeline_over_an_8_bit_kv_cache_is_the_whole_model(
        tmp_path, monkeypatch, family, counts):
    """KNURLOGIC_KV_BITS=8 on both ranks: the split model's logits are the
    unsplit quantized model's. qwen3_5_moe 2,2 puts an attention layer last
    on the follower, whose cache write hangs on the send (mx.depends over
    the quantized triple)."""
    import numpy as np
    monkeypatch.setenv("KNURLOGIC_KV_BITS", "8")
    d = _ring(tmp_path, "logits", family, counts)
    whole, split = np.array(d["whole"]), np.array(d["split"])
    assert np.abs(whole - split).max() < 1e-4, np.abs(whole - split).max()
    assert (whole.argmax(-1) == split.argmax(-1)).all()


@pytest.mark.parametrize("always", [False, True])
def test_mtp_drafting_on_a_pipeline_is_the_unsplit_engine(tmp_path, always):
    """The tiny qwen3_5 with a random head (rejected almost every step:
    the replay path at its hardest; `always` drafts every step on a vocab of
    8, so accepts happen too): rank 0 alone holds the head, drafts and
    samples; the follower holds none and its logits are zeros -- and every
    row's tokens are the unsplit engine's. Every rank made the same
    broadcasts: B1 once per decode step, B2 once per drafting step, B0 and
    BA once per admission."""
    d = _ring(tmp_path, "mtp", "1" if always else "0")
    assert d["split"] == d["whole"]
    (b0, b1, b2, steps, ba), follower = d["calls"]
    assert follower == [b0, b1, b2, steps, ba]
    assert b0 == ba == 3                            # three admissions
    # the follower's prefill chunks (16 tokens; prompts of 37, 9, 70) were
    # sent while the next one computed; rank 0 sends nothing
    assert d["overlapped"][0] == 0 and d["overlapped"][1] > 0
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


def test_a_prefix_only_the_follower_could_use_is_prefilled_on_every_rank(
        tmp_path):
    """Rank 0 alone holds the head, so only rank 0 can tell whether a
    prompt-cache entry is usable for a drafting row: the follower takes
    rank 0's answer (BA) and prefills from scratch with it -- the second
    prompt's tokens are the unsplit engine's, and the ring stays in step."""
    d = _ring(tmp_path, "hit")
    assert d["hit"] == 5
    assert d["split"] == d["whole"]
    assert all(len(t) == 20 for t in d["whole"])


@pytest.mark.parametrize("split", ["pipeline", "tensor"])
def test_images_on_a_split_model_are_the_unsplit_engines(tmp_path, split):
    """Two images in a prompt, rank 0 alone holding the tower: every row's
    tokens are the unsplit engine's, the rows travelled once (the first
    prompt; the second's images are inside its prompt-cache hit), and the
    follower's trie hit the same image prefix as rank 0's (follow raises
    Desync if its hit differs)."""
    d = _ring(tmp_path, "image", split, timeout=300)
    assert d["hit"] == d["cut"]
    assert d["images"] == 1
    assert d["split"] == d["whole"]
    assert all(len(t) == 10 for t in d["whole"])


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


def test_a_heavy_first_layer_is_balanced_by_bytes_not_count():
    # Qwen3.8 Flash: layer 1 carries a 42 GiB n-gram embedding. Counted, the
    # smaller M3 Ultra takes 19 layers = 63.5 GiB with 13 GiB to spare while
    # the M4 Max keeps 70; by bytes each rank fills about the same fraction.
    per = [0.5 * GIB, 42 * GIB] + [1.4 * GIB] * 46
    s = R.pipeline_shares(per, _ranks(120, 84))
    fill = [s["bytes"][i] / (w * GIB) for i, w in enumerate((120, 84))]
    assert abs(fill[0] - fill[1]) < 0.15, s["reason"]


# ------------------------------------------ a stage's wrapped layers, one process

def _tiny(family):
    import pipeline_ring_worker as W
    return W.build(family)


def _stage(model, start, end, *, recv, send):
    """`model` as the pipeline stage holding layers [start, end), its ends
    wrapped the way split() wraps them (no ring: nothing is called)."""
    from knurlogic.engine.runtime import pipeline as PL
    keep = list(PL.core_of(model).layers)[start:end]
    if recv:
        keep[0] = PL.Recv(keep[0], 1, None, mx.float32)
    if send:
        keep[-1] = PL.Send(keep[-1], 0, None, mx.float32)
    PL.restage(model, keep, start, end)
    return model


def test_unwrap_sees_through_both_ends_of_a_one_layer_stage():
    import mlx.nn as nn
    from knurlogic.engine.runtime import pipeline as PL
    lin = nn.Linear(2, 2)
    both = PL.Send(PL.Recv(lin, 1, None, mx.float32), 0, None, mx.float32)
    assert PL.unwrap(both) is lin and PL.unwrap(lin) is lin
    assert type(both) is not nn.Linear and both.weight is lin.weight


def test_overlap_finds_a_stages_send_and_is_undone_on_the_way_out(
        monkeypatch):
    """`overlapped` turns on the stage's Send (found through a one-layer
    stage's Recv) for the prefill chunks only, and KNURLOGIC_PIPELINE_
    OVERLAP=off leaves every send synchronous (the A/B)."""
    import types
    import mlx.nn as nn
    from knurlogic.engine.runtime import pipeline as PL
    send = PL.Send(PL.Recv(nn.Linear(2, 2), 1, None, mx.float32), 0, None,
                   mx.float32)
    model = types.SimpleNamespace(model=types.SimpleNamespace(
        layers=[nn.Linear(2, 2), send]))
    assert PL.sends_of(model) == [send]
    with PL.overlapped(model):
        assert send.overlap
    assert not send.overlap
    monkeypatch.setenv("KNURLOGIC_PIPELINE_OVERLAP", "off")
    with PL.overlapped(model):
        assert not send.overlap


@pytest.mark.parametrize("start,end,recv,send", [(0, 3, False, True),
                                                 (1, 4, True, False),
                                                 (1, 2, True, True)])
def test_the_flash_next_head_binds_on_any_stage(start, end, recv, send):
    """qwen4_exp's head is a full-attention block of the layers' own class,
    built by the global index -- on a stage whose first layer is rank 0's
    Recv, and on a follower's stage holding no full-attention layer at all
    (the tiny model's only one is layer 3). Before: TypeError from building
    a Recv, then an IndexError / a linear-attention block -- the head did
    not bind and MTP was off on every rank of the pipeline."""
    import importlib
    from knurlogic.engine.families.qwen.heads.qwen4_exp import MTPHead
    from knurlogic.engine.runtime import pipeline as PL
    model = _stage(_tiny("qwen4_exp"), start, end, recv=recv, send=send)
    core = PL.core_of(model)
    arch = importlib.import_module(type(core).__module__)
    head = MTPHead(model, arch)
    assert type(head.block) is arch.DecoderLayer
    assert head.block.layer_type == "full_attention"
    assert head.fa_idx == core.args.layer_types.index("full_attention") == 3
    # and it drafts: random glue, the block's own random weights
    D, hc = head.D, head.hc
    head.norm_e = head._norm(D, mx.ones((D,)))
    head.norm_h = head._norm(hc * D, mx.ones((hc * D,)), group_size=D)
    head.fc = 0.02 * mx.random.normal((D, 2 * D))
    out = head.draft_logits(mx.random.normal((1, 1, hc * D)),
                            mx.array([[5]]))
    mx.eval(out)
    assert out.shape[:2] == (1, 1) and bool(mx.isfinite(out).all())


@pytest.mark.parametrize("family", ["qwen3_5_moe", "glm5_next"])
def test_restage_reads_the_layer_kinds_through_the_stage_ends(family):
    """The per-family indices are read through a wrapped first / last layer:
    the tiny models' layers 0-2 are linear and layer 3 full attention, so
    the stage [2, 4) with both ends wrapped has its recurrent cache at 0 and
    its attention cache at 1."""
    from knurlogic.engine.runtime import pipeline as PL
    model = _stage(_tiny(family), 2, 4, recv=True, send=True)
    core = PL.core_of(model)
    assert (core.ssm_idx, core.fa_idx) == (0, 1)
    assert isinstance(core.layers[0], PL.Recv)
    assert isinstance(core.layers[1], PL.Send)
