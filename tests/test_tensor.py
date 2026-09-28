"""Tensor split: the plan codec, the refusal and placement arithmetic, rank
order, the ring's prompt-cache journal, and the VQ codebook rule (a split
VQ layer's halves add up to the whole; a sliced codebook is caught)."""

import pytest

from knurlogic.engine.runtime import plan as P
from knurlogic.tuning import resolve as R


# ------------------------------------------------------------------ plan

def _admit(**kw):
    op = {"op": "admit", "uid": 3, "prompt": [1, 2, 3, 4, 5],
          "segs": [[4], [5]], "hit": 3, "max_tokens": 16,
          "sampling": {"temp": 0.7, "seed": 11},
          "penalties": {"repetition_penalty": 1.1}, "initial": "normal"}
    op.update(kw)
    return op


def test_plan_round_trips():
    plan = {"ops": [{"op": "pop", "n": 2}, _admit(),
                    {"op": "insert", "uid": 1, "event": "finished",
                     "kind": "assistant"},
                    {"op": "remove", "uids": [0, 2]}],
            "tokens": [[1, 42], [3, 7]]}
    data = P.encode(plan)
    assert isinstance(data, bytes) and data.startswith(b"{")
    assert P.decode(data) == plan
    assert P.encode(P.decode(data)) == data          # canonical bytes


def test_plan_carries_a_live_knob():
    plan = {"ops": [{"op": "set", "name": "VQ_DECODE_CHUNK", "value": "256"},
                    {"op": "park"}]}
    assert P.decode(P.encode(plan)) == plan


@pytest.mark.parametrize("op", [
    {"op": "set", "name": "KNURLOGIC_CONTEXT_LENGTH", "value": "8192"},
    {"op": "set", "name": "PATH", "value": "/tmp"},
    {"op": "set", "name": "VQ_DECODE_CHUNK", "value": 256},
    {"op": "set", "name": "VQ_DECODE_CHUNK"},
])
def test_plan_refuses_a_set_that_is_not_a_ranks_live_knob(op):
    with pytest.raises(P.PlanError):
        P.encode({"ops": [op]})


def test_the_knobs_that_travel_are_live_ones():
    from knurlogic.engine.serve.load import LIVE_KNOBS
    assert set(P.SETS) < set(LIVE_KNOBS)


def test_plan_empty_and_control():
    assert P.empty({"ops": []}) and P.empty({})
    assert not P.empty({"ops": [], "tokens": [[0, 1]]})
    assert P.control(-5, 9, 0) == [-5, 9, 0]
    assert P.CONTROL_LEN == 3


@pytest.mark.parametrize("bad", [
    b"not json",
    b'{"ops": [{"op": "launch"}]}',
    b'{"ops": [], "extra": 1}',
    b'{"ops": [{"op": "pop", "n": 0}]}',
    b'{"ops": [{"op": "remove"}]}',
    b'{"ops": [{"op": "insert", "uid": 1, "event": "x", "kind": "user"}]}',
    b'{"tokens": [[1]]}',
    b'{"tokens": [[1, "a"]]}',
])
def test_plan_refuses_what_is_not_a_plan(bad):
    with pytest.raises(P.PlanError):
        P.decode(bad)


def test_plan_admit_must_be_the_prompt_after_the_hit():
    with pytest.raises(P.PlanError):
        P.encode({"ops": [_admit(segs=[[4]])]})          # lost a token
    with pytest.raises(P.PlanError):
        P.encode({"ops": [_admit(hit=9)]})
    with pytest.raises(P.PlanError):
        P.encode({"ops": [_admit(prompt=[1, True, 3, 4, 5])]})


@pytest.mark.parametrize("bad", [{"hit": True}, {"hit": 1.0}, {"hit": "1"},
                                 {"segs": 5}, {"segs": None}, {"segs": "ab"}])
def test_plan_admit_types_are_plan_errors(bad):
    with pytest.raises(P.PlanError):
        P.encode({"ops": [_admit(**bad)]})


# ------------------------------------------------------------- refusals

QWEN36 = {"model_type": "qwen3_5_moe", "quantization": {"group_size": 64},
          "text_config": {
              "model_type": "qwen3_5_moe_text", "hidden_size": 2048,
              "num_attention_heads": 16, "num_key_value_heads": 2,
              "head_dim": 256, "linear_num_key_heads": 16,
              "linear_num_value_heads": 32, "linear_value_head_dim": 128,
              "num_experts": 256, "moe_intermediate_size": 512,
              "shared_expert_intermediate_size": 512},
          "vq_modules": {
              "language_model.model.layers.0.mlp.switch_mlp.down_proj":
                  {"experts": 256, "out": 2048, "in": 512, "k": 2048,
                   "dim": 4, "group": 64, "pack_bits": 11},
              "language_model.model.layers.0.mlp.switch_mlp.gate_proj":
                  {"experts": 256, "out": 512, "in": 2048, "k": 2048,
                   "dim": 4, "group": 64, "pack_bits": 11}}}


def test_the_35b_splits_two_ways():
    assert R.tensor_refusals(QWEN36, 2) == []
    assert R.tensor_refusals(QWEN36, 1) == []


def test_three_ways_is_refused_with_the_arithmetic():
    why = "\n".join(R.tensor_refusals(QWEN36, 3))
    assert "num_attention_heads = 16 is not divisible by 3" in why
    assert "16 / 3 = 5.33333" in why
    assert "num_key_value_heads = 2 is fewer than 3" in why
    assert "3 % 2 = 1" in why


def test_a_packed_down_proj_that_would_cut_a_word_is_refused():
    import copy
    cfg = copy.deepcopy(QWEN36)
    # 8 ranks: heads divide (16, 32, 16), kv repeats (8 % 2 == 0) -- but
    # down_proj's 512 inputs / 8 = 64 < 32 codes x dim 4 = 128
    cfg["text_config"]["linear_num_value_heads"] = 32
    why = R.tensor_refusals(cfg, 8)
    down = [w for w in why if "down_proj" in w and "switch_mlp" in w]
    assert down and "512 / 8 = 64" in down[0]
    assert "not a multiple of 128" in down[0]
    assert "32 x dim 4" in down[0] and "packed code word" in down[0]
    # the same layer unpacked needs only whole scale groups (64): fits
    for m in cfg["vq_modules"].values():
        m.pop("pack_bits")
    assert not [w for w in R.tensor_refusals(cfg, 8)
                if "switch_mlp.down_proj" in w]


def test_vq_dense_and_other_families_are_refused():
    cfg = dict(QWEN36, vq_linear={"a.self_attn.q_proj": {}})
    assert any("vq_linear" in w for w in R.tensor_refusals(cfg, 2))
    cfg = dict(QWEN36, vq_embed={"a.embed": {}})
    assert any("vq_embed" in w for w in R.tensor_refusals(cfg, 2))
    other = {"model_type": "glm5_next", "text_config": {}}
    assert "glm5_next" in R.tensor_refusals(other, 2)[0]


# ------------------------------------------------------------ placement

def test_placement_splits_layers_and_replicates_the_rest():
    t = {
        "language_model.model.embed_tokens.weight": 100,
        "language_model.lm_head.weight": 100,
        "language_model.model.layers.0.input_layernorm.weight": 2,
        "language_model.model.layers.0.mlp.gate.weight": 10,          # router
        "language_model.model.layers.0.mlp.shared_expert_gate.weight": 1,
        "language_model.model.layers.0.mlp.switch_mlp.down_proj.codes": 400,
        "language_model.model.layers.0.mlp.switch_mlp.down_proj.codebook": 16,
        "language_model.model.layers.0.mlp.switch_mlp.down_proj.vq_scales": 40,
        "language_model.model.layers.0.mlp.shared_expert.up_proj.weight": 60,
        "language_model.model.layers.0.linear_attn.in_proj_qkv.weight": 81,
        "language_model.model.layers.3.self_attn.o_proj.scales": 20,
    }
    p = R.tensor_placement_of(t, 2)
    assert p["sharded_bytes"] == 400 + 40 + 60 + 81 + 20
    assert p["replicated_bytes"] == 100 + 100 + 2 + 10 + 1 + 16
    assert p["per_rank_bytes"] == 301 + 229          # ceil(601 / 2) + 229
    assert not R.tensor_sharded("x.layers.0.mlp.switch_mlp.up_proj.codebook")


# ------------------------------------------------------------ rank order

M4 = {"name": "m4", "chip": "Apple M4 Max", "p_core_ghz": 4.5,
      "working_set_bytes": 120 << 30}
M3 = {"name": "m3", "chip": "Apple M3 Ultra", "p_core_ghz": 4.05,
      "working_set_bytes": 240 << 30}


def test_leader_is_the_fastest_single_core():
    assert R.rank_order([M3, M4]) == ["m4", "m3"]      # generation beats RAM
    assert R.chip_generation("Apple M4 Max") == 4
    assert R.chip_generation(None) == 0


def test_leader_ties_go_to_clock_then_free_memory_then_order():
    a = dict(M4, name="a", p_core_ghz=4.4)
    b = dict(M4, name="b")
    assert R.rank_order([a, b])[0] == "b"                  # higher clock
    c = dict(M4, name="c", free_bytes=10)
    d = dict(M4, name="d", free_bytes=20)
    assert R.rank_order([c, d])[0] == "d"                  # more free memory
    e, f = dict(M4, name="e"), dict(M4, name="f")
    assert R.rank_order([e, f]) == ["e", "f"]              # given order


def test_after_the_leader_the_ring_follows_the_links():
    lead = dict(M4, name="lead", links={"x": "wifi", "y": "tb5"})
    x = dict(M3, name="x", links={"lead": "wifi", "y": "ethernet"})
    y = dict(M3, name="y", chip="Apple M2 Ultra",
             links={"lead": "tb5", "x": "ethernet"})
    assert R.rank_order([x, y, lead]) == ["lead", "y", "x"]


def test_an_explicit_order_wins_and_must_name_everyone():
    assert R.rank_order([M4, M3], explicit=["m3", "m4"]) == ["m3", "m4"]
    with pytest.raises(ValueError):
        R.rank_order([M4, M3], explicit=["m3"])
    with pytest.raises(ValueError):
        R.rank_order([M4, dict(M4)])


# ------------------------------------------------- the ring's prompt cache

mx = pytest.importorskip("mlx.core")


class _Entry:
    def __init__(self, n):
        self.nbytes = n

    def is_trimmable(self):
        return False


def test_journal_cache_turns_byte_trims_into_counted_pops():
    from knurlogic.engine.runtime.scheduler import PromptCache
    from knurlogic.engine.runtime.tensor import Journal, JournalPromptCache
    j = Journal()
    c = JournalPromptCache(PromptCache(10), j)
    with pytest.raises(ValueError):
        c.insert("m", [1], [_Entry(1)], "user")        # no origin: refused
    for i in range(4):
        c.insert("m", [i, i + 1], [_Entry(100)], "assistant",
                 origin=("finished", i))
    assert c.nbytes == 400
    c.trim_to(250)
    ops = j.take()
    assert ops[:4] == [{"op": "insert", "uid": i, "event": "finished",
                        "kind": "assistant"} for i in range(4)]
    assert ops[4] == {"op": "pop", "n": 2} and c.nbytes == 200
    c.trim_to(1000)
    assert j.take() == []                      # nothing popped, nothing said

    # a follower applying the same ops holds the same entries
    mirror = PromptCache(10)
    for i in range(4):
        mirror.insert("m", [i, i + 1], [_Entry(100)], "assistant")
    mirror.lru.trim_to(n_sequences=len(mirror.lru) - 2)
    assert mirror.fetch("m", [3, 4, 9])[1] == [9]
    assert mirror.fetch("m", [0, 1, 9])[1] == [0, 1, 9]     # popped


def test_every_ring_row_gets_a_seed():
    from knurlogic.engine.runtime.tensor import assign_seed
    s = assign_seed({"temp": 0.5})
    assert isinstance(s["seed"], int) and s["temp"] == 0.5
    assert assign_seed({"seed": 7})["seed"] == 7


# --------------------------------------------------------- the codebook rule

E, OUT, IN, K, D, G = 4, 64, 256, 256, 4, 64


def _vq_layer(rt, out, inn, seed):
    mx.random.seed(seed)
    codes = mx.random.randint(0, K, (E, out, inn // D)).astype(mx.uint8)
    codebook = (mx.random.normal((K, D)) * 0.2).astype(mx.float16)
    scales = mx.random.uniform(0.5, 1.5, (E, out, inn // G)).astype(mx.float16)
    return rt.VQSwitchLinear(codes, codebook, scales)


def _halves(layer, kind):
    from knurlogic.engine.runtime.tensor import predicate, split_params
    import copy
    out = []
    for r in range(2):
        h = copy.copy(layer)
        h.update(split_params(layer.parameters(), predicate(kind), r, 2))
        out.append(h)
    return out


def test_the_predicate_never_splits_a_codebook():
    from knurlogic.engine.runtime.tensor import predicate
    w = mx.zeros((K, D))
    for kind in ("all-to-sharded", "sharded-to-all"):
        assert predicate(kind)("mlp.switch_mlp.down_proj.codebook", w) is None
    assert predicate("all-to-sharded")("x.codes", mx.zeros((E, OUT, 8))) == 1
    assert predicate("sharded-to-all")("x.codes", mx.zeros((E, OUT, 8))) == -1


def test_split_vq_down_proj_halves_add_up_to_the_whole():
    """sharded-to-all: each rank takes half the input; the sum of the two
    partial outputs (what all_sum makes) is the whole layer's output."""
    from knurlogic.engine.vq import runtime
    rt = runtime.runtime_module({})
    whole = _vq_layer(rt, OUT, IN, 1)
    x = (mx.random.normal((1, 1, IN)) * 0.5).astype(mx.float16)
    idx = mx.array([[[0, 2]]], dtype=mx.uint32)
    h0, h1 = _halves(whole, "sharded-to-all")
    assert h0.codebook.shape == (K, D) and h0.codes.shape[-1] == IN // D // 2
    y = whole(x, idx).astype(mx.float32)
    parts = (h0(x[..., :IN // 2], idx).astype(mx.float32)
             + h1(x[..., IN // 2:], idx).astype(mx.float32))
    assert mx.allclose(y, parts, atol=2e-2, rtol=1e-2).item()


def test_split_vq_gate_proj_halves_concatenate_to_the_whole():
    from knurlogic.engine.vq import runtime
    rt = runtime.runtime_module({})
    whole = _vq_layer(rt, OUT, IN, 2)
    x = (mx.random.normal((1, 1, IN)) * 0.5).astype(mx.float16)
    idx = mx.array([[[1, 3]]], dtype=mx.uint32)
    h0, h1 = _halves(whole, "all-to-sharded")
    y = whole(x, idx)
    cat = mx.concatenate([h0(x, idx), h1(x, idx)], axis=-1)
    assert mx.array_equal(y, cat).item()


def test_a_sliced_codebook_is_caught():
    """mlx's default all-to-sharded predicate cuts the codebook's K axis;
    the layer refuses to run on it, and check_codebooks names it."""
    import copy

    import mlx.nn as nn
    from knurlogic.engine.runtime.tensor import check_codebooks, split_params
    from knurlogic.engine.vq import runtime
    rt = runtime.runtime_module({})
    whole = _vq_layer(rt, OUT, IN, 3)
    naive = lambda p, w: max(w.ndim - 2, 0)          # noqa: E731 (mlx default)
    h = copy.copy(whole)
    h.update(split_params(whole.parameters(), naive, 0, 2))
    assert h.codebook.shape[0] == K // 2
    x = mx.zeros((1, 1, IN), dtype=mx.float16)
    with pytest.raises(RuntimeError, match="codebook was sharded"):
        h(x, mx.array([[[0]]], dtype=mx.uint32))

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = h
    with pytest.raises(RuntimeError, match="codebook was split"):
        check_codebooks(M())


def test_a_ring_serves_its_first_model_and_refuses_switching():
    from knurlogic.engine.runtime.scheduler import Command, Scheduler

    class H:
        state, path, error, model = "empty", None, "", None

        def expect(self, p):
            self.state, self.path = "loading", p

        def load(self, p, **k):
            self.state, self.path, self.model = "ready", p, object()

    class R:
        world, journal = 2, None
    s = Scheduler(H(), tensor=R())
    s.host.expect("/m/a")                   # what Scheduler.load does first
    first = Command("load", "/m/a")
    s._commands.put(first)
    s._do_commands()
    assert first.error == "" and s.host.state == "ready"
    for c in (Command("load", "/m/b"), Command("unload")):
        s._commands.put(c)
        s._do_commands()
        assert "split across 2 ranks" in c.error
    with pytest.raises(ValueError, match="count-based"):
        Scheduler(H(), tensor=R(), prompt_cache_bytes=1 << 30)


def test_a_two_rank_split_computes_the_whole_models_logits(tmp_path):
    """Two processes on 127.0.0.1, the tiny qwen3_5_moe split in two
    (float32, so rounding cannot hide a wrong split): the split model's
    logits over a prefill and six decode steps are the unsplit model's."""
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
    out = tmp_path / "logits.json"
    env = dict(os.environ, MLX_HOSTFILE=str(hosts),
               PYTHONPATH=os.pathsep.join(
                   [str(here.parent / "src"), str(here)] + sys.path))
    procs = [subprocess.Popen(
        [sys.executable, str(here / "tensor_ring_worker.py"), str(out)],
        env=dict(env, MLX_RANK=str(r)), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT) for r in range(2)]
    try:
        logs = [p.communicate(timeout=120)[0].decode() for p in procs]
    finally:
        for p in procs:
            p.kill()
    assert all(p.returncode == 0 for p in procs), "\n".join(logs)
    import numpy as np
    d = json.loads(out.read_text())
    whole, split = np.array(d["whole"]), np.array(d["split"])
    assert whole.shape == split.shape == (7, whole.shape[1])
    assert np.abs(whole - split).max() < 1e-3, np.abs(whole - split).max()
    assert (whole.argmax(-1) == split.argmax(-1)).all()


def test_rank_0s_frees_never_lower_the_peers_estimate(monkeypatch):
    """Tensor: what rank 0 took since the exchange raises the peers'
    over-limit, what it freed does not lower it. Pipeline (unequal
    stages): the peers' own number, untouched by rank 0's memory."""
    import mlx.core as mx
    from types import SimpleNamespace
    from knurlogic.engine.runtime import tensor as T
    mem = {"a": 50}
    monkeypatch.setattr(mx, "get_active_memory", lambda: mem["a"])
    for split, took, freed in (("tensor", 17, 7), ("pipeline", 7, 7)):
        r = T.Ring(SimpleNamespace(size=2), split=split)
        r.peer_over, r.local_then = 7, 50
        mem["a"] = 60
        assert r.peers_over_now() == took
        mem["a"] = 30
        assert r.peers_over_now() == freed


class _FakeLink:
    """follow()'s side of the ring, replayed: one plan per exchange."""

    def __init__(self, plans):
        self.plans, self.rank, self.slept = list(plans), 1, 0

    def exchange(self, over, payload):
        return [], P.encode(self.plans.pop(0))

    def sleep(self):
        self.slept += 1


def test_a_follower_applies_a_set_even_while_parked(monkeypatch):
    import importlib
    from knurlogic.engine.runtime import tensor as T
    load = importlib.import_module("knurlogic.engine.serve.load")
    got = []
    monkeypatch.setattr(load, "apply_live",
                        lambda env: got.append(env) or {
                            k: "applied" for k in env})
    link = _FakeLink([
        {"ops": [{"op": "set", "name": "KNURLOGIC_CACHE_LIMIT_GB",
                  "value": "20"}, {"op": "park"}]},
        {"ops": [{"op": "stop"}]}])
    assert T.follow(None, None, "m", link, prompt_cache_size=2,
                    completion_batch_size=1, prefill_step_size=512,
                    working_set=0) == 0
    assert got == [{"KNURLOGIC_CACHE_LIMIT_GB": "20"}] and link.slept == 1


def test_a_set_journaled_while_parked_rings_the_ring():
    """Rank 0 idle, the others parked: a Settings apply is journaled on the
    scheduler thread and the next park() rings them to take it."""
    from knurlogic.engine.runtime import tensor as T
    from knurlogic.engine.runtime.scheduler import Scheduler

    sent = []

    class L:
        size, socks, parked = 2, [object()], True

        def exchange(self, over, payload):
            sent.append(P.decode(payload))
            return [[0, 0, 0], [0, 0, 0]], None

    class H:
        state, path, error, model = "ready", "/m/a", "", object()

    ring = T.Ring(L())
    s = Scheduler(H(), tensor=ring)
    ring.park()
    assert sent == []                       # parked, nothing new: asleep
    s.share_live({"VQ_DECODE_CHUNK": "128",
                  "KNURLOGIC_CONTEXT_LENGTH": "4096"})
    s._journal_sets()
    ring.park()
    assert sent == [{"ops": [{"op": "set", "name": "VQ_DECODE_CHUNK",
                              "value": "128"}, {"op": "park"}]}]
    assert ring.link.parked

def _bell_server():
    import socket
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    port = srv.getsockname()[1]
    srv.close()
    return port


def test_a_rank_dialing_before_rank_0_listens_keeps_trying(caplog):
    """The bell: a rank that dials before rank 0's socket is up retries
    until it is, and rank 0 logs which rank came from where."""
    import logging
    import socket
    import threading
    import time
    from knurlogic.engine.runtime import tensor as T
    port = _bell_server()
    got = {}

    def dial():
        got["c"] = T.bell_dial("127.0.0.1", port, 1234, 1, 10.0,
                               pause_s=0.05)
    t = threading.Thread(target=dial)
    t.start()
    time.sleep(0.4)                          # refused a few times meanwhile
    srv = socket.create_server(("127.0.0.1", port))
    with caplog.at_level(logging.INFO, logger=T.__name__):
        socks = T.bell_answer(srv, "127.0.0.1", 1234, 2, 10.0)
    t.join(5)
    assert len(socks) == 1
    socks[0].sendall(b"w")
    assert got["c"].recv(1) == b"w"
    assert "bell: rank 1 connected from 127.0.0.1" in caplog.text
    for c in socks + [got["c"]]:
        c.close()


def test_rank_0s_bell_names_the_ranks_that_never_came():
    """Rank 0 of three: rank 2 connects, a stranger with the wrong nonce is
    turned away, rank 1 never comes -- the error says the address and
    rank 1, and not rank 2."""
    import socket
    import threading
    import pytest
    from knurlogic.engine.runtime import tensor as T
    srv = socket.create_server(("127.0.0.1", 0))
    port = srv.getsockname()[1]
    held = []

    def others():
        held.append(T.bell_dial("127.0.0.1", port, 99, 2, 5.0))
        s = socket.create_connection(("127.0.0.1", port))
        s.sendall(b"\0" * 16)
        held.append(s)
    t = threading.Thread(target=others)
    t.start()
    with pytest.raises(TimeoutError) as e:
        T.bell_answer(srv, "127.0.0.1", 99, 3, 1.0)
    t.join(5)
    msg = str(e.value)
    assert f"127.0.0.1:{port}" in msg and "[1]" in msg
    assert "connected: [2]" in msg
    for c in held:
        c.close()


def test_a_rank_that_cannot_reach_rank_0_says_where_it_dialed():
    import pytest
    from knurlogic.engine.runtime import tensor as T
    port = _bell_server()
    with pytest.raises(ConnectionError, match=f"rank 1 could not reach rank "
                       f"0 at 127.0.0.1:{port}"):
        T.bell_dial("127.0.0.1", port, 1, 1, 0.3, pause_s=0.05)
