"""Tensor split: the plan codec, the refusal and placement arithmetic, rank
order, the ring's prompt-cache journal, and the VQ codebook rule (a split
VQ layer's halves add up to the whole; a sliced codebook is caught)."""

import pytest

from knurlogic.engine.runtime import plan as P
from knurlogic.tuning import rank_order, tensor_split

# ------------------------------------------------------------------ plan

def _admit(**kw):
    op = {"op": "admit", "uid": 3, "prompt": [1, 2, 3, 4, 5],
          "segs": [[4], [5]], "hit": 3, "max_tokens": 16,
          "sampling": {"temp": 0.7, "seed": 11},
          "penalties": {"repetition_penalty": 1.1}, "initial": "normal",
          "images": [], "refs": [], "chunk": 512}
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
    from knurlogic.tuning.live import LIVE_KNOBS
    assert set(P.SETS) < set(LIVE_KNOBS)


def test_plan_empty_and_control():
    assert P.empty({"ops": []}) and P.empty({})
    assert not P.empty({"ops": [], "tokens": [[0, 1]]})
    assert P.control(-5, 9, 0) == [-5, 9, 0, 0, 0]
    assert P.control(-5, 9, 0, 11, 12)[P.ACTIVE:] == [11, 12]
    assert P.CONTROL_LEN == 5


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


def test_an_image_key_travels_and_comes_back_as_rank_0s_key():
    """A follower's prompt cache is keyed by the same sentinels as rank
    0's: the admit op carries the key as ids plus its image runs and refs,
    and rebuilds it exactly -- a run the hit cut into (k0 > 0) and the same
    image twice included."""
    from knurlogic.engine.vision import key as K
    a = [("img", "aa", "p1", k) for k in range(3)]
    b = [("img", "bb", "p1", k) for k in range(2)]
    key = [1, 2] + a + [3] + b + a + [4]
    grids = {"aa": (1, 2, 6), "bb": (1, 2, 4)}
    ids, images, refs = P.key_to_wire(
        key, lambda sha, ph: ({"aa": 3, "bb": 2}[sha], grids[sha]))
    assert ids[:2] == [1, 2] and ids[2] == -1 and ids[5] == 3
    assert refs == [["aa", "p1", 3, [1, 2, 6]], ["bb", "p1", 2, [1, 2, 4]]]
    assert images == [[2, 5, 0, 0], [6, 8, 1, 0], [8, 11, 0, 0]]
    op = _admit(prompt=ids, segs=[ids[4:]], hit=4, images=images, refs=refs)
    back = P.decode(P.encode({"ops": [op]}))["ops"][0]
    assert P.key_from_wire(back["prompt"], back["images"],
                           back["refs"]) == key
    # a key sliced inside an image keeps its k0
    cut = key[3:]
    ids, images, refs = P.key_to_wire(cut, lambda s, p: (3 if s == "aa"
                                                        else 2, None))
    assert images[0] == [0, 2, 0, 1]
    assert P.key_from_wire(ids, images, refs) == cut
    assert P.key_to_wire([5, 6], None) == ([5, 6], [], [])
    assert K.has_image(P.key_from_wire(ids, images, refs))


@pytest.mark.parametrize("images,refs", [
    ([[0, 9, 0, 0]], [["aa", "p", 3, None]]),        # past the prompt
    ([[0, 2, 1, 0]], [["aa", "p", 3, None]]),        # no such ref
    ([[0, 3, 0, 1]], [["aa", "p", 3, None]]),        # past the image
    ([[0, 2, 0, 0]], [["aa", "p", "3", None]]),      # n_tokens not an int
])
def test_plan_refuses_an_image_run_that_does_not_fit(images, refs):
    with pytest.raises(P.PlanError):
        P.encode({"ops": [_admit(images=images, refs=refs)]})


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
    assert tensor_split.tensor_refusals(QWEN36, 2) == []
    assert tensor_split.tensor_refusals(QWEN36, 1) == []


def test_three_ways_is_refused_with_the_arithmetic():
    why = "\n".join(tensor_split.tensor_refusals(QWEN36, 3))
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
    why = tensor_split.tensor_refusals(cfg, 8)
    down = [w for w in why if "down_proj" in w and "switch_mlp" in w]
    assert down and "512 / 8 = 64" in down[0]
    assert "not a multiple of 128" in down[0]
    assert "32 x dim 4" in down[0] and "packed code word" in down[0]
    # the same layer unpacked needs only whole scale groups (64): fits
    for m in cfg["vq_modules"].values():
        m.pop("pack_bits")
    assert not [w for w in tensor_split.tensor_refusals(cfg, 8)
                if "switch_mlp.down_proj" in w]


def test_a_deepseek_v4_split_must_keep_whole_rounding_blocks():
    """deepseek_v4 rounds each FP8 / FP4 linear's input in blocks of 128
    (architecture edit 21); a split whose slice of wo_b's input or an
    expert's width cuts a block is refused, with its numbers. Flash's and
    Vision-Exp's shapes split 2, 4 and 8 ways."""
    import json
    from pathlib import Path
    flash = {"model_type": "deepseek_v4", "num_attention_heads": 64,
             "num_key_value_heads": 1, "o_groups": 8, "o_lora_rank": 1024,
             "moe_intermediate_size": 2048, "n_shared_experts": 1}
    for n in (2, 4, 8):
        assert tensor_split.tensor_refusals(flash, n) == [], n
    # the released Vision-Exp config.json (deepseek-ai, MIT), copied in
    real = Path(__file__).resolve().parents[1] / "support" \
        / "fixtures_deepseek_v4_vision" / "config.json"
    cfg = json.loads(real.read_text())
    for n in (2, 4, 8):
        assert tensor_split.tensor_refusals(cfg, n) == [], n
    tiny = dict(flash, num_attention_heads=4, o_groups=2, o_lora_rank=64,
                moe_intermediate_size=128)
    why = "\n".join(tensor_split.tensor_refusals(tiny, 2))
    assert "wo_b's input" in why and "a rank's 64 is not whole 128-blocks" \
        in why
    assert "routed experts' down_proj input" in why
    assert "act_quant" in why
    # the DSpark goldens' shapes: whole blocks on two ranks
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "support"
                           / "goldens"))
    from build_deepseek_v4_dspark import CONFIG
    assert tensor_split.tensor_refusals(CONFIG, 2) == []


def test_vq_dense_and_other_families_are_refused():
    cfg = dict(QWEN36, vq_linear={"a.self_attn.q_proj": {}})
    assert any("vq_linear" in w for w in tensor_split.tensor_refusals(cfg, 2))
    cfg = dict(QWEN36, vq_embed={"a.embed": {}})
    assert any("vq_embed" in w for w in tensor_split.tensor_refusals(cfg, 2))
    other = {"model_type": "glm5_next", "text_config": {}}
    assert "glm5_next" in tensor_split.tensor_refusals(other, 2)[0]


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
    p = tensor_split.tensor_placement_of(t, 2)
    assert p["sharded_bytes"] == 400 + 40 + 60 + 81 + 20
    assert p["replicated_bytes"] == 100 + 100 + 2 + 10 + 1 + 16
    assert p["per_rank_bytes"] == 301 + 229          # ceil(601 / 2) + 229
    assert not tensor_split.tensor_sharded("x.layers.0.mlp.switch_mlp.up_proj.codebook")


# ------------------------------------------------------------ rank order

M4 = {"name": "m4", "chip": "Apple M4 Max", "p_core_ghz": 4.5,
      "working_set_bytes": 120 << 30}
M3 = {"name": "m3", "chip": "Apple M3 Ultra", "p_core_ghz": 4.05,
      "working_set_bytes": 240 << 30}


def test_leader_is_the_fastest_single_core():
    assert rank_order.rank_order([M3, M4]) == ["m4", "m3"]      # generation beats RAM
    assert rank_order.chip_generation("Apple M4 Max") == 4
    assert rank_order.chip_generation(None) == 0


def test_leader_ties_go_to_clock_then_free_memory_then_order():
    a = dict(M4, name="a", p_core_ghz=4.4)
    b = dict(M4, name="b")
    assert rank_order.rank_order([a, b])[0] == "b"                  # higher clock
    c = dict(M4, name="c", free_bytes=10)
    d = dict(M4, name="d", free_bytes=20)
    assert rank_order.rank_order([c, d])[0] == "d"                  # more free memory
    e, f = dict(M4, name="e"), dict(M4, name="f")
    assert rank_order.rank_order([e, f]) == ["e", "f"]              # given order


def test_after_the_leader_the_ring_follows_the_links():
    lead = dict(M4, name="lead", links={"x": "wifi", "y": "tb5"})
    x = dict(M3, name="x", links={"lead": "wifi", "y": "ethernet"})
    y = dict(M3, name="y", chip="Apple M2 Ultra",
             links={"lead": "tb5", "x": "ethernet"})
    assert rank_order.rank_order([x, y, lead]) == ["lead", "y", "x"]


def test_an_explicit_order_wins_and_must_name_everyone():
    assert rank_order.rank_order([M4, M3], explicit=["m3", "m4"]) == ["m3", "m4"]
    with pytest.raises(ValueError):
        rank_order.rank_order([M4, M3], explicit=["m3"])
    with pytest.raises(ValueError):
        rank_order.rank_order([M4, dict(M4)])


# ------------------------------------------------- the ring's prompt cache

mx = pytest.importorskip("mlx.core")


class _Entry:
    def __init__(self, n):
        self.nbytes = n

    def is_trimmable(self):
        return False


def test_journal_cache_turns_byte_trims_into_counted_pops():
    from knurlogic.engine.prompt_cache.memory import PromptCache
    from knurlogic.engine.prompt_cache.ring import JournalPromptCache
    from knurlogic.engine.runtime.tensor import Journal
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


def test_the_predicate_never_splits_a_codebook():
    from knurlogic.engine.runtime.tensor_rules import predicate
    w = mx.zeros((K, D))
    for kind in ("all-to-sharded", "sharded-to-all"):
        assert predicate(kind)("mlp.switch_mlp.down_proj.codebook", w) is None
    assert predicate("all-to-sharded")("x.codes", mx.zeros((E, OUT, 8))) == 1
    assert predicate("sharded-to-all")("x.codes", mx.zeros((E, OUT, 8))) == -1


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


@pytest.mark.parametrize("family,kv_bits", [("qwen3_5_moe", "bf16"),
                                            ("qwen3_5_moe", "8"),
                                            ("qwen4_exp", "bf16"),
                                            ("qwen4_exp", "8"),
                                            ("deepseek_v4", "bf16")])
def test_a_two_rank_split_computes_the_whole_models_logits(tmp_path, family,
                                                           kv_bits):
    """Two processes on 127.0.0.1, the tiny model split in two (float32,
    so rounding cannot hide a wrong split): the split model's logits over a
    prefill and six decode steps are the unsplit model's. qwen4_exp deals
    its n-gram table's parts between the ranks and sums the lookup;
    deepseek_v4 cuts its heads (whole o_groups) and experts and runs its
    shared kv, compressors, indexers and hyper-connections whole.
    At 8-bit KV too: groups run along a head's dims, so a rank's half of
    the heads quantizes as it does in the whole cache (up to a last-bit
    difference in the sharded projections that feed it)."""
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
    out = tmp_path / "logits.json"
    env = dict(os.environ, MLX_HOSTFILE=str(hosts), KNURLOGIC_KV_BITS=kv_bits,
               PYTHONPATH=os.pathsep.join(
                   [str(here.parents[1] / "src"), str(here)] + sys.path))
    procs = [subprocess.Popen(
        [sys.executable, str(here / "tensor_ring_worker.py"), str(out),
         family],
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
    # 8-bit: the sharded projections round differently in the last float32
    # bit, which can move an element across a quantization step (~0.4% of
    # its group's range) -- bounded by the 8-bit gate of test_kvquant, 1%
    # of the largest |logit|, not bit-equal
    tol = 1e-3 if kv_bits == "bf16" else 0.01 * np.abs(whole).max()
    assert np.abs(whole - split).max() < tol, np.abs(whole - split).max()
    assert (whole.argmax(-1) == split.argmax(-1)).all()


def test_rank_0s_frees_never_lower_the_peers_estimate(monkeypatch):
    """Tensor: what rank 0 took since the exchange raises the peers'
    over-limit, what it freed does not lower it. Pipeline (unequal
    stages): the peers' own number, untouched by rank 0's memory."""
    from types import SimpleNamespace

    import mlx.core as mx

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


def test_rank_0_publishes_every_ranks_memory(monkeypatch):
    """A pipeline follower serves no /status.json: its memory comes back
    in the control rows and rank 0 publishes it (`ranks`)."""
    from types import SimpleNamespace

    from knurlogic.engine.runtime import tensor as T
    from knurlogic.engine.serve import state
    monkeypatch.setitem(state.SERVED, "ranks", None)
    rows = [P.control(0, 3, 0, 100, 150), P.control(-4, 3, 0, 200, 260)]
    link = SimpleNamespace(size=2, exchange=lambda over, payload: (rows, None))
    r = T.Ring(link, split="pipeline")
    r.exchange(0, {"ops": []})
    assert state.SERVED["ranks"] == [
        {"rank": 0, "active_bytes": 100, "peak_bytes": 150,
         "over_limit_bytes": 0},
        {"rank": 1, "active_bytes": 200, "peak_bytes": 260,
         "over_limit_bytes": -4}]
    assert r.peer_over == -4


def test_a_parked_follower_holds_no_margin_a_long_prefill_measured(
        monkeypatch):
    """A pipeline follower's limit leaves the largest transient it ever
    measured. One 41k-token prefill's put it past its own active memory for
    good: its over-limit stayed positive, rank 0 read the pipeline as over
    its limit with nothing running, and refused every prompt. Parked, a
    rank holds no row to step: it reports against the floor margin."""
    import mlx.core as mx

    from knurlogic.engine.runtime import tensor as T
    GIB = T.GIB
    monkeypatch.setattr(mx, "get_active_memory", lambda: 60 * GIB)
    seen = []

    class Mark(T.Mark):
        def __init__(self, ws):
            super().__init__(ws)
            self.spike = 40 * GIB       # measured by the long prefill

    class Link(_FakeLink):
        def exchange(self, over, payload):
            seen.append(over)
            return super().exchange(over, payload)

    monkeypatch.setattr(T, "Mark", Mark)
    link = Link([{"ops": [{"op": "park"}]}, {"ops": [{"op": "stop"}]}])
    T.follow(None, None, "m", link, prompt_cache_size=2,
             completion_batch_size=1, prefill_step_size=512,
             working_set=96 * GIB)
    assert seen[0] > 0                  # 60 held, limit 96 - 50 = 46
    assert seen[1] < 0                  # parked: limit 96 - 5 = 91


# ------------------------------------------------- the headers, one rule set

def _artifact(tmp_path, cfg, shapes):
    """A config.json and one header-only safetensors file: the refusals
    read 8 bytes and the JSON, never the data."""
    import json
    import struct
    head, off = {}, 0
    for k, shp in shapes.items():
        n = 1
        for d in shp:
            n *= d
        head[k] = {"dtype": "U8", "shape": list(shp),
                   "data_offsets": [off, off + n]}
        off += n
    raw = json.dumps(head).encode()
    (tmp_path / "model.safetensors").write_bytes(
        struct.pack("<Q", len(raw)) + raw)
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return tmp_path


def _layer0(proj: dict) -> dict:
    """QWEN36's layer 0 (linear attention, MoE), switch_mlp per `proj`."""
    pre = "language_model.model.layers.0."
    out = {pre + "linear_attn.conv1d.weight": (8192, 4, 1),
           pre + "linear_attn.in_proj_qkv.weight": (8192, 256),
           pre + "linear_attn.in_proj_qkv.scales": (8192, 32),
           pre + "linear_attn.in_proj_qkv.biases": (8192, 32),
           pre + "linear_attn.dt_bias": (32,),
           pre + "linear_attn.out_proj.weight": (2048, 768),
           pre + "linear_attn.out_proj.scales": (2048, 64),
           pre + "linear_attn.out_proj.biases": (2048, 64),
           pre + "mlp.gate.weight": (256, 2048),
           pre + "mlp.shared_expert.down_proj.weight": (2048, 96),
           pre + "mlp.shared_expert.down_proj.scales": (2048, 8)}
    for k, shp in proj.items():
        out[pre + "mlp.switch_mlp." + k] = shp
    return out


VQ_SWITCH = {"gate_proj.codes": (256, 512, 176),
             "gate_proj.vq_scales": (256, 512, 32),
             "gate_proj.codebook": (2048, 4),
             "down_proj.codes": (256, 2048, 44),
             "down_proj.vq_scales": (256, 2048, 8),
             "down_proj.codebook": (2048, 4)}


def test_a_vq_moe_header_set_splits_two_ways(tmp_path):
    a = _artifact(tmp_path, QWEN36, _layer0(VQ_SWITCH))
    assert tensor_split.tensor_split_refusals(a, 2) == []
    assert tensor_split.tensor_unverified(a) == {}
    from knurlogic.interfaces.page.documents import splits_of
    assert splits_of(a) == ["tensor", "pipeline"]


def test_an_affine_quant_header_set_splits_two_ways(tmp_path):
    a = _artifact(tmp_path, QWEN36, _layer0({
        "gate_proj.weight": (256, 512, 256),
        "gate_proj.scales": (256, 512, 32),
        "gate_proj.biases": (256, 512, 32),
        "down_proj.weight": (256, 2048, 64),
        "down_proj.scales": (256, 2048, 8),
        "down_proj.biases": (256, 2048, 8)}))
    assert tensor_split.tensor_split_refusals(a, 2) == []


def test_an_array_that_does_not_divide_is_refused_with_its_numbers(tmp_path):
    a = _artifact(tmp_path, QWEN36, _layer0(dict(
        VQ_SWITCH, **{"down_proj.vq_scales": (256, 2048, 7)})))
    why = tensor_split.tensor_split_refusals(a, 2)
    assert why == ["layers.0.mlp.switch_mlp.down_proj.vq_scales: 7 on axis "
                   "-1 do not divide by 2"]
    # a fused qkv divides per segment: q, k, v each cut n ways, so a whole
    # that divides is not enough (8192 rows, but q and k 127 each)
    import copy
    cfg = copy.deepcopy(QWEN36)
    cfg["text_config"].update(linear_num_key_heads=1, linear_key_head_dim=127)
    why = tensor_split.tensor_header_refusals(a, cfg, 2)
    assert "layers.0.linear_attn.conv1d.weight: 8192 rows (in segments " \
        "[127, 127, 7938]) do not divide by 2" in why


def _skipzero(nlive):
    return {"gate_proj.sz_codes": (nlive, 1024),
            "gate_proj.sz_scales": (nlive, 64),
            "gate_proj.sz_rowmask": (512, 128),
            "gate_proj.sz_shape": (4,),
            "gate_proj.codebook": (256, 4)}


def test_a_skipzero_module_odd_rows_are_refused_by_its_headers(tmp_path):
    # Qwen3.5-397B 2.4bpw: a tensor launch died loading on rank 1 at
    # "Array split ... (70095, 1024)" -- the headers say it first
    a = _artifact(tmp_path, QWEN36, _layer0(_skipzero(70095)))
    why = tensor_split.tensor_split_refusals(a, 2)
    assert "layers.0.mlp.switch_mlp.gate_proj.sz_codes: 70095 rows do not " \
        "divide by 2" in why
    from knurlogic.interfaces.page.documents import splits_of
    assert "tensor" not in splits_of(a)


def test_a_skipzero_module_even_rows_is_unverified_not_refused(tmp_path):
    # even rows divide, but the layout is none a rule knows: offered, and
    # run whole and split at launch (engine/runtime/viability)
    a = _artifact(tmp_path, QWEN36, _layer0(_skipzero(70096)))
    assert tensor_split.tensor_split_refusals(a, 2) == []
    assert tensor_split.tensor_unverified(a) == {
        ("mlp.switch_mlp.gate_proj",
         ("sz_codes", "sz_rowmask", "sz_scales", "sz_shape")): 0}


def _sz_runtime(a, out=1024):
    # a bundled runtime that splits its own SKIPZERO rows (vqlab 65a9024)
    import copy
    import json
    (a / "model.py").write_text("SKIPZERO_SHARD = 1\n")
    cfg = copy.deepcopy(QWEN36)
    mods = {f"language_model.model.layers.0.mlp.switch_mlp.{p}":
            {"experts": 512, "out": out} for p in ("gate_proj", "down_proj")}
    cfg["vq_skipzero"] = {"format": "vq-skipzero", "version": 1,
                          "modules": mods}
    (a / "config.json").write_text(json.dumps(cfg))
    return cfg


def test_a_runtime_that_splits_skipzero_itself_is_offered_tensor(tmp_path):
    # odd packed rows (70095) are the runtime's to split per expert; down's
    # compact codes are cut on the input axis like any VQ module
    a = _artifact(tmp_path, QWEN36, _layer0(dict(_skipzero(70095), **{
        "down_proj.sz_codes": (2080653, 256),
        "down_proj.sz_scales": (2080653, 16),
        "down_proj.sz_rowmask": (512, 512),
        "down_proj.sz_shape": (4,),
        "down_proj.codebook": (256, 4)})))
    _sz_runtime(a)
    assert tensor_split.tensor_split_refusals(a, 2) == []
    assert tensor_split.tensor_unverified(a) == {}
    from knurlogic.interfaces.page.documents import splits_of
    assert "tensor" in splits_of(a)
    import types
    g = types.SimpleNamespace(rank=lambda: 1, size=lambda: 2)
    from knurlogic.engine.runtime.tensor import load_config
    assert load_config(a, g)["vq_skipzero"]["shard"] == {"rank": 1, "n": 2}


def test_a_runtime_split_needs_output_rows_that_divide(tmp_path):
    a = _artifact(tmp_path, QWEN36, _layer0(_skipzero(70095)))
    _sz_runtime(a, out=1023)
    why = tensor_split.tensor_split_refusals(a, 2)
    assert any("1023 output rows do not divide by 2" in w for w in why)


def test_without_the_runtime_split_load_config_adds_nothing(tmp_path):
    import types

    from knurlogic.engine.runtime.tensor import load_config
    a = _artifact(tmp_path, QWEN36, _layer0(_skipzero(70095)))
    g = types.SimpleNamespace(rank=lambda: 0, size=lambda: 2)
    assert load_config(a, g) is None


def test_a_module_the_runtime_split_is_not_cut_again():
    import types

    import mlx.nn as nn

    from knurlogic.engine.runtime import tensor as T
    from knurlogic.engine.runtime.tensor_rules import RULES
    lin = nn.Linear(8, 8)
    object.__setattr__(lin, "_vq_sharded", (1, 2))
    layer = types.SimpleNamespace(mlp=types.SimpleNamespace(
        switch_mlp=types.SimpleNamespace(gate_proj=lin)))
    T._apply(layer, "mlp.switch_mlp.gate_proj",
             RULES["mlp.switch_mlp.gate_proj"], 1, 2)
    assert lin.weight.shape == (8, 8)
    with pytest.raises(ValueError, match="rank 1 of 2, not 0 of 2"):
        T._apply(layer, "mlp.switch_mlp.gate_proj",
                 RULES["mlp.switch_mlp.gate_proj"], 0, 2)


#: Flash-Next VQ's text config and its layer 1 (linear attention, MoE, the
#: n-gram table) cut down to 4 parts of 8 rows
FLASH_NEXT = {"model_type": "qwen4_exp", "text_config": {
    "model_type": "qwen4_exp_text", "num_attention_heads": 24,
    "num_key_value_heads": 2, "linear_num_key_heads": 16,
    "linear_key_head_dim": 128, "linear_num_value_heads": 48}}


def _flash_next_layer1(parts=4):
    pre = "model.layers.1."
    out = {pre + "linear_attn.in_proj_qkv.weight": (10240, 640),
           pre + "linear_attn.in_proj_qkv.scales": (10240, 40),
           pre + "linear_attn.conv1d.weight": (10240, 4, 1),
           pre + "linear_attn.out_proj.weight": (2560, 1536),
           pre + "attn_hyper_connection.input_mix_weight_up.weight":
               (10240, 40),
           pre + "ple.key_proj.weight": (10240, 640),
           pre + "ple.ple_embedding.layer_multipliers": (3,),
           pre + "mlp.switch_mlp.down_proj.codes": (512, 2560, 100),
           pre + "mlp.switch_mlp.down_proj.codebook": (1024, 2)}
    for i in range(parts):
        t = f"{pre}ple.ple_embedding.ngram_embedding.shard_{i}."
        out.update({t + "codes": (8, 80), t + "vq_scales": (8, 5),
                    t + "codebook": (256, 2)})
    return out


def test_flash_next_deals_its_ngram_parts_whole(tmp_path):
    """The n-gram table's parts go whole to one rank each (codebook and
    all: no axis of a part is cut), so they count as split bytes and no
    layout in them is unverified; the hyper-connections and the rest of
    the PLE are replicated."""
    from knurlogic.engine.runtime import tensor_rules as TR
    a = _artifact(tmp_path, FLASH_NEXT, _flash_next_layer1())
    assert tensor_split.tensor_split_refusals(a, 2) == []
    assert tensor_split.tensor_unverified(a) == {}
    t = "model.layers.1.ple.ple_embedding.ngram_embedding.shard_3."
    assert TR.locate(t + "codes") == (
        1, "ple.ple_embedding.ngram_embedding", "shard_3.codes", None)
    assert tensor_split.tensor_sharded(t + "codebook")
    assert not tensor_split.tensor_sharded("model.layers.1.ple.key_proj.weight")
    assert not tensor_split.tensor_sharded(
        "model.layers.1.attn_hyper_connection.input_mix_weight_up.weight")
    assert [TR.owner(i, 4, 2) for i in range(4)] == [0, 0, 1, 1]
    p = tensor_split.tensor_placement(type("A", (), {"path": a}), 2)
    # replicated: the hyper-connection, key_proj, the hash multipliers and
    # down_proj's codebook
    assert p["replicated_bytes"] == 10240 * 40 + 10240 * 640 + 3 + 1024 * 2


def test_ngram_parts_that_do_not_divide_are_refused(tmp_path):
    a = _artifact(tmp_path, FLASH_NEXT, _flash_next_layer1(parts=3))
    assert tensor_split.tensor_header_refusals(a, FLASH_NEXT, 2) == [
        "layers.1.ple.ple_embedding.ngram_embedding: 3 parts do not "
        "divide by 2"]


#: DeepSeek-V4-Flash's config (the tensor-relevant keys) and its layer 2
#: (ratio 4: compressor and indexer) as the VQ-3.2 build's headers hold it
DSV4 = {"model_type": "deepseek_v4", "num_attention_heads": 64,
        "num_key_value_heads": 1, "o_groups": 8, "head_dim": 512,
        "q_lora_rank": 1024, "o_lora_rank": 1024, "n_routed_experts": 256,
        "moe_intermediate_size": 2048, "index_n_heads": 64}


def _dsv4_layer2(sink=64):
    pre = "model.layers.2."
    out = {}
    for name, o, i in (("attn.wq_a", 1024, 4096), ("attn.wkv", 512, 4096),
                       ("attn.wq_b", 32768, 1024), ("attn.wo_a", 8192, 4096),
                       ("attn.wo_b", 4096, 8192),
                       ("attn.compressor.wkv", 1024, 4096),
                       ("attn.compressor.wgate", 1024, 4096),
                       ("attn.indexer.wq_b", 8192, 1024),
                       ("attn.indexer.weights_proj", 64, 4096),
                       ("attn.indexer.compressor.wkv", 256, 4096),
                       ("attn.indexer.compressor.wgate", 256, 4096),
                       ("ffn.shared_experts.gate_proj", 2048, 4096),
                       ("ffn.shared_experts.up_proj", 2048, 4096),
                       ("ffn.shared_experts.down_proj", 4096, 2048)):
        # affine 8-bit in groups of 64
        out[pre + name + ".weight"] = (o, i // 4)
        out[pre + name + ".scales"] = (o, i // 64)
        out[pre + name + ".biases"] = (o, i // 64)
    out.update({pre + "attn.attn_sink": (sink,),
                pre + "attn.q_norm.weight": (1024,),
                pre + "attn.kv_norm.weight": (512,),
                pre + "attn.compressor.ape": (4, 1024),
                pre + "attn_hc.fn": (24, 16384),
                pre + "ffn.gate.weight": (256, 4096),
                pre + "ffn.gate.e_score_correction_bias": (256,)})
    for p, rows, words, groups in (("gate_proj", 2048, 352, 64),
                                   ("up_proj", 2048, 352, 64),
                                   ("down_proj", 4096, 176, 32)):
        t = pre + "ffn.switch_mlp." + p
        out.update({t + ".codes": (256, rows, words),
                    t + ".vq_scales": (256, rows, groups),
                    t + ".codebook": (2048, 4)})
    return out


def test_deepseek_v4_cuts_heads_and_experts_and_keeps_the_kv_whole(tmp_path):
    """The heads (wq_b, the sink, wo_a by whole o_groups, wo_b's input),
    the routed experts and the shared expert are cut; the low-rank q and
    the one shared kv head (wq_a, wkv), the compressor, the indexer, the
    router and the hyper-connections are replicated."""
    a = _artifact(tmp_path, DSV4, _dsv4_layer2())
    assert tensor_split.tensor_split_refusals(a, 2) == []
    assert tensor_split.tensor_split_refusals(a, 4) == []
    assert tensor_split.tensor_unverified(a) == {}
    pre = "model.layers.2."
    for k in ("attn.wq_b.weight", "attn.attn_sink", "attn.wo_a.scales",
              "attn.wo_b.weight", "ffn.switch_mlp.down_proj.codes",
              "ffn.shared_experts.up_proj.biases"):
        assert tensor_split.tensor_sharded(pre + k), k
    for k in ("attn.wq_a.weight", "attn.wkv.weight", "attn.kv_norm.weight",
              "attn.compressor.wkv.weight", "attn.indexer.wq_b.weight",
              "attn.indexer.weights_proj.weight", "attn_hc.fn",
              "ffn.gate.weight", "ffn.switch_mlp.gate_proj.codebook"):
        assert not tensor_split.tensor_sharded(pre + k), k


def test_deepseek_v4_refuses_groups_and_shapes_that_do_not_divide(tmp_path):
    # 16 ranks: 64 heads and wo_a's 8192 rows divide, 8 o_groups do not --
    # a rank would hold half a group
    assert tensor_split.tensor_refusals(DSV4, 16) == [
        "o_groups = 8 is not divisible by 16 ranks (8 / 16 = 0.5)"]
    assert len(tensor_split.tensor_refusals(DSV4, 3)) == 2         # heads and groups
    a = _artifact(tmp_path, DSV4, _dsv4_layer2(sink=63))
    assert tensor_split.tensor_header_refusals(a, DSV4, 2) == [
        "layers.2.attn.attn_sink: 63 rows do not divide by 2"]


def test_hf_bf16_names_map_to_the_rules_sanitize_feeds_shard():
    from knurlogic.engine.runtime.tensor_rules import locate
    pre = "model.language_model.layers.3."
    assert locate(pre + "mlp.experts.gate_up_proj") == (
        3, "mlp.switch_mlp.gate_proj", "weight", 2)
    assert locate(pre + "mlp.experts.down_proj")[1] == \
        "mlp.switch_mlp.down_proj"
    assert locate(pre + "linear_attn.A_log") == (
        3, "linear_attn.A_log", None, None)
    assert locate("mtp.layers.0.self_attn.q_proj.weight") is None
    assert locate(pre + "mlp.shared_expert_gate.weight") is None


# --------------------------------------------- viability of an unknown layout

def _probe_module(perm=None, experts=None):
    """y = x @ W.T with W stored under a parameter name no rule knows;
    `perm` stores the output rows out of order (read back through it),
    `experts` adds a per-expert axis the module indexes."""
    import mlx.nn as nn

    class Probe(nn.Module):
        def __init__(self):
            super().__init__()
            IN, OUT = 64, 32
            w = mx.random.normal((OUT, IN), key=mx.random.key(1))
            self.order = mx.arange(OUT) if perm is None else perm
            self.packed_w = w[mx.argsort(self.order)] if perm is not None \
                else w

        @property
        def input_dims(self):
            return self.packed_w.shape[1]

        @property
        def output_dims(self):
            return self.packed_w.shape[0]

        def __call__(self, x):
            w = self.packed_w if perm is None else self.packed_w[self.order]
            return (x.astype(mx.float32) @ w.T).astype(x.dtype)
    return Probe()


def test_a_layout_that_splits_like_the_rule_holds():
    from knurlogic.engine.runtime.tensor_rules import A2S, S2A, Rule
    from knurlogic.engine.runtime.viability import check_module
    assert check_module(_probe_module(), Rule(A2S), 2) is None
    assert check_module(_probe_module(), Rule(S2A), 2) is None


def test_a_layout_whose_rows_are_not_in_order_fails_with_its_error():
    from knurlogic.engine.runtime.tensor_rules import A2S, Rule
    from knurlogic.engine.runtime.viability import check_module
    # rows stored by a table (as SKIPZERO's are): a row cut takes the
    # wrong rows, and the table itself is cut too
    perm = mx.array([(i * 7) % 32 for i in range(32)])
    why = check_module(_probe_module(perm), Rule(A2S), 2)
    assert why and "differs from whole by" in why


def test_skipzero_cut_by_rows_mixes_experts():
    # the real module's geometry: codes [NLIVE, W] and row_table [E, OUT];
    # a row cut of the packed codes leaves the (whole) table's rows
    # unhalved, so the part is refused unrun
    import mlx.nn as nn

    from knurlogic.engine.runtime.tensor_rules import A2S, Rule
    from knurlogic.engine.runtime.viability import check_module

    class SZ(nn.Module):
        def __init__(self):
            super().__init__()
            self.codes = mx.zeros((6, 8), dtype=mx.uint8)
            self.row_table = mx.zeros((4, 2), dtype=mx.int32)

        num_experts = property(lambda s: s.row_table.shape[0])
        output_dims = property(lambda s: s.row_table.shape[1])
        input_dims = property(lambda s: s.codes.shape[1] * 4)
    why = check_module(SZ(), Rule(A2S), 2)
    assert "output_dims 2 -> 2 (want 1)" in why


# ------------------------------------------------------------ fit

def _two(ws0, ws1):
    gib = 1 << 30
    return [{"name": n, "chip": c, "p_core_ghz": None, "bandwidth_gbs": None,
             "working_set_bytes": int(w * gib)}
            for n, c, w in (("A", "Apple M4 Max", ws0),
                            ("B", "Apple M3 Ultra", ws1))]


def test_tensor_placement_puts_the_head_on_rank_0_alone():
    from knurlogic.cluster import launch as C
    gib = 1 << 30
    shape = {"tensor_per_rank_bytes": 10 * gib, "leader_bytes": 5 * gib,
             "refusals": []}
    p = C.placement(_two(64, 64), shape, "tensor")
    assert [s["bytes"] for s in p["shares"]] == [15 * gib, 10 * gib]
    assert "rank 0 5.0 GiB more" in p["reason"]
    # no head bytes: every rank the same share, as before
    p = C.placement(_two(64, 64), dict(shape, leader_bytes=0), "tensor")
    assert [s["bytes"] for s in p["shares"]] == [10 * gib] * 2


def test_a_head_that_does_not_fit_rank_0_is_refused_with_the_arithmetic():
    from knurlogic.cluster import launch as C
    gib = 1 << 30
    shape = {"tensor_per_rank_bytes": 10 * gib, "leader_bytes": 5 * gib,
             "refusals": []}
    # 15 GiB on rank 0 against 18 - 4 (step margin); rank 1's 10 fits
    with pytest.raises(ValueError, match=r"A: its tensor share 10.0 GiB "
                       r"\+ 5.0 GiB rank 0 alone holds .*working set "
                       r"18.0 GiB less the 4.0 GiB step margin"):
        C.placement(_two(18, 18), shape, "tensor")
    C.placement(_two(19, 18), shape, "tensor")


def test_tensor_shape_carries_rank_0s_head_bytes(tmp_path, monkeypatch):
    import json
    import struct

    from knurlogic.cluster import launch as C
    from knurlogic.machine import artifact

    def shard(path, tensors):
        header, at = {}, 0
        for k, n in tensors.items():
            header[k] = {"dtype": "U8", "shape": [n],
                         "data_offsets": [at, at + n]}
            at += n
        h = json.dumps(header).encode()
        path.write_bytes(struct.pack("<Q", len(h)) + h + b"\0" * at)
    shard(tmp_path / "model.safetensors",
          {"model.layers.0.mlp.weight": 10,
           "vision_tower.blocks.0.attn.qkv.weight": 30})
    shard(tmp_path / "mtp-head-q6.safetensors", {"block.fc.weight": 50})

    class A:
        path = tmp_path
        raw_config = {"text_config": {"num_hidden_layers": 1}}
    monkeypatch.setattr(artifact.Artifact, "load", staticmethod(lambda p: A))
    monkeypatch.setattr(tensor_split, "tensor_split_refusals", lambda *a: [])
    assert C.shape_of(str(tmp_path), 2, "tensor")["leader_bytes"] == 50 + 30
    assert C.shape_of(str(tmp_path), 2, "tensor",
                      vision=False)["leader_bytes"] == 50


def test_a_follower_prefills_each_row_at_the_chunk_rank_0_fitted(
        monkeypatch):
    """The chunk rides the admit op, and a refit (the `chunk` op) follows
    it: every rank sets it on its engine before the step that prefills
    that row -- ranks prefilling in different chunk counts deadlock. A
    prefilling step's transient is not held as the follower's margin."""
    from types import SimpleNamespace as NS

    from knurlogic.engine.mtp import batch_generator as BG
    from knurlogic.engine.runtime import pipeline as PL
    from knurlogic.engine.runtime import request as RQ
    from knurlogic.engine.runtime import tensor as T
    chunks, marks = [], []

    class Ex:
        def __init__(self, gen):
            self.gen, self.q, self.n = gen, [], 0

        def insert(self, a):
            self.q.append(self.n)
            self.n += 1
            return self.n - 1

        def next_admission(self):
            return self.q[0] if self.q else None

        def set_chunk(self, n):
            chunks.append((self.q[0], n))

        def step(self):
            self.q.pop(0)
            return []

        def close(self):
            pass

    class Mark(T.Mark):
        def around(self, fn, prefill=False):
            marks.append(prefill)
            return fn()

    monkeypatch.setattr(BG, "MTPBatchGenerator", lambda *a, **k: NS(
        _batch=NS(uids=[], t1=None)))
    monkeypatch.setattr(PL, "coordinate", lambda *a, **k: None)
    monkeypatch.setattr(RQ, "control_machine", lambda tok, init: (None, []))
    monkeypatch.setattr(T, "LocalExecutor", Ex)
    monkeypatch.setattr(T, "Mark", Mark)

    def admit(uid, c):
        return _admit(uid=uid, prompt=[1, 2, 3], segs=[[1, 2, 3]], hit=0,
                      chunk=c)
    link = _FakeLink([
        {"ops": [admit(0, 256), admit(1, 2048)]},
        {"ops": [{"op": "chunk", "uid": 1, "chunk": 512}]},
        {"ops": [{"op": "stop"}]}])
    link.group = None
    T.follow(None, None, "m", link, prompt_cache_size=2,
             completion_batch_size=4, prefill_step_size=2048,
             working_set=0)
    assert chunks == [(0, 256), (1, 512)] and marks == [True, True]


def test_a_prefill_steps_transient_is_not_a_followers_margin(monkeypatch):
    import mlx.core as mx

    from knurlogic.engine.runtime import tensor as T
    GIB = T.GIB
    mem = {"active": 60 * GIB, "peak": 77 * GIB}
    monkeypatch.setattr(mx, "get_active_memory", lambda: mem["active"])
    monkeypatch.setattr(mx, "get_peak_memory", lambda: mem["peak"])
    monkeypatch.setattr(mx, "reset_peak_memory", lambda: None)
    m = T.Mark(84 * GIB)
    m.around(lambda: None, prefill=True)         # a 59k prefill: 17 GiB
    assert m.limit() == 84 * GIB - int(4.2 * GIB)
    mem["peak"] = 61 * GIB
    m.around(lambda: None)                       # a decode step: 1 GiB
    assert m.spike == GIB


def test_a_plan_chunk_is_a_positive_int():
    assert P.decode(P.encode({"ops": [{"op": "chunk", "uid": 1,
                                       "chunk": 256}]}))
    for bad in (0, -1, True, "256"):
        with pytest.raises(P.PlanError):
            P.encode({"ops": [{"op": "chunk", "uid": 1, "chunk": bad}]})
    with pytest.raises(P.PlanError):
        P.encode({"ops": [_admit(chunk=0)]})
