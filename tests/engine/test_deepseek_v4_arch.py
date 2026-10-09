"""DeepSeek-V4 (deepseek_v4), vendored from an mlx-lm fork: a tiny
random-weight model (tests/support/goldens/build_deepseek_v4.py) loads through
knurlogic's registration and mlx-lm's loader, computes the fork's own
logits, splits into two pipeline stages with the unsplit logits, carries
its DeepseekV4Cache through a prompt-cache copy and a restore, and batches
two rows as it runs them one at a time. The resolver half needs no mlx.
(The chat template is tests/test_deepseek_v4.py.)"""
import copy
import hashlib
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests" / "support" / "goldens"))

from knurlogic.tuning import fit, pipeline_split  # noqa: E402

ARCH = ROOT / "src/knurlogic/engine/families/deepseek/architecture"
# A local DeepSeek-V4-Flash artifact directory (optional).
REAL = Path(os.environ.get("KNURLOGIC_TEST_DEEPSEEK_V4") or "/nonexistent")


# ------------------------------------------------------------ no mlx

def test_the_vendored_file_is_the_one_provenance_names():
    want = re.search(r"vendored sha256: `([0-9a-f]{64})`",
                     (ARCH / "PROVENANCE.md").read_text()).group(1)
    got = hashlib.sha256((ARCH / "deepseek_v4.py").read_bytes()).hexdigest()
    assert got == want


def test_model_type_maps_to_the_vendored_module():
    from knurlogic.engine import arch
    assert arch.ARCH_FOR_MODEL_TYPE["deepseek_v4"] == "deepseek_v4"
    assert arch.required_modules("deepseek_v4") == ["deepseek_v4"]


def _real_like_config():
    ratios = [0, 0] + [4, 128] * 20 + [4, 0]
    return {"model_type": "deepseek_v4", "num_hidden_layers": 43,
            "head_dim": 512, "num_key_value_heads": 1,
            "num_attention_heads": 64, "index_head_dim": 128,
            "sliding_window": 128, "compress_ratios": ratios,
            "max_position_embeddings": 1048576}


def test_kv_per_token_counts_the_compressed_pools_not_k_and_v_per_layer():
    per, why = fit.kv_bytes_per_token(_real_like_config())
    # 21 ratio-4 layers x (512 + 128) / 4 + 20 ratio-128 layers x 512 / 128,
    # bf16 -- the generic count (43 x K,V x 512) would be ~13x this
    assert per == (21 * (512 + 128) // 4 + 20 * 512 // 128) * 2 == 6880
    assert "41 compressed layers of 43" in why and "bounded" in why


def test_kv_quantization_is_refused():
    from knurlogic.tuning import measured
    bits, why = measured.kv_quant_for("deepseek_v4")
    assert bits == [] and "DeepseekV4Cache" in why


def test_it_may_be_pipelined():
    assert pipeline_split.pipeline_refusals(_real_like_config(), 2) == []


@pytest.mark.skipif(not (REAL / "config.json").is_file(),
                    reason="the DeepSeek-V4-Flash artifact is not mounted"
                           " (set KNURLOGIC_TEST_DEEPSEEK_V4)")
def test_the_real_artifact_splits_m3_ultra_96_and_m4_max_128_by_bytes():
    """The planner over the real headers (nothing loaded): the M3 Ultra's
    86016 MB wired limit and the M4 Max's 122880 MB. The M4 Max holds at
    most ~100 GB, whichever leads."""
    from knurlogic.cluster import launch as L
    sh = L.shape_of(str(REAL), 2, "pipeline")
    assert len(sh["layer_bytes"]) == 43 and not sh["refusals"]
    machines = [
        {"name": "M3", "chip": "Apple M3 Ultra", "bandwidth_gbs": 819.0,
         "working_set_bytes": 86016 << 20, "links": {}},
        {"name": "M4", "chip": "Apple M4 Max", "bandwidth_gbs": None,
         "working_set_bytes": 122880 << 20, "links": {}}]
    for order in (None, ["M3", "M4"]):
        p = L.placement(machines, sh, "pipeline", order)
        by = {s["machine"]: s for s in p["shares"]}
        assert by["M4"]["bytes"] <= 100e9, p["reason"]
        assert sum(s["layers"] for s in p["shares"]) == 43


# ------------------------------------------------------------ the model

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

import build_deepseek_v4 as G  # noqa: E402


def _load(model_config=None):
    """The tiny artifact the way `knurlogic serve` loads one: Artifact,
    loading.register (the vendored set), then mlx-lm's loader."""
    from mlx_lm.utils import load_model

    from knurlogic.interfaces import loading
    from knurlogic.machine.artifact import Artifact
    a = Artifact.load(str(G.TINY))
    assert loading.register(a) == []
    model, _ = load_model(G.TINY, model_config=model_config)
    mod = sys.modules[type(model).__module__]
    assert mod.__name__ == "mlx_lm.models.deepseek_v4"
    assert Path(mod.__file__).resolve() == (ARCH / "deepseek_v4.py").resolve()
    return model


def test_the_loaded_model_computes_the_forks_logits():
    """The golden was computed by the fork itself (mlx-lm 0.31.9)
    over the same weights: prefill, then six decode steps, pools growing
    on both paths and the 8-token window rotating -- with every pool row
    kept by the indexer (G.GOLDEN_CONFIG), since where it chooses the
    vendored file is deliberately not the fork."""
    import numpy as np
    golden = np.load(ROOT / "tests/support/goldens/deepseek_v4_tiny.npz")
    got = G.logits_of(_load(dict(G.GOLDEN_CONFIG)))
    want = golden["logits"]
    assert got.shape == want.shape == (7, G.CONFIG["vocab_size"])
    assert np.abs(got - want).max() < 1e-3, np.abs(got - want).max()
    assert (got.argmax(-1) == want.argmax(-1)).all()


class _Out(nn.Module):
    """Stage 1's last layer: keeps what a Send would send."""

    def __init__(self, inner, box):
        super().__init__()
        self.inner, self._box = inner, box

    def __call__(self, x, *a, **kw):
        y = self.inner(x, *a, **kw)
        self._box.append(y)
        return y


class _In(nn.Module):
    """Stage 0's first layer: takes what a Recv would receive."""

    def __init__(self, inner, box):
        super().__init__()
        self.inner, self._box = inner, box

    def __call__(self, x, *a, **kw):
        h = self._box.pop(0)
        assert h.shape == x.shape       # the placeholder a Recv reads
        return self.inner(h, *a, **kw)


@pytest.mark.parametrize("cut", [1, 2, 3])
def test_two_stages_in_one_process_are_the_whole_model(cut):
    """restage() both halves (the follower holds [0, cut), rank 0 the rest)
    and hand the hidden state across as the ring would; the logits over a
    prefill and six decode steps are the unsplit model's."""
    import numpy as np

    from knurlogic.engine.split import pipeline as PL
    whole = G.logits_of(_load())
    first, last = _load(), _load()
    assert PL.family_of(first) == "deepseek_v4"
    n = len(PL.core_of(first).layers)
    box = []
    keep1 = list(PL.core_of(first).layers)[:cut]
    keep1[-1] = _Out(keep1[-1], box)
    PL.restage(first, keep1, 0, cut)
    keep0 = list(PL.core_of(last).layers)[cut:]
    keep0[0] = _In(keep0[0], box)
    PL.restage(last, keep0, cut, n)
    c1, c0 = first.make_cache(), last.make_cache()
    assert (len(c1), len(c0)) == (cut, n - cut)

    rows = []
    for ids in [G.PROMPT] + [[t] for t in G.DECODE]:
        x = mx.array([ids])
        first(x, cache=c1)
        rows.append(last(x, cache=c0)[0, -1])
    split = np.array(mx.stack(rows).astype(mx.float32))
    assert np.abs(split - whole).max() < 1e-5, np.abs(split - whole).max()


def test_a_copied_prompt_cache_continues_like_the_original():
    """What the prompt cache stores is a deepcopy of the row's caches; its
    DeepseekV4Cache (window + compressor and indexer pools) must continue
    exactly as the live one; and extract / merge, the batch engine's
    per-row moves, round-trip it."""
    import numpy as np
    model = _load()
    arch = sys.modules["mlx_lm.models.deepseek_v4"]
    cache = model.make_cache()
    assert all(type(c) is arch.DeepseekV4Cache for c in cache)
    model(mx.array([G.PROMPT]), cache=cache)
    mx.eval([c.state for c in cache])
    entry = copy.deepcopy(cache)
    assert all(not c.is_trimmable() for c in entry)   # exact prefix only
    assert entry[0].nbytes == cache[0].nbytes > 0
    moved = [arch.DeepseekV4Cache.merge([c]).extract(0)
             for c in copy.deepcopy(entry)]

    def cont(c):
        out = [model(mx.array([[t]]), cache=c)[0, -1] for t in G.DECODE]
        return np.array(mx.stack(out).astype(mx.float32))
    live = cont(cache)
    assert np.abs(cont(entry) - live).max() == 0
    assert np.abs(cont(moved) - live).max() < 1e-5


def _drive(gen, prompts, max_tokens, caches=None, prefixes=None):
    uids = gen.insert_segments(
        segments=[[p] for p in prompts],
        max_tokens=[max_tokens] * len(prompts),
        caches=caches or [None] * len(prompts),
        all_tokens=prefixes or [[] for _ in prompts])
    out, done = {u: [] for u in uids}, set()
    for _ in range(10_000):
        _, grs = gen.next()
        for r in grs:
            out[r.uid].append(r.token)
            if r.finish_reason is not None:
                done.add(r.uid)
        if len(done) == len(uids):
            break
    gen.close()
    return [out[u] for u in uids]


@pytest.mark.parametrize("b", [
    G.PROMPT[3:9] + [1 + t for t in G.DECODE],   # 12: emits a step apart
    G.PROMPT[:3],                                 # 3: no window yet
], ids=["12-tokens", "3-tokens"])
def test_two_rows_batch_as_they_run_alone(b):
    """knurlogic's batch engine (no head): two prompts of different lengths
    admitted together emit what each emits alone, greedy. Rows of different
    lengths complete their compressor windows on different steps; the fork
    as taken let the other row's padding into a short row's attention and
    moved every row's overlap carry on any row's emit (PROVENANCE.md)."""
    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    model = _load()
    a = G.PROMPT
    alone = [_drive(MTPBatchGenerator(model, None), [p], 16)[0]
             for p in (a, b)]
    both = _drive(MTPBatchGenerator(model, None), [a, b], 16)
    assert both == alone
    assert all(len(t) == 16 for t in both)


def test_a_restored_prompt_cache_is_a_fresh_prefill():
    """A row admitted with a stored cache for its first tokens prefills only
    the rest and emits what a from-scratch prefill emits."""
    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    model = _load()
    head, tail = G.PROMPT[:9], G.PROMPT[9:]
    cache = model.make_cache()
    model(mx.array([head]), cache=cache)
    mx.eval([c.state for c in cache])
    gen = MTPBatchGenerator(model, None)
    restored = _drive(gen, [tail], 8, caches=[copy.deepcopy(cache)],
                      prefixes=[list(head)])
    assert gen._counters.prompt_tokens == len(tail)
    fresh = _drive(MTPBatchGenerator(model, None), [G.PROMPT], 8)
    assert restored == fresh


@pytest.mark.parametrize("steps", [0, 1, 2, 3, 4])
def test_a_row_joining_a_decoding_batch_keeps_both_rows_exact(steps):
    """The batch engine's extend: row a decodes `steps` tokens alone (its
    carry buffer in the decode hot path's fixed form), then a 3-token row
    joins. Every later step, each row's logits are its lone run's. The fork
    as taken read the joined buffers' lengths after concatenating them
    (PROVENANCE.md), so row a emitted a compressed row early."""
    from mlx_lm.generate import _extend_cache, _merge_caches
    model = _load()
    a, b = G.PROMPT, G.PROMPT[:3]
    toks = [5, 9, 13, 22, 40, 41, 2, 3, 7, 8, 9, 10]

    def prefilled(p):
        c = model.make_cache()
        model(mx.array([p]), cache=c)
        return c
    batch, ra = _merge_caches([prefilled(a)]), prefilled(a)
    for t in toks[:steps]:
        model(mx.array([[t]]), cache=batch)
        model(mx.array([[t]]), cache=ra)
    batch = _extend_cache(batch, _merge_caches([prefilled(b)]))
    rb = prefilled(b)
    for t in toks[steps:]:
        both = model(mx.array([[t], [t + 1]]), cache=batch)[:, -1]
        la = model(mx.array([[t]]), cache=ra)[0, -1]
        lb = model(mx.array([[t + 1]]), cache=rb)[0, -1]
        assert mx.abs(both[0] - la).max().item() < 1e-4
        assert mx.abs(both[1] - lb).max().item() < 1e-4


def test_a_prefill_chooses_what_decoding_the_same_tokens_chooses():
    """With more compressed rows than index_topk, a prefilled prompt ends
    in the state (and the logits) of the same tokens decoded one at a time
    and of a prefill split in two. The fork let a prefill query rank rows
    from its future (PROVENANCE.md, the indexer's prefill topk)."""
    model = _load()
    p = G.PROMPT + G.DECODE                   # 17 tokens: 4 ratio-4 rows
    whole = model(mx.array([p]), cache=model.make_cache())[0, -1]
    c = model.make_cache()
    for t in p:
        one = model(mx.array([[t]]), cache=c)[0, -1]
    c = model.make_cache()
    model(mx.array([p[:9]]), cache=c)
    halves = model(mx.array([p[9:]]), cache=c)[0, -1]
    assert mx.abs(whole - one).max().item() < 1e-4
    assert mx.abs(whole - halves).max().item() < 1e-4


def test_ragged_prev_is_not_compiled():
    """Its `lens` is a Python list: compiled, it traced one graph per emit
    pattern per layer (knurlogic edit 11)."""
    src = (ARCH / "deepseek_v4.py").read_text()
    i = src.index("def _ragged_prev(")
    assert "@mx.compile" not in src[max(0, i - 40):i]


def test_the_shared_expert_clamps_as_deepseeks_does():
    """DeepSeek's reference builds the shared expert with swiglu_limit, as
    its routed experts (vendored edit 18): gate capped at the limit, up
    clipped to [-limit, limit]."""
    import numpy as np

    from knurlogic.engine import register
    register.register("deepseek_v4")
    import mlx_lm.models.deepseek_v4 as M
    args = M.ModelArgs.from_dict(dict(
        model_type="deepseek_v4", vocab_size=256, hidden_size=64,
        moe_intermediate_size=64, n_routed_experts=8, num_experts_per_tok=2,
        num_hash_layers=0, compress_ratios=[0, 0], num_hidden_layers=2))
    mlp = M.DeepseekV4MoE(args, 1).shared_experts
    assert mlp.swiglu_limit == args.swiglu_limit > 0
    x = mx.array(np.random.default_rng(0).standard_normal((3, 64)) * 50,
                 dtype=mx.float32)
    g, u = mlp.gate_proj(x), mlp.up_proj(x)
    lim = args.swiglu_limit
    want = mlp.down_proj(nn.silu(mx.minimum(g, lim)) * mx.clip(u, -lim, lim))
    assert mx.allclose(mlp(x), want, atol=1e-4)


# A row's history before it is batched: (prefill tokens, then single-token
# decode steps). Same lengths, different histories, leave rotating windows
# of different buffer lengths: a 7-token prefill holds 7 keys, 6 + 1 step
# holds a ring grown to 8 slots, 9 tokens prefilled hold 9 keys for an
# 8-wide window. Totals 3..16 run below, at and past the window (8) and
# fill the ratio-4 / ratio-8 pools to different levels.
_HISTORIES = [(3, 0), (6, 1), (7, 0), (5, 2), (8, 0), (7, 1), (9, 0),
              (6, 3), (12, 0), (11, 5)]


def _row(model, hist, chunk=None):
    n, d = hist
    p = (G.PROMPT * 3)[:n]
    c = model.make_cache()
    step = chunk or n
    for i in range(0, n, step):
        model(mx.array([p[i:i + step]]), cache=c)
    for t in G.DECODE[:d]:
        model(mx.array([[t]]), cache=c)
    return c


def _check_batch(model, batch, lone, steps=10):
    """Every later step, each batch row's logits are its lone run's."""
    for s in range(steps):
        ids = [[(G.DECODE[s % 6] + 3 * j) % 64] for j in range(len(lone))]
        out = model(mx.array(ids), cache=batch)[:, -1]
        for j, c in enumerate(lone):
            want = model(mx.array([ids[j]]), cache=c)[0, -1]
            got = out[j]
            assert mx.abs(got - want).max().item() < 1e-4, (j, s)


@pytest.mark.parametrize("chunk", [None, 4], ids=["whole", "chunked"])
def test_rows_merged_in_any_order_run_as_alone(chunk):
    """mlx-lm's _merge_caches over rows whose windows hold different
    buffer lengths at the same or different offsets, every ordered pair:
    each row's logits are its lone run's (float32). The fork concatenated
    two rows' raw window buffers when their offsets agreed, and a 7-token
    row met a 6-token one that had decoded a step with a broadcast /
    concatenate shape error (vendored edit 22)."""
    import itertools

    from mlx_lm.generate import _merge_caches
    model = _load()
    model.set_dtype(mx.float32)
    for ha, hb in itertools.permutations(_HISTORIES, 2):
        batch = _merge_caches([_row(model, ha, chunk), _row(model, hb, chunk)])
        _check_batch(model, batch,
                     [_row(model, ha, chunk), _row(model, hb, chunk)], 4)


@pytest.mark.parametrize("ha,hb", [
    ((7, 0), (6, 1)), ((6, 1), (7, 0)), ((8, 0), (7, 1)), ((7, 1), (8, 0)),
    ((9, 0), (8, 1)), ((3, 0), (12, 0)), ((12, 0), (3, 0)),
    ((6, 3), (9, 0)), ((5, 2), (7, 0))])
@pytest.mark.parametrize("steps", [0, 1, 3])
def test_a_row_joining_mid_decode_runs_as_alone(ha, hb, steps):
    """The batch engine's extend: a batch of row a decodes `steps` steps,
    then row b joins (one row each side, so both windows can be plain
    rotating caches at one offset); then a third row joins the two. Every
    later step, each row's logits are its lone run's."""
    from mlx_lm.generate import _extend_cache, _merge_caches
    model = _load()
    model.set_dtype(mx.float32)
    batch, ra = _merge_caches([_row(model, ha)]), _row(model, ha)
    for t in G.DECODE[:steps]:
        model(mx.array([[t]]), cache=batch)
        model(mx.array([[t]]), cache=ra)
    batch = _extend_cache(batch, _merge_caches([_row(model, hb)]))
    lone = [ra, _row(model, hb)]
    _check_batch(model, batch, lone, 3)
    batch = _extend_cache(batch, _merge_caches([_row(model, (7, 0))]))
    _check_batch(model, batch, lone + [_row(model, (7, 0))], 6)


def test_the_batch_engine_admits_short_rows_beside_any_row():
    """Through the batch engine, rows shorter than, at and past the window
    admitted while others decode all finish (an admission that raised
    failed only that request, so this is what a live short chat saw)."""
    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    model = _load()
    P = G.PROMPT * 3
    for first, later in [(7, 6), (6, 7), (8, 7), (9, 3), (3, 9), (16, 5)]:
        for wait in (0, 1, 2, 4):
            gen = MTPBatchGenerator(model, None, prefill_step_size=4)
            uids = gen.insert_segments(
                segments=[[P[:first]]], max_tokens=[16], caches=[None],
                all_tokens=[[]])
            got = {uids[0]: 0}
            done = set()

            def step(gen=gen, got=got, done=done):
                for r in gen.next()[1]:
                    got[r.uid] = got.get(r.uid, 0) + 1
                    if r.finish_reason is not None:
                        done.add(r.uid)
            for _ in range(wait):
                step()
            uids += gen.insert_segments(
                segments=[[P[2:2 + later]]], max_tokens=[8], caches=[None],
                all_tokens=[[]])
            for _ in range(40):
                step()
                if len(done) == 2:
                    break
            gen.close()
            assert got.get(uids[1]) == 8 and got[uids[0]] == 16, \
                (first, later, wait, got)


def test_module_arrays_evaluate_first_in_another_thread():
    """The engine evaluates on its own thread. A lazy array made when a
    module was imported belongs to the importing thread's stream: live, a
    DeepSeek split's first admission raised "There is no Stream(gpu, 0) in
    current thread" on deepseek_v4's _NO_W (mlx 0.32.3). A fresh
    interpreter imports every vendored architecture on the main thread, so
    nothing evaluated them there first; a second thread evaluates every
    module-level array."""
    import subprocess
    import sys
    code = (
        "import importlib, threading, mlx.core as mx\n"
        "from knurlogic.engine import register\n"
        "import sys\n"
        "from knurlogic.engine import families\n"
        "register.register(override=True)\n"
        "for name in register.available():\n"
        "    from knurlogic.engine.arch import host_for\n"
        "    importlib.import_module(f'{host_for(name)}.models.{name}')\n"
        "dirs = [str(d.resolve()) for d in families.architecture_dirs()]\n"
        "mods = [m for m in list(sys.modules.values())\n"
        "        if any(str(getattr(m, '__file__', '') or '').startswith(d)"
        " for d in dirs)]\n"
        "arrs = [(m.__name__, k, v) for m in mods for k, v in vars(m).items()\n"
        "        if isinstance(v, mx.array)]\n"
        "bad = []\n"
        "def run():\n"
        "    for m, k, v in arrs:\n"
        "        try:\n"
        "            mx.eval(v + 0)\n"
        "        except Exception as e:\n"
        "            bad.append(f'{m}.{k}: {e}')\n"
        "t = threading.Thread(target=run); t.start(); t.join()\n"
        "print(len(mods), 'modules', len(arrs), 'arrays', bad)\n"
        "raise SystemExit(1 if bad or not mods else 0)\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr


def test_no_module_makes_a_lazy_array_at_import():
    """The same failure for any family: a module-level mx.zeros / ones /
    arange / ... is lazy and bound to the importing thread. Arrays made
    from Python data (mx.array([...])) hold their values and are fine."""
    import ast
    from pathlib import Path
    src = Path(__file__).resolve().parents[2] / "src" / "knurlogic"
    lazy = {"zeros", "ones", "full", "arange", "eye", "linspace", "tri",
            "concatenate", "stack", "broadcast_to", "zeros_like", "ones_like"}
    found = []
    for f in src.rglob("*.py"):
        for node in ast.parse(f.read_text()).body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and \
                    isinstance(node.value, ast.Call):
                fn = node.value.func
                if isinstance(fn, ast.Attribute) and fn.attr in lazy and \
                        isinstance(fn.value, ast.Name) and fn.value.id == "mx":
                    found.append(f"{f.relative_to(src)}:{node.lineno}")
    assert not found, found
