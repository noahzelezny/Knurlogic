"""DeepSeek-V4-Flash-Vision-Exp's DSpark drafter
(families/deepseek/heads/deepseek_v4_dspark.py, engine/mtp/block_loop.py)
held to DeepSeek's own reference code, and drafting with it exact.

The fixtures are a tiny HF checkpoint in the official names and formats
(FP8 + E8M0 block scales, FP4 experts) and the MLX artifact made from it:
the trunk through the vendored sanitize, the sidecar through
heads/dspark_pack.py. The golden is the reference's Transformer.forward /
forward_spec on that checkpoint under torch
(tests/support/goldens/build_deepseek_v4_dspark.py)."""
import json
import os
import shutil
import struct
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests" / "support" / "goldens"))

mx = pytest.importorskip("mlx.core")

import build_deepseek_v4_dspark as G  # noqa: E402

GOLD = dict(np.load(G.OUT))
#: a sidecar packed from the real checkpoint (optional; headers only)
REAL = Path(os.environ.get("KNURLOGIC_TEST_DSPARK_SIDECAR", "/nonexistent"))
REAL_CONFIG = Path("/Volumes/Models/Teacher Models/"
                   "deepseek-ai--DeepSeek-V4-Flash-Vision-Exp/config.json")


def _load(path=G.TINY):
    from mlx_lm.utils import load_model

    from knurlogic.interfaces import loading
    from knurlogic.machine.artifact import Artifact
    assert loading.register(Artifact.load(str(path))) == []
    model, _ = load_model(Path(path))
    return model


def _head(model, path=G.TINY):
    from knurlogic.engine.mtp import registry
    return registry.load_head(model, model_path=path)


def test_the_dspark_sidecar_is_found_and_chosen():
    """find_head sees it (fit and launch count it like any head); the
    registry picks DSpark for a config with dspark_block_size."""
    from knurlogic.engine import mtp
    from knurlogic.engine.families.deepseek.heads.deepseek_v4_dspark import \
        DSparkHead
    found = mtp.find_head(G.TINY)
    assert found is not None and found.family == "deepseek_v4"
    assert found.path.name == "mtp-head-dspark-mxfp4.safetensors"
    head, spec = _head(_load())
    assert isinstance(head, DSparkHead) and head.block_size == G.K
    assert spec.name == "deepseek_v4" and spec.head.endswith(":DSparkHead")
    gp = head.m.stages[0].ffn.switch_mlp.gate_proj
    assert type(gp).__name__ == "QuantizedSwitchLinear" and gp.mode == "mxfp4"
    assert head.capture_paths() == ["layers.2", "layers.3", "hc_head"]


def test_a_dspark_sidecar_needs_a_dspark_config(tmp_path):
    from knurlogic.engine.mtp import registry
    d = tmp_path / "m"
    shutil.copytree(G.TINY, d)
    cfg = json.loads((d / "config.json").read_text())
    cfg.pop("dspark_block_size")
    (d / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="dspark_block_size"):
        # as the server asks: the sidecar find_head found
        registry.load_head(_load(d), sidecar=d / G.TINY.joinpath(
            "mtp-head-dspark-mxfp4.safetensors").name)


def _trace(model, head):
    """The golden's run through the MLX trunk and head: prefill, then each
    forced decode token -> the trunk's logits, the main hidden state and
    the head's draft (temperature 0)."""
    from knurlogic.engine.mtp.capture import capture_input
    from contextlib import ExitStack
    out = {k: [] for k in ("logits", "main_hidden", "draft_ids",
                           "draft_logits", "confidence")}
    with ExitStack() as st:
        gets = [st.enter_context(capture_input(model.model, p))
                for p in head.capture_paths()]
        cache, hc = model.make_cache(), head.make_draft_cache()
        lg = model(mx.array([G.PROMPT]), cache=cache)
        mh = head.main_hidden([g() for g in gets])
        out["prefill_main_hidden"] = mh[0]
        head.advance(mh, hc)
        out["logits"].append(lg[0, -1])
        for t in G.DECODE:
            lg = model(mx.array([[t]]), cache=cache)
            mh = head.main_hidden([g() for g in gets])
            head.advance(mh, hc)
            ids, dl, conf = head.draft(mx.argmax(lg[:, -1], axis=-1), hc)
            for k, v in (("logits", lg[0, -1]), ("main_hidden", mh[0, -1]),
                         ("draft_ids", ids[0]), ("draft_logits", dl[0]),
                         ("confidence", conf[0])):
                out[k].append(v)
    return {k: np.array(mx.stack(v) if isinstance(v, list) else v)
            .astype(np.float32) for k, v in out.items()}


def test_the_drafts_are_the_references():
    """Trunk logits, the HC-mean main hidden state of layers 1-3, and at
    each decode step forward_spec's 5 draft ids (exact), their logits with
    the Markov term, and the confidence scores, against the torch
    reference on the same checkpoint. Everything runs float32 on weights
    both sides hold exactly (FP8 and FP4 values times powers of two), so
    only summation order differs (measured: under 1e-5). The trunk has
    compressed layers (ratio 4 with the indexer choosing 8 of 31 rows,
    ratio 128) and both sides round the kv, pooled rows and indexer query
    through FP8 / FP4 as the reference does, and every FP8 / FP4 linear's
    input through act_quant (measured without the latter: 1.6 on the
    logits, 5.6 on the draft logits, other draft ids)."""
    model = _load()
    head, _ = _head(model)
    got = _trace(model, head)
    err = {k: float(np.abs(got[k] - GOLD[k]).max())
           for k in ("logits", "prefill_main_hidden", "main_hidden",
                     "draft_logits", "confidence")}
    print("max abs err", err)
    assert max(err.values()) < 1e-4
    assert (got["draft_ids"].astype(np.int32) == GOLD["draft_ids"]).all()


def test_the_low_precision_simulation_is_the_reference_kernels():
    """act_quant (ue8m0 scales, block 64 and 128), fp4_act_quant (block 32) and
    rotate_activation, as op graphs and as the fused Metal kernels the
    model runs, each against the torch port of its CUDA kernel
    (the golden's `qat_*`), bit for bit: float32 inputs over 2**-20..2**12,
    zero blocks, blocks under the amax floors, rounding ties, and the same
    rows as bf16 (the real model's dtype)."""
    model = _load()
    A = sys.modules[type(model).__module__]
    for name, dt in (("x", mx.float32), ("xb", mx.bfloat16)):
        x = mx.array(GOLD[f"qat_{name}"]).astype(dt)
        for fn, key in ((A.fp8_simulate, "fp8"), (A.fp4_simulate, "fp4")):
            got = np.array(fn(x).astype(mx.float32))
            assert (got == GOLD[f"qat_{name}_{key}"]).all(), (name, key)
        for d in (64, 128):
            got = np.array(A.rotate_activation(x.reshape(-1, d))
                           .astype(mx.float32)).reshape(x.shape)
            assert (got == GOLD[f"qat_{name}_rot{d}"]).all(), (name, d)
        # the fused kernels the model runs: FP8 on the first 192 dims
        # with the last 64 passed through, and rotate + FP4 per 128
        want = np.concatenate([GOLD[f"qat_{name}_fp8"][:, :192],
                               GOLD[f"qat_{name}"][:, 192:]], axis=1)
        if name == "xb":
            want[:, 192:] = np.array(x[:, 192:].astype(mx.float32))
        got = np.array(A.fp8_simulate_nope(x, 64).astype(mx.float32))
        assert (got == want).all(), name
        got = np.array(A.fp4_simulate_rotated(x.reshape(-1, 128))
                       .astype(mx.float32)).reshape(x.shape)
        assert (got == GOLD[f"qat_{name}_rotfp4"]).all(), name
        # a linear's input (edit 21): act_quant in blocks of 128
        for fn in (lambda v: A.fp8_simulate(v, 128), A.fp8_act):
            got = np.array(fn(x).astype(mx.float32))
            assert (got == GOLD[f"qat_{name}_fp8_128"]).all(), name


def _run(gen, prompts, max_tokens, sampler=None, caches=None,
         prefixes=None):
    uids = gen.insert_segments(
        segments=[[p] for p in prompts], max_tokens=[max_tokens] * len(prompts),
        caches=caches or [None] * len(prompts),
        all_tokens=prefixes or [[] for _ in prompts],
        **({"samplers": sampler} if sampler is not None else {}))
    out, done, ents = {u: [] for u in uids}, set(), {}
    for _ in range(10_000):
        _, grs = gen.next()
        for r in grs:
            out[r.uid].append(r.token)
            if r.finish_reason is not None:
                done.add(r.uid)
                ents[r.uid] = (r.prompt_cache, r.all_tokens)
        if len(done) == len(uids):
            break
    gen.close()
    return [out[u] for u in uids], [ents[u] for u in uids]


def _guess(head, seqs):
    """Drafts that are the plain run's tokens (`seqs`: its prompts +
    outputs) for the first r positions of a block and wrong after, r
    from 0 to K varying with the position and the row: steps commit 1 to
    K + 1 tokens, and rows of one batch accept different counts. Each row
    is found by its position and its t1."""
    real = head.block

    def block(t, cache):
        x, base = real(t, cache)
        want = []
        for i, (p, tok) in enumerate(zip(cache.lengths, t.tolist())):
            seq = next((q for q in seqs if len(q) > p and q[p] == tok), [])
            right = (3 * p + i) % (G.K + 1)
            want.append([seq[p + 1 + k] if k < right and p + 1 + k < len(seq)
                         else G.NOISE for k in range(G.K)])
        hot = mx.arange(base.shape[-1]) == mx.array(want)[..., None]
        return x, base + 100.0 * hot.astype(base.dtype)
    head.block = block


PROMPTS = {"one-row": [G.PROMPT],
           "three-rows": [G.PROMPT, G.PROMPT[:3], G.PROMPT[2:9] + G.DECODE]}


@pytest.mark.parametrize("rows", list(PROMPTS))
@pytest.mark.parametrize("guess", [False, True], ids=["head", "guess"])
def test_greedy_block_drafting_is_the_plain_steps_token_for_token(
        rows, guess, monkeypatch):
    """`head`: the random stages' own drafts, nearly all rejected (every
    step restores the trunk and replays). `guess`: drafts partly right, so
    steps commit 1 to 6 tokens and rows of one batch accept different
    counts (the batch commits its fewest)."""
    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    from knurlogic.engine.mtp.block_loop import BlockBatch
    monkeypatch.setenv("KNURLOGIC_MTP_BATCH_MAX_ROWS", "8")   # draft always
    model = _load()
    head, _ = _head(model)
    prompts = PROMPTS[rows]
    plain, _ = _run(MTPBatchGenerator(model, None, prefill_step_size=4),
                    prompts, 24)
    if guess:
        _guess(head, [p + o for p, o in zip(prompts, plain)])
    stats = {}
    gen = MTPBatchGenerator(model, head, stats=stats, prefill_step_size=4)
    assert isinstance(gen._batch, BlockBatch)
    draft, ents = _run(gen, prompts, 24)
    assert stats["steps"] > 0
    if guess:
        assert stats["accepted"] > stats["steps"]   # > 1 token a step
    assert draft == plain
    assert all(len(t) == 24 for t in draft)


def test_plain_steps_and_block_steps_interleave_exactly(monkeypatch):
    """A ceiling of one drafting row: three rows take plain steps (the
    head's cache still takes every committed position), and as they
    finish the last one drafts -- a block verify right after 1-wide steps
    (architecture edit 17)."""
    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    monkeypatch.setenv("KNURLOGIC_MTP_BATCH_MAX_ROWS", "1")
    model = _load()
    head, _ = _head(model)
    prompts = PROMPTS["three-rows"]
    plain, _ = _run(MTPBatchGenerator(model, None, prefill_step_size=4),
                    prompts, 24)
    _guess(head, [p + o for p, o in zip(prompts, plain)])
    stats = {}
    draft, _ = _run(MTPBatchGenerator(model, head, stats=stats,
                                      prefill_step_size=4), prompts, 24)
    assert 0 < stats["steps"] and draft == plain


def _steps(batch, rows, n):
    from knurlogic.engine.mtp.batch_loop import RowParams, admit
    out = {}
    for uid, (ids, drafts) in enumerate(rows):
        p = RowParams(max_tokens=n, dist=None, processors=[], eos=set(),
                      drafts=drafts)
        batch.extend([admit(batch.model, batch.head, batch.get_h,
                            mx.array(ids), p, uid=uid,
                            make_draft_cache=(batch.head.make_draft_cache
                                              if batch.head else
                                              lambda: None),
                            prefill_step_size=4)])
        out[uid] = []
    while len(batch):
        for rs in batch.step():
            out[rs.uid] += [e.token for e in rs.tokens]
    return [out[u] for u in sorted(out)]


def test_a_row_that_does_not_draft_rides_along(monkeypatch):
    """A row that may not draft (an image request's, when its image is in
    the uncached span) beside one that does: the batch commits t1 alone
    each step, both rows exact."""
    from contextlib import ExitStack

    from knurlogic.engine.mtp.batch_loop import MTPBatch
    from knurlogic.engine.mtp.block_loop import BlockBatch
    from knurlogic.engine.mtp.capture import capture_input
    monkeypatch.setenv("KNURLOGIC_MTP_BATCH_MAX_ROWS", "8")
    model = _load()
    head, _ = _head(model)
    rows = [(G.PROMPT, True), (G.PROMPT[2:9], False)]
    plain = _steps(MTPBatch(model, None, lambda: None, copy_caches=True),
                   rows, 16)
    _guess(head, [r[0] + o for r, o in zip(rows, plain)])
    with ExitStack() as st:
        gets = [st.enter_context(capture_input(model.model, p))
                for p in head.capture_paths()]
        b = BlockBatch(model, head,
                       lambda: head.main_hidden([g() for g in gets]),
                       copy_caches=True)
        got = _steps(b, rows, 16)
    assert (2, True) in b._cost       # block steps with both rows
    assert got == plain


def test_a_seeded_row_reproduces_with_drafting(monkeypatch):
    """A seeded row's token at each position is the target's draw under
    that position's key, drafted or not; beside a greedy row."""
    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    monkeypatch.setenv("KNURLOGIC_MTP_BATCH_MAX_ROWS", "8")
    model = _load()
    head, _ = _head(model)
    prompts = [G.PROMPT, G.PROMPT[2:9]]
    samp = [{"temp": 0.8, "seed": 1234}, None]
    plain, _ = _run(MTPBatchGenerator(model, None, prefill_step_size=4),
                    prompts, 20, sampler=samp)
    _guess(head, [p + o for p, o in zip(prompts, plain)])
    stats = {}
    draft, _ = _run(MTPBatchGenerator(model, head, stats=stats,
                                      prefill_step_size=4),
                    prompts, 20, sampler=samp)
    assert stats["accepted"] > 0
    assert draft == plain
    again, _ = _run(MTPBatchGenerator(model, head, prefill_step_size=4),
                    prompts, 20, sampler=samp)
    assert again == draft


def test_a_finished_rows_entry_carries_its_dspark_cache(monkeypatch):
    """The prompt-cache entry of a drafting row holds the DSpark cache at
    the trunk's offset, and a request continuing from it drafts the same
    tokens a fresh prefill of the whole prompt does."""
    from knurlogic.engine.families.deepseek.heads.deepseek_v4_dspark import \
        DSparkCache
    from knurlogic.engine.mtp.batch_generator import (MTPBatchGenerator,
                                                      trunk_offset)
    from knurlogic.engine.mtp.caches import position
    monkeypatch.setenv("KNURLOGIC_MTP_BATCH_MAX_ROWS", "8")
    model = _load()
    head, _ = _head(model)
    (gold,), _ = _run(MTPBatchGenerator(model, None, prefill_step_size=4),
                      [G.PROMPT], 30)
    _guess(head, [G.PROMPT + gold])
    (first,), ((entry, fed),) = _run(
        MTPBatchGenerator(model, head, prefill_step_size=4), [G.PROMPT], 9)
    assert isinstance(entry[-1], DSparkCache)
    hit = trunk_offset(entry[:-1])
    assert position(entry[-1]) == hit and hit <= len(fed)
    tail = fed[hit:] + G.DECODE
    stats = {}
    cont, _ = _run(MTPBatchGenerator(model, head, stats=stats,
                                     prefill_step_size=4),
                   [tail], 12, caches=[entry], prefixes=[fed[:hit]])
    fresh, _ = _run(MTPBatchGenerator(model, head, prefill_step_size=4),
                    [fed + G.DECODE], 12)
    assert stats["steps"] > 0 and cont == fresh


def _header(f):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))


@pytest.mark.skipif(not (REAL.is_file() and REAL_CONFIG.is_file()),
                    reason="no sidecar packed from the real checkpoint "
                           "(KNURLOGIC_TEST_DSPARK_SIDECAR)")
def test_the_real_sidecar_binds_by_its_header_alone():
    """Every name and shape dspark_pack wrote from the real checkpoint,
    against stages built from its config -- lazily, nothing loaded."""
    from knurlogic.engine import register
    from knurlogic.engine.families.deepseek.heads.deepseek_v4_dspark import \
        DSparkHead
    register.register("deepseek_v4")
    import mlx_lm.models.deepseek_v4 as M
    hdr = _header(REAL)
    meta = hdr.pop("__metadata__", {})
    assert json.loads(meta["knurlogic_mtp"])["kind"] == "dspark"
    args = M.ModelArgs.from_dict(json.loads(REAL_CONFIG.read_text()))
    core = types.SimpleNamespace(args=args)
    head = DSparkHead(types.SimpleNamespace(model=core), M)
    shapes = {k: types.SimpleNamespace(shape=tuple(v["shape"]))
              for k, v in hdr.items()}
    mw = head.bind(shapes)
    assert len(mw) == len(hdr)
    st = head.m.stages
    assert len(st) == 3 and st[0].ffn.switch_mlp.gate_proj.mode == "mxfp4"
    assert hasattr(st[0].ffn.gate, "bias_vl")
    assert head.capture_paths() == ["layers.41", "layers.42", "hc_head"]


def test_a_row_ending_mid_block_stores_only_what_it_committed(monkeypatch):
    """A row that ends partway through a step's accepted drafts (max_tokens
    anywhere from 4 to 13, steps committing 1 to 6 tokens): its entry is at
    exactly the tokens the client saw, its DSpark cache beside it, and
    the next turn restored from it is a fresh prefill's, token for
    token."""
    import copy

    from knurlogic.engine.mtp.batch_generator import (
        MTPBatchGenerator,
        trunk_offset,
    )
    from knurlogic.engine.mtp.caches import position
    monkeypatch.setenv("KNURLOGIC_MTP_BATCH_MAX_ROWS", "8")
    model = _load()
    head, _ = _head(model)
    (gold,), _ = _run(MTPBatchGenerator(model, None, prefill_step_size=4),
                      [G.PROMPT], 30)
    _guess(head, [G.PROMPT + gold])
    for n in range(4, 14):
        (toks,), ((entry, fed),) = _run(
            MTPBatchGenerator(model, head, prefill_step_size=4),
            [G.PROMPT], n)
        assert toks == gold[:n]
        assert fed == G.PROMPT + toks, n
        assert trunk_offset(entry[:-1]) == len(fed)
        assert position(entry[-1]) == len(fed)
        stats = {}
        cont, _ = _run(MTPBatchGenerator(model, head, stats=stats,
                                         prefill_step_size=4),
                       [G.DECODE], 8, caches=[copy.deepcopy(entry)],
                       prefixes=[fed])
        fresh, _ = _run(MTPBatchGenerator(model, head, prefill_step_size=4),
                        [fed + G.DECODE], 8)
        assert stats["steps"] > 0 and cont == fresh, n


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_every_verify_width_is_the_plain_steps_token_for_token(k,
                                                               monkeypatch):
    """KNURLOGIC_MTP_VERIFY=k: each step verifies only the first k of the
    head's K drafts (three rows, partly right drafts); greedy output is the
    plain run's at every k, and the rows still accept."""
    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    monkeypatch.setenv("KNURLOGIC_MTP_BATCH_MAX_ROWS", "8")
    monkeypatch.setenv("KNURLOGIC_MTP_VERIFY", str(k))
    model = _load()
    head, _ = _head(model)
    prompts = PROMPTS["three-rows"]
    plain, _ = _run(MTPBatchGenerator(model, None, prefill_step_size=4),
                    prompts, 24)
    _guess(head, [p + o for p, o in zip(prompts, plain)])
    stats = {}
    gen = MTPBatchGenerator(model, head, stats=stats, prefill_step_size=4)
    draft, _ = _run(gen, prompts, 24)
    assert stats["accepted"] > 0
    assert draft == plain


def _width_batch(K=5):
    from knurlogic.engine.mtp.block_loop import BlockBatch
    b = BlockBatch(None, None, lambda: None, copy_caches=True, block_size=K)
    # every width timed: a step costs 100 ms + 10 ms per verified draft
    for k in range(1, K + 1):
        b._vcost[(1, k)] = (0.100 + 0.010 * k, 99)
    return b


def _feed(b, pred, hits, n):
    """n verified-at-K steps whose confidence said `pred` and whose batch
    accepted `hits` drafts."""
    for _ in range(n):
        b._pred = list(pred)
        b._record_accept(1, b.block_size, hits)


def test_the_verify_width_follows_calibrated_confidence(monkeypatch):
    """Confidence maps to k by expected tokens per second: per position
    P(batch accepts >= j) = the product over drafting rows of the running
    product of sigmoid(score); k = argmax (1 + sum_{j<=k} P_j) / T(k) with
    T timed per width. Calibrated (the predictions matched the verdicts),
    each step's own scores choose: a confident step verifies all 5, a
    doubtful one 1."""
    monkeypatch.delenv("KNURLOGIC_MTP_VERIFY", raising=False)
    import mlx.core as mx
    b = _width_batch()
    # history: the scores said ~2 of 5 and 2 were accepted -- calibrated
    _feed(b, [0.97, 0.95, 0.05, 0.03, 0.01], 2, 40)
    assert b.calibrated()
    sure = b._predicted(mx.full((1, 5), 8.0), [True])        # P ~ 1
    assert b._width(1, sure) == 5
    unsure = b._predicted(mx.array([[0.5, -6.0, -6.0, -6.0, -6.0]]), [True])
    assert b._width(1, unsure) == 1
    # two rows: the batch accepts j only if both do
    two = b._predicted(mx.array([[8.0] * 5, [8.0, 8.0, -8.0, -8.0, -8.0]]),
                       [True, True])
    assert two[1] > 0.99 and two[2] < 0.01
    assert b._width(1, two) == 2


def test_badly_calibrated_confidence_falls_back_to_measured_acceptance(
        monkeypatch):
    """The scores promised every draft and the batch accepted none: not
    calibrated, so the width is the measured acceptance's (k = 1 here),
    whatever this step's scores say."""
    monkeypatch.delenv("KNURLOGIC_MTP_VERIFY", raising=False)
    import mlx.core as mx
    b = _width_batch()
    _feed(b, [0.99] * 5, 0, 40)
    assert not b.calibrated()
    sure = b._predicted(mx.full((1, 5), 8.0), [True])
    assert b._width(1, sure) == 1
    # measured acceptance high again: the measured width is all 5
    _feed(b, [0.99] * 5, 5, 40)
    assert b._width(1, None) == 5


def test_a_widths_compiling_first_step_does_not_decide_it(monkeypatch):
    """Each width's first step compiles its verify shape (live: hundreds of
    ms). Left in, it kept every narrow width dearer than K, which alone is
    re-measured, so K won forever. The first step is left out, and every
    losing width is re-measured in turn."""
    from knurlogic.engine.mtp import block_loop as BL
    monkeypatch.delenv("KNURLOGIC_MTP_VERIFY", raising=False)
    b = BL.BlockBatch(None, None, lambda: None, copy_caches=True,
                      block_size=5)
    cost = {1: 0.070, 2: 0.077, 3: 0.085, 4: 0.096, 5: 0.107}
    seen = set()
    picks = []
    for _ in range(400):
        k = b._width(1, None)
        # live acceptance: P(>= j) falls fast after the first draft
        b._record_accept(1, k, min(k, 1 if len(picks) % 3 else 2))
        b._record_width(1, k, cost[k] + (0.8 if k not in seen else 0.0))
        seen.add(k)
        picks.append(k)
    settled = picks[-50:]
    assert max(set(settled), key=settled.count) in (1, 2)
    # the rechecks reach widths other than K
    late = picks[60:]
    assert {3, 4} & set(late)


def _stream(b, cost, acc, steps, rng, spikes=True):
    """`steps` steps of the picker against a synthetic stream: a step
    verifying k costs cost[k] seconds plus 2 ms of noise; with `spikes`,
    each width's first step compiles (+0.8 s), one step in 20 stalls
    (+40..150 ms) and one in 100 recompiles (+0.5 s). The batch accepts
    >= j drafts with chance acc[j - 1]. -> the picks."""
    picks = []
    for _ in range(steps):
        k = b._width(1, None)
        u = rng.random()
        b._record_accept(1, k, sum(u < a for a in acc[:k]))
        t = cost[k] + rng.gauss(0, 0.002)
        if spikes:
            if (1, k) not in b._vwarm:
                t += 0.8
            elif rng.random() < 0.05:
                t += rng.uniform(0.040, 0.150)
            elif rng.random() < 0.01:
                t += 0.5
        b._record_width(1, k, t)
        picks.append(k)
    return picks


def _best(cost, acc):
    return max(cost, key=lambda k: (1 + sum(acc[:k])) / cost[k])


@pytest.mark.parametrize("acc", [
    [0.9, 0.75, 0.4, 0.2, 0.1],         # 3 wins, 2 within 4%
    [0.6, 0.35, 0.1, 0.05, 0.02],       # 2, 1 and 3 within 11%
    [0.98, 0.95, 0.9, 0.85, 0.8],       # all 5
])
def test_the_verify_width_converges_through_outliers_and_spikes(
        monkeypatch, acc):
    """Costs as live (whole steps of ~70, 77, 85, 97, 107 ms at widths
    1-5) under noise, stalls and recompiles: each width's cost is the
    median of its recent steps, so no outlier holds a width dear (live,
    an EMA read width 2 at 113 ms for as long as it went unmeasured), and
    the picker settles on the width that is truly best."""
    import random
    from knurlogic.engine.mtp import block_loop as BL
    monkeypatch.delenv("KNURLOGIC_MTP_VERIFY", raising=False)
    cost = {1: 0.070, 2: 0.077, 3: 0.085, 4: 0.097, 5: 0.107}
    b = BL.BlockBatch(None, None, lambda: None, copy_caches=True,
                      block_size=5)
    picks = _stream(b, cost, acc, 1500, random.Random(3))
    best = _best(cost, acc)
    late = picks[-400:]
    assert max(set(late), key=late.count) == best
    # the rechecks: about one step in RECHECK_STEP away from the best
    assert late.count(best) >= len(late) * (1 - 2 / BL.RECHECK_STEP)
    for k in cost:          # every estimate within a few ms of the truth
        assert abs(b._wcost(1, k) - cost[k]) < 0.004, (k, b._wcost(1, k))


def test_a_width_timed_too_few_times_takes_the_fitted_line(monkeypatch):
    """Widths with fewer than COST_MIN timed steps are priced on the line
    through the widths that have them; before any are, the picker times
    K and 1 only, and the rest come from the line until timed."""
    import random
    from knurlogic.engine.mtp import block_loop as BL
    monkeypatch.delenv("KNURLOGIC_MTP_VERIFY", raising=False)
    b = BL.BlockBatch(None, None, lambda: None, copy_caches=True,
                      block_size=5)
    cost = {1: 0.070, 2: 0.077, 3: 0.085, 4: 0.097, 5: 0.107}
    picks = _stream(b, cost, [0.9, 0.8, 0.7, 0.6, 0.5],
                    2 * (BL.EXPLORE_STEPS + 1), random.Random(0),
                    spikes=False)
    assert set(picks) == {1, 5}
    for k in (2, 3, 4):
        assert (1, k) not in b._vcost
        line = cost[1] + (cost[5] - cost[1]) * (k - 1) / 4
        assert abs(b._wcost(1, k) - line) < 0.003
    # one outlier among a width's samples does not move its cost (the
    # first step, compiling, is left out)
    for t in (0.900, 0.077, 0.078, 0.200, 0.076):
        b._record_width(1, 2, t)
    assert abs(b._wcost(1, 2) - 0.0775) < 0.001


def test_the_profile_logs_past_its_window(monkeypatch):
    """KNURLOGIC_MTP_PROFILE=1 through more than one logging window: the
    phase and width lines print and the engine keeps stepping (a name
    reused in the width line once broke the second window, live)."""
    from knurlogic.engine.mtp import block_loop as BL
    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    monkeypatch.setattr(BL, "PROFILE", True)
    monkeypatch.setattr(BL, "PROFILE_EVERY", 4)
    monkeypatch.delenv("KNURLOGIC_MTP_VERIFY", raising=False)
    model = _load()
    head, _ = _head(model)
    gen = MTPBatchGenerator(model, head, stats={}, prefill_step_size=4)
    out, _ = _run(gen, PROMPTS["three-rows"][:1], 40)
    assert len(out[0]) == 40


def test_the_profile_window_on_a_rank_that_records_no_acceptance(monkeypatch):
    """A split's follower drafts nothing and records no acceptance: its
    width line must still print (live, an empty list there was indexed and
    the follower died, desyncing the ring)."""
    from knurlogic.engine.mtp import block_loop as BL
    monkeypatch.setattr(BL, "PROFILE", True)
    monkeypatch.setattr(BL, "PROFILE_EVERY", 1)
    b = BL.BlockBatch(None, None, lambda: None, copy_caches=True,
                      block_size=5)
    b._clock = lambda: 0.0
    for _ in range(3):
        b._prof_start()
        b._prof_end(True, 2)
