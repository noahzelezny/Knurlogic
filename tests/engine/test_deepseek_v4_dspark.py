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
    only summation order differs (measured: under 1e-5)."""
    model = _load()
    head, _ = _head(model)
    got = _trace(model, head)
    err = {k: float(np.abs(got[k] - GOLD[k]).max())
           for k in ("logits", "prefill_main_hidden", "main_hidden",
                     "draft_logits", "confidence")}
    print("max abs err", err)
    assert max(err.values()) < 1e-4
    assert (got["draft_ids"].astype(np.int32) == GOLD["draft_ids"]).all()


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
