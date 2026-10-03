"""DeepseekV4Cache rolls a speculative forward back without a replay
(architecture edit 19, engine/mtp/caches.rollback).

A verify forward feeds the trunk W tokens; when the drafts after the
first p are rejected, the cache must end where feeding only those p
tokens leaves it. Until edit 19 that took a restore and a p-wide replay
forward. Here the cache is rolled back by bookkeeping alone and compared
with a twin that was fed only the p tokens: every array of its state (the
window ring, each compressor / indexer branch's carry buffer, overlap
carry and pool) within float32 tolerance -- the pooled rows of the
windows the p tokens complete are the ones the W-wide forward computed,
which a p-wide forward computes with sums of another width (measured:
~1e-6) -- and the logits of the steps after it, 1-wide and multi-token.

A tiny random-weight model with compress ratios 4, 128, 4 and 0 (window
8, an indexer that chooses: index_topk 2), started at positions whose
next W tokens cross a ratio-4 and a ratio-128 pool boundary, from a
prefill (variable carry buffer) and from 1-wide decode steps (the hot
path's fixed buffer); and a merged batch of rows of different lengths
(per-row buffers and pools, a BatchRotatingKVCache window)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

mx = pytest.importorskip("mlx.core")

CONFIG = dict(
    model_type="deepseek_v4", vocab_size=64, hidden_size=64,
    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=1,
    q_lora_rank=32, o_lora_rank=16, o_groups=2, head_dim=32,
    qk_rope_head_dim=16, sliding_window=8, compress_ratios=[4, 128, 4, 0],
    index_n_heads=8, index_head_dim=16, index_topk=2,
    moe_intermediate_size=32, n_routed_experts=4, n_shared_experts=1,
    num_experts_per_tok=2, num_hash_layers=1, hc_mult=4,
    hc_sinkhorn_iters=3,
    rope_scaling={"type": "yarn", "factor": 4,
                  "original_max_position_embeddings": 256,
                  "beta_fast": 32, "beta_slow": 1},
    max_position_embeddings=1024, num_nextn_predict_layers=0,
    tie_word_embeddings=False, eos_token_id=1, bos_token_id=0)

W = 6          # a DSpark verify: t1 and K = 5 drafts
TOL = 1e-4


@pytest.fixture(scope="module")
def model():
    from mlx.utils import tree_flatten

    from knurlogic.engine import register
    register.register("deepseek_v4")
    import mlx_lm.models.deepseek_v4 as M
    m = M.Model(M.ModelArgs.from_dict(CONFIG))
    rng = np.random.default_rng(0)
    w = []
    for k, v in tree_flatten(m.parameters()):
        if k.endswith("tid2eid"):
            a = rng.integers(0, CONFIG["n_routed_experts"],
                             size=v.shape).astype(np.int32)
        elif "switch_mlp" in k and v.dtype == mx.uint8:   # E8M0 scales
            a = rng.integers(118, 124, size=v.shape).astype(np.uint8)
        elif "switch_mlp" in k:                           # packed mxfp4
            a = rng.integers(0, 2 ** 32, size=v.shape,
                             dtype=np.uint64).astype(np.uint32)
        elif k.endswith("norm.weight"):
            a = (1 + 0.1 * rng.standard_normal(v.shape)).astype(np.float32)
        else:
            a = (0.15 * rng.standard_normal(v.shape)).astype(np.float32)
        w.append((k, mx.array(a)))
    m.load_weights(w)
    mx.eval(m.parameters())
    return m


def _ids(n, seed):
    return np.random.default_rng(seed).integers(2, 64, size=n).tolist()


def _fed(model, prompt, decode):
    """A fresh cache fed `prompt` in one prefill, then `decode` 1-wide."""
    c = model.make_cache()
    model(mx.array([prompt]), cache=c)
    for t in decode:
        model(mx.array([[t]]), cache=c)
    return c


def _arrays(c):
    """Every piece of a DeepseekV4Cache's state, by name."""
    from mlx_lm.models.deepseek_v4 import _CompressorBranch
    out = {f"local.{n}": v for n, v in vars(c.local).items()}
    for key, b in c._branches.items():
        for n in _CompressorBranch.__slots__:
            out[f"{key}.{n}"] = getattr(b, n)
    return out


def _same_state(a, b):
    for i, (ca, cb) in enumerate(zip(a, b)):
        assert ca._spec is None and cb._spec is None
        sa, sb = _arrays(ca), _arrays(cb)
        assert sa.keys() == sb.keys()
        for n in sa:
            x, y = sa[n], sb[n]
            where = f"layer {i} {n}"
            if isinstance(x, mx.array) or isinstance(y, mx.array):
                assert isinstance(x, mx.array) and isinstance(y, mx.array), where
                assert x.shape == y.shape, (where, x.shape, y.shape)
                if x.dtype in (mx.int32, mx.int64, mx.bool_):
                    assert mx.array_equal(x, y).item(), where
                else:
                    xf, yf = x.astype(mx.float32), y.astype(mx.float32)
                    # -inf (an empty overlap carry) on both sides
                    assert mx.array_equal(mx.isinf(xf), mx.isinf(yf)).item()
                    d = mx.abs(mx.where(mx.isinf(xf), 0, xf - yf)).max()
                    assert d.item() < TOL, (where, d.item())
            else:
                assert x == y, (where, x, y)


def _same_after(model, a, b, rows):
    """The same next steps on both: three 1-wide, then a 3-wide."""
    rng = np.random.default_rng(7)
    for width in (1, 1, 1, 3):
        ids = mx.array(rng.integers(2, 64, size=(rows, width)).tolist())
        la, lb = model(ids, cache=a), model(ids, cache=b)
        d = mx.abs(la - lb).max().item()
        assert d < TOL, d
        assert mx.array_equal(la.argmax(-1), lb.argmax(-1)).item()


def _spec_then_rollback(model, cache, block, keep):
    from knurlogic.engine.mtp import caches as C
    snaps = C.snapshot(cache)
    assert {s[0] for s in snaps} == {"spec"}
    model(mx.array(block), cache=cache)
    assert C.rollback(cache, snaps, keep)


# start positions (prompt, decode steps): the W tokens after them cross a
# ratio-128 boundary at 128 and ratio-4 ones; from a prefill's variable
# carry and from the decode hot path's fixed buffer
STARTS = {"prefill-125": (125, 0), "prefill-126": (126, 0),
          "decode-123+3": (123, 3), "decode-120+5": (120, 5),
          "short-5+2": (5, 2)}


@pytest.mark.parametrize("keep", list(range(W + 1)))
@pytest.mark.parametrize("start", list(STARTS))
def test_a_rolled_back_cache_is_one_fed_only_the_kept_tokens(model, start,
                                                             keep):
    n, steps = STARTS[start]
    prompt, dec = _ids(n, 1), _ids(steps, 2)
    block = [_ids(W, 3)]
    a, b = _fed(model, prompt, dec), _fed(model, prompt, dec)
    _spec_then_rollback(model, a, block, keep)
    if keep:
        model(mx.array([block[0][:keep]]), cache=b)
    _same_state(a, b)
    _same_after(model, a, b, 1)


@pytest.mark.parametrize("keep", [0, 1, 2, 4, W])
@pytest.mark.parametrize("decode", [0, 3])
def test_a_merged_batch_of_rows_of_different_lengths_rolls_back(
        model, keep, decode):
    """Rows at 126, 7 and 63 tokens (ragged buffers, pools and window)
    merged as the batch engine merges them, `decode` 1-wide steps, then the
    W-wide forward rolled back to `keep`."""
    from mlx_lm.generate import _merge_caches
    prompts = [_ids(126, 1), _ids(7, 4), _ids(63, 5)]

    def merged():
        c = _merge_caches([_fed(model, p, []) for p in prompts])
        for t in _ids(decode, 6):
            model(mx.array([[t]] * 3), cache=c)
        return c
    a, b = merged(), merged()
    block = [_ids(W, 10 + i) for i in range(3)]
    _spec_then_rollback(model, a, block, keep)
    if keep:
        model(mx.array([r[:keep] for r in block]), cache=b)
    _same_state(a, b)
    _same_after(model, a, b, 3)


def test_a_cache_that_cannot_roll_forward_is_restored(model):
    """Two forwards since the snapshot: not a recorded verify. `rollback`
    refuses and restores the snapshot (the caller replays)."""
    from knurlogic.engine.mtp import caches as C
    prompt = _ids(9, 1)
    a, b = _fed(model, prompt, []), _fed(model, prompt, [])
    snaps = C.snapshot(a)
    model(mx.array([_ids(3, 2)]), cache=a)
    model(mx.array([_ids(3, 3)]), cache=a)
    assert not C.rollback(a, snaps, 2)
    _same_state(a, b)


def test_keeping_every_token_stops_the_record(model):
    from knurlogic.engine.mtp import caches as C
    prompt = _ids(9, 1)
    a, b = _fed(model, prompt, []), _fed(model, prompt, [])
    snaps = C.snapshot(a)
    block = [_ids(W, 3)]
    model(mx.array(block), cache=a)
    C.release(a, snaps)
    model(mx.array(block), cache=b)
    _same_state(a, b)
