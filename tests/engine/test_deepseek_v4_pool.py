"""DeepseekV4Cache's compressed pools grow in place (a buffer with spare
rows, POOL_STEP at a time), not by a whole-pool copy per emitted window.

The pool a reader sees must be the one the old concatenate built, uniform
and per-row (rows of different lengths, zero past each row's own), and a
decode of many emits must reallocate the buffer only once per POOL_STEP."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

mx = pytest.importorskip("mlx.core")


@pytest.fixture(scope="module")
def A():
    from knurlogic.engine import register
    register.register("deepseek_v4")
    import mlx_lm.models.deepseek_v4 as M
    return M


D = 8


def _rows(B, k, seed):
    return mx.array(np.random.default_rng(seed).standard_normal(
        (B, k, D)).astype(np.float32))


def test_uniform_appends_match_concatenate_and_grow_by_steps(A):
    cache = A.DeepseekV4Cache(8)
    br = cache.get_branch(A._K_COMP)
    want = None
    bufs = set()
    for i in range(600):
        new = _rows(2, 1 if i else 5, i)
        want = new if want is None else mx.concatenate([want, new], axis=1)
        cache.update_pool(new, A._K_COMP)
        bufs.add(br._pool_buf.shape[1])
    assert br.pool_lengths is None
    np.testing.assert_array_equal(np.array(br.pool), np.array(want))
    # 604 rows: capacities 256, 512, 768 -- not one buffer per emit
    assert bufs == {256, 512, 768}


def test_per_row_appends_match_the_old_merge(A):
    cache = A.DeepseekV4Cache(8)
    br = cache.get_branch(A._K_COMP)
    br.pool = _rows(3, 4, 0)
    br.pool_lengths = [4, 2, 0]
    want = np.array(br.pool)
    want[1, 2:] = 0
    want[2, :] = 0
    br.pool = mx.array(want)
    lens = [4, 2, 0]
    for step, counts in enumerate(([1, 0, 2], [0, 1, 1], [2, 2, 0])):
        new = _rows(3, max(counts), 10 + step)
        tot = [c + n for c, n in zip(lens, counts)]
        merged = np.zeros((3, max(tot), D), np.float32)
        for i, (c, n) in enumerate(zip(lens, counts)):
            merged[i, :c] = want[i, :c]
            merged[i, c:c + n] = np.array(new)[i, :n]
        want, lens = merged, tot
        br._new_pool_lengths = counts
        got = cache.update_pool(new, A._K_COMP)
        assert br.pool_lengths == lens
        np.testing.assert_array_equal(np.array(got), want)
        np.testing.assert_array_equal(np.array(br.pool), want)
