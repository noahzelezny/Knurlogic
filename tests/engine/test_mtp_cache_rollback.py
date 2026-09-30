"""Rollback of GLM's composite attention cache inside the batch engine.

glm5_next's full-attention layers keep a CacheList (main KV + indexer KV).
Batched, its members are mlx-lm BatchKVCaches, whose offset is one per row;
rollback must use their shared write index. Without it, serving
GLM-5.3-Flash 2.7 raises TypeError on the first drafting step.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
mx = pytest.importorskip("mlx.core")


def _batched_list(rows=2, steps=5):
    from mlx_lm.models.cache import BatchKVCache
    from knurlogic.engine.families.glm5.architecture.glm5_next._mlx_vlm.models.cache \
        import CacheList
    cl = CacheList(BatchKVCache([0] * rows), BatchKVCache([0] * rows))
    for _ in range(steps):
        for c in cl.caches:
            k = mx.random.normal((rows, 2, 1, 8))
            c.update_and_fetch(k, k)
    return cl


def test_a_batched_cachelist_rolls_back_to_the_snapshot():
    from knurlogic.engine.mtp import caches
    cl = _batched_list()
    before = [c.size() for c in cl.caches]
    snaps = caches.snapshot([cl])
    assert snaps[0][0] == "attn-list"
    for _ in range(2):                                 # a 2-token verify
        for c in cl.caches:
            k = mx.random.normal((2, 2, 1, 8))
            c.update_and_fetch(k, k)
    caches.restore([cl], snaps)
    assert [c.size() for c in cl.caches] == before


def test_batched_cachelist_rollback_can_fail():
    """Holding anything that is not an attention cache, the composite is
    refused, not guessed at."""
    from knurlogic.engine.mtp import caches
    from knurlogic.engine.families.glm5.architecture.glm5_next._mlx_vlm.models.cache \
        import ArraysCache, CacheList
    with pytest.raises(TypeError):
        caches.snapshot([CacheList(ArraysCache(size=2))])


def test_snapshot_semantics_check_takes_batched_and_composite_caches():
    """the check copied s[2] of every non-"attn" snapshot, and "battn" /
    "attn-list" snapshots carry None there -- a TypeError on exactly the
    caches the batch engine uses, instead of an answer."""
    import mlx.core as mx
    from types import SimpleNamespace
    from knurlogic.engine.mtp import caches as C

    class Batched:                       # keys + trim, per-row offsets
        def __init__(self):
            self.keys, self.offset = mx.zeros((1,)), mx.array([3])

        def trim(self, n):
            pass

        def size(self):
            return 3

    state = SimpleNamespace(cache=[mx.array([1.0])])

    def reassign():
        state.cache = [mx.array([2.0])]
    assert C.check_snapshot_semantics([Batched(), state], reassign) is True

    def mutate():
        state.cache[0][0] = 5.0
    state.cache = [mx.array([1.0])]
    assert C.check_snapshot_semantics([Batched(), state], mutate) is False
