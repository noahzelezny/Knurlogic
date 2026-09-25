"""engine/serve/cache_guard: an exact prompt-cache hit is handed back one
token short, so the batch path always has a segment to process."""
import pytest

pytest.importorskip("mlx.core")


def test_an_exact_hit_leaves_one_token(monkeypatch):
    from mlx_lm.models import cache as C
    from knurlogic.engine.serve import cache_guard
    trimmed = []
    real = C.LRUPromptCache.fetch_nearest_cache
    monkeypatch.setattr(C.LRUPromptCache, "fetch_nearest_cache",
                        lambda self, m, t: (["kv"], []))
    monkeypatch.setattr(C, "can_trim_prompt_cache", lambda c: True)
    monkeypatch.setattr(C, "trim_prompt_cache",
                        lambda c, n: trimmed.append(n))
    cache_guard.install()
    got = C.LRUPromptCache.fetch_nearest_cache(None, "m", [1, 2, 3])
    assert got == (["kv"], [3]) and trimmed == [1]
    # a cache that cannot be trimmed is a miss, not a guess
    monkeypatch.setattr(C, "can_trim_prompt_cache", lambda c: False)
    assert C.LRUPromptCache.fetch_nearest_cache(None, "m", [1, 2, 3]) == \
        (None, [1, 2, 3])


def test_the_real_cache_on_an_exact_prompt():
    """Through mlx-lm's own LRUPromptCache and KVCache: the prompt that
    crashed GLM (an earlier prompt minus its last token) now has work."""
    import mlx.core as mx
    from mlx_lm.models import cache as C
    from knurlogic.engine.serve import cache_guard
    cache_guard.install()
    kv = C.KVCache()
    kv.update_and_fetch(mx.zeros((1, 1, 4, 8)), mx.zeros((1, 1, 4, 8)))
    lru = C.LRUPromptCache(max_size=4)
    lru.insert_cache("m", [1, 2, 3, 4], [kv])
    cache, rest = lru.fetch_nearest_cache("m", [1, 2, 3, 4])
    assert rest == [4] and cache[0].offset == 3
