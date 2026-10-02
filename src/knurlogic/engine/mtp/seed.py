"""Seed an MTP head over a prompt in prefill-sized chunks.

The trunk prefills a long prompt in `prefill_step_size` chunks so that no
single forward sees the whole sequence at once. Seeding the head
over the SAME prompt in ONE call (`head.advance(h_all[:, :n-1], ...)`)
is quadratic in the prompt on any head with real attention: Flash-Next's
head carries the trunk's sparse indexer, and past its 2048-token budget one
call over S positions materialises S x (S/4) fp32 index scores plus two
S x S boolean masks -- ~6 bytes x S^2, or ~86 GB at a 120k-token prompt,
enough to wedge an M4 Max (128 GB) with no tokens for 300 s. Chunking the seed
the way the trunk chunks its prefill bounds the temporaries at
`step x kv_len`, the same shape the trunk already pays.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx


def _cache_state(cache: Any) -> list[mx.array]:
    """The arrays to force after a chunk so the graph never spans chunks."""
    from mlx.utils import tree_flatten
    out: list[mx.array] = []
    caches = cache if isinstance(cache, (list, tuple)) else [cache]
    for c in caches:
        st = getattr(c, "state", None)
        if st is None:
            continue
        # nested: deepseek_v4's state is (window, compressor, indexer) tuples
        out.extend(a for _, a in tree_flatten(st) if isinstance(a, mx.array))
    return out


def seed_head(head: Any, h_chunks: list[mx.array], ids: mx.array, n: int,
              cache: Any, step: int, start: int = 0) -> None:
    """Advance `head` over positions start..n-2 -- input (h_t, x_{t+1}) -- in
    chunks of at most `step` positions.

    `h_chunks` are the captured trunk hidden states, in order, starting at
    position `start` and covering at least positions start..n-2 (the caller
    may append the final position's capture; the extra is ignored). `ids` is
    [B, >= n]. The head's cache carries the offset (== start on entry, the
    prefix-pool case), so each chunk sees exactly the mask the trunk would
    have built for the same positions.
    """
    if n - start < 2:
        return
    step = max(1, int(step))
    h_all = mx.concatenate(h_chunks, axis=1) if len(h_chunks) > 1 else h_chunks[0]
    for i in range(start, n - 1, step):
        j = min(i + step, n - 1)
        head.advance(h_all[:, i - start:j - start], ids[:, 1 + i:1 + j], cache)
        mx.eval(_cache_state(cache))
        mx.clear_cache()
    del h_all
