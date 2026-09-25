"""A prompt-cache hit never leaves nothing to process.

mlx-lm's `LRUPromptCache.fetch_nearest_cache` returns `(cache, [])` when a
stored entry is EXACTLY the new prompt. The batch path then pops every
segment and `BatchGenerator.insert_segments` indexes `segments[-1]` of an
empty list -- IndexError on the generation thread, which kills it and hangs
the server. Checkpoints are stored at segment ends (one token short of the
prompt), so a prompt equal to an earlier prompt minus its last token hits
exactly. Measured on the M4 (2026-09-25): GLM's closed-think `none` prompt
is its `low` prompt plus `</think>`, so `none` then `low` crashed the
server.

An exact hit is returned one token short -- trimmed, when the cache can be
trimmed -- so there is always a token to process; one that cannot be
trimmed (recurrent state) is treated as a miss rather than guessed at.
"""

from __future__ import annotations


def install() -> None:
    from mlx_lm.models import cache as C
    real = C.LRUPromptCache.fetch_nearest_cache
    if getattr(real, "_knurlogic", False):
        return

    def fetch_nearest_cache(self, model, tokens):
        cache, rest = real(self, model, tokens)
        if cache is None or rest or not tokens:
            return cache, rest
        if C.can_trim_prompt_cache(cache):
            C.trim_prompt_cache(cache, 1)
            return cache, list(tokens[-1:])
        return None, list(tokens)

    fetch_nearest_cache._knurlogic = True
    C.LRUPromptCache.fetch_nearest_cache = fetch_nearest_cache
