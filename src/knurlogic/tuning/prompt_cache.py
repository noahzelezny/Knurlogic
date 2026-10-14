"""The prompt cache's size for a server: how many entries it keeps and
how many bytes it may hold (prompt_cache_policy), read once at serve."""

from __future__ import annotations

from knurlogic.machine.units import gib

#: prompt-cache entries one conversation keeps: its system prompt, turns
#: and segment checkpoints (on a hybrid model each checkpoint is a whole
#: entry). A server-wide cap of 10 evicts four agents' entries on a
#: hybrid model: each agent needs its own
PROMPT_CACHE_PER_AGENT = 10
#: a ring counts at most this many concurrent agents (its decode
#: concurrency, which every rank has from the same argv)
PROMPT_CACHE_AGENTS_MAX = 8
#: one machine: the entry cap under the byte cap -- bytes decide
PROMPT_CACHE_ENTRIES_BYTE_SIZED = 256
#: one machine: the share of the headroom (working set less the model)
#: the prompt cache may hold; the memory guard takes it back before a
#: request's own KV would not fit
PROMPT_CACHE_HEADROOM_SHARE = 0.5


def prompt_cache_policy(serving: dict, world: int, working_set: int,
                        holds: int) -> tuple:
    """(size, bytes or None, why) of the prompt cache.

    One machine: sized by bytes -- PROMPT_CACHE_HEADROOM_SHARE of what the
    working set leaves beside the model -- with a generous entry cap, so
    it is memory, not a count, that evicts. A ring: count-based by design
    (every rank must evict the same entries; bytes are each rank's own),
    so the count is PROMPT_CACHE_PER_AGENT per concurrent agent. An
    explicit --prompt-cache-size / --prompt-cache-gib wins."""
    size = int(serving.get("prompt_cache_size") or 0)
    cap = serving.get("prompt_cache_bytes")
    dc = int(serving.get("decode_concurrency") or 32)
    agents = max(1, min(dc, PROMPT_CACHE_AGENTS_MAX))
    per = (f"{PROMPT_CACHE_PER_AGENT} per agent x {agents} concurrent "
           f"agents (decode concurrency {dc}, at most "
           f"{PROMPT_CACHE_AGENTS_MAX} counted)")
    if world > 1:
        if size:
            return size, None, (f"{size} entries (--prompt-cache-size); a "
                                f"ring's prompt cache is count-based")
        n = PROMPT_CACHE_PER_AGENT * agents
        return n, None, (f"{n} entries: {per}; a ring's prompt cache is "
                         f"count-based -- every rank evicts the same "
                         f"entries -- and the memory guard still takes "
                         f"entries back first")
    if cap:
        n = size or PROMPT_CACHE_ENTRIES_BYTE_SIZED
        return n, int(cap), (f"{gib(int(cap))} "
                             f"(--prompt-cache-gib), up to {n} entries")
    room = int(working_set or 0) - int(holds or 0)
    if room > 0:
        b = int(room * PROMPT_CACHE_HEADROOM_SHARE)
        n = size or PROMPT_CACHE_ENTRIES_BYTE_SIZED
        return n, b, (f"{gib(b)} ({PROMPT_CACHE_HEADROOM_SHARE:.0%}"
                      f" of the {gib(room)} the working set leaves "
                      f"beside the model), up to {n} entries: sized by "
                      f"memory, not a count")
    n = size or PROMPT_CACHE_PER_AGENT * agents
    return n, None, (f"{n} entries" + ("" if size else f": {per}")
                     + "; the working set is unknown, so not sized by bytes")
