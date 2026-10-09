"""engine/prompt_cache/ -- the prompt cache, in memory and on disk, and
the scheduler's commands on it (docs/design/prompt-cache-disk.md).

  memory.py    PromptCache: the in-memory LRU and who owns its entries
               (entry_owner)
  layers.py    the entries on disk, one file per layer, readable under
               any split (whole model, pipeline): save, commit, restore
               (docs/design/prompt-cache-shards.md)
  transfer.py  an entry's missing pieces, fetched from a peer's page
               (machines without shared storage); the fetcher is
               installed by the interfaces side
  location.py  where they live: <models folder>/Prompt cache/, else
               the fallback under ~/.cache
  disk.py      the per-split format (tensor splits, older entries), the
               shared encoding, settings, sweep, pins, drops, registry
  commands.py  the Scheduler's cache methods (PromptCacheCommands,
               a mixin) and Command, what they queue
  ring.py      a ring's prompt cache: JournalPromptCache (rank 0) and
               apply_cache_op (the following ranks)
  report.py    usage.knurlogic.cache: what the prompt cache did for a
               request

The HTTP side (/v1/prompt-cache) is interfaces/http/prompt_cache.py;
the pieces a peer fetches are served by interfaces/page/prompt_cache.py.
Import from the submodules.
"""
