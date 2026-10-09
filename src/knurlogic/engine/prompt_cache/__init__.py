"""engine/prompt_cache/ -- the prompt cache, in memory and on disk, and
the scheduler's commands on it (docs/design/prompt-cache-disk.md).

  memory.py    PromptCache: the in-memory LRU and who owns its entries
               (_owner)
  disk.py      the entries on disk: save, restore, read back, sweep
  commands.py  the Scheduler's cache methods (PromptCacheCommands,
               a mixin) and Command, what they queue
  ring.py      a ring's prompt cache: JournalPromptCache (rank 0) and
               apply_cache_op (the following ranks)
  report.py    usage.knurlogic.cache: what the prompt cache did for a
               request

The HTTP side (/v1/prompt-cache) is interfaces/http/prompt_cache.py.
Import from the submodules.
"""
