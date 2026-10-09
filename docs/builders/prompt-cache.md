# Building on the prompt cache

For people changing knurlogic or adding a model. What a client sees is in
[the user guide](../guide/prompt-cache.md); the reasoning behind each
choice is in [prompt-cache-disk](../design/prompt-cache-disk.md) and
[prompt-cache-shards](../design/prompt-cache-shards.md).

## Where the code is

Everything is in one package, `src/knurlogic/engine/prompt_cache/`:

| file | what |
|---|---|
| `memory.py` | `PromptCache`: mlx-lm's LRU trie plus who owns each entry (`owners`), pins, shared checkpoints, keep-latest |
| `disk.py` | the file format, the key (`identity`), save, restore, the sweep (TTL and budget), pins.json, the registry's disk side, the vote across ranks (`agree`) |
| `commands.py` | `PromptCacheCommands`, a mixin of the `Scheduler`: save, park, drop, pin, list, read-back, divergence, restore at load |
| `ring.py` | a split model: `JournalPromptCache` journals every cache change, `apply_cache_op` replays it on the other ranks |
| `report.py` | `usage.knurlogic.cache` for one request |

Outside it, only thin hooks:

- `interfaces/http/prompt_cache.py`: the `/v1/prompt-cache` endpoints, a
  mixin of the model server's handler (only `interfaces/` serves HTTP).
- `interfaces/page/prompt_cache.py` (`prompt_cache_forward`): the page forwards
  to the right model server, or through the peer relay to another Mac.
- `engine/runtime/scheduler.py` inherits the commands mixin and calls
  into it at admission (read-back, divergence) and at load (restore).
- `engine/runtime/plan.py`: the schema every journaled op must match.
- `tuning/groups.py` (`PROMPT_CACHE_KNOBS`): the disk on/off, GB and TTL
  settings.

New prompt-cache behavior goes in the package. Grow the scheduler,
tensor code or HTTP server only by a call into it.

## Rules that keep it correct

- **Commands run between steps.** Every save, drop, pin and park is a
  scheduler `Command`, executed by the scheduler thread when no batch
  step is running. Nothing touches the cache from an HTTP thread.
- **A split model changes its cache only through the journal.** Rank 0's
  `JournalPromptCache` records each op (`insert`, `pop`, `drop`,
  `pin`, `park_session`, `read_back`, `drop_files`, `save_cache`, ...);
  every follower applies the same op in the same order, so all ranks hold
  the same entries. A new op needs:
  1. its fields in `plan.py` (`OPS`, `_FIELDS`, `_OPTIONAL`): a test
     fails if a journaled op is missing;
  2. a case in `ring.apply_cache_op`;
  3. an attribute on `JournalPromptCache` if the scheduler reads one
     (a test lists the ones it must have).
  Op names are global across the journal: `park` is already the ring's
  idle op.
- **Only rank 0 deletes files on a ring**, and it journals the names
  (`drop_files`), so no rank reads an entry another has half-deleted.
  Followers save with `sweep_after=False` and read with `sweep_first=False`.
- **A file is complete or absent.** Write to a temp name, then rename. A
  read checks the key, token count, hash and bytes; any mismatch deletes
  the file and is a miss, never an error.
- **No automatic saving.** Unload, switch and stop save nothing; only a
  client's call does. Do not add a timer or an exit hook.
- **Hybrid models** (Qwen3.5's recurrent layers) reuse only exact
  checkpoints: their state cannot be trimmed back to a shorter prefix.

## A new cache class

A model family whose layers use a new cache class must save and restore
it bit for bit: add it to `tests/engine/test_prompt_disk.py` (the next
tokens' logits from a restored cache equal the original's). A class made
by a factory needs a `_disk_class` name (see
`engine/families/qwen/kvcache.py`). This is part of the onboarding
checklist: [new-model.md](../design/new-model.md), "Cache saving".

## Tests

`tests/engine/test_prompt_cache_sessions.py` (ownership, keep-latest,
park, the journal), `test_prompt_disk.py` (format, sweep, restore),
`tests/interfaces/test_prompt_cache_http.py` and
`test_prompt_cache_page.py` (endpoints and forwarding). Run all:
`pytest -q -n 8 tests`.
