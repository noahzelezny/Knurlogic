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
| `layers.py` | the per-layer format (whole models and pipeline splits): `Part` (which slots a rank holds), `shard_key`, `Store` (write / read / complete check, save, index), `commit` (the ring's entry.json vote), `restore` and `agree` (the load's vote) |
| `location.py` | where the cache root is: `<models folder>/Prompt cache/`, else the fallback; `locations.json` remembers every root |
| `transfer.py` | machines without shared storage: what this rank lacks of an entry, fetched from a peer (`fill`), verified, renamed into place; `listing` / `piece_path` are what a page serves. No networking: the fetcher is injected (`use`) |
| `disk.py` | the per-split format (a tensor split's, and entries earlier builds wrote), the object encoding both formats use, the settings, the sweep (TTL and budget, both formats, per root), pins.json, drops, the registry's disk side, the per-split vote (`agree`) |
| `commands.py` | `PromptCacheCommands`, a mixin of the `Scheduler`: save, park, drop, pin, list, read-back, divergence, restore at load |
| `ring.py` | a split model: `JournalPromptCache` journals every cache change, `apply_cache_op` replays it on the other ranks |
| `report.py` | `usage.knurlogic.cache` for one request |

Outside it, only thin hooks:

- `interfaces/http/prompt_cache.py`: the `/v1/prompt-cache` endpoints, a
  mixin of the model server's handler (only `interfaces/` serves HTTP).
- `interfaces/page/prompt_cache.py` (`prompt_cache_forward`): the page forwards
  to the right model server, or through the peer relay to another Mac.
  The same module serves saved pieces to peers (`piece_route`:
  `/v1/prompt-cache/entries`, `/piece`, `/peers`, behind the peer gate)
  and holds `PageFetcher`, which `interfaces/serve.py` installs into
  `transfer` when a page started the server (`KNURLOGIC_PAGE_PORT`).
- `engine/runtime/scheduler.py` inherits the commands mixin and calls
  into it at admission (read-back, divergence) and at load (restore).
- `engine/split/plan.py`: the schema every journaled op must match.
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
- **An entry is whole or absent.** In the per-layer format each rank
  writes the layer files it holds; `entry.json` is written last, by the
  `commit_entries` op the scheduler journals after the save ops
  (`_journal_save`, `_cmd_park`): every rank votes on holding its part,
  and only entries every rank holds get `entry.json`. Rank 0's side of
  that collective runs from `Ring.after_ops`, right after the exchange
  that shipped the op. Anything that reads an entry checks `entry.json`
  and every file it needs (`Store.holds`); a miss on one rank is a miss
  on all (the votes).
- **Only rank 0 deletes files on a ring**, and it journals the names
  (`drop_files`), so no rank reads an entry another has half-deleted.
  Followers save with `sweep_after=False` and read with `sweep_first=False`.
- **A fetched piece is checked before it is a file.** `transfer.fill`
  downloads under a temp name, checks the size against the peer's
  listing and the header (key id, tokens, slot) and arrays as a read
  does, then renames; entry.json last. A request's names never become a
  path: `transfer.piece_path` matches each against its format and keeps
  the file inside a root.
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
`test_prompt_cache_layers.py` (save under one split, restore under
another; incomplete entries; location; sweep and pins on entry
directories; a follower's ops), `test_prompt_cache_transfer.py` (two
machines' roots: restore under another split with the missing pieces
fetched, refused pieces, the route's guards, read-back fetch),
`test_pipeline.py`'s two-process
`test_a_prompt_cache_saved_under_one_cut_restores_under_another`,
`tests/interfaces/test_prompt_cache_http.py` and
`test_prompt_cache_page.py` (endpoints and forwarding). Run all:
`pytest -q -n 8 tests`.
