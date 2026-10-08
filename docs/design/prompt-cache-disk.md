# The prompt cache on disk

The ask: an agent harness `docs/design/telemetry.md`, "Ask: save prompt caches to
disk and restore them" (2026-10-07). A model swap loses every resident
context, and each session's next request re-prefills it (110k tokens at
136 tok/s is ~13.5 min). Saved to disk, the reload is a read.

Code: `engine/serve/prompt_disk.py`; the scheduler's `_save_disk`,
`_restore_disk`, `_disk_hit` (engine/runtime/scheduler.py); a ring's
followers in `engine/runtime/tensor.serve_follower` / `follow`.

## When

- **Save** (entries with a session only; see Sessions): when the model
  unloads or is switched (`Scheduler._do_commands`,
  after the rows are failed and the executor closed, before the prompt
  cache is replaced), when the server stops (`Scheduler._run`'s cleanup;
  on a ring after the other ranks are sent `stop`, and each of them saves
  at that op), and on request: `POST /v1/prompt-cache/save` (loopback
  only), a scheduler command run between steps; on a ring a `save_cache`
  op tells every rank. Never on the request path.
- **Restore**: `ModelHost.after_bind`, on the loading thread once weights,
  vision and head are bound and before the warm-up: the entries are read
  and inserted into the new in-memory prompt cache. On a ring every rank
  does this at the same point of its load.

## Where and how

`~/.cache/knurlogic/prompt-cache/<key id>/` (XDG_CACHE_HOME honoured, as
jobs/ and load.lock), a `key.json` for people, and one safetensors file per
entry: `<gen>-<seq>-<tokens hash>.safetensors`. `gen` is the save (one more
than the newest in the directory), `seq` the entry's place in the
in-memory LRU at that save; restoring inserts in (gen, seq) order, so the
LRU comes back as it was -- identically on every rank. An entry already on
disk (same tokens, same key) is renamed into the new save, not rewritten.

An entry is serialized by its objects' attributes (`vars` and `__slots__`),
not by mlx-lm's `save_prompt_cache`: that one saves `state` only and
rebuilds classes from mlx-lm's own module, which loses knurlogic's caches
(quantized K/V's group and dims, DeepSeek V4's compressor pools, a factory-
made class). Arrays go in the file, plain values and the object tree in its
header, classes by module and name (`_disk_class` names a class a factory
makes: engine/families/qwen/kvcache.py). Anything else -- a function, an
object referenced twice -- makes the entry unsaveable: skipped with the
reason logged, never written in part. An entry whose key holds an image
(a vision prompt) is skipped: the image store does not outlive the model.

Writes go to a temp file and are renamed. A read checks the header's key
against the loaded model's, the token count and hash, and the array bytes;
any failure deletes the file and is a miss.

Supported, round-tripped bit for bit in tests/engine/test_prompt_disk.py
(the next tokens' logits from the restored cache equal the original's):
KVCache, QuantKVCache (8-bit KV), mlx-lm QuantizedKVCache, RotatingKVCache
(gemma4 windows), ArraysCache (qwen3.5 deltanet state), CacheList (GLM's
MLA latent + DSA indexer, plain and 8-bit), qwen4_exp's attention + indexer
caches (plain and QuantAttnCache), DeepseekV4Cache (window + compressor
and indexer pools), and a drafting head's cache riding in the entry
(DSparkCache).

## The key

`prompt_disk.identity`: the artifact's identity (`machine/artifact.identity`:
config, index, every shard's header and samples, every shipped *.py -- so
quant and model.py are in it), the KV bits, the drafting head (its cache is
part of an entry; class and block size), `KNURLOGIC_LONG_CONTEXT` (YaRN moves
the rope, so the K), per-chip rounding, knurlogic's version, and the split:
`{"split", "world", "rank"}` plus every rank's layer runs on a pipeline.
The key's hash names the directory; a different key finds nothing, and a
file whose header disagrees is deleted unread.

## Clusters

Each rank saves its own part (a pipeline rank only its layers; a tensor rank
its shard) under its own key. At load every rank reads what it has, then
they vote (`prompt_disk.agree`: one all_gather of the count and a digest of
the entry names, after `Link.align`). Equal everywhere: every rank inserts
its entries, in the same order, into its own prompt cache, bypassing the
journal (each does it itself). Any difference -- a split that changed, a
budget that evicted on one Mac only, a corrupt file -- and none restores: a
miss, not an error. Files are read before the vote, so a rank cannot fail
after agreeing.

A ring's rank 0 saves at stop with the stop's 15 s budget
(`watch_ring`); a save that outlasts it is cut by the exit, and the temp
file never becomes an entry.

## Budget and TTL

Knurlogic-wide settings (machine/preferences over the environment, read at
each save and load; tuning/settings.PROMPT_CACHE_KNOBS):

- `KNURLOGIC_PROMPT_CACHE_DISK` on / off (on).
- `KNURLOGIC_PROMPT_CACHE_DISK_GB`: every model's files together; least
  recently used (file mtime: set at save, restore) go first. Unset: 20% of
  the disk's free space plus what the cache holds, at most 64 GiB.
- `KNURLOGIC_PROMPT_CACHE_TTL_H`: an entry not saved or restored for this
  long is deleted (24).

The sweep runs after every save and before every restore.

A load restores at most the in-memory cache's entry count (and its byte cap
where it has one), newest save first; the memory guard trims the prompt
cache as before if a request needs the room.

## Prefix-compatible

A restored entry goes into the in-memory `LRUPromptCache` with
`insert_cache`; its trie's nearest-prefix search serves it to any prompt
that starts with its tokens (or trims a longer one). No second matcher.

## Observable

`usage.knurlogic.cache.disk = {tokens, read_ms}`: the cached tokens served
from an entry restored from disk, and how long that entry took to read.
`tokens > 0` is a disk hit; `used > 0` with `disk.tokens == 0` a memory
hit; `used == 0` a cold prefill. Once the session's next answer is cached,
its hits come from that newer, in-memory entry. The ledger keeps it per
request as `disk_tokens`.

## Sessions

The cache belongs to the agent (the client session), not the model.

- **Owner.** Each entry records `{session, role, run}` from the request
  that made it (`X-Client-Session` / `-Role` / `-Run`, the ledger's parsing
  and byte limits) in a side map on `PromptCache` keyed by the entry's
  token tuple, pruned lazily against mlx-lm's LRU (which evicts
  silently). On a ring the `insert` op carries the owner, so every rank's
  side map mirrors rank 0's. A request with no session owns nothing, and
  its entries are never saved. Files carry `owner`, `pinned`, `saved_at`
  and `model` in their header; restore puts them back.
- **Saves.** Unload / stop / `POST .../save` with no body: every live
  entry that has a session. `POST .../save {"session"}`: that session's
  newest entry (its longest), written only if not already on disk --
  under a new gen, renaming nothing, so a restore still inserts oldest
  save first. There is no timed save.
- **Break-even.** An entry is not written when recomputing it (tokens /
  the last measured prefill tok/s) is quicker than reading it back (bytes
  / `prompt_disk.READ_BPS`, 3 GB/s: the M3 read 70-128 MB in ~24 ms).
  Counted `not_worth`. No rate measured yet: written. Off on a ring (every
  rank must keep the same entries, and each has its own bytes).
- **Park.** `POST /v1/prompt-cache/park {"session"}`, between steps: the
  session's entries saved (only what is not on disk), then freed from
  memory; one the save did not write stays. Its next request reads the
  longest on-disk prefix back. The client decides when (the harness parks its
  PM while sub-agents work); the server keeps no idle policy beyond a
  pinned session's. Refused on a split model (no read-back there yet).
- **Drop.** `POST /v1/prompt-cache/drop {"session"}`, between steps:
  out of memory (mlx-lm 0.32 has no single-entry removal;
  `prompt_disk.remove_entry` does what its `insert_cache` does to a
  replaced entry: `PromptTrie.pop`, `CacheOrder.remove`, the byte
  counters), its files off disk under every key, its pin forgotten. A
  `drop` op on a ring.
- **Pins.** `X-Cache-Retain: pin` (sticky for the session on this
  server) or `POST .../pin`. Stored in one `pins.json` at the cache root
  (`{session: bool}`, written atomically): a pin never rewrites an entry
  file; a header's `pinned` is only what it was at write. The sweep never
  deletes a pinned session's files (TTL nor budget; the budget counts
  unpinned files only); only a drop does. A `pin` op on a ring.
- **Latest step only.** `X-Cache-Keep: latest` (per request; a client
  that only appends -- the harness's workers -- sends it on every one): the
  entries a request makes replace its session's earlier steps, in memory
  and on disk, so a session holds one step (its conversation checkpoint
  and its answer). Its system-prompt checkpoint is nobody's: one copy per
  distinct system prompt and tools, which every session starting from it
  shares. It is saved (at unload, and at a save with no session) and
  restored as shared, so the first worker after a reload skips that
  prefill; its file's mtime is refreshed at each save and restore, and a
  model in use longer than the TTL writes it again at unload. Opt-in,
  because the page's chat regenerates and edits from
  earlier messages. On a ring the owner rides the `insert` op with its
  step, and every rank replaces the same entries.
- **Parking and read-back.** A pinned session with no new entry for
  `PARK_IDLE_S` (10 min) has its entries saved and freed from memory. An
  admission whose best memory hit is shorter than an on-disk entry of
  this model that prefixes the prompt reads that entry back first (an
  in-memory index of the key directory's tokens, built at restore,
  updated at save and drop: no file is opened to look). Any on-disk
  entry, pinned or not; reported in `usage.knurlogic.cache.disk`. Single
  server only for now: a ring's ranks would each have to read in step.
- **Registry.** `GET /v1/prompt-cache`: memory entries (between steps)
  and every model's files (headers only), one row per entry.

## What it cannot do

Cut a segment off the front of a cache: past the first layer every later
token's K/V carries the removed tokens' influence. A side call keeps the
whole prefix.
