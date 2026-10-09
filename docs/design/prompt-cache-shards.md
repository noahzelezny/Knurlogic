# The shard-agnostic prompt cache

Status: phases 1 and 2 built (0.1.6): whole models and pipeline splits,
a shared folder beside the models, the local fallback, and missing pieces
fetched from a peer when machines do not share storage. Phase 3 (tensor,
per head range) is not built; a tensor split still saves in
prompt-cache-disk.md's per-split format. What phases 1 and 2 changed
against this design is under "As built". Everything a client sees
(save / park / drop / pin / keep-latest / registry / diverged) stays as it
is; registry rows gain a `location`.

## Why

A saved entry is keyed by the split today: the key holds `{split, world,
rank}` plus every rank's layer runs. Re-split the 397B (26/34 instead of
30/30, another machine first, pipeline instead of tensor) and every saved
session is gone, though the K/V of layer 17 for those tokens is the same
numbers wherever layer 17 runs. The split decides who holds a layer, not
what is in it.

So the cache is saved per layer, and per head range, and any split reads
the pieces it needs.

## What is saved

One entry = one prompt's tokens. On disk it becomes one directory, and one
file per layer (or per layer and head range):

```
<cache root>/<model key>/<tokens hash>/
    entry.json                 tokens, owner, pinned, saved_at, shared, model
    L017.safetensors           layer 17, all heads
    L018.h0-8.safetensors      layer 18, heads 0..7 (a tensor rank's slice)
    L018.h8-16.safetensors
    draft.safetensors          the drafting head's cache, when the entry has one
```

The **model key** drops the layout. It keeps everything that changes the
numbers: artifact identity, KV bits, drafting head, long context (YaRN),
per-chip rounding, knurlogic's cache format version. Not knurlogic's
version itself: a release that does not touch the cache must not orphan it.

A layer file's header records its layer, its head range (or "all"), the
object tree (as today: classes by module and name, arrays in the file) and
the byte digest. `entry.json` is written last; an entry without it is
incomplete and never read.

### Which axis is "heads"

Pipeline needs nothing more: a rank owns whole layers. Tensor cuts each
cache along some axis, and that axis differs by cache kind:

| cache kind | split axis | example |
|---|---|---|
| K/V (KVCache, QuantKVCache, Rotating) | KV heads | qwen, gemma4 |
| MLA latent (CacheList) | none: replicated | GLM-5, DeepSeek |
| deltanet / recurrent state (ArraysCache) | value heads | qwen3.5 |
| compressor / indexer pools | per family | DeepSeek V4, qwen4_exp |

The axis comes from the family's spec, next to its tensor rules
(engine/split/tensor_rules.py), not from a guess at array shapes. A kind
without a spec is saved whole per rank under that rank's layout (today's
behavior) and is listed as missing in the onboarding checklist
(new-model.md, "Cache saving").

A tensor rank that needs heads 8..16 of layer 18 reads the files that cover
them: one exact slice, several smaller ones concatenated, or a whole layer
cut down. A replicated kind is written once (by rank 0) and read by every
rank.

## Where it lives

**Beside the models**, visible: `<models folder>/Prompt cache/`, a folder a
person sees in Finder next to their models (not a hidden dot-folder), with a README.txt that says it is
regenerable and safe to delete. Reason: software should not hide
gigabytes of disk use from the people who own the disk.

- **Shared model storage** (a shared external drive, read by every machine over the
  network): every machine reads and writes the same folder. A rank writes only
  the layers and heads it holds, so writers never collide on a file. Rank 0
  writes `entry.json` after every rank has reported its files written (one
  journal op), so the entry appears whole or not at all.
- **No shared storage** (option 2): each machine keeps its own folder beside its
  own models. When a load needs a layer or head range this machine does not
  have, it asks the peer page that does: `GET /v1/prompt-cache/piece` over
  the cluster link. The page serves files from its cache folder only.
- **No writable models folder**: fall back to `~/.cache/knurlogic/prompt-cache/`
  (today's location), and the page says where the cache is.

Budget, TTL, pins and the sweep work per folder as today. On shared
storage only one sweeper may run: the coordinator's rank 0 (as today on a
ring), and it journals the names it deletes so no other machine reads a
half-deleted entry.

## Loading

At load, rank 0 lists the complete entries for the model key (newest
first, within the in-memory cache's count and bytes). For each, every rank
checks that it can get every piece it needs -- locally, from the shared
folder, or from a peer. Then the vote, as today (`agree`): an entry is
restored only if every rank has all its pieces; otherwise it is skipped on
every rank. A mismatch is a miss, never an error. Same software and an
existing cluster make this rare and cheap to recover from.

Read-back (a parked session's longer prefix) uses the same path for one
entry.

## Migration

Old per-split entries are still read, only under their exact split, as
today. Once restored into memory they are just entries: the next client
save writes them in the new format, and the old files age out by TTL. No
converter.

## Cost

Per-layer files mean more files (60 layers x entries). A directory per
entry keeps listings cheap. Writes are the same bytes as today. Network
transfer for option 2 runs over the cluster link (to be measured); even
slow, it is far cheaper than a 10-minute prefill. The break-even check counts the
transfer.

## Order of work

1. Pipeline, shared folder beside the models, local fallback.
2. Option 2: fetch missing pieces from a peer.
3. Tensor, one family at a time, from its spec (qwen first: KV heads and
   deltanet value heads).

Each step ships with its tests and a live check on the split 397B: save,
re-split differently, reload, and the session hits.

## Alternatives considered

- **Keep the per-split key** (today): simplest, but every re-split
  orphans every saved session.
- **A converter from old entries**: not needed; old entries restore under
  their own split and are rewritten in the new format at the next save.
- **Keep the hidden cache directory**: rejected for the default, kept as
  the fallback when the models folder is not writable.

## As built (phase 1)

**Layout.** `<root>/<key id>/<tokens hash>/` holds `entry.json` and one
`L{i:03d}.safetensors` per cache slot, plus `draft.safetensors` when the
entry carries a drafting head's cache. A file is indexed by the slot of
the model's `make_cache()` list, which on every pipeline-capable family is
the layer index. A stage whose cache list is not one slot per layer falls
back to the per-split format (logged). `entry.json` holds the tokens,
owner, pinned, saved_at, shared, model, kind, the key, `layers` (every
slot index of the whole model), `draft` (whether the entry has a head
cache) and `gen` / `seq` (the save and the LRU place, the restore order).

**The key** (`layers.shard_key`): the artifact's identity, the KV bits,
long-context rope, per-chip rounding and `CACHE_FORMAT` (2). Not
knurlogic's version, and not the layout. **The drafting head is not in
the key either** (a change from the design): only `draft.safetensors`
depends on it, and only rank 0 knows which head is bound, so a key with
the head in it would differ between rank 0 and a follower. Instead the
draft file's header names the head ({class, block}); a reader running
another head, or none, restores the entry's layers without it (a
drafting row then re-prefills; a plain row hits).

**Who writes what.** A single machine writes every file and `entry.json`
at once. On a pipeline each rank writes the layer files of the layers it
holds (`Part`: its slots, from the layout's bounds); rank 0 holds the
drafting head, so it alone owns the non-layer state (`draft.safetensors`).
`entry.json` is written by the commit:

1. rank 0's save writes its files and queues a record of each entry;
2. the scheduler journals `save_cache` (or `park_session`), then
   `commit_entries` with those records;
3. each follower applies the save op (its layers), then the commit op:
   it says, per entry, whether it holds every file of its part, in one
   all_gather after `Link.align`;
4. rank 0 runs the same collective right after the exchange that shipped
   the op (`Ring.after_ops`), before its next step;
5. every rank writes `entry.json` for each entry every rank holds. The
   records come from rank 0 and the tokens from each rank's own save, so
   the bytes are identical: on a shared folder the second write replaces
   the first with the same file; on separate folders each gets its copy.

An entry some rank did not write never gets `entry.json` on any rank; its
layer files are removed by the sweep once stale (an hour).

**Restore and the vote.** At load the per-split restore runs first (its
own all-or-none vote, unchanged), then this format's: each rank lists
the complete entries of the key it can read whole (every layer file of
its part, and `layers` matching the model), reads the newest (the
in-memory cache's count; the byte cap on a single machine only), and
votes: one all_gather of a fixed-width row (the count and two 28-bit
ints of each entry's name, the newest 256). Each rank keeps the entries
every rank read, in rank 0's order. A rank that lacks a file of an entry
leaves it out on every rank: a miss, not an error. Read-back (a parked
session's longer prefix) reads one entry the same way: rank 0 reads its
part and journals `read_back` with the entry's name; each follower reads
its part from its own folder. Rank 0 does not read back an entry whose
commit has not run yet.

**Location** (`location.py`). `<models folder>/Prompt cache/`, where the
models folder holds the artifact's directory (for a Hugging Face cache
path, the folder holding `models--org--name`), made with a README.txt.
Not writable: the fallback, `~/.cache/knurlogic/prompt-cache/`
(XDG_CACHE_HOME honoured). Each root used is recorded in the fallback's
`locations.json`, so a drop, a pin and the registry reach every root.
The log line at first use says which root and why; registry rows carry
`location`; a save's result and the in-memory listing too.

**Sweep, pins, drops per root.** The sweep (TTL, then budget, pinned
sessions exempt) covers both formats in a root; an entry directory's age
is its `entry.json`'s mtime (touched at save and restore) and its size
its files'. `pins.json` is per root; a pin or unpin is written to every
known root, and a read merges them. On a ring only rank 0 sweeps, after
its saves, and journals the entry names it deleted (`drop_files`); the
followers delete those names from their own folders. A drop by session
removes both formats' entries in every root (`entry.json` first).

**Migration.** Per-split entries are read under their exact old key, by
the old code. New saves write only the per-layer format (a tensor split
excepted). No converter.

**Left for later phases.** Phase 3: tensor splits, per head
range (the seam: `part_of` returns None for a tensor layout, and those
splits keep the per-split format); file names would gain `.h{a}-{b}`.

## As built (phase 2)

**Who fetches, when.** Each rank, for itself, at two points: at load,
inside `layers.restore` before the rank lists what it can read (so before
the vote), and at a read-back, before the entry is read (rank 0 in
`_read_back`, a follower at the `read_back` op; on a ring the index holds
only restored entries, so nothing is missing there). `transfer.fill`
lists this model's entries on every answering peer, takes the newest
(the in-memory cache's count, local and the peers' together), and for
each one this rank lacks pieces of (entry.json, a layer file of its part,
`draft.safetensors` when the entry has one and the rank holds the head)
fetches them from a peer that lists them. An entry some piece of which no
peer holds is not fetched at all. On shared storage every file is already
here: nothing is fetched (a read-back of a whole entry asks no peer).

**The routes**, on every page (`interfaces/page/prompt_cache.py`):

- `GET /v1/prompt-cache/entries?key=<key id>`: this machine's whole
  entries of that key, over every root it knows (`location.known`):
  `{hash: {gen, seq, n_tokens, draft, layers, sizes}}`.
- `GET /v1/prompt-cache/piece?key=<key id>&entry=<hash>&layer=<n | draft |
  entry>`: one file's bytes.
- `GET /v1/prompt-cache/peers`: the answering peer pages, for this
  machine's model servers.

The first two take the peer gate every page-to-page route takes
(`peers.peer_refusal`: no Origin header; arrived on loopback or
Thunderbolt, or from a `--peer` address); the third is loopback only.
They are routed before the browser gate, like `/peer/v1/msg`. No request
string becomes a path: the key id must be 16 hex digits, the entry 24, the
layer an integer (or `draft` / `entry`); the file is served only when its
entry has `entry.json` there, it is not a symbolic link, and it resolves
inside the root (`transfer.piece_path`).

**How a model server reaches the peers.** The page sets
`KNURLOGIC_PAGE_PORT` (its port) in its environment; the model servers it
starts, and the cluster ranks a page starts, inherit it, and
`interfaces/serve.py` installs a `PageFetcher` (engine/prompt_cache never
imports interfaces: the engine takes a fetcher object, `transfer.use`).
The fetcher asks the local page for the peer list, then each peer page
directly. A server started by hand (no page) fetches nothing.

**Verification.** Each piece is downloaded into the entry directory under
a temp name, then checked: its size equals the peer's listing; a layer or
draft file's header names this key id, these tokens (hash and count),
this slot and the format version, and its arrays match the header
(`layers._read`); a fetched entry.json must carry this model's key, its
tokens must hash to the entry's name, and its layers be this model's.
Then it is renamed into place; entry.json is written last, once every
piece is. A piece that fails is deleted, the next peer listing it is
tried, and with none left the entry is a miss on this rank (the vote
drops it on every rank).

**Break-even.** Each peer's transfer rate is measured on every fetch (the
last one counts). An entry whose missing bytes at the measured rates take
longer than recomputing its tokens at the measured prefill rate is not
fetched. Nothing measured yet (a fresh load): fetched. At load the
prefill rate is usually not measured yet either, so the check bites at
read-back on a server that has run prefills.

**After.** A fetched file is an ordinary local file: the sweep, TTL and
pins apply, and the next load reads it without asking.

## Code

All prompt-cache code lives in `engine/prompt_cache/`:

| file | what |
|---|---|
| `memory.py` | the in-memory LRU and session ownership (`PromptCache`) |
| `disk.py` | files, keys, the sweep, pins, restore |
| `commands.py` | the scheduler's cache commands (save, park, drop, pin, list, read-back) |
| `ring.py` | a split model's journaled cache ops and follower apply |
| `report.py` | what a request's usage says about the cache |
| `layers.py` | the per-layer format: Part, key, write / read / complete check, save, commit and restore votes (built) |
| `location.py` | where the cache root is (built) |
| `transfer.py` | phase 2: what a rank lacks, fetched through an injected fetcher, verified, written (built) |

The HTTP endpoints are in `interfaces/http/prompt_cache.py`; the page
forwards them to the machine that runs the model (`interfaces/page/prompt_cache.py`).
The same page module serves the pieces to peers and holds the fetcher
(`PageFetcher`) the model server installs.
