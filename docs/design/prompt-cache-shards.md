# The shard-agnostic prompt cache

Status: design, decided 2026-10-08 (the maintainer). Not built yet. Today's format is
in prompt-cache-disk.md; this replaces its key and its location. Everything
a client sees (save / park / drop / pin / keep-latest / registry / diverged)
stays as it is.

## Why

A saved entry is keyed by the split today: the key holds `{split, world,
rank}` plus every rank's layer runs. Re-split the 397B (26/34 instead of
30/30, another Mac first, pipeline instead of tensor) and every saved
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
(engine/runtime/tensor_rules.py), not from a guess at array shapes. A kind
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
regenerable and safe to delete. the maintainer: software should not hide GBs from
people.

- **Shared model storage** (the external drive, read by both Macs over the
  network): every Mac reads and writes the same folder. A rank writes only
  the layers and heads it holds, so writers never collide on a file. Rank 0
  writes `entry.json` after every rank has reported its files written (one
  journal op), so the entry appears whole or not at all.
- **No shared storage** (option 2): each Mac keeps its own folder beside its
  own models. When a load needs a layer or head range this Mac does not
  have, it asks the peer page that does: `GET /v1/prompt-cache/piece` over
  the cluster link. The page serves files from its cache folder only.
- **No writable models folder**: fall back to `~/.cache/knurlogic/prompt-cache/`
  (today's location), and the page says where the cache is.

Budget, TTL, pins and the sweep work per folder as today. On shared
storage only one sweeper may run: the coordinator's rank 0 (as today on a
ring), and it journals the names it deletes so no other Mac reads a
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

## Code

All prompt-cache code lives in `engine/prompt_cache/`:

| file | what |
|---|---|
| `memory.py` | the in-memory LRU and session ownership (`PromptCache`) |
| `disk.py` | files, keys, the sweep, pins, restore |
| `commands.py` | the scheduler's cache commands (save, park, drop, pin, list, read-back) |
| `ring.py` | a split model's journaled cache ops and follower apply |
| `report.py` | what a request's usage says about the cache |

The HTTP endpoints are in `interfaces/http/prompt_cache.py`; the page
forwards them to the Mac that runs the model (`interfaces/page/prompt_cache.py`).
The new format adds `layers.py` (per-layer, per-head pieces) and
`transfer.py` (option 2).
