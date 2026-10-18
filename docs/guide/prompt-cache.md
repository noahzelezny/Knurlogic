# The prompt cache

Every answer starts by reading the prompt into the model ("prefill"). For a
long agent session that is most of the wait: 100k tokens can take minutes.
Knurlogic keeps what it has already read, so the next request in the same
conversation only reads what is new.

You do not have to do anything for that part: it is always on, in memory.
Everything below is for clients that want more control -- keeping a
session's cache across a model reload, freeing memory for a while, or
cleaning up.

## In memory (automatic)

- Each request's prompt and answer stay in the model's memory after it
  finishes. A later request that starts with the same tokens reuses them.
- When memory runs short, the least recently used entries go first.
- Nothing is written to disk unless a client asks (below).

`usage.knurlogic.cache` in every response tells you what happened:
`used` tokens came from the cache, the rest was read fresh.

## Naming your session

Send headers with your requests so knurlogic knows which entries are
whose:

| header | meaning |
|---|---|
| `X-Client-Session: <id>` | the conversation these entries belong to |
| `X-Client-Role`, `X-Client-Run` | optional labels, shown in the registry and usage |
| `X-Cache-Keep: latest` | keep only the session's newest step (for clients that only append) |
| `X-Cache-Retain: pin` | never delete the session's saved files |

A request without `X-Client-Session` still uses the cache, but nobody owns
its entries and they are never saved.

**Keep latest** is for agents that only ever add to their conversation.
Each new step replaces the session's older ones, so one session holds one
step instead of dozens. The system prompt's checkpoint is shared by every
session that starts with it, and is kept. Do not use it from a chat that
edits or regenerates earlier messages.

**Forking a worker.** To keep a worker's state at some point -- for a
reviewer that will see exactly what the worker had read -- send the
worker's messages at that point under a session of its own (one token is
enough: they are cached), pin it and save it. That request is served from
the worker's checkpoint and takes it over, so the worker's later steps no
longer replace it; a request that starts with those messages is then
served from it, also after a reload.

## Saving to disk

Saved entries survive a model unload, a switch, a restart. When the same
model loads again, they are read back and the session continues where it
was, without re-reading its prompt.

All endpoints are on the page (`http://localhost:8899`) or the model's own
port, from the Mac itself only (loopback). The page forwards to the Mac
that runs the model, also when that is another Mac in your cluster. With
several models loaded, name one: `"model": "<id>"` in the body (or
`?model=` for GET).

```bash
curl -X POST localhost:8899/v1/prompt-cache/save -d '{"session": "pm-1"}'
```

| call | what it does |
|---|---|
| `POST /v1/prompt-cache/save {"session"}` | save that session's newest entry and the checkpoint where its messages end (call it after your context compacts, or at a point you will fork from) |
| `POST /v1/prompt-cache/save` (no body) | save every entry that has a session |
| `POST /v1/prompt-cache/park {"session"}` | save the session, then free its memory; its next request reads it back from disk |
| `POST /v1/prompt-cache/pin {"session", "pinned": true}` | its files are never deleted by age or size limits |
| `POST /v1/prompt-cache/drop {"session"}` | forget the session: out of memory and off disk |
| `POST /v1/prompt-cache/drop {"sessionless": true}` | drop entries no session owns (optionally `"older_than_s": 3600`) |
| `GET /v1/prompt-cache` | the registry: every entry, in memory and on disk |

A save skips an entry when reading it back would be slower than just
re-reading the prompt (short prompts on a fast Mac).

**Park** is for a session that will sit idle while others work: its
memory goes to them, and it comes back from disk in milliseconds when it
resumes.

### Where the files are

In a folder named `Prompt cache` next to your models: the folder that
holds the model's own folder (for a Hugging Face download, the `hub`
folder). Inside, one folder per model and one folder per saved prompt,
with a file per layer. A `README.txt` there says the same as this: the
files can be deleted at any time; the worst case is a slower next
request.

If that folder cannot be written (a read-only drive, for example), the
cache goes to `~/.cache/knurlogic/prompt-cache/` instead. The server log
says which one is used, and `GET /v1/prompt-cache` lists each entry's
`location`.

A saved prompt does not depend on how a model is split across Macs: a
session saved on one Mac is found again when the same model runs split
over two (pipeline or tensor), and the other way round, or with the
layers cut differently, or with a pipeline in place of a tensor split. On a shared drive both Macs use the same folder. When
each Mac keeps its models on its own disk, each has its own folder: a
Mac that needs layers another Mac saved copies them from that Mac's page
(over the link the pages use) before the model's first request, and
keeps them. A copy that would take longer than reading the prompt again
is skipped. Under a tensor split each Mac saves its share of every
layer's heads. A model family the tensor saving does not cover yet (see
the "Cache saving" list in [new-model.md](../design/new-model.md)) still
saves per split under tensor. Entries saved by 0.1.5 are still read back
for the split they were saved under; new saves use the new layout.

### Limits

| setting | default | |
|---|---|---|
| `KNURLOGIC_PROMPT_CACHE_DISK` | on | saving allowed at all |
| `KNURLOGIC_PROMPT_CACHE_DISK_GB` | 20% of free space plus what the cache holds, at most 64 GiB | all models' files together; oldest go first |
| `KNURLOGIC_PROMPT_CACHE_TTL_H` | 24 | files unused this long are deleted |
| `KNURLOGIC_PROMPT_CACHE_CONVERT` | down | sessions saved at another KV width: `down` converts 16-bit ones to the quantized width the model now runs, `both` also converts quantized ones back to 16-bit, `off` leaves them |
| `KNURLOGIC_PROMPT_CACHE_ANY_QUANT` | on | sessions saved by another quant of the same model (same architecture and tokenizer) are used; `off` reads only this quant's |

Converting takes a fraction of a second per entry where re-reading the
prompt takes minutes. 16 -> 8 matches what an 8-bit model stores up to
quantization noise; 8 -> 16 carries the 8-bit error into a 16-bit run.
The converted entry replaces the old one, so a session is saved once;
`usage.knurlogic.cache.converted` says when a request was served from
one, and registry rows carry each entry's `kv_bits`.

Borrowing: a model loaded at another quant (say 5.4 bpw after 3.4 bpw)
continues from the sessions the other quant saved, when it has none of
its own for them. It is an approximation: measured on Qwen3.6-35B quants,
the next-token distribution differs from reading the prompt again by mean
KL 0.015-0.042 (the top token agrees 89-96% of the time), about half of
what the two quants differ anyway. The other quant's files are only read,
never moved or deleted; the borrowing model saves its own copy at its
next save. `usage.knurlogic.cache.borrowed_from` names the quant a
request was served from. All five settings are under Settings ->
Knurlogic -> Prompt cache.

Pinned sessions are exempt from both limits; only a drop removes them.

## When a session misses

If a session's new prompt does not continue any of its cached entries
(the client re-rendered an earlier turn differently, dropped reasoning, a
header changed), it is read in full. `usage.knurlogic.cache.diverged` says
where it parted: the token position and a few words from each side. That
is almost always a client-side rendering difference worth fixing.

## Two or more Macs

A model split across Macs saves and restores the same way; each Mac keeps
its own part, and a saved session comes back under any split of the same
model (see above;
[prompt-cache-shards.md](../design/prompt-cache-shards.md) has the
design).

## What it cannot do

Remove a piece from the middle or front of a cached prompt: every later
token depends on what came before it. A prompt that changes early is read
again from that point.
