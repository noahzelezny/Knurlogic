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
| `X-Cache-Keep: latest` | keep only this session's newest step (for clients that only append) |
| `X-Cache-Retain: pin` | never delete this session's saved files |

A request without `X-Client-Session` still uses the cache, but nobody owns
its entries and they are never saved.

**Keep latest** is for agents that only ever add to their conversation.
Each new step replaces the session's older ones, so one session holds one
step instead of dozens. The system prompt's checkpoint is shared by every
session that starts with it, and is kept. Do not use it from a chat that
edits or regenerates earlier messages.

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
| `POST /v1/prompt-cache/save {"session"}` | save that session's newest entry (call it after your context compacts) |
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

`~/.cache/knurlogic/prompt-cache/`, one folder per model. They can be
deleted at any time; the worst case is a slower next request. (A visible
folder next to your models is coming: see
[the design](../design/prompt-cache-shards.md).)

### Limits

| setting | default | |
|---|---|---|
| `KNURLOGIC_PROMPT_CACHE_DISK` | on | saving allowed at all |
| `KNURLOGIC_PROMPT_CACHE_DISK_GB` | 20% of free space, at most 64 GiB | all models' files together; oldest go first |
| `KNURLOGIC_PROMPT_CACHE_TTL_H` | 24 | files unused this long are deleted |

Pinned sessions are exempt from both limits; only a drop removes them.

## When a session misses

If a session's new prompt does not continue any of its cached entries
(the client re-rendered an earlier turn differently, dropped reasoning, a
header changed), it is read in full. `usage.knurlogic.cache.diverged` says
where it parted: the token position and a few words from each side. That
is almost always a client-side rendering difference worth fixing.

## Two or more Macs

A model split across Macs saves and restores the same way; each Mac keeps
its own part. Today a saved session only comes back under the same split
(same Macs, same layers each). Split-independent saves are designed in
[prompt-cache-shards.md](../design/prompt-cache-shards.md).

## What it cannot do

Remove a piece from the middle or front of a cached prompt: every later
token depends on what came before it. A prompt that changes early is read
again from that point.
