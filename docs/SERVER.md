# knurlogic's own server (design, 2026-09-25, for review)

Noah's green light (2026-09-25): start the server, pinned first by an API
conformance suite, with a Fable 5.1 design review that has Scout's ingest
requirements in the brief. The line drawn: **reuse mlx-lm as a library,
own the server around it.**

## Why now

knurlogic patches mlx-lm 0.31.3's 1904-line `server.py` in ~34 places
across six modules (model pin and swap, BatchGenerator factory,
handle_completion, generate_response, usage, _tokenize, generate, the
sampler, the prompt-cache lookup, the thinking layer). One day on a free
box found five server-level faults, each now patched from outside:

| fault | where it lives |
|---|---|
| a seed is ignored (compiled sampler reads the main thread's RNG state) | sample_utils + threads |
| an exact prompt-cache hit leaves no segment; the generation thread dies | server batch path |
| HTTP answered before the model is loaded; early requests mis-translated | server lifecycle |
| stop sequences matched as token ids, not text (`stop: "D"` misses `" D"`) | server state machine |
| NaN logits sampled as token 0 ("!!!!!") with a normal finish | sampling |

Two more were knurlogic's own and are fixed (concurrent probe renders; a
headless model discarding every prefix hit). The patches depend on
internals that move between releases, and the cluster pipeline needs a
scheduler mlx-lm's server does not have. Patching now costs more than
owning.

## Keep from mlx-lm (library, pinned)

Model architectures and `load`; the tokenizer wrapper, chat-template
application, think tokens, tool parsers; KV cache classes, trim/extract/
merge; `LRUPromptCache` (with the exact-hit guard) until replaced;
`sample_utils` building blocks (not its compiled categorical). Nothing
else from `server.py`.

## Own (new)

```
interfaces/http/            the wire: routes, SSE, errors, auth-free by design
  openai.py                 /v1/chat/completions, /v1/completions, /v1/models
  anthropic.py              /v1/messages (today's interfaces/messages.py)
  knurlogic.py              /status.json, /v1/residency, /v1/ensure, ui routes
engine/runtime/             everything that touches mlx (engine rule holds)
  host.py                   ModelHost: load / ensure / unload, one state
                            machine per model: loading -> ready -> unloading;
                            requests wait for `ready`, never race the load
  scheduler.py              ONE generation thread that owns the MLX stream;
                            request queue -> admission -> batch step ->
                            per-request emit; failures are per request
  request.py                per-request state: detokenizer, reasoning/answer
                            split, tool-call parse, TEXT stop matcher,
                            usage, cache report, NaN verdict
  sampling.py               per-request RNG key (seed -> key; unseeded ->
                            split from the scheduler's key), so a seed is
                            reproducible under batching, on any thread
  executor.py               the step: local batch engine today
                            (MTPBatchGenerator, drafting or not, vision);
                            the cluster pipeline later, same interface
```

The existing pieces move in unchanged where they already are the design:
the batch engine and segment checkpoints (engine/mtp), vision
(engine/vision + families), thinking translation (serve/thinking.py
becomes a request-stage function instead of a monkeypatch), the cache
report.

**One path, not two.** mlx-lm serves seeded requests on a separate
sequential path; that split is where two of the bugs live. Here every
request goes through the batch executor; a seed is a per-row key.

## Scout's ingest requirements (docs/PLAN.md)

1. `/v1/residency`: flat list -- model, capabilities, memory_bytes, nodes,
   state (loading/ready/unloading). Straight from ModelHost.
2. `/v1/models` entries carry `capabilities` (text/vision/thinking) and
   `size_bytes`, from the artifact and the family manifest.
3. `/v1/ensure {model, wait}`: idempotent; returns at once if ready,
   otherwise loads (through the load lock), optionally waits.
4. Several `image_url` parts per message (works today); an image over the
   family's max pixels refused with 413 naming the limit, before decode.
5. Prefix reuse across a shared prompt (works; now also headless), and an
   `X-Knurlogic-Concurrency` header: the scheduler's current batch width
   and whether one more row is measured to help (the batch engine already
   measures per-width step cost).
6. `usage` with prompt/completion/total on every response (today; keep).
7. OpenAI shape only; nothing custom.

## Cluster readiness

`executor.py` is the seam. A pipeline executor runs stages on several
nodes (mlx ring/jaccl, as mlx-lm's `sharded_load`), and the scheduler
stays the same: admission, batching and per-request state do not care
where the layers run. Node discovery is cluster/ (peers, Bonjour,
`--host cluster`). exo is not in the path.

## Migration

1. The conformance suite (tests/api) passes on today's server: done on
   gemma e4b; Flash 2.1 and GLM 2.7 running.
2. Build behind `knurlogic serve --server knurlogic`; the old path stays
   the default.
3. Switch the default when the new server passes the suite on every
   family (plus the strict xfails flipped to passes: text stops, Scout's
   endpoints) AND is no slower on a measured decode/prefill comparison
   (n>=3 per arm, one process per arm).
4. Remove the monkeypatches.

## Open questions for review

1. HTTP: stdlib `ThreadingHTTPServer` (what the page and today's server
   use, no dependency) vs asyncio. Streaming many clients over threads
   is fine at this scale; is there a reason to go async now?
2. Keep `LRUPromptCache` or own the trie now (the exact-hit bug, and
   checkpoints are already ours)?
3. Per-request RNG: `mx.random.categorical(key=...)` per row costs a key
   split per token -- measure, or batch the keys?
4. Where the per-request text stop matcher sits relative to the
   reasoning split and the detokenizer (stops must not fire inside
   reasoning? OpenAI does not say).
5. Anything in the ordering above that makes the cluster executor
   harder later.
