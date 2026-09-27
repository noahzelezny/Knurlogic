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
interfaces/http/            the wire (as built)
  server.py                 ThreadingHTTPServer, routes, body cap, the
                            browser guards (Origin / Host), --host cluster
  openai.py                 /v1/chat/completions, /v1/completions,
                            /v1/models, OpenAI error objects, SSE
  scout.py                  /v1/residency, /v1/ensure, the concurrency hint
  __init__.py               serve(), switch() (through interfaces/loading)
interfaces/messages.py      /v1/messages: handler_over(), in-process
interfaces/loading.py       what a model must pass before it loads
engine/runtime/             everything that touches mlx (engine rule holds)
  host.py                   ModelHost: empty/loading/ready/unloading/failed
  scheduler.py              ONE thread owns the MLX stream: commands,
                            tokenize, prompt cache, admission, steps
  prompt.py                 template, segments, initial reasoning state
  request.py                per-request text: reasoning split, text stops,
                            tool calls, usage
  executor.py               the step: the local batch engine today; the
                            cluster pipeline later, same interface
engine/mtp/sampling.py      per-request seeds (Keys: key(seed, position))
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

## Cluster: tensor split (2026-09-27, first slice)

One model, N ranks, every layer's weights split N ways
(`engine/runtime/tensor.py`, protocol `engine/runtime/plan.py`).

- **Rank 0** owns HTTP, the scheduler, tokenizing and ALL sampling. Its
  executor is `TensorExecutor`: the local batch engine, with every
  admission, removal and prompt-cache change journaled.
- **Each step**: one fixed-size `all_gather` of a control vector per rank
  (`[active - limit, step, plan length]`), then, when rank 0 has
  something to say, one `all_sum` of the plan's JSON bytes (never pickle;
  the other ranks contribute zeros). The plan's ops, in order: `admit`
  (tokens, the prompt-cache hit rank 0 found -- the follower repeats the
  fetch and checks it --, sampling with an assigned seed, penalties, the
  control machine's start), `remove`, `insert` (store the cache from last
  step's checkpoint/finished event), `pop` (evict n LRU entries), `reset`,
  `stop`; and `tokens`: rank 0's next token for every live row. Ranks >= 1
  (`follow`) apply it, overwrite their batch's next tokens with rank 0's,
  and run the same step. A plan ending in `reset` or `stop` is not
  followed by a step.
- **Prompt cache**: count-based on a ring (a byte cap is refused); rank 0's
  byte trims become counted pops. Identical ops in identical order keep
  every rank's LRU identical.
- **Memory**: the guard reads the tightest rank -- the peers' over-limit
  from the last exchange, moved by what rank 0 freed since (equal shards,
  same ops) -- so eviction and admission are decided once, on rank 0.
- **VQ**: a codebook is replicated, never sliced (`tensor.predicate`);
  codes and scales split. `tuning/resolve.tensor_refusals` refuses with the
  arithmetic when heads do not divide or a packed down_proj slice would cut
  a code word (IN/N % max(group, 32 x dim)); VQ dense/embedding modules are
  refused in this slice.
- **Off in this slice**: MTP drafting, images (400), switching models.
- **Bring-up**: `knurlogic serve <artifact> --rank r --world n --split
  tensor --link ring|jaccl --hosts a:p,b:p --prefill-chunk N
  --working-set-gib G` (hidden flags: the cluster page passes them; rank
  order is `tuning/resolve.rank_order`). `mx.distributed.init(strict=True)`,
  a barrier, then each rank loads lazily, splits, and evaluates its shard.
  Only rank 0 binds HTTP.

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

## Fable 5.1 review (2026-09-25): build it -- accepted

1. **Executor protocol = today's BatchGenerator shapes** (insert_segments /
   next / extract_cache / remove / close); the scheduler is a port of
   `ResponseGenerator._generate`'s loop, not a new one. `mlx_lm.generate.
   BatchGenerator` joins the keep-list (subclassed). Delete the three
   patches on mlx-lm's data flow: the cache_report thread-local (pass the
   request into admission), `tag_samplers` (pass sampling params), and
   exception-as-progress (a `RowFailure` event).
2. **Executor output is (uid, token, logprob_of_token, top_k?)**, never a
   [V] row. (Corrected 2026-09-27: this said "on a pipeline only the
   last rank has logits". mlx-lm's pipeline all_gathers the last stage's
   hidden state, so every rank computes logits; under tensor every rank
   does too. The rule stands for a different reason: rank 0 samples, and
   what crosses a boundary stays small.)
3. **One path, per-row keys by `mx.vmap`**: measured equal to the per-row
   loop draw for draw, 460 us vs 444 us plain at B=8; keys advance by
   `vmap(split)`, lazy. Rules: rebuild the key array with take/stack when
   rows leave (a strided view under vmap gave WRONG draws on mlx 0.31.2 --
   batch_loop.filter must gather); `rejection_correct` takes keys too.
4. **Stops**: control tokens stay a token state machine; user `stop`
   strings match detokenized ANSWER text after the reasoning split, with a
   hold-back of max(len(stop))-1 chars; never inside reasoning. Stated in
   /status.json.
5. **OpenAI error objects** `{"error": {message, type, param, code}}`;
   accept `max_completion_tokens`; refuse `n>1` with 400.
6. **/v1/messages in-process** (today it self-requests over loopback).
7. **Scout amendments**: concurrency header `X-Knurlogic-Concurrency:
   rows=3, more=?1` (RFC 8941), `more` omitted when unmeasured, also on
   /v1/residency -- and ingest is prefill/image-bound, where the hint is
   weaker; `/v1/ensure` on a one-model process is a switch (409 while rows
   are in flight unless `force`); 413 from the image header (PNG IHDR /
   JPEG SOF) before decode; memory_bytes from mx active memory.
8. **Cluster: the executor owns the SPMD loop.** Rank 0: HTTP + scheduler;
   ranks 1..n: `executor.serve_forever()` applying broadcast admissions;
   per-rank prompt caches kept identical by identical insert order;
   tokens and the NaN verdict broadcast from the last rank (as built for
   tensor: from rank 0, which samples -- see "Cluster: tensor split"); admission is
   an event the executor acknowledges, never assumed synchronous.
9. **Keep LRUPromptCache, own the wrapper** (the exact-hit guard moves into
   a knurlogic PromptCache class). Own the trie later.
10. **HTTP: ThreadingHTTPServer**, daemon threads, per-token flush, a write
    failure stops and removes the row, requests block on `ModelHost.ready`.

Build order: (1) executor protocol + MTPBatchGenerator behind it;
(2) request.py -- detokenizer, reasoning split, text stops, usage (no
model; flips the stop xfail); (3) host + scheduler + per-row sampling
behind `--server knurlogic`, suite green on gemma e4b incl. seeds under
load; (4) http openai / anthropic in-process / knurlogic endpoints
(flips Scout's xfails); (5) Flash, GLM, Qwen, measured decode/prefill,
switch the default, delete the patches.

Suite additions required: seeded request under concurrent load equals it
alone; stops across a token boundary, not inside reasoning, never in a
streamed delta; client disconnect frees the row; a failing request beside
a succeeding one; a request during load; tool calls (OpenAI tool_calls +
streaming deltas, Anthropic tool_use/input_json_delta); OpenAI error
shape, max_completion_tokens, n>1 refused; multi-byte UTF-8 split across
tokens never yields U+FFFD; /v1/completions.

## Suite results on today's server (2026-09-25, M4)

| model | result |
|---|---|
| gemma e4b | 23 passed (after fixing headless prefix reuse) |
| Qwen Flash-Next 2.1 | 23 passed |
| GLM-5.3 2.7 | 22 passed; shared prefix 0 used -- fixed 2026-09-25 (CacheList offset): 23 passed |

Known gaps pinned as strict xfails: text stop sequences; Scout's five.

## Build progress

1. **Executor protocol -- done (2026-09-25).** `engine/runtime/executor.py`:
   `Admission` in; `Progress`, `Checkpoint`, `Token` (token + its logprob,
   top-k on request, never a [V] row), `Finished` (the row's cache) and
   `RowFailure` out. `LocalExecutor` wraps MTPBatchGenerator, which now
   takes sampling params as a dict and cache reports per row directly --
   the tagged sampler and the thread-local stay only for mlx-lm's server
   until step 5 deletes them. Checkpoints come at each segment end and at
   the prompt less its last token (mlx-lm's convention: that token is fed
   as its own segment). tests/test_executor.py.
2. **request.py -- done (2026-09-25).** `engine/runtime/request.py`: the
   control-token state machine (`control_machine`, no user stops in it),
   the reasoning/answer/tool split, text stops on the answer only with a
   max(len(stop))-1 hold-back, tool-call parsing through the tokenizer's
   parser, usage with reasoning tokens and the cache report. The
   detokenizer is finalized at the end (mlx-lm's server never does).
   tests/test_request.py, no model.
3. **Host, scheduler, per-row sampling -- done.** `engine/runtime/host.py`
   (empty/loading/ready/unloading/failed; poses as mlx-lm's provider for
   the status code), `scheduler.py` (one thread owns the MLX stream:
   commands, tokenizing incl. vision, the prompt cache with the exact-hit
   rule, admission, steps, per-request text, cancellation), `prompt.py`
   (template, segments, initial reasoning state). Seeds: the token at
   position n is drawn with key(seed, n) (Gumbel-max); a seeded row
   verifies a draft by drawing the target under the same key -- identical
   across batch composition and drafting/plain regimes up to the logits
   themselves (a batched forward is not bit-identical on every kernel:
   GLM 2.7 drifts up to 0.3 in logprob under load; gemma and Flash held).
   MLX arrays made on the scheduler's stream are freed on it: freed after
   it ends, the process segfaults (measured), so stop() unloads there.
4. **HTTP -- done.** `interfaces/http/`: `server.py` (ThreadingHTTPServer,
   daemon threads, per-token flush, a failed write cancels the row),
   `openai.py` (chat/completions/models, OpenAI error objects,
   max_completion_tokens, n>1 refused, a render failure is a 400 before
   any stream, prefill keepalives), `/v1/messages` in-process
   (messages.handler_over), `scout.py` (/v1/residency, /v1/ensure,
   capabilities + size in /v1/models, 413 from the image header,
   X-Knurlogic-Concurrency from the engine's per-width timings). The
   page's load/unload go through the scheduler.

Fable 5.1 build review (2026-09-25): no blockers; eight findings, all
fixed (20ea528) -- notably a pre-existing one: the draft step handed the
logits processors a history without t1, so penalties differed by regime.

### Conformance on knurlogic's own server (M4, 2026-09-25)

| model | result |
|---|---|
| gemma e4b | 37 passed, 1 skipped (text-model refusal: it has vision) |
| Qwen Flash-Next 2.1 | 37 passed, 1 skipped (same); drafting 0.90 |
| GLM-5.3 2.7 | 36 passed, 2 skipped (+ seeded-under-load: batched logits drift 0.31); drafting 0.89 |

### Measured: mlx-lm's server vs knurlogic's own (tools/server_bench.py, removed with mlx-lm's server; it is in git history)

M4, one server process per run, 3 runs per arm alternating; decode =
greedy streamed tok/s after the first token (median of 3), prefill = time
to first token on a ~3000-token fresh prompt, batch4 = 4 concurrent.
Ratio = knurlogic / mlx-lm of the medians.

| model | decode | prefill TTFT | batch4 | verdict |
|---|---|---|---|---|
| gemma e4b | 1.08 (spread 51-64) | 0.99 | 1.06 | equal within noise |
| Flash-Next 2.1 | 0.999 | 1.01 | 0.996 | equal within noise |
| GLM-5.3 2.7 | 0.94 (ranges overlap: 24.9-28.7 vs 26.1-27.9) | 1.06 (overlap) | 0.98 | equal within noise |
| Qwen3.5-397B 2.2 | 1.06 (overlap) | 0.97 (overlap) | 1.00 | equal within noise |

### Final build (mlx-lm's server removed, c70ec9a+), M4, 2026-09-25

| model | conformance |
|---|---|
| gemma e4b | 37 passed, 1 skipped (text-model refusal) |
| Qwen Flash-Next 2.1 | 37 passed, 1 skipped (same) |
| GLM-5.3 2.7 | 36 passed, 2 skipped (same + batched-logit drift 0.21) |
| Qwen3.5-397B 2.2 | 37 passed, 1 skipped (same) |

No tracebacks in any server log. Migration steps 1-4 done; the default is
knurlogic's own server and the only one.
