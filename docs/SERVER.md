# knurlogic's own server (design, 2026-09-25, for review)

the maintainer's green light (2026-09-25): start the server, pinned first by an API
conformance suite, with a a review design review that has the harness's ingest
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

## the harness's ingest requirements (docs/PLAN.md)

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

### Requests: what is running and what is waiting

Any number of requests may wait; nothing is refused for queue length. The
server decides how many run at once and reports, per model, a `requests`
object (`Scheduler.requests()`, lock-free, cheap enough to poll):

| field | meaning |
|---|---|
| `in_flight` | rows admitted and generating (at most `capacity`) |
| `pending` | requests waiting: queued, held for memory, or admitted past the batch and waiting for a slot |
| `capacity` | the most rows decoded together (`decode_concurrency`) |
| `oldest_pending_s` | seconds the oldest pending request has waited (0 when none) |
| `holding` | why they wait: `loading`, `memory`, `batch_full`, `queued` (arrived, admitted next step), or `null` when nothing waits |

Where it appears:
- `GET /status.json` -> `requests` (the server's own status document);
- `GET /v1/residency` -> each `data[]` row's `requests` (the harness);
- the page's `GET /loaded.json` -> each knurlogic `resident[]` row's
  `requests` (read from that server's /status.json; `null` for other
  runtimes). A cluster's row is rank 0's server, which runs the scheduler;
- the MCP's `state()` -> `requests`: one entry per knurlogic model, the
  fields above plus `model` and `where`.

The existing 503s carry `Retry-After`: 5 s for no model loaded, 10 s for
insufficient memory, 30 s for a cluster that is stopping. No new 429 or 503.

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
  from the last exchange, raised by what rank 0 has taken since (equal
  shards, same ops) and never lowered by what it freed (a free here is not
  yet a free there) -- so eviction and admission are decided once, on
  rank 0.
- **VQ**: a codebook is replicated, never sliced (`tensor.predicate`);
  codes and scales split. `tuning/resolve.tensor_refusals` refuses with the
  arithmetic when heads do not divide or a packed down_proj slice would cut
  a code word (IN/N % max(group, 32 x dim)); VQ dense/embedding modules are
  refused in this slice.
- **Measured** (M4, TheDrainFlorist--Qwen3.6-35B-A3B-VQ-3.4bpw, two
  ranks on one machine over the ring on 127.0.0.1, prompt chunk 512,
  greedy): each rank holds 6.9 GiB (placement said 6.9). Decode 29.6
  tok/s against 70.7 in one process (n=4 each; the ring's loopback
  collectives, 80 per token, are the cost -- one machine is the proof
  rig, not the use). Rank 1 disagreed with rank 0's token 0 times in 2211
  steps (concurrent rows, a cancel, prompt-cache hits, sampled rows), and
  a ring is reproducible run to run. It is NOT token-identical to one
  process: the split sums bf16-rounded partials, first-token logits move
  by a few bf16 ulps, and greedy text forks at the first near-tie (tokens
  36, 61, 74 of the three prompts; the tied pairs were 0.00, 0.13 and 0.13
  nats apart). One process is not batch-invariant either: the same prompt
  sequential vs beside two others forked at token 49-63.
- **Off in this slice**: MTP drafting, images (400), switching models.
- **Bring-up**: `knurlogic serve <artifact> --rank r --world n --split
  tensor --link ring|jaccl --hosts a:p,b:p --prefill-chunk N
  --working-set-gib G` (hidden flags: the cluster page passes them; rank
  order is `tuning/resolve.rank_order`). `mx.distributed.init(strict=True)`,
  a barrier, then each rank loads lazily, splits, and evaluates its shard.
  Only rank 0 binds HTTP.

## Cluster: pipeline split (2026-09-27, second slice)

One model, N ranks, each holding a contiguous run of layers
(`engine/runtime/pipeline.py`; the step plan is the tensor split's,
unchanged). `--split pipeline`.

- **Layout**: rank 0 -- the leader, which samples -- holds the LAST layers,
  the final norm and lm_head; rank N-1 holds the first layers and embeds.
  Hidden states go rank N-1 -> ... -> 0 by `send`/`recv`, each in its own
  rank's dtype (the ranks' stage dtypes are gathered when the model is
  split; a send is cast to its receiver's, never read off a placeholder:
  the maintainer's 574a7bd7). Every send is evaluated inside the forward that makes
  it and every receive is waited for inside the forward that uses it, so
  point-to-point messages and the CPU collectives stay in program order on
  every rank.
- **Who samples, and how logits reach rank 0**: they are born there.
  mlx-lm's pipeline all_gathers the last stage's hidden state so every
  rank computes logits; we do not, because no follower needs them -- the
  plan carries rank 0's tokens, as under tensor. A follower's trunk
  (`pipeline.Silent`) returns zeros of the logits' shape, so its lm_head
  never runs and its NaN guard never fires alone. Cost per decode step:
  zero bytes for logits, against one all_gather of [B, L, H] (plus a
  follower lm_head each) for the gather design. B0: the step that admits a
  row broadcasts every row's next token after the admission, because the
  admission samples its first token after that step's plan went out.
- **Layer shares** (`tuning/resolve.pipeline_shares`, pure Python): each
  rank's weight is what it can hold (working set less the replicated
  embed/norm/lm_head), times its memory bandwidth when EVERY rank's is
  known (a table of unbinned chips, or `--bandwidth-gbs`; a binned Max is
  unknown, never guessed); largest remainder, ties to the lower rank,
  every rank >= 1 layer, capped by what fits by the average layer and then
  checked with the real per-layer bytes from the safetensors headers. After
  the ring is up every rank all_gathers (working set, bandwidth, layer
  count, layer-bytes checksum), refuses if the artifacts differ, and
  computes the same split; rank 0 prints it with the reason.
  `--layers a,b,...` (rank order) overrides it.
- **Families**: qwen3_5 / qwen3_5_moe (PipelineMixin's start/end contract
  kept, its uniform split and all_gather not used; fa_idx/ssm_idx
  recomputed on the slice), glm5_next (fa_idx/ssm_idx, the maintainer's f3ab3a83),
  qwen4_exp (ple_layers and a sliced make_cache, the maintainer's dd946407). Gemma4
  is refused: it shares KV across layers. Tested two ways on 127.0.0.1,
  float32, against the unsplit model: logits (every family, uneven cuts)
  equal to 1e-4 -- in fact 0.0: the same arithmetic in the same order.
- **MTP on pipeline** (qwen3_5 families; the maintainer's exo design): the head
  lives on rank 0, which has the true final hidden state. A follower runs
  a head too, on its own stage's output: its drafts are never used, but
  its head cache moves exactly as rank 0's, so prompt-cache entries,
  offsets and replays are the same code on every rank. Per step, fixed:
  B1 `[drafting, d2 per row]` before the verify forward (rank 0's timed
  regime choice and its drafts), and, only when B1 said drafting, B2
  `[ok per row, t2 per row]` after it (the verdicts that drive every
  rank's rollback and the one replay forward). The count depends on B1,
  which every rank receives, never on a rank's own verdict. Every rank
  drafts or none does (`tensor.agree_head` after load). Tested: rank 0's
  tokens are the unsplit engine's (random head, mostly rejected; and
  drafting forced on vocab 8 so accepts happen), both ranks made the same
  B0/B1/B2 counts, and the serving path (TensorExecutor + follow) streams
  the unsplit executor's tokens.
- **Images**: refused (400) in this slice, as under tensor. The choice for
  when they land: rank 0 runs the tower and SENDS the image rows'
  embeddings to rank N-1 (its ring neighbour), with MTP off for image
  rows. Running the tower on every rank would need the image bytes and the
  store on every rank; the embeddings are needed at the first stage only,
  but MRoPE positions are needed at every stage, so they go in the admit
  op (they are pure in the key).
- **Memory guard**: as tensor (the tightest rank, via the control vector),
  but the peers' own over-limit is used as reported, refreshed every step:
  stages are unequal, so rank 0's memory says nothing about a peer's.
- **Bring-up**: `knurlogic serve <artifact> --rank r --world n --split
  pipeline --link ring --hosts a:p,b:p --prefill-chunk N
  --working-set-gib G [--layers a,b] [--bandwidth-gbs X]`.

## Cluster: launch and failure (2026-09-27, third slice)

Page to page, two phases (`interfaces/cluster_jobs.py`; files and markers
`cluster/jobs.py`). No token: running knurlogic is consent; every
`/peer/cluster/*` route has `/peer/loaded.json`'s gate (no Origin;
loopback, Thunderbolt, or a `--peer` address).

- **Launch**: the page's `POST /loaded.json {action: load, identity,
  nodes: [ids], split: tensor|pipeline, link: ring|jaccl}` (one node: the
  single-peer path). The coordinator reads each machine's `cluster` block
  from its status (chip, working set under the allowance, bandwidth,
  Thunderbolt addresses, RDMA, knurlogic/mlx versions, self-heal), orders
  ranks (`resolve.rank_order`), places the model (tensor share /
  `pipeline_shares`) and shows it. `prepare` to every page: each checks
  the artifact by identity, the fit of ITS share, versions, its link;
  any refusal and nothing starts. Then `start`: each page spawns its own
  rank (`serve --rank/--world/--split/--link/--hosts/--job/--layers ...`,
  MLX_RANK and MLX_HOSTFILE in the job dir; jaccl: MLX_IBV_DEVICES from
  the rdma_<iface> on the peer's Thunderbolt subnet, MLX_JACCL_COORDINATOR
  on rank 0's Thunderbolt address). Prompt chunk 512, ring-wide.
- **RDMA probe**: `rdma_ctl status`, `ibv_devices`, `ibv_devinfo`
  (PORT_ACTIVE); the page greys RDMA with the reason.
- **Failure**: stock mlx has no collective timeout, so it is out of band.
  Each rank writes `~/.cache/knurlogic/jobs/<job>/rank<r>.json` (phase
  joining/loading/ready, step, rank 0's busy) every 2 s and at each phase
  change. The page that started a rank watches it: pid gone, never joined
  (300 s), or rank 0 busy with a still step counter for 120 s (idle is not
  stalled). Any of them: that page SIGTERMs its ranks (SIGKILL after 10 s)
  and sends `/peer/cluster/stop` to every other page of the job. Rank 0's
  SIGTERM answers every request in flight with a 503 (`cluster_failed`)
  before it exits. Unloading the job from any page is the same stop.
- **jaccl self-heal** (fork present): JACCL_COLLECTIVE_TIMEOUT_MS=0 while
  loading, set to 60000 after load (the maintainer's d2e82f92 / 43dc7f56).
- **Registry**: `jobs/jobs.json`, keyed `<job>/<rank>`; rank 0 also in
  `servers.json` by its port (chat, relay, residency find it there).
- **Measured across two Macs** (2026-09-27; M4 Max 128 GB leads, M3 Ultra
  96 GB follows, Thunderbolt, launched from the M3's page, greedy, 200
  tokens, n=3, prompt chunk 512). 35B-A3B VQ 3.4bpw: one process 71.5 (M4)
  / 55.6 (M3) tok/s; tensor ring 38.5, tensor jaccl 51.5; pipeline ring
  57.8, pipeline jaccl 59.1. 397B-A17B VQ 2.4bpw (the biggest rung whose
  identity matches on both): tensor jaccl 25.6 (prefill 254 tok/s at 4.4k
  tokens; 56 GiB a rank), tensor ring 19.9 (277), pipeline jaccl 27.3
  (265; 36/24 layers, M4 73 GiB, M3 ~50), and 27.0 with the M3 leading.
  Before MLX_METAL_FAST_SYNCH the 35B's tensor split made 14 (jaccl) and
  10.7 (ring). Tensor is token-identical ring vs jaccl; pipeline on one
  machine is token-identical to one process, across two chips it forks at
  the first near-tie (0.13 nats at token 7), as M4 vs M3 alone do. Killing
  either rank mid-stream stops both within ~3 s with nothing left behind:
  the follower's death is a 503 `cluster_failed`, and so is rank 0's: the
  page answers it with the job's stop reason (a stream already under way
  ends with one `data: {"error": ... "cluster_failed"}` event).

## Migration

1. The conformance suite (tests/api) passes on today's server: done on
   gemma e4b; Flash 2.1 and GLM 2.7 running.
2. Build behind `knurlogic serve --server knurlogic`; the old path stays
   the default.
3. Switch the default when the new server passes the suite on every
   family (plus the strict xfails flipped to passes: text stops, the harness's
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

## a review review (2026-09-25): build it -- accepted

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
   does too. knurlogic's own pipeline does not gather: rank 0 holds the
   last layers, and only it has logits. The rule stands for a different
   reason: rank 0 samples, and what crosses a boundary stays small.)
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
7. **the harness amendments**: concurrency header `X-Knurlogic-Concurrency:
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
(flips the harness's xfails); (5) Flash, GLM, Qwen, measured decode/prefill,
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

Known gaps pinned as strict xfails: text stop sequences; the harness's five.

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

a review build review (2026-09-25): no blockers; eight findings, all
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
