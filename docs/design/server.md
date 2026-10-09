# knurlogic's server

knurlogic reuses mlx-lm as a library and owns the server around it.

## Why own the server

mlx-lm 0.31.3's `server.py` has server-level faults that cannot be fixed
from outside without patching internals that move between releases:

| fault | where it lives |
|---|---|
| a seed is ignored (the compiled sampler reads the main thread's RNG state) | sampling + threads |
| an exact prompt-cache hit leaves no segment; the generation thread dies | batch path |
| HTTP answered before the model is loaded | lifecycle |
| stop sequences matched as token ids, not text (`stop: "D"` misses `" D"`) | state machine |
| NaN logits sampled as token 0 with a normal finish | sampling |

It also serves seeded requests on a separate sequential path, and has no
scheduler a cluster pipeline could use.

**Kept from mlx-lm (pinned):** model architectures and `load`; the tokenizer
wrapper, chat-template application, think tokens, tool parsers; KV cache
classes and trim/extract/merge; `BatchGenerator` (subclassed);
`LRUPromptCache`, wrapped in knurlogic's own PromptCache (which owns the
exact-hit rule); `sample_utils` building blocks, not its compiled
categorical. Nothing else from `server.py`.

## Layout

```
interfaces/http/            the wire
  server.py                 ThreadingHTTPServer, routes, body cap, the
                            browser guards (Origin / Host), --host cluster
  openai.py                 /v1/chat/completions, /v1/completions,
                            /v1/models, OpenAI error objects, SSE
  residency.py              /v1/residency, /v1/ensure, the concurrency hint
  prompt_cache.py           /v1/prompt-cache: save, drop, pin, park, list
  messages.py               /v1/messages, in-process
  __init__.py               serve(), switch() (through interfaces/load_checks)
interfaces/page/            the page (`knurlogic ui`)
interfaces/load_checks.py   what a model must pass before it loads
cluster/launch.py           a cluster job, page to page; recovery.py
context_management/         what the model sees; no mlx, no HTTP
                            (see compaction.md)
engine/runtime/             everything that touches mlx
  model_host.py             ModelHost: empty/loading/ready/unloading/failed
  scheduler.py              ONE thread owns the MLX stream: commands,
                            tokenize, prompt cache, admission, steps
  memory_guard.py           the Scheduler's memory guard (a mixin)
  prompt.py                 template, segments, initial reasoning state
  request.py                per-request text: reasoning split, text stops,
                            tool calls, usage
  executor.py               the step: LocalExecutor
  control_tokens.py         the control-token state machine
engine/split/               the cluster splits, their step plan and ring
                            (TensorExecutor is split/ring.py)
engine/prompt_cache/        the prompt cache: memory.py (PromptCache),
                            disk.py, commands.py (the Scheduler's cache
                            methods), ring.py (a ring's journaled cache),
                            report.py (usage.knurlogic.cache)
engine/mtp/                 the batch engine, drafting, segment checkpoints
engine/mtp/sampling.py      per-request seeds
```

**One path.** Every request, seeded or not, with images or not, goes
through the batch executor; a seed is a per-row key.

## Executor

`engine/runtime/executor.py`. In: `Admission`. Out: `Progress`,
`Checkpoint`, `Token` (token and its logprob, top-k on request -- never a
`[V]` row: what crosses a boundary stays small), `Finished` (the row's
cache) and `RowFailure` (a failing request beside a succeeding one fails
alone). `LocalExecutor` wraps the MTP batch generator, which takes sampling
params per row and reports the cache per row. Checkpoints come at each
segment end and at the prompt less its last token (that token is fed as its
own segment). The scheduler is a port of mlx-lm's generation loop over this
protocol; admission and batching do not care where the layers run.

## Scheduler and host

`ModelHost` has states empty/loading/ready/unloading/failed; requests block
on ready (a request during load waits, it is not mistranslated). The
scheduler thread owns the MLX stream and does commands, tokenizing
(including vision, [vision.md](vision.md)), the prompt cache, admission,
steps, per-request text and cancellation. MLX arrays made on the
scheduler's stream are freed on it -- freed after the stream ends, the
process segfaults -- so `stop()` unloads there.

**Seeds.** The token at position n of a seeded row is drawn with
`key(seed, n)` (Gumbel-max); keys for the batch are drawn by `mx.vmap`
(measured equal to a per-row loop draw for draw, 460 us vs 444 us plain at
B=8) and advance by `vmap(split)`. When rows leave, the key array is
rebuilt with take/stack: a strided view under vmap gives wrong draws on mlx
0.31.2. A seeded row verifies a draft by drawing the target under the same
key, so a seeded request is identical across batch composition and
drafting/plain regimes up to the logits themselves -- a batched forward is
not bit-identical on every kernel (GLM-5.3 drifts up to 0.3 in logprob
under load).

## Per-request text

`engine/runtime/request.py`, no model:

- Control tokens are a token state machine (`control_machine`); user stops
  are not in it.
- The output is split into reasoning, answer and tool calls (tool calls
  through the tokenizer's parser).
- User `stop` strings match detokenized **answer** text after the reasoning
  split, with a hold-back of max(len(stop))-1 characters: never inside
  reasoning, never in a streamed delta. Stated in `/status.json`.
- The detokenizer is finalized at the end; multi-byte UTF-8 split across
  tokens never yields U+FFFD.
- Usage: prompt/completion/total on every response, reasoning tokens, and
  the cache report.

## HTTP

stdlib `ThreadingHTTPServer`, daemon threads, per-token flush; a failed
write cancels and removes the row (a client disconnect frees it).

- OpenAI error objects `{"error": {message, type, param, code}}`;
  `max_completion_tokens` accepted; `n>1` refused with 400; a template
  render failure is a 400 before any stream; prefill keepalives while a
  long prompt prefills.
- `/v1/messages` (Anthropic) runs in-process, including tool_use /
  input_json_delta streaming.
- 503s carry `Retry-After`: 5 s for no model loaded, 10 s for insufficient
  memory, 30 s for a cluster that is stopping. Nothing is refused for queue
  length.

### Endpoints for ingest clients

OpenAI shapes plus:

1. `/v1/residency`: a flat list -- model, capabilities, memory_bytes (mx
   active memory), nodes, state (loading/ready/unloading), `requests`,
   `instance`, `recovery`. Straight from ModelHost.
2. `/v1/models` entries carry `capabilities` (text/vision/thinking) and
   `size_bytes`.
3. `/v1/ensure {model, wait}`: idempotent; returns at once if ready,
   otherwise loads through the load lock and optionally waits. On a
   one-model process it is a switch: 409 while rows are in flight unless
   `force`.
4. Several `image_url` parts per message; an image over the family's max
   pixels is refused with 413 naming the limit, read from the image header
   (PNG IHDR / JPEG SOF) before decode.
5. `X-Knurlogic-Concurrency: rows=N, more=?1|?0` (RFC 8941): the batch's
   current width and whether one more row is measured to help, from the
   engine's per-width step timings; `more` is omitted when unmeasured. Also
   on `/v1/residency`. Ingest is often prefill- or image-bound, where the
   hint is weaker.

### Requests: running and waiting

Any number of requests may wait. The server decides how many run at once
and reports, per model, a `requests` object (`Scheduler.requests()`,
lock-free, cheap to poll):

| field | meaning |
|---|---|
| `in_flight` | rows admitted and generating (at most `capacity`) |
| `pending` | requests waiting: queued, held for memory, or admitted past the batch and waiting for a slot |
| `capacity` | the most rows decoded together (`decode_concurrency`) |
| `oldest_pending_s` | seconds the oldest pending request has waited (0 when none) |
| `holding` | why they wait: `loading`, `memory`, `batch_full`, `queued`, or `null` |

It appears in `GET /status.json`; in each `/v1/residency` row; in the
page's `GET /loaded.json`, on each knurlogic `resident[]` row (`null` for
other runtimes; a cluster's row is rank 0's, which runs the scheduler); and
in the MCP's `state()`, per model on this Mac and on the peers its page
sees.

### Instance ids

Every server the page starts (page launch or MCP `load`) gets a 16-hex
`instance` id, stored in its registry record next to its pid; a cluster
job's instance id is its job id (8-32 hex, stable across its ranks). It
appears in `/v1/residency` (read from this machine's registry by port), on
each knurlogic `resident[]` row of `/loaded.json` (`machine/loaded.py`), in
each MCP `state()` model (the same instance seen from two pages is one
entry), and on the page's Instances card (first 6 hex).
`unload(instance=...)` stops it, on this Mac or a peer, alongside `port`,
`model` and `job`.

## Cluster

`executor.py` is the seam: a cluster executor runs the layers on several
machines and the scheduler is unchanged. Machines are found as described in
[discovery.md](discovery.md). Rank 0 owns HTTP, the scheduler, tokenizing
and **all sampling**; ranks >= 1 follow.

### The step plan

Each step: one fixed-size `all_gather` of a control vector per rank
(`[active - limit, step, plan length]`), then, when rank 0 has something to
say, one `all_sum` of the plan's JSON bytes (never pickle; the other ranks
contribute zeros). Ops, in order: `admit` (tokens; the prompt-cache hit
rank 0 found, which the follower repeats and checks; sampling with an
assigned seed; penalties; the control machine's start), `remove`, `insert`
(store the cache from last step's checkpoint/finished event), `pop` (evict n
LRU entries), `set` (a live knob that acts on a rank's own engine --
`VQ_DECODE_CHUNK` and the cache limits; a parked ring is woken to take it),
`reset`, `stop`; and `tokens`: rank 0's next token for every live row. A
follower applies the plan, overwrites its batch's next tokens with rank
0's, and runs the same step. A plan ending in `reset` or `stop` is not
followed by a step.

- **Prompt cache**: count-based on a ring (a byte cap is refused); rank 0's
  byte trims become counted pops. Identical ops in identical order keep
  every rank's LRU identical.
- **Memory**: admission and eviction are decided once, on rank 0, against
  the tightest rank, from the peers' over-limit in the last exchange.
- **Images**: rank 0 alone holds the tower and the image store and encodes
  at tokenize as on one Mac (its bytes count on rank 0 only). A follower
  binds the family without a tower (`engine.vision.request.MirrorVision`).
  The admit op carries the cache key as ids plus its image runs and refs
  (`plan.key_to_wire`), so a follower's prompt cache is keyed by the same
  sentinels and its MRoPE positions come from the refs. An admission whose
  uncached span holds images ships those images' feature rows from rank 0
  (`Coord.images`: one CPU all_sum, float32, exact for bf16); every rank
  embeds with the family's own code. Only feature rows travel, never the
  prompt's embeddings. Image rows do not draft, on every rank alike.
- **Bring-up**: `knurlogic serve <artifact> --rank r --world n --split
  tensor|pipeline --link ring|jaccl --hosts a:p,b:p --prefill-chunk N
  --working-set-gib G [--layers a,b] [--bandwidth-gbs X]` (hidden flags; the
  page passes them; rank order is `tuning/rank_order.rank_order`).
  `mx.distributed.init(strict=True)`, a barrier, then each rank loads
  lazily, splits, and evaluates its shard. Only rank 0 binds HTTP.

### Tensor split

Every layer's weights split N ways (`engine/split/tensor.py`); rank 0's
executor is `TensorExecutor`, the local batch engine with every admission,
removal and prompt-cache change journaled into the plan.

- **VQ**: a codebook is replicated, never sliced (`tensor.predicate`); codes
  and scales split. `tuning/tensor_split.tensor_refusals` refuses with the
  arithmetic when heads do not divide or a packed slice would cut a code
  word or a quantization group.
- **Memory**: shards are equal, so the guard raises the peers' last
  over-limit by what rank 0 has taken since, and never lowers it by what it
  freed (a free here is not yet a free there).
- **Not supported**: MTP drafting, switching models.
- **Numerics**: not token-identical to one process -- the split sums
  bf16-rounded partials, first-token logits move by a few bf16 ulps, and
  greedy text forks at the first near-tie. One process is not
  batch-invariant either (the same prompt alone vs beside two others forks
  within ~60 tokens). A ring is reproducible run to run, and tensor is
  token-identical ring vs jaccl.

### Pipeline split

Each rank holds a contiguous run of layers (`engine/split/pipeline.py`;
the step plan is the tensor split's).

- **Layout**: rank 0 -- the leader, which samples -- holds the last layers,
  the final norm and lm_head; rank N-1 holds the first layers and embeds.
  Hidden states go rank N-1 -> ... -> 0 by `send`/`recv`, each cast to its
  receiver's stage dtype (gathered when the model is split). Every send is
  evaluated inside the forward that makes it and every receive is waited
  for inside the forward that uses it, so point-to-point messages and the
  CPU collectives stay in program order on every rank. The exception is a
  prompt's prefill chunks (`pipeline.overlapped`): a follower's chunk is
  sent while its next chunk computes, at most two sends in flight, all
  complete before the prefill's last forward. `KNURLOGIC_PIPELINE_OVERLAP=off`
  sends synchronously.
- **Logits are born on rank 0.** mlx-lm's pipeline all_gathers the last
  stage's hidden state so every rank computes logits; knurlogic does not,
  because no follower needs them -- the plan carries rank 0's tokens. A
  follower's trunk (`pipeline.Silent`) returns zeros of the logits' shape,
  so its lm_head never runs and its NaN guard never fires alone. B0: the
  step that admits a row broadcasts every row's next token after the
  admission, because the admission samples its first token after that
  step's plan went out.
- **Layer shares** (`tuning/pipeline_split.pipeline_shares`, pure Python): each
  rank's weight is what it can hold (working set less the replicated
  embed/norm/lm_head), times its memory bandwidth when every rank's is known
  (a table of unbinned chips, or `--bandwidth-gbs`; a binned chip is
  unknown, never guessed); largest remainder, ties to the lower rank, every
  rank >= 1 layer, capped by what fits by the average layer and then checked
  with the real per-layer bytes from the safetensors headers. After the
  ring is up every rank all_gathers (working set, bandwidth, layer count,
  layer-bytes checksum), refuses if the artifacts differ, and computes the
  same split; rank 0 prints it with the reason. `--layers` overrides it.
- **Families**: qwen3_5 / qwen3_5_moe (fa_idx/ssm_idx recomputed on the
  slice), glm5_next, qwen4_exp (ple_layers and a sliced make_cache). gemma4
  is refused: it shares KV across layers. Tested against the unsplit model
  in float32 with uneven cuts: logits equal (0.0 difference -- the same
  arithmetic in the same order).
- **MTP** (qwen3_5 families): the head lives on rank 0 alone, which has the
  true final hidden state; its bytes count there only
  (`pipeline_shares(leader_bytes=)`). A follower never runs it but takes
  part in every verify: its stage runs the drafted token as the second
  position of the 2-wide forward, and rolls back and replays as rank 0
  says. Per step: B1 `[drafting, d2 per row]` before the verify forward,
  and, only when B1 said drafting, B2 `[ok per row, t2 per row]` after it.
  Per admission, BA `[ok, hit, drafts]` before its prefill: only rank 0 can
  tell whether a prompt-cache entry has an aligned head cache. The count of
  collectives depends on B1 and BA, which every rank receives, never on a
  rank's own verdict. `follower.agree_head` tells every rank after load
  whether rank 0 bound a head.
- **Memory**: stages are unequal, so the peers' own over-limit is used as
  reported, refreshed every step.

### Launch and failure

Page to page, two phases (`cluster/launch.py`; files and markers
`cluster/jobs.py`). No token: running knurlogic is consent; every
peer message (`POST /peer/v1/msg`) has one gate (no Origin;
loopback, Thunderbolt, or a `--peer` address).

- **Launch**: `POST /loaded.json {action: load, identity, nodes: [ids],
  split: tensor|pipeline, link: ring|jaccl}` (one node: the single-peer
  path). The coordinator reads each machine's `cluster` status block (chip,
  working set under the allowance, bandwidth, Thunderbolt addresses, RDMA,
  knurlogic/mlx versions, self-heal), orders ranks, places the model and
  shows it. `prepare` goes to every page: each checks the artifact by
  identity, the fit of its share, versions, its link; any refusal and
  nothing starts. Then `start`: each page spawns its own rank (MLX_RANK and
  MLX_HOSTFILE in the job dir; jaccl: MLX_IBV_DEVICES from the rdma device
  on the peer's Thunderbolt subnet, MLX_JACCL_COORDINATOR on rank 0's
  Thunderbolt address; `MLX_METAL_FAST_SYNCH=1`). Prompt chunk 512,
  ring-wide.
- **RDMA probe**: `rdma_ctl status`, `ibv_devices`, `ibv_devinfo`
  (PORT_ACTIVE); the page greys RDMA with the reason.
- **Failure is out of band** (stock mlx has no collective timeout). Each
  rank writes `~/.cache/knurlogic/jobs/<job>/rank<r>.json` (phase
  joining/loading/ready, step, rank 0's busy) every 2 s and at each phase
  change. The page that started a rank watches it: pid gone, never joined
  (300 s), or rank 0 busy with a still step counter for 120 s (idle is not
  stalled). Any of them: that page SIGTERMs its ranks (SIGKILL after 10 s)
  and sends a `Stop` message to every other page of the job. Rank 0's
  SIGTERM answers every request in flight with a 503 `cluster_failed`
  before it exits; a stream already under way ends with one
  `data: {"error": ... "cluster_failed"}` event. Unloading the job from any
  page is the same stop. Killing either rank mid-stream stops both within
  ~3 s with nothing left behind.
- **jaccl self-heal**: with an mlx build that supports it,
  `JACCL_COLLECTIVE_TIMEOUT_MS` is 0 while loading and 60000 after load.
- **Registry**: `jobs/jobs.json`, keyed `<job>/<rank>`; rank 0 is also in
  `servers.json` by its port (chat, relay and residency find it there).

### Measured across two Macs

M4 Max 128 GB leading, M3 Ultra 96 GB following, Thunderbolt, greedy, 200
tokens, n=3, prompt chunk 512. Decode tok/s:

| model | one process | tensor ring | tensor jaccl | pipeline ring | pipeline jaccl |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B VQ 3.4bpw | 71.5 (M4 Max) / 55.6 (M3 Ultra) | 38.5 | 51.5 | 57.8 | 59.1 |
| Qwen3.5-397B-A17B VQ 2.4bpw | does not fit | 19.9 | 25.6 | -- | 27.3 |

The 397B: tensor jaccl prefills 254 tok/s at 4.4k tokens with 56 GiB a
rank; pipeline jaccl splits 36/24 layers (73 GiB / ~50 GiB) and makes 27.0
with the M3 Ultra as rank 0. Without `MLX_METAL_FAST_SYNCH` the 35B's tensor
split makes 14 (jaccl) and 10.7 (ring): the flag makes the GPU hand each
collective to the CPU by a spinning shared event instead of a
command-buffer completion. Pipeline on one machine is token-identical to
one process; across two chips it forks at the first near-tie, as the two
chips alone do.

### The MCP across machines

The MCP (`interfaces/mcp/`, stdio) runs in its own process; the page on
this Mac holds the peers, the jobs it coordinates and the watcher. So
everything past this Mac is a request to that page over loopback
(`KNURLOGIC_PAGE`, default `127.0.0.1:8899`) -- the same `POST /loaded.json`
its Launch and Unload buttons send. Without the page, `load` and `unload`
work on this Mac by port, and `state` lists this Mac.

- `load(artifact, port, tune, sets, machines, split, link[, cable])`:
  `machines` empty is this Mac (fit and ready checks, then a child server);
  one other Mac is the page's single-peer load; two or more is a cluster
  job, `link` `tcp` (ring) or `rdma` (jaccl). The answer is the job, rank
  0's `port`, the `leader`, `machines` and `placement {order, leader,
  layers, cable, cable_note}`. Every refusal -- a share that does not fit,
  a machine not answering, the model not on a machine, RDMA without a
  Thunderbolt 5 cable, another load in progress -- comes back as
  `{loaded: false, refused}` with nothing started. The server never
  evicts: the client unloads first.
- `unload(port | model | job | instance[, machine])`: a job with a rank on
  this Mac is stopped on every machine; a peer's model or job is unloaded
  through its leader's page.
- `state()` -> `models`: every resident model on this Mac and the answering
  peers, each with `machine`, `port`, `machines`, `split`, `link`, `job`,
  `leader`, `phase`, `requests`, `instance`, `recovery`. A cluster job is
  one entry, never one per rank; `machines` names any peer that did not
  answer.

## Auto-recovery

A model whose server or cluster rank dies or stalls without being asked to
stop is relaunched by the page that launched it, and says so
(`cluster/recovery.py`). Eviction and model choice stay with the client;
recovery only brings back what was running -- the same machines, rank
order (so the same split), link, port, tune and settings.

- **Who.** The page that coordinated a cluster launch (whether its Launch
  or the MCP asked) and the page that started a one-Mac server. A relaunch
  takes the same path as a launch: `cluster/launch.launch` with the
  recorded request (prepare checks fit, versions, links and one load at a
  time), or `mcp.load` for a one-Mac server. A link-init failure the cable
  failover is handling is left to it.
- **Never recovered.** Any requested stop (an unload from any page or the
  MCP, a page closing, a launch abandoned). A stop because the model does
  not fit or a machine ran out of memory -- a reason saying so, a prepare
  refusal for memory, or a log line (`Insufficient Memory`,
  `Unable to allocate`, ...) appended to the stop reason as
  `out of memory: <line>` -- is `failed` at once: relaunching into the same
  memory can take a machine down.
- **A machine gone.** A stop because a peer's page went away is retried
  only once every machine of the job answers again, within the window.
- **Limits.** At most 3 relaunches per model in 15 minutes, after 10 s,
  30 s and 90 s; the next failure makes the model `failed`, with the last
  reason, until someone loads it again. A relaunch starts only once no rank
  of the old job is left on any machine -- by record and by process
  (`pgrep` for `--job <job>`, a `JobState` message's `processes`) -- and a
  one-Mac server once no `knurlogic serve` on its port is left.
- **Reported.** `recovery: {attempts, last_reason, last_at, next_at, state}`
  (state `recovering` | `recovered` | `failed`; epoch seconds; `next_at`
  only while one is scheduled), or `null`, on each `/loaded.json` resident
  row and job; in `/loaded.json`'s `recovery` list (tracked models with
  nothing serving now); in each MCP `state()` model; and in the server's
  own `/v1/residency` row. The page keeps the record in
  `~/.cache/knurlogic/recovery.json` by port, and a cluster relaunch
  carries it to the page running rank 0. `recovered` stays reported for the
  window.
- **Switch.** `KNURLOGIC_RECOVER=off` in the page's environment turns it
  off.
- **Not covered**: a one-Mac server that hangs without exiting (its
  `stalled` phase is shown, not acted on); recovery state lives in the page
  process, so a page restart forgets what it was tracking.

## Conformance and performance

`tests/api` is an API conformance suite run against a live server, per
family: seeded request under concurrent load equals it alone; stops across
a token boundary, not inside reasoning, never in a streamed delta; client
disconnect frees the row; a failing request beside a succeeding one; a
request during load; OpenAI tool_calls and Anthropic tool_use, streaming
and not; OpenAI error shape, `max_completion_tokens`, `n>1` refused;
multi-byte UTF-8 across tokens; `/v1/completions`; the ingest endpoints.
It passes on gemma e4b, Qwen Flash-Next 2.1, GLM-5.3 2.7 and
Qwen3.5-397B 2.2 (skips: the text-model refusal on vision models, and
seeded-under-load on GLM, whose batched logits drift ~0.2-0.3).

Against mlx-lm's server (one process per run, 3 runs per arm alternating,
M4 Max 128 GB), knurlogic's is equal within noise on decode, prefill
time-to-first-token and 4 concurrent requests, on the same four models
(ratios 0.94-1.08, ranges overlapping). MTP drafting acceptance on the
suite: 0.90 (Flash-Next), 0.89 (GLM-5.3).

## Module notes

### src/knurlogic/engine/crosschip.py

mlx 0.31.2's `mx.quantized_matmul` moves from its matrix-vector kernel (qmv)
to its matrix-matrix kernel (qmm) at a row count that depends on the GPU
architecture (M3-generation applegpu_g15d vs M4-generation applegpu_g16s).
A forward of about 10-31 rows -- a short prompt, a prefill chunk's tail,
10-31 decode rows at once -- therefore rounds differently on the two chips
in every affine-quantized layer (attention, shared experts, lm_head), and
the ranks of a split across them drift apart.

The fix: a call with 9-31 rows against a 2-D weight is flattened,
zero-padded to 32 rows, run (qmm on every chip) and sliced back. Rows of a
matmul are independent, so the result is what a real 32-row call gives for
those rows; 1-8 and >=32 rows are untouched. Measured: M3- and M4-generation
chips bit-identical across all layers; +2-6% on the affected calls only.

Off by default: rank 0 samples every token, so rounding differences cannot
desync a cluster. `on` forces it; `auto` turns it
on for a cluster job whose machines have different GPU architectures.

### knurlogic/interfaces/http/residency.py

- `/v1/models` -- capabilities (text / vision / thinking) and
  `size_bytes`.
- `/v1/residency` -- one flat list: model, capabilities, `memory_bytes`,
  nodes, state (`loading` / `ready` / `unloading` / `failed`).
- `/v1/ensure` -- `{model, wait}`: idempotent; a different model is a
  switch -- only to an artifact this machine's stores hold, through the
  same checks as startup (`interfaces/load_checks.py`); 409 while requests are
  running, unless `force`.
- **413** -- an image over the decode limit (judged from its header before
  decoding: `engine/vision/images.py`), or a request whose images together
  exceed the image store's memory budget.
- `X-Knurlogic-Concurrency: rows=N, more=?1|?0` (RFC 8941) -- the batch's
  width now, and whether one more row measured faster per token; `more` is
  left out until both widths have been timed. Ingest is prefill- and
  image-bound, where the hint says less than it does for decoding.

### knurlogic/interfaces/http/server.py

Routes:

```
POST /v1/chat/completions  /chat/completions  /v1/completions   openai.py
POST /v1/messages          Anthropic, in-process over openai.py
GET  /v1/models            the served model, capabilities and size
GET  /v1/residency         what is loaded, its state and memory
POST /v1/ensure            load a model if it is not; optionally wait
GET  /health
and the page's routes (web.routes): /, /status.json, /settings.json, ...
```

A write that fails (the client went away) cancels the Job, so the
scheduler frees its row on the next step. A non-streamed request writes
nothing until its reply is complete, so its connection is watched instead:
once a second the handler peeks at the socket, and end-of-file or an error
cancels the Job (data waiting is not a hang-up). Without it, a client that
timed out and resent its request left every copy running to the end.
`KNURLOGIC_DISCONNECT_POLL=off` turns the watch off, for a client that
closes its sending side after the request and still reads the reply (none
known).

**Browsers.** A web page open in the user's browser can send requests to
localhost; without care, any site could make this server load a model.
Browsers mark their requests with `Origin`, and ordinary clients (SDKs,
curl, harnesses) send none. So a request carrying an Origin is answered
only when that origin is this server itself (the page it serves) or one the
operator allowed (`--allow-origin`), and only those get CORS headers. The
Host header must name this machine (a hostname, `.local` name or IP
literal), which stops DNS rebinding -- a foreign domain re-pointed at
127.0.0.1 to look same-origin.

### knurlogic/interfaces/serve.py

Point Cline, Continue, Zed, OpenWebUI or anything else that speaks OpenAI
at `http://host:port/v1`.

The environment is set before the server loads anything, and that ordering
is not incidental: a VQ artifact's bundled runtime reads its knobs at
import, and the import happens inside the server's own model load. Setting
them after would silently do nothing -- the same class of bug as an env
file sourced after the one that overwrites it.

### src/knurlogic/engine/split/pipeline.py

Rank 0 holds the LAST layers, so the logits are born on the rank that
samples and nothing is gathered: mlx-lm's pipeline all_gathers the final
hidden state to every rank so every rank can compute logits; here the
other ranks never need them (rank 0's step plan carries every token --
engine/split/plan.py), so a follower's trunk returns zeros of the logits'
shape and its lm_head never runs (`Silent`).

Written for knurlogic. The layer slice keeps the contract of mlx-lm's
PipelineMixin (start_idx / end_idx / pipeline_layers, which the vendored
qwen3_5 reads) without calling its `pipeline()`, whose split is uniform and
whose forward all_gathers (engine/runtime/PROVENANCE.md).

Hidden states cross ranks with send / recv, each evaluated where it is
built (a follower's step finishes its send inside the forward; rank 0's
receive is waited for inside the forward), so the order of point-to-point
messages and of the CPU collectives (the plan exchange, B0-B2 below) is the
program order on every rank. A prompt's prefill chunks are the one
exception (`overlapped`): a chunk's send runs while the next chunk
computes, and every send has completed before the prefill's last forward.
A receive is made in the RECEIVING rank's own activation dtype and a send
is cast to its receiver's (the dtypes are agreed when the model is split),
never the dtype of a placeholder: an unloaded embedding is float32, and a
float32 receive of bf16 bytes runs a whole shard in float32.

MTP on pipeline (the head lives on rank 0, which holds the last layers and
so the true final hidden state -- and on rank 0 ALONE: a follower never
loads a head and never runs one). A follower still takes part in every
verify: its stage runs the drafted token as the second position of the
2-wide forward, and rolls back and replays as rank 0 says. What a head
would have decided on a follower is told to it instead (`Coord`), so the
collective count per step is the same on every rank and never depends on a
verdict:

    BA  [ok, hit, drafts]        per admission, before its prefill: rank
                                 0's usable prefix (a drafting row needs
                                 an entry with an aligned head cache, which
                                 only rank 0 can see) and whether the row
                                 drafts (which moves its checkpoints)
    B1  [drafting, d2 per row]   before the verify forward: whether this
                                 step drafts (rank 0's timing decides) and
                                 the drafted tokens the first stage embeds
    B2  [ok per row, t2 per row] after the verify forward, only when B1
                                 said drafting: the verdicts that drive
                                 every rank's rollback, and the tokens that
                                 commit

and, on the step that admits a row, B0 [t1 per row]: the admitted row's
first token is sampled inside the admission, after the step's plan was
sent, and a follower's own sample is noise. A follower's prompt-cache
entries carry no head cache; BA makes that invisible.

### src/knurlogic/engine/split/plan.py -- step plan format

Control vector, one int64 per slot, one row per rank:

    [0] over    this rank's active memory minus its limit (signed bytes)
    [1] step    the step counter (every rank counts; a mismatch is desync)
    [2] length  plan bytes that follow -- rank 0's slot only; 0 = none

Ops, applied in order (every field explicit):

    admit   uid, prompt (every token), segs (what is prefilled, after the
            prompt-cache hit and the lean decision), hit (tokens the prompt
            cache supplied: rank 0's fetch, repeated and checked), max_tokens,
            sampling (make_distribution kwargs + an assigned seed),
            penalties, initial (the control machine's start state),
            images, refs (a prompt with images: `key_to_wire`; [] for text)
    remove  uids
    insert  uid, event ("checkpoint" | "finished"), kind: store this
            rank's cache from that event of the last step in the prompt cache
    pop     n: evict the n least recently used prompt-cache entries
    reset   close the executor (rank 0 closed its own)
    set     name, value: a live knob rank 0 applied to itself (a Settings
            apply); every rank applies it to its own engine before the step.
            Only SETS travel: a knob read by rank 0's scheduler alone
            (KNURLOGIC_CONTEXT_LENGTH) changes nothing on a follower
    stop    leave the loop

`tokens` (optional): [uid, token] for every row rank 0's batch holds at the
start of this step, in batch order.

A prompt with images is a cache KEY (engine/vision/key.py): ids with a
sentinel ("img", sha, proc_hash, k) per image token. JSON has no tuples,
and a follower's prompt cache must be keyed exactly as rank 0's, so the
admit op carries the key as ids (-1 at every image token), `images`
[[start, end, ref, k0]] per image run and `refs` [[sha, proc_hash,
n_tokens, grid_thw]] per image -- what a follower needs to rebuild the key
and to compute positions. The image rows themselves never travel in the
plan (they follow in the admission: pipeline.Coord.images).

### src/knurlogic/engine/runtime/request.py

  control tokens   think/tool markers and end-of-turn tokens are matched as
                   TOKEN sequences by the engine's state machine (built by
                   `control_machine`); their text never reaches the client.
                   A marker can be several tokens, so the last few tokens'
                   text is held until no marker can still be completing.
  reasoning split  a token's state says where its text goes: reasoning,
                   tool, or the answer.
  stop strings     the request's `stop` matches the detokenized ANSWER text
                   -- never reasoning, never a tool call -- across token
                   boundaries (matching token ids instead misses `stop: "D"`
                   against a token " D"). The last max(len(stop)) - 1
                   characters are held back, so a stop never leaks into a
                   streamed delta before it is seen.
  tool calls       the text between tool markers, parsed by the tokenizer's
                   own tool parser when the call closes.
  usage            prompt, completion (every token the engine emitted),
                   reasoning tokens, and the engine's cache report.

### src/knurlogic/engine/runtime/scheduler.py, memory_guard.py -- memory guard

Everything that touches the model happens on the scheduler thread, in
order: loads and unloads (commands), tokenizing (vision's image work
included -- its pins are thread-local), prompt-cache lookups and inserts,
admission and steps. A request waits for the host to be ready; while
nothing is running the loop blocks on the queue instead of spinning.

A step that outgrows the GPU working set is not an exception: Metal aborts
the process. Measured on an M4 Max (128 GB): Flash 4.4 at 97 GiB, four long
reviews and a full prompt cache climbed to 118 of 120 GiB over two hours,
then "Insufficient Memory" killed the server and every request in it.
Before each step, when active memory is past the limit (the working set
less a margin for one step's temporaries), the prompt cache gives up
entries first -- they are a convenience -- and then the newest rows are
stopped with `OutOfMemory` (a 503: retry), the least work lost. While
memory is past the admission mark, new requests wait.

A step cannot be interrupted, and admitting a row prefills its whole prompt
inside one step (plus a deep copy per segment checkpoint), so the per-step
check alone lets one long prompt jump past the limit (measured: a
50k-token agent turn took the same server from 114 to 118 GiB within a
minute, no step between). So admission estimates too, from what this
model's caches measured (fixed state per row plus bytes per token: hybrid
models carry linear-attention state whatever the length). A prompt wants
its KV twice -- the row, and the checkpoint copies the prompt cache keeps;
the prompt cache gives way for it; if only one copy fits, the row is
admitted LEAN, without checkpoints (the request beats the cache); failing
that it waits for running rows, or with none running is refused -- never
admitted to abort the process.

### knurlogic/interfaces/http/telemetry.py -- the telemetry contract

docs/design/telemetry.md (`telemetry: 1`), server side. Every inference
request (chat/completions, completions, Messages, Responses, Ollama) opens
a `Request` at the handler: a ULID, its api, and the four `X-Client*`
labels cut to their byte limits (128, Role 32; truncated, never refused).
The ULID is the answer's `X-Request-Id` (on success, stream or error) and
the job's `request_id`, which the scheduler writes as
usage.knurlogic.request_id beside the timing, whose contract fields
(`queue_ms`, `prefill_ms`, `decode_ms`, `prefill_tps`, `decode_tps`) sit
beside the older ones. A client's own X-Request-Id is not echoed. When the
handler finishes, the Request is one row of machine/ledger.py
(KNURLOGIC_HOME/ledger.db; 30 days or 256 MiB, `KNURLOGIC_LEDGER_DAYS` /
`KNURLOGIC_LEDGER_MIB`); `GET /v1/usage?group=session|run|...` sums it
(loopback only until keys exist).

A streamed request's prefill progress is an SSE `event:
knurlogic.progress` beside the `: keepalive d/n` comment, and while queued
a `phase: "queue"` event every second with the requests ahead. Messages
streams always carry it (the Anthropic SDK skips unknown event names);
OpenAI chat/completions and Responses streams only for a client that sends
`X-Client` (the OpenAI SDK hands an unknown event's data on as a chunk);
Ollama's NDJSON has none. `/v1/models` says `knurlogic.telemetry: 1`.

The page's router (`interfaces/http/telemetry.py`'s `id_of`, `_stream`) passes a client's
X-Request-Id on and returns the model server's.

### src/knurlogic/engine/runtime/timing.py -- where a request's time went

`usage.knurlogic.timing` gives the rates (TTFT, prefill and decode tok/s).
A rate says how fast, not where the rest went, so the timing also carries a
partition of the request's wall time, `spans_s`, from the HTTP handler
starting to build the job to the moment its usage is written:

    http_build       request body -> job (parse, translate)       HTTP thread
    queue            submitted -> the scheduler began admitting it
    tokenize         chat template + tokenizer (vision: the tower)
    admit_memory     fitting the prompt into memory (_make_room)
    cache_fetch      the prompt-cache lookup
    admit_other      the rest of _insert (executor insert, request setup)
    {prefill,decode}_gap      scheduler work before a step (memory guard,
                              chunk refit, admitting others)
    {prefill,decode}_forward  the executor's step (model, sampling, on a
                              ring the collectives)
    {prefill,decode}_host     after the step (events, detokenize, deltas)
    decode_forward_shared     a decode step that also prefilled ANOTHER
                              request's prompt: time waited on someone
                              else's admission
    prefill_waiting           a step that admitted another request while
                              this one waited its turn (one admission
                              per step)

It is a cursor, not a set of timers: each boundary charges the time since
the previous one to a bucket, so the buckets sum to `spans_whole_s` by
construction; `spans_unaccounted_s` is the instrument's own check and is 0
unless a mark was skipped. A step shared by several rows is charged in full
to each (each waited for all of it). A request that waited for memory and
was re-admitted carries its failed attempts in `queue`. Host clocks only:
nothing is evaluated or synchronized for it, so it does not change what the
GPU runs. `KNURLOGIC_TIMING_SPANS=off` turns it off, read live.
`vqlab serve-timeline` drives n requests of a stated length and prints the
median of each bucket.

### src/knurlogic/engine/model/__init__.py -- engine boundary

  load.py          engine info, load, memory, the cache limit, knobs a
                   running process can change, tool dialects
  state.py         what is served, vision: the process's dicts
  segments.py      the system prompt gets its own segment (checkpoint) on
                   templates where the empty-turn diff finds none (GLM)
  thinking.py      reasoning_effort -> each chat template's own controls
  vision.py        the served model's vision family, bound at load

The engine is mlx-lm (and mlx-vlm for multimodal architectures). The cost
of keeping the option of another engine open is exactly this package:
every other module asks here instead of importing mlx directly.
Knurlogic's own code is ~1000 lines against ~3900 lines of vendored
architectures and a 4500-line runtime shipped inside each artifact -- so
the package itself is nearly uncoupled, and the job is to keep it that way
rather than to abstract anything clever.

WHAT A REPLACEMENT WOULD HAVE TO HONOUR, because these are the ecosystem's
choices, not this package's:

  * `mlx_lm.models.<type>` / `mlx_vlm.models.<type>` is where a model class
    is looked up, by `importlib.import_module`. That is how `register` gets
    vendored architectures in front of installed ones without writing to
    anyone's site-packages.
  * An artifact may ship its OWN runtime and name it in `config.json`
    (`model_file`). That file is executed, and it is where a VQ artifact's
    kernels live. It is also a PER-ARTIFACT runtime boundary: a new artifact
    can bundle a runtime for a new engine while every already published
    artifact keeps running the one it shipped with. An engine migration is
    therefore per-rung, not global.
  * The vendored architectures are written against the mlx array API. An
    engine that mirrors that API runs them unchanged; one that does not
    rewrites 3900 lines and every architecture after.

VERSION SKEW IS THIS PACKAGE'S JOB. mlx-lm 0.31.3 executes `model_file`
unconditionally; 0.32.0 puts it behind `trust_remote_code=` and raises
without it. Passing the kwarg blindly is a TypeError on one, omitting it a
ValueError on the other, so a VQ artifact cannot load on both unless
something inspects the signature. Nobody downstream should ever learn that.

### src/knurlogic/engine/model/thinking.py

Which controls a template has is its DIALECT, detected from the template
text -- not from the architecture: one module (qwen3_5) ships templates
with on/off only (Qwen3.6) and with graded effort (Qwen3.8). The dialects
and their native levels live in the family manifests
(engine/families/<family>/__init__.py, "thinking").

Native controls only, never a token budget. A level the template cannot
express goes to the nearest native level AT OR ABOVE it (never less thought
than asked), or the highest there is; "none" on a template with no off
switch goes to its lowest level. Every response says what was applied, in
usage.knurlogic.thinking. Deciding HOW MUCH to think for a given question is
the harness's call; this is only the translation.

A client that sends `chat_template_kwargs` itself wins over the
translation, key by key -- it asked for something specific -- and the report
then says what the MERGED kwargs render to, not what was asked.

THE TEMPLATE TEXT PROPOSES, RENDERING DECIDES. Detection is a substring
pre-filter on the template; with a loaded tokenizer, `probe` renders a tiny
conversation through mlx-lm's own TokenizerWrapper once per native level
and once bare. The dialect is only trusted when every level renders
differently, and the model's default is whichever level the BARE render
equals -- because mlx-lm injects enable_thinking=<has_thinking> into any
request that is silent about it (tokenizer_utils.apply_chat_template), so
gemma, whose template defaults off, thinks by default when served.

Reasoning is streamed by default; `reasoning: {"exclude": true}`
(OpenRouter's spelling) strips it from messages and deltas. Its token count
goes in usage.completion_tokens_details.reasoning_tokens.
