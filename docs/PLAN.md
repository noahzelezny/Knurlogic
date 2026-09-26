# Where Knurlogic is, and where it goes next

*Rewritten 2026-09-22, end of the third session. A STATE document, not a
log: what is true now, what was measured so nobody re-derives it, and what
is next. The git history holds the narrative. `CONTEXT.md` is the map.*

## What it is

Local models on your own machines, managed equally well by a person (the
page, the CLI) and by an agent (the MCP). It resolves the settings that
decide whether a model runs, reports what is true about the machine, and
drafts with multi-token-prediction heads no stock runtime uses. It serves
with its own server over mlx-lm as a library (docs/SERVER.md), replaces
exo for clustering, and carries the work that was trapped in forks of both.

## What is true now

    engine/       runtime/ is knurlogic's own server's engine half (host,
                  scheduler, executor, prompt and request stages; HTTP is
                  interfaces/http); serve/ says what is served (load,
                  thinking, vision, drafting head, cache report); drafting
                  on every batch (MTPBatchGenerator); families/ holds
                  everything per model family; generic mtp/, vision/, vq/
                  name no family
    families/     qwen, gemma4, glm5 -- each a MANIFEST (architectures keyed
                  by module, model_type spellings, heads, prefill widths with
                  evidence) plus architecture/ (vendored code, PROVENANCE,
                  pins, licenses), vision/, heads/. Explicit list in
                  families/__init__.py; being listed is being tested
                  (tests/test_families.py). Reviewed twice by Fable 5.1.
    machine/      64 artifacts found across every store; residency in every
                  runtime (exo's read, never driven); one load budget; which
                  build of each dependency is installed
    tuning/       settings with their evidence; resolve() -> env + argv
    interfaces/   MCP (9 tools), page, CLI, serve, Anthropic Messages,
                  connect, doctor
    cluster/      knurlogic's own multi-machine: peers, Bonjour discovery,
                  the `--host cluster` gate; exo's /state is read only as
                  one witness of which machines exist. Nothing drives exo
                  (the exo wrap -- `serve --cluster`, overrides, `place` /
                  `unplace` -- was removed 2026-09-25)

**Not yet run on a real model since the 2026-09-24 reorganisation**
(engine/serve split, families move): one vision_gate pass (gemma e4b) on
a free box, and `knurlogic smoke --pin` on GLM 2.7 -- glm5_next's import
paths moved, so it reads UNPINNED until then.

166 tests. The mlx tripwire is a folder rule: nothing outside `engine/`
imports mlx; checked that it fires on a lazy import planted in `machine/`.

**Verified by driving it, not just by tests:**

* An MCP client over stdio loaded a real model (Qwen3.8-27B-VQ-3.9bpw),
  the server answered, the session ended, the server kept serving, and a
  SECOND session found it and unloaded it. Its log showed the budget limited
  by memory available now and `prefill_step_size=4096` (the measured qwen3_5
  width) reaching the engine.
* mlx-lm's own argument parser read knurlogic's argv: `prefill_step_size
  512` and `prompt_concurrency 1`, against its defaults of 2048 and 8.
* Batch drafting is greedy token-identical to mlx-lm's BatchGenerator on a
  tiny random qwen3_5 -- three rows admitted mid-decode, reject-heavy
  (vocab 512) and accept-exercising (vocab 8, drafting forced).
* `knurlogic deps` distinguishes the jaccl fork from stock by reading the
  built library, not a version: YES in exo's interpreter, no in knurlogic's
  (measured while deps still probed exo's).

## Where the forks stand

The one home for what each fork carries and why it is or is not ported is
`PIECES` in `machine/deps.py` (mlx, mlx-lm, mlx-vlm); `knurlogic deps` says
what is installed. In short:

    exo fork (mtp-stage1, 86 commits)
      ported    MTP: sequential, batched, heads, registry (engine/mtp/);
                the per-family prefill table (tuning/settings.py)
      not used  placement, sharding, networking, jaccl deadline arming,
                subnet pinning, KV-pool budget, VQ codebook sharding --
                knurlogic replaces them (see 'Next: replace exo')
    mlx-lm fork (exo-qwen4-exp)
      ported    the architectures, vendored and pinned by digest
    mlx fork (jaccl-selfheal)
      cannot    compiled C++ in mlx itself; ships as a wheel. deps detects it
    vqlab
      boundary  vqlab builds models and heads; knurlogic runs what it built

## Measured, so it is not re-derived

### Drafting

* **The head is a separate artifact** beside the weights
  (`mtp-head-q6.safetensors`, outside the `model*.safetensors` glob on
  purpose, so a loader never picks it up). 11 built on this disk: 2.14 GiB
  (qwen4_exp), 5.41 (qwen3_5), 6.09 (glm5_next), all 6-bit. MLX conversion
  drops upstream `mtp.*` weights while keeping the config key that declares
  them, so "declares a head" does not mean "has one": 40 declare, 1 ships
  graftable raw weights, 11 have a built head.
* **Every released rung of every model a head was built for has one.** By
  architecture: glm5_next 3/0, qwen4_exp 3/0, qwen3_5_moe 4/4, qwen3_5 0/3.
  The seven without are Qwen3.6-35B-A3B and Qwen3.8-27B, never built for.
  Lab intermediates (`qwen4exp_vq_*`, `397b-v2-*`) are not releases.
* **A head binds to the base model, not the rung.** The Flash-Next VQ-2.1
  sidecar drafted coherently on the community `Qwen3.8-Flash-Next-3bit`
  (main vs draft max|a-b| 8.0456, cosine 0.820; control 0.0000 / 1.000). So
  community quants can draft. No acceptance or speed claim from that run.
* **Speed, from the exo fork** (vqlab report 2026-09-04, stage 1, TB4 TCP):

        rung                    stock 300/2000   MTP 300/2000   acceptance
        GLM 2.7   (one box)      19.5 / 5.7      22.3 / 11.0    0.887/0.701
        GLM 3.1   (cluster)       6.9 / 6.2      18.5 / 17.4    0.727/0.642
        397B 2.6  (cluster)      23.7 / 23.5     23.7 / 23.4    0.853/0.851

  GLM cluster rungs up to 2.8x on long generation; the 397B at parity.
  Batched drafting, Flash-Next VQ-2.1bpw, M3: 23.1 vs 22 tok/s, identical
  tokens (2026-09-17). `vqlab mtp-accept` is the instrument for acceptance;
  quote speed only from a thermally stable box (the same config measured
  twice in one process gave 1.723x and 1.135x on a hot M4 Max).
* **Verify-width numerics.** A 2-token verify forward and a 1-token forward
  differ by up to 2e-2 in logprob through the recurrent kernels (float32).
  At vocab 4 a row with a top-2 margin of 8.6e-4 flipped. A near-tie, not a
  logic fault -- a fault diverges at once and at every vocab.
* **The port is a copy, verified by identity**: knurlogic's probe returns
  `max|a-b| 8.12155818939209`, cosine `0.8312658071517944`, the fork's floats
  to the last digit. `tools/mtp_probe.py` is the gate.

### Memory and the prefill spike

* **The prefill spike is the dense-expert transient, not the KV cache**:
  3.35 MB/token where KV theory predicts 0.059. `transient = chunk * out *
  in * 2`, with out/in read off the artifact. A family effect, not a box
  effect.
* **`VQ_DECODE_CHUNK`: smaller is also faster** (128 -> 32 is 1.37x on every
  rung). Nothing may raise it above 32. Sizing from the artifact TIGHTENS
  only; loosening is unmeasured and wrong there is an OOM.
* **Prefill chunk per family** (from the exo fork's constants.py):
  glm5_next 2048 (4096 OOMed both boxes of a 224 GB pair on the 135 GB
  3.6bpw -- before the per-chunk eval fix; 512 after it was over-caution);
  qwen3_5 / qwen3_5_moe 4096 (+115% prefill at 11k tokens against a 512 cap,
  no peak cost, bit-identical; a blanket SSM->512 made it 8x the chunks).
* **The prompt chunk is ring-wide.** Ranks that disagree desync -- seen live
  in the fork, GLM-5.3 at 2048 on one rank and 4096 on the other.
* **mlx-lm prefills 8 prompts in one forward by default**, and the transient
  is per prompt. On a tight box knurlogic sets 1.
* **Available memory is free + inactive** (vm_stat), which is what psutil
  and exo report. Installed-minus-footprints read 75.9 GiB and top's
  "unused" read 1.6 on a box with ~70 available. Footprints are `top`'s
  phys_footprint, not `ps` rss (2.3x apart on one process).
* **`iogpu.wired_limit_mb` is what the GPU working set follows** (86016 ->
  84.0 GiB of 96). knurlogic computes the number and prints the command.
* **RTILE=64 is 0.75-0.97x and never faster** (F25/F33). The per-flag
  findings (F54, F56, F103/F105, F124) live in `tuning/settings.py` beside
  the defaults they justify.

### Environments and processes

* **exo's model directory** is `<data home>/models`, and the data home is
  `~/.exo` on anything but Linux. Here `~/.exo/models` is a symlink to the
  external volume; not knowing that default hid 51 of 64 artifacts.
* **Knob names are the artifact's.** `VQLAB_CACHE_LIMIT_GB` is read by 24 of
  37 bundled runtimes; `VQLAB_PREFILL_CHUNK` by none. A knob has a logical
  name and a list of env names; the resolver emits the one the target reads,
  or hands it to the engine as argv.
* **Live vs restart**: `VQ_DECODE_CHUNK` and the cache limit apply to a
  running server; the GEMM/numerics flags are compiled into Metal source at
  import and need a restart; the prompt chunk and concurrency are server
  argv, read once.
* **Version skew is engine/serve/load.py's job**: mlx-lm 0.31.3 runs `model_file`
  unconditionally, 0.32.0 raises without `trust_remote_code=`.
* **Tool calling**: all 54 templates swept, none silently unparsed (40
  qwen3_coder, 7 glm47, 3 gemma4, 1 json_tools, 3 none).

### Driving exo (2026-09-22, live, two nodes)

knurlogic no longer drives exo (removed 2026-09-25). What was learned is
kept as what knurlogic's own placement has to get right.

* **A placement is invisible until exo publishes it.** A second placement
  posted one call after the first was accepted: no instance, no runners in
  /state yet, memory still reading free. A placement has to count as
  moving from the moment it is posted.
* **exo leaves ghost runners.** Removed instances left runners in
  RunnerShuttingDown indefinitely, referenced by no instance, with the
  node's memory back to exactly what it was. Counting them made `ready`
  itself wait forever; an orphan unchanged for 30s is reported, not counted.
* **exo reports load progress; nobody showed it.** RunnerLoading carries
  layers loaded of total: 27B went 25 -> 63 of 64 layers in 8s, warmed, and
  served at 14s. RunnerFailed carries exo's error message.
* **exo's refusals need arithmetic added.** "No cycles found with
  sufficient memory" becomes: 100.9 GiB needed, 37.0 free, 63.9 short, and
  these two instances are holding it. exo's error body is
  `{"error": {"message"}}`, not `{"detail"}`.
* **A local server answers before it holds its weights** (mlx maps them
  lazily): `warming` until 90% resident.

## Load-bearing design decisions

* **One load budget** (`machine/wired.load_budget()`): the smaller of the
  GPU working set and memory available now. `fit`, `models`, `settings`,
  `load` and `serve` all compute against it. When they used different
  numbers, a 47.5 GiB model had 6.8 GiB to spare in one answer and 36 GiB of
  roomy defaults in the next.
* **`resolve()` takes a byte count or nodes**; a cluster gets one Resolution
  per node plus ring-wide values enforced across them.
* **`ready` gates on memory in motion.** A server `load` started that is
  still loading, or any process holding the model-load lock, blocks `load`
  (`force` passes it; nothing passes `fit`).
* **Servers outlive the session that started them and stay stoppable**:
  `~/.cache/knurlogic/servers.json`, each unload checking the pid is still a
  knurlogic serve. Output goes to `~/.cache/knurlogic/serve-<port>.log`.
* **The tuning axis is capped by measurements**, and a profile that cannot
  have what it asked for says so.
* **Kernel knobs belong to whoever packs the kernels**:
  `Artifact.declared_knobs()` outranks anything scanned. Empty today; a
  hand-off point for vqlab.

## Discipline that cost this project real time

* **Every probe needs a channel proving the arms differ**, independent of
  the quantity measured. An A/B once ran identical code in both arms because
  the artifact executes its own bundled model.py.
* **A check run in the wrong interpreter answers for that interpreter.** A
  version probe looped over env paths with bare `python3`. `deps` asks each
  interpreter in its own subprocess for that reason.
* **Speed claims**: n>=3 per arm, alternating, one process per arm, quote the
  ratio never an absolute, refuse when within-arm spread exceeds the
  between-arm difference.
* **An index is a summary, not the disk.** The first head survey read the
  safetensors index and found zero; the sidecars are kept out of it on
  purpose.
* **A setting that is emitted is not a setting that is read.** It happened
  three times: the prompt chunk as an env var mlx-lm's server never reads,
  the cache limit under a name nothing applied, and knurlogic's names handed
  to exo, which reads its own. Trace every knob to its consumer.
* **A fixture in a shape the real system never sends proves nothing.** The
  `ready` test used `{"m": {"pct": 12}}`; exo sends lists of
  `{DownloadPending: {...}}`, and the real bug -- `ready` false forever --
  hid behind the fake.
* **A version floor is a proxy.** `mlx-vlm >= 0.7.1` called exo's 0.6.17 too
  old for GLM (exo loads its own, older glm5_next, which needs none of the
  modules knurlogic's copy does) and could not say what was missing. The
  check is now the modules the vendored code imports.
* **Drive it.** Every MCP bug this session was found by using the MCP as a
  client, not by reading it; the tests passed throughout.

## Vision: built on branch `vision-integration` (2026-09-23), not yet on main

Design: `docs/design/vision.md` (v2 + three reviews folded in) and
`docs/design/vision-contracts.md`. Built by a swarm (P0 contracts; P-VQ,
P1 Qwen, P4 serve on Opus; P2 gemma, P3 GLM, P5 interfaces on Sonnet 5),
then integrated: end-to-end tests per family through the real serve path,
image pinning by prompt-cache refcount, vision in the memory budget, the
chat tab, and a guarded /chat proxy on the control page. 364 tests green on
tiny random fixtures; every new gate mutated once.

Found at the joins and fixed: gemma's image mask broke after one chunk of
cached text; GLM silently dropped images (inputs_embeds) and crashed on
short forwards (an unvendored module); all released rungs reported no
vision (the `_text` model_type spelling, third time); the chat could not
reach any running model from the control page.

Real-model gates, 2026-09-23, on the M4 (NozzleBook, M4 Max) in a clean
`pip install` venv, tools/vision_gate.py, one rung at a time:

    gemma e4b VQ-PLE        PASS
    gemma 26b-a4b 6.2       PASS
    Qwen3.8-Flash-Next 2.1  PASS
    Qwen3.8-27B 3.9         PASS
    Qwen3.6-35B-A3B 3.4     cache PASS; reads "42" as "4" at 448 px, "42"
                            at 896 px -- the model's acuity, not the path:
                            identical preprocessing to 27B/Flash, which
                            read the 448 px image

Each gate checks: text answer, the red square, the "42", no image
re-prefill on turns 1-4 (new tokens < the image's own token count,
measured with/without the image), recall after five turns.

Found by the real gates (all invisible to the tiny fixtures) and fixed:
- e4b stores `embed_vision`'s projection 8-bit affine; towers are now
  quantized to match the checkpoint before loading (engine/vision/quant.py)
- gemma's placeholder `<image_soft_token>` is not a token in the released
  tokenizer; it is now boi + `<|image|>` + eoi, read from tokenizer.json
- released gemma loads through mlx-lm's `gemma4` wrapper (text under
  `.language_model`), which drops `mm_mask`: vendored with that one edit,
  registered from the config's outer model_type
- Qwen3.6's template re-renders the previous assistant turn without the
  empty think block it generated with, so on a hybrid (non-trimmable)
  model no turn ever reused the cache. The batch generator now stops
  prefill at the server's segment ends and hands checkpoints back
  (end-of-segment responses); a drafting row's checkpoint holds the head
  one step back plus h_{c-1}, replayed with the new token on restore.
  Tested token-identical to a fresh prefill and to mlx-lm.

Also seen: Flash-Next's first turn reuses nothing from an earlier request
with the same image and different text -- expected on linear attention
(no trim), turn continuation reuses fully. The runtime's comment that
Flash-Next vision "stays unreachable" is stale: the standalone tower
serves it.

Done 2026-09-23, all on the M4 (clean pip venv, no mlx-vlm):
- A failed admission or decode step fails its own requests; the
  generation thread lives on (was: every later request hung).
- G-VQ: 13 rungs bit-identical to their published model.py (logit diff
  0.0, 40/40 tokens): e4b, gemma 26b, Flash-Next 5.5, 27B 3.9/4.5/4.8,
  35B-A3B 3.4/3.8/4.6/5.4, 397B 2.2, GLM 2.7 -- all 9 runtime families
  but 397B arc6. Verified rungs now LOAD on knurlogic's runtime (serve
  routes through it; before this, nothing did).
- The chat proxy streams a full answer from exo (curl and the page's chat
  tab: TTFT 446 ms, 63.8 tok/s on e4b).
- GLM-5.3-Flash serves from `pip install knurlogic`: glm5_next re-vendored
  from mlx-vlm 0.6.17 (the rungs' build version; 0.7.1 cannot load them),
  loaded through the VQ runtime, MTP drafting on (acceptance ~0.83), and
  the vision gate PASSES -- after fixing its pixel normalization (a red
  square was "salmon").

Before merging to main:
1. G-VQ still unrun: 397B 2.4/2.6/3.1 and GLM 3.1/3.6 (larger than one
   machine: cluster or a free M3); Flash-Next 2.1/3.2/4.4 need the HUB
   weights -- the local copies on both machines are a lab build whose
   vq_modules differ from the published config (Hub config and weights
   agree). Told the vqlab session: any Flash number scored from ~/.exo for
   those rungs is not the published artifact.
2. Thinking effort, one control for every family (decided 2026-09-24):
   - accept OpenAI `reasoning_effort` on chat completions and Anthropic
     `thinking` on /v1/messages. The ladder is the STANDARD names, nothing
     invented: none / minimal / low / medium / high / xhigh. Omitted = the
     model's own default.
   - NATIVE controls only. No token budgets (some models truncate mid
     thought; it is rudimentary). A level a model cannot express maps to
     its nearest native setting and the response says what was applied.
   - "auto" (thinking by difficulty) is a classification call: the
     harness's, not knurlogic's.
   - KEYED BY CHAT-TEMPLATE DIALECT, not by architecture (Fable 5.1's
     review): Qwen3.6 (on/off) and Qwen3.8 (on/off + effort) are the same
     qwen3_5 module. The dialect is detected from the artifact's template,
     as the tool-call dialect is (serve/load.py tool_support); the mapping
     ladder -> native kwargs is per dialect. New module serve/thinking.py
     is the plug point -- nothing touches chat_template_kwargs today.
   - BUILT 2026-09-24 (engine/serve/thinking.py, dialects in the family
     manifests; tests render the released templates), then reviewed by
     Fable 5.1 and fixed: the template text only PROPOSES a dialect -- a
     render probe through mlx-lm's own TokenizerWrapper decides, and finds
     the served default (gemma thinks by default when served: mlx-lm
     injects enable_thinking for silent requests); a client override is
     reported by what it renders; reasoning streams by default,
     `reasoning: {"exclude": true}` strips it; reasoning_tokens counted;
     Anthropic /v1/messages returns thinking blocks (streamed as
     thinking_delta + signature_delta) only when thinking is enabled, and
     passes usage.knurlogic on; /status.json carries the same answer as
     the MCP. Not yet measured on
     a served model: needs knurlogic serving a rung on a free box -- exo
     cannot stand in, it IGNORES chat_template_kwargs (Flash-Next 4.4 via
     exo still reasoned with enable_thinking=false, 2026-09-24), so no
     client can turn thinking off through exo today. Measure per dialect,
     n>=3 per arm: reasoning tokens and answer quality at each level, and
     GLM's closed-think-block "off" before offering it.
   - Return reasoning as `reasoning_content`; count it in
     completion_tokens_details.reasoning_tokens; the MCP `models` tool
     lists each model's native levels.
   - Read from the templates: Qwen3.8 enable_thinking + effort
     low/medium/xhigh; Qwen3.6 on/off; GLM-5.3 effort low/high/max, no
     off; gemma 4 on/off, OFF by default; DeepSeek-V4 thinking_mode.
     GLM "off" via an already-closed think block is its own format, not a
     budget -- measure answer quality before offering it.

Done 2026-09-24 (CPU, tiny fixtures, all five families):
- usage.knurlogic.cache on every batch-engine response: offered, used,
  discarded, prefilled, via (none / prefix / checkpoint), images
  {total, in_cached_span, prefilled, encoded}, checkpoints_stored; and
  cached_tokens is what the engine USED, not what the trie offered.
  tools/vision_gate.py checks it on real models (discarded 0, encoded 0).
- The encode-twice gate: same image, preprocessed and encoded twice from
  scratch, bit-identical features.

## Done 2026-09-25: the free-box session (M4 alone)

1. **gemma e4b through `knurlogic serve`: vision gate PASS** after two
   fixes it found -- `serve` died at start (the package's `load()`
   shadowed the `load` module; tripwire test added) and the token count
   disagreed with the tower (gemma's aspect-preserving resize was missing;
   pinned against mlx-vlm 0.6.17). Five turns: turns 1-4 prefill only
   their 16 new tokens, the 258-token image never again; recall at turn 5.
   Thinking: `<|channel>thought` split out, `none` off (0 reasoning
   tokens), reasoning_tokens counted, `/v1/messages` gives thinking +
   signature + text blocks, `reasoning.exclude` strips it.
2. **GLM 2.7 smoke PASS; first pin since the families move** (mlx-lm
   0.31.3, mlx-vlm 0.6.17).
3. **Thinking per dialect** -- tools/thinking_bench.py, 4 checkable
   questions x 3 unseeded runs at t=0.6, one process per arm, results in
   docs/measured/2026-09-25-thinking/. Every arm 100% correct: these
   questions separate levels by COST, not by accuracy.

   | model (dialect) | arm: mean reasoning tokens (per-run range) |
   |---|---|
   | gemma e4b (toggle) | off 0 . on 506 (426-572) |
   | Flash-Next 2.1, dev v2 (effort) | none 0 . low 192 (180-204) . medium 258 (233-292) . xhigh 136 (126-150) |
   | GLM-5.3 2.7 (effort) | closed 0 . low 12 (9-13) . high 17 (16-19) . max 151 (129-163) |
   | Qwen3.5-397B 2.2 (toggle) | off 0 . on 957 (917-1015) |

   * Flash: xhigh reasons LESS than low/medium on these questions; the
     ranges do not overlap. The template is applied as written (low and
     xhigh add their system lines, medium adds none) -- this is the model
     on easy questions, not the controls. Not a claim about hard ones.
   * **GLM closed-think "off" works**: prompt at low effort with
     `<think></think>` already closed, 12/12 right, 0 reasoning, 35 tokens
     an answer vs 59 at low. Offering it as GLM's `none` needs a prompt
     suffix rather than a template kwarg -- a design call for Noah (it is
     the template's own format for past turns, not a budget).
   * Timing is secondary here (4 concurrent requests, run 0 includes the
     load); not a speed claim.

## NEXT (set 2026-09-25, second compaction)

Done today: knurlogic's own server is the only one (docs/SERVER.md: four
families conformant on the M4, equal speed to mlx-lm's within noise); two
Fable 5.1 reviews (full branch, then re-review: MERGE); the exo wrap is
gone (exo is read, never driven); mlx-vlm is not a dependency (GLM served
without it); browser guards on both servers; ~/Knurlogic/Models is
knurlogic's own (visible) store.

1. **Merge vision-integration to main** -- Noah's call; the re-review said
   merge.
2. **Multi-machine serving, knurlogic's own** (replaces exo): the cluster
   executor behind engine/runtime/executor.py (docs/SERVER.md "Cluster
   readiness": rank 0 HTTP + scheduler, ranks 1..n serve_forever, tokens
   and the NaN verdict broadcast from the last rank), over cluster/
   (peers, Bonjour, --host cluster). Placement lessons: "Driving exo"
   findings below.
3. **HF download + a hub search tab** on the page (exo's shape), into
   ~/Knurlogic/Models (KNURLOGIC_MODELS moves it). Needed by 2 as well
   (a model onto a second Mac).
4. Firewall-prompt UX for a satellite that missed it (Noah decides the
   details).
5. Upstream the mlx-lm bugs found (seed on threads, exact-hit crash, stop
   strings as token ids, NaN as token 0, system-segment diff).

Machines: the M3 Studio is off-limits for model loads unless told. The M4
(ssh 10.0.0.2, ~/kl-test/venv, wheel from a fresh clone) is the test box;
its GLM env (gvqvlm) was removed -- everything runs from venv now. Models
live on /Volumes/Thunderbay SSD/Exo Models (~/.exo/models links there).

## Requirements for knurlogic's own server: Scout's ingest (2026-09-25)

Sent by the Scout session at Noah's request. scout/ingest/vlm.py is the
only client and talks through a small Backend protocol, so matching these
keeps the exo swap to one class. Today it uses exo's GET /state, GET
/v1/models (`capabilities` incl. "vision", `storage_size_megabytes`) and
POST /v1/chat/completions (`image_url` data URLs,
`chat_template_kwargs: {enable_thinking: false}`); it picks the largest
resident vision model and falls back to a default loaded on first use.

1. One honest residency endpoint for every instance kind: flat list of
   model id, capabilities, memory used, node placement, state
   (loading/ready/unloading).
2. Catalog entries with `capabilities` (text/vision/thinking) and size.
3. Idempotent "ensure loaded": same POST whether resident or not, with a
   way to wait for ready.
4. Several `image_url` parts in one message; size limits (max edge)
   refused explicitly, never an OOM.
5. Prefix-cache reuse across requests sharing a long fixed prompt, and a
   concurrency hint (response header) saying whether a second in-flight
   request helps.
6. `usage` with prompt and completion tokens on every response.
7. OpenAI shape as the wire format; nothing custom.

Already true or close: 5 (prefix cache + usage.knurlogic.cache), 6, 7,
`enable_thinking` passes through (and `reasoning_effort: none` is the
portable form). To check: 4 (multi-image, explicit limits). New: 1-3.

## OPEN from the M4 session (2026-09-25)

* **GLM: a shared system prompt is offered but not used.** After the
  segment fix the 472-token system checkpoint IS offered (it was never
  stored before), then discarded: "no aligned head cache". GLM's head has
  `reassign` cache semantics and the checkpoint/HeadCarry restore path
  works for Qwen's head (Flash passes the same test), not GLM's. Options:
  keep the trunk hit and do not draft that row (right for short-output
  ingest), or make GLM's head restorable at checkpoints. A decision, not
  a patch -- for the server build (executor step 1).

* **GLM's `none` is SHIPPED** as the template's own closed-think format
  (Noah, 2026-09-25): off 0 reasoning tokens on the served model, status
  lists off/low/high/max. It exposed a latent mlx-lm crash -- an EXACT
  prompt-cache hit leaves no segment and kills the generation thread --
  now guarded (engine/serve/cache_guard.py).
* **"!!!!!" runs on Flash (Noah, seen in Scout).** Not the seed bug. In
  Qwen's vocabulary token 0 is "!", and a sampler fed NaN/inf logits
  returns 0 forever: a numerical overflow upstream of sampling. Next: a
  NaN guard in the decode loop that stops the row and reports the step
  (and, if cheap, the first layer that went non-finite) -- measured for
  cost before it ships.

* **A seeded request ignored its seed -- FIXED (engine/serve/sampling.py).**
  gemma e4b at temperature 1.5 answered "Flummoxed" for every seed. Cause,
  reproduced with no model in stock mlx 0.31.2 + mlx-lm 0.31.3: mlx-lm's
  sampler is `mx.compile`d with `mx.random.state` as input, MLX random
  state is per thread, and the server samples on its generation thread --
  where the compiled function reads a state nothing updates (one token
  forever). knurlogic serve now swaps in the plain call. Worth reporting
  upstream. The bench still sends no seed, so tonight's numbers do not
  depend on the fix.
* **The first concurrent requests after a load were "not controllable" --
  FIXED (thinking.py, `_render_lock`).** The probe rendered the template
  from every handler thread at once, concurrent renders failed, and the
  failure read as "the template ignores its controls". Every bench arm's
  first four requests got the model's default level instead of the one
  asked for (a Flash `none` request reasoned 242 tokens). The first bench
  pass was discarded and rerun after the fix.
* **Requests that arrive while the model is still loading got no
  translation -- FIXED (thinking.py, `_served_tokenizer`).** mlx-lm answers
  HTTP before its generation thread loads the model; with no tokenizer
  there was no template, and every request was served the model's default
  (Flash-Next 2.1: `none` requests reasoned 89-232 tokens, `low` got
  xhigh). The artifact's tokenizer is now read off disk until the served
  one exists. The Flash pass was discarded and rerun.

## Done 2026-09-24: the page is knurlogic's own

The chat panel's near-verbatim exo ports were rewritten from what each must
do, not from exo's source, and THIRD-PARTY-UI.md is gone; the README says
the look is inspired by exo. The `<think>` splitter was dropped -- the page
shows the server's `reasoning_content` (gemma's `<|channel>thought` never
matched `<think>`) -- and the "enable thinking" checkbox, which sent a field
nothing read, is a `reasoning_effort` picker on the standard ladder. The
SSE reader was tested against a stream cut into 3-byte chunks (split UTF-8
and lines, CRLF, keepalive, [DONE]); the page chatted end to end with
Flash-Next 4.4 via exo, thinking streamed into its panel. The listed PDF
routine was never in the page (PDFs say "not supported yet").

## OPEN, and it may move published numbers: the Flash-Next PLE hash seed

Found 2026-09-23 during the vision integration (P1), confirmed new by the
vqlab session. qwen4_exp's n-gram PLE hashes n-grams with per-layer
multipliers derived from a seed. Three sources disagree:

    Flash-Next checkpoint buffer layer_multipliers   seed 1234's values
      (model.layers.1: [23703573157769, 20109073645365, 8052911324071])
    mlx-vlm 0.6.17 qwen4_exp/config.py:55             seed 1234
    Noah's mlx-lm fork (vqlab's fit/score env),       seed 0, and the
      exo (same fork), knurlogic (vendored copy)      stored buffer unused

The official configs declare no seed, so the fork uses 0 and recomputes
`_mults`, ignoring the checkpoint's buffer. The "seed 1234" in vqlab's
cards is the k-means fit seed -- a coincidence, not evidence.

Implication, NOT yet measured: every fork run of Flash-Next looks up PLE
rows with a different hash than the checkpoint carries. VQ-vs-teacher KL
stays internally consistent (both ran seed 0), but the teacher reference
may be off, and a stock mlx-vlm user gets different PLE rows than the
artifacts were fitted under.

Before anything changes:
1. The decisive A/B, on the bf16 TEACHER: ppl with seed 0 vs with the
   checkpoint's layer_multipliers, same text, one process per arm. If the
   checkpoint's hash wins, teacher caches rebuild and every published
   Flash KL number moves (vqlab's instrument; Noah's call).
2. Record which default each shipping path hits (above) -- a downloader-
   divergence question as well as a reference-quality one.
3. knurlogic keeps its current default until 1 has run and Noah decides.
   Scheduled (vqlab, 2026-09-23): the teacher A/B runs on 2026-09-24,
   after the paper handoff work; seed 0 stays until it reports.
   The fix, if confirmed: use the checkpoint's stored multipliers (the
   artifact is the authority), not a guessed seed.

## Decided: 35B-A3B 3.8 / 4.6 / 5.4 stay on v2 numerics

Their published model.py carries v2 knobs, not what the rungs were built
with. vqlab decided (2026-09-23, docs/RUNTIME-SHIP-PLAN.md at 6f53200): no
rebundle to v1.5 -- v2 is faster and closer to bf16; the rungs are
rescored on the shipped v2 bundle. knurlogic reads knobs from the
published Hub model.py, so nothing changes here.

## Release (target: about a week, with the paper)

- Build release wheels from a FRESH clone (or `rm -rf build` first):
  setuptools reuses build/lib and never drops deleted files -- a wheel
  built here on 2026-09-24 still carried the pre-move architectures/ and
  seam.py until build/ was removed.

The HF model cards will point at knurlogic, so every RELEASED model must
run from `pip install knurlogic` -- five families: qwen3_5_moe, qwen3_5,
qwen4_exp, glm5_next, gemma4.

* **Vision is standard, and owned.** Checked 2026-09-23: all 20 released
  rungs carry `vision_config` and their vision weights (333 tensors in a
  `model-vision-graft.safetensors` sidecar for the Qwen families and one
  gemma; inside the main shards for GLM and gemma e4b). mlx-vlm 0.6.17
  implements all five families. So: vendor the vision tower and image
  processor per family with provenance (GLM's seven mlx-vlm siblings come
  with it, and mlx-vlm stops being a dependency); an image-capable serve
  path beside the text one (text requests keep drafting); a gate per family
  of one text answer and one image answer.
* **The page opens a chat on the loaded model**, with image attach.
* **`models` stops listing non-chat models** (an embedder, whisper, siglip,
  a background remover) as servable.
* **Credit the interface**: README and THIRD-PARTY.md -- design inspired by
  exo's dashboard (Apache-2.0).
* **Rewrite history once before making the repo public**, to drop the one
  remaining Co-Authored-By trailer (the first commit, `reserve the name`).
  File contents are unchanged; commit ids change, so any clone re-clones and
  ids quoted in old commit messages go stale.
* Then: HF card instructions point at knurlogic.

## BIG TODO: setting up several machines must be easy (Noah, 2026-09-24)

The scariest part for a new user, and the first thing to get right before
release. Found by doing it on the M4 tonight, every step a stumble:

* knurlogic was not on the PATH (it lived in a venv); `command not found`.
* The page listens on loopback by default, so a peer needs `--host` set on
  purpose -- and nothing told you which address.
* macOS put up "allow incoming connections?" on the OTHER machine, and
  until someone clicked it every request hung. Nothing said why.
* The Studio asked the peer on the wrong port, and gave up in 1 s on a
  busy peer that took 1.4. Both fixed (`1be7110`, `32610e4`).
* Machines were found through exo. Peers and Bonjour (cluster/) now find
  them without it; exo's /state is only one more witness.

What "easy" should mean: install on each Mac, run one command on each, and
the machines find each other; anything blocking (firewall, wrong address,
different versions) is NAMED on the page, on the machine that can fix it.

Open for Noah (user-facing details, 2026-09-25): **resending the firewall
prompt when it is missed on the satellite.** macOS asks once per app; a
dismissed or denied prompt is remembered and re-asking needs the firewall
entry removed (admin). What knurlogic can do on its own: detect "listening
but unreachable" and say exactly that, on both machines, with the fix.

## Next: replace exo (direction set 2026-09-22)

`pip install knurlogic` and nothing else, including clustering. Verified:
pip `mlx` ships the ring (TCP) and jaccl (RDMA) backends and a launcher
(`mlx._distributed_utils.launch`); pip `mlx-lm` ships `sharded_load`,
pipeline and tensor. So the engine half of a cluster is already
pip-installable; what exo adds is orchestration. In order:

1. One box from a clean venv with only `pip install knurlogic`. DONE on the
   M3, 2026-09-23: 14s install (knurlogic, mlx 0.32.2, mlx-lm 0.31.3,
   numpy; no exo, no mlx-vlm), every command ran from outside the repo, and
   Flash-Next VQ-2.1bpw loaded through the MCP, drafting on the batch path
   (acceptance 0.75), answering from the venv's own interpreter. Still to
   do: the same on the M4, which has no exo at all. GLM-5.3 needs nothing
   extra: served from an environment WITHOUT mlx-vlm on the M4
   (2026-09-25): conformance 36 passed / 2 skipped, vision on, drafting
   0.91 -- the `vlm` extra is gone.
2. `knurlogic node` -- a stdlib HTTP agent per Mac -- and a two-node
   pipeline over the ring backend, on a model already on both disks. exo
   stays installed until this is dependable.
3. Per-node downloads and knurlogic's own placement. exo becomes optional.
4. The page takes over the whole interface, closer to exo's.

The jaccl self-heal fork stays an optional mlx build `deps` detects; MTP
across a pipeline comes with the cluster executor (docs/SERVER.md,
"Cluster readiness"); the exo-era pipeline port was removed unused.

## Not done

Ordered by what would surprise somebody most.

1. **Batch drafting is not timed.** It runs on real weights (Flash-Next
   VQ-2.1bpw on the M3: the batch path, 8 requests, acceptance 0.88).
   Greedy output is not byte-identical to drafting off -- and stock mlx-lm
   is not identical to itself between sequential and concurrent either.
   Every divergence sat at the same few positions between the same two
   continuations; the one measured was a 0.25 logprob margin at bf16's
   0.125 resolution. Identity is proven on the float32 gate only. Speed
   needs a free box and the discipline above (reloads between arms). Costs
   inherited from the fork: a row is prefilled whole inside one `next()`,
   and there is no end-of-segment cache insert mid-prompt.
2. **knurlogic's own interpreter cannot serve GLM-5.3.** Its mlx-vlm 0.5.0
   lacks six modules the vendored glm5_next imports (`knurlogic deps` lists
   them). Upgrading mlx-vlm there is an environment change for a person to
   make.
3. **The page and the MCP are nearly at parity.** Load on the page goes
   through the MCP's own functions and shows its refusals. Still one-sided:
   the wired-limit control (page only), and live acceptance (neither shows
   it, though `/status.json`'s `drafting` block carries it).
4. **`state` sees this machine only.** On the page, another node's memory
   comes from its own knurlogic (a peer) or, failing that, exo's RAM figures;
   the per-runtime split needs a knurlogic on that node bound past loopback.
5. **`models` cannot tell a release from a lab intermediate.** Both are
   "servable" and "fits"; nothing on disk marks which is which.
6. **Knobs the exo fork had that knurlogic's own cluster will need**: a
   memory limit, a KV-pool token budget, the jaccl timeouts. Each needs a
   measurement before it gets a default.
7. **The MCP is not registered with a client.** It has been driven over
   stdio as a client would; adding it to a Claude Code or Codex config is
   the person's call.
8. **The soft gate** -- run records including OOMs, `resolve()` preferring a
   measurement, `doctor` labelling MEASURED vs PREDICTED. Designed, not
   built. `OK / UNPINNED / DRIFTED` is the vocabulary.
