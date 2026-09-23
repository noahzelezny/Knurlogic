# Where Knurlogic is, and where it goes next

*Rewritten 2026-09-22, end of the third session. A STATE document, not a
log: what is true now, what was measured so nobody re-derives it, and what
is next. The git history holds the narrative. `CONTEXT.md` is the map.*

## What it is

Local models on your own machines, managed equally well by a person (the
page, the CLI) and by an agent (the MCP). It resolves the settings that
decide whether a model runs, reports what is true about the machine, and
drafts with multi-token-prediction heads no stock runtime uses. It wraps
mlx-lm and exo rather than rebuilding them, and carries the work that was
trapped in forks of both.

## What is true now

    engine/       seam.py over mlx-lm's server; drafting on single requests
                  (stream_generate swap) AND batches (MTPBatchGenerator);
                  vendored architectures; overrides
    machine/      64 artifacts found across every store; residency in every
                  runtime; one load budget; which build of each dependency
                  every interpreter has
    tuning/       settings with their evidence; resolve() -> env + argv
    interfaces/   MCP (9 tools), page, CLI, serve, serve --cluster,
                  Anthropic Messages, connect, doctor

141 tests. The mlx tripwire is a folder rule: nothing outside `engine/`
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
* `knurlogic deps` distinguishes the jaccl fork from stock by the same check
  on two interpreters: YES in exo's, no in knurlogic's.

## Where the forks stand

The one home for what each fork carries and why it is or is not ported is
`PIECES` in `machine/deps.py`; `knurlogic deps` says which interpreter has
which. In short:

    exo fork (mtp-stage1, 86 commits)
      ported    MTP: sequential, batched, heads, registry (engine/mtp/);
                the per-family prefill table (tuning/settings.py);
                EXO_PREFILL_STEP_SIZE / EXO_MLX_CACHE_LIMIT_GB / EXO_MTP
                handed to exo under exo's names, ring-consistent
      stays     placement, sharding, networking, jaccl deadline arming,
                subnet pinning, KV-pool budget, VQ codebook sharding --
                knurlogic wraps exo rather than rebuilding it
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

* **exo's runner is spawned**: a fresh interpreter that inherits the
  environment and nothing of `sys.modules`. Overrides therefore install
  through a stdlib `sitecustomize.py` on PYTHONPATH. Verified on exo's own
  interpreter, control and override arms, four pids.
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
* **Version skew is the seam's job**: mlx-lm 0.31.3 runs `model_file`
  unconditionally, 0.32.0 raises without `trust_remote_code=`.
* **Tool calling**: all 54 templates swept, none silently unparsed (40
  qwen3_coder, 7 glm47, 3 gemma4, 1 json_tools, 3 none).

## Load-bearing design decisions

* **One load budget** (`machine/wired.load_budget()`): the smaller of the
  GPU working set and memory available now. `fit`, `models`, `settings`,
  `load` and `serve` all compute against it. When they used different
  numbers, a 47.5 GiB model had 6.8 GiB to spare in one answer and 36 GiB of
  roomy defaults in the next.
* **`resolve()` takes a byte count or nodes**; a cluster gets one Resolution
  per node plus ring-wide values enforced across them.
* **`ready` separates a local load from ring placement.** Runners moving
  memory on this box block `load`; downloads and unseen nodes block only exo
  placement. Only `DownloadOngoing` is in flight -- exo lists every model
  card as `DownloadPending` on every node.
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

## Not done

Ordered by what would surprise somebody most.

1. **Batch drafting has not run on real weights through knurlogic.** It is
   gated on token identity with a random head. Flash-Next VQ-2.1bpw is 47.5
   GiB and would not fit beside exo's instance when this was written. Run it
   on a free box: identity against the plain generator first, then speed by
   the discipline above. The costs it inherits from the fork: a row is
   prefilled whole inside one `next()` (other rows wait), and there is no
   end-of-segment cache insert mid-prompt.
2. **knurlogic's own interpreter cannot serve GLM-5.3.** Its mlx-vlm 0.5.0
   lacks six modules the vendored glm5_next imports (`knurlogic deps` lists
   them). Upgrading mlx-vlm there is an environment change for a person to
   make.
3. **The page and the MCP are not at parity.** The page has the wired-limit
   control; the MCP does not. The MCP has `ready`; the page never says
   memory is moving. Neither surfaces live acceptance, though
   `/status.json`'s `drafting` block carries it.
4. **`state` sees this machine only.** Another node's memory comes from exo's
   RAM figures; the per-runtime split needs a knurlogic on that node bound
   past loopback.
5. **`models` cannot tell a release from a lab intermediate.** Both are
   "servable" and "fits"; nothing on disk marks which is which.
6. **exo-fork knobs knurlogic does not set**: `EXO_MLX_MEM_LIMIT_GB`,
   `EXO_KV_POOL_MAX_TOKENS`, the jaccl timeouts. Named in `PIECES`; each
   needs a measurement before it gets a default.
7. **The MCP is not registered with a client.** It has been driven over
   stdio as a client would; adding it to a Claude Code or Codex config is
   the person's call.
8. **The soft gate** -- run records including OOMs, `resolve()` preferring a
   measurement, `doctor` labelling MEASURED vs PREDICTED. Designed, not
   built. `OK / UNPINNED / DRIFTED` is the vocabulary.
