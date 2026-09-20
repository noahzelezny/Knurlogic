# Where Knurlogic is, and where it goes next

*Rewritten 2026-09-20 at the end of the second session. This is a STATE
document, not a log: it says what is true now, what was measured so nobody
re-derives it, and what to do next. The git history holds the narrative.*

## What it is

An artifact you cannot load is worth nothing. Knurlogic resolves the settings
and verifies the environment between a downloaded model and a working one,
then hands off to an engine that already knows how to serve.

    pip install knurlogic && knurlogic serve <artifact>   ->  http://host:port/v1

Verified from a clean venv on stock PyPI mlx-lm 0.31.3. 81 tests.

It is a tool for taking control of your own machine: the settings that decide
whether a model runs, the ones nobody exposed, and a way to replace a module
in somebody else's package without forking it.

## What works

    serve      OpenAI endpoint (adapter over mlx-lm's server) + /v1/messages,
               settings resolved and env set BEFORE the model loads
    serve --cluster   wraps exo: resolves per node, proxies, aggregates status
    doctor     will it run, and which of the three failure modes is biting
    smoke      generate a token AND prove where the code came from
    vendor     take an architecture under version control with provenance
    override   replace a module in mlx-lm / mlx-vlm / exo, no fork
    models     find every model on this machine, in every tool's store
    connect    the lines that point a client at this server
    /          what loaded, the memory split, and the knobs, with the
               measurement behind each one

* **`engine.py` is the one module that knows what runs a model.** A test
  fails if any other module imports an engine. It has caught four leaks; that
  is the only reason the claim is still true.
* **Architectures**: qwen4_exp, qwen3_5, qwen3_5_moe, gemma4_text pinned by
  actual token generation. glm5_next vendored from stock mlx-vlm 0.7.1,
  UNPINNED -- no box here fits the smallest GLM rung (108 GiB vs 84 usable).
* **`serve --cluster` is verified against the real two-node exo** on this
  desk (Laptop B 128 GiB, Studio A 96 GiB): inventory read,
  `/v1/models` answered through knurlogic's port by exo with its 153 models,
  both nodes in `/status`.
* **`/status.json` is the contract**, and it carries the cluster shape even
  for one box: `{schema, cluster, nodes[]}` with the single-node keys still at
  the top level.

## NEXT: port MTP, and make it batch

This is the headline and everything else is secondary.

### What was measured 2026-09-20, which corrected the premise

All 54 artifacts, two independent channels -- config.json for the
declaration, the safetensors HEADERS for the tensors, neither reading the
other:

    declares mtp in config .................... 40
    ships upstream `mtp.*` graft weights ....... 1   (the 806 GB bf16 397B)
    ships a BUILT head beside the weights ..... 11   (mtp-head-q6.safetensors)
    declares one and has NOTHING ............... 31

40 = 31 + 8 declaring rungs that have a built head + the 1 graftable one. The
other 3 built heads sit beside GLM rungs whose config never declared one, so
the declaration does not even reliably signal the head's ABSENCE.

**The earlier reading was wrong.** "40 artifacts, 3925 GiB of downloaded MTP
weights that `sanitize()` throws away" does not survive contact with the
disk. The weights were never downloaded: MLX conversion drops `mtp.*` at
CONVERSION while keeping the upstream config that declares one. `sanitize()`
discarding those keys fires on exactly ONE artifact here. So the first step
recorded in the last version of this file -- make `sanitize()` keep the head
-- would have been real work for one rung out of fifty-four.

Two process notes worth keeping, because both nearly went the other way:

* The first pass read `model.safetensors.index.json` and found **zero** heads.
  The sidecar is named to stay OUTSIDE the `model*.safetensors` glob ON
  PURPOSE, so a loader never picks it up and the head costs nothing until
  asked for -- and an index-only scan is blind to all eleven. Read the files.
* Family is taken from the module tree, not the metadata label. The qwen4_exp
  packs predate the `family` field entirely. Three families, three distinct
  trees: `mixer` (qwen4_exp), `norm_out` (qwen3_5), `eh_proj` (glm5_next).

### What that changes

MTP is not "stop discarding what you already have". **The head is a separate
artifact**: grafted once from the one checkpoint that carries it, quantized,
and written beside each rung. Eleven rungs on this disk already have that
sidecar built and sitting unread -- 2.14 GiB (qwen4_exp), 5.41 (qwen3_5),
6.09 (glm5_next), all 6-bit.

`knurlogic mtp` reports the three states and `models` flags `[MTP]`. Nothing
loads a head yet.

### IT IS ALREADY WORKED OUT. Do not rebuild any of it.

Surveyed 2026-09-20 rather than assumed, because the previous version of this
section read as though MTP and batched MTP were open problems. They are not.
Both are solved, in the maintainer's own code, and measured.

**Where each piece lives.**

    vqlab/src/vqlab/mtp/          1295 lines  capture, sampling, caches,
                                              registry, loop, runtime, bench
    vqlab/src/vqlab/mtp_*.py      2015 lines  the HEAD BUILDERS -- graft,
                                              pack, extract -- plus accept,
                                              probe, run, smoke. exo has
                                              none of this half.
    exo/.../engines/mlx/mtp/      3091 lines  the same core, PLUS
                                              batch_loop.py (648) and
                                              heads/{qwen4_exp,qwen35,glm5}

**The two copies have drifted**, which is the argument for knurlogic holding
one. Measured by diff:

    capture.py    identical
    sampling.py   identical
    caches.py      30 lines differ    exo newer (09-17 vs 09-02)
    registry.py    63 lines differ    same day
    loop.py       296 lines differ    exo newer (09-17 vs 09-03)

**Measured, and not to be re-derived** (vqlab MORNING-REPORT 2026-09-04,
exo stage-1, both boxes, TB4 TCP):

    rung                    stock 300/2000   MTP 300/2000   acceptance
    GLM 2.7   (one box)      19.5 / 5.7      22.3 / 11.0    0.887/0.701
    GLM 3.1   (cluster)       6.9 / 6.2      18.5 / 17.4    0.727/0.642
    397B 2.6  (cluster)      23.7 / 23.5     23.7 / 23.4    0.853/0.851

GLM cluster rungs: up to 2.8x long-generation decode. 397B: parity, and its
stock decode does not degrade with context, so there is nothing to win.

**Batching with MTP is done too**, and the caveat that shipped with the
report above -- "drafting currently requires EXO_NO_BATCH=1" -- has been
retired. `mtp/batch_loop.py` drafts inside the batch engine, selected by
`~/.exo/engine-mode`. Measured 2026-09-17 on
Qwen3.8-Flash-Next-VQ-2.1bpw, M3: **23.1 vs 22 tok/s, identical tokens** --
a lone request on the drafting batch engine decodes as the sequential loop
would, and a second request simply joins the batch.

`vqlab mtp-accept` is the reliable instrument and its docstring says why:
greedy decoding is deterministic so repeats buy nothing, prompts are the
replicates, and the comparison is paired. Wall-clock is a property of the
machine -- the same configuration measured twice in one process gave 1.723x
and 1.135x on a thermally constrained M4 Max. Quote acceptance from there;
quote speed only from a thermally stable box.

### So what is actually left for knurlogic

Nothing in the algorithm. The work is packaging:

1. **One copy.** Take the newer of each file (exo for caches/loop/batch_loop,
   vqlab for the graft/pack builders exo does not have), into
   `knurlogic/mtp/`. capture.py and sampling.py are already identical in both
   and can be taken as-is.
2. **Defaults that need no folklore.** Today a user has to know about
   `EXO_NO_BATCH`, `~/.exo/engine-mode`, and that a sidecar must sit beside
   the weights. knurlogic already detects the sidecar (`knurlogic mtp`); it
   should pick the drafting batch engine when a head is present and say so.
**And a line that decides what knurlogic does NOT do: vqlab builds models,
knurlogic coalesces them.** The head builders (`mtp-graft`, `mtp-pack`) are a
build step and stay in vqlab. They are not ported here and knurlogic never
grows a button for them.

That line is visible in the artifacts themselves. Measured across every one
on this disk: **11 of 11 built heads sit beside a VQ artifact, and not one
community rung has one.** A sidecar is a thing vqlab made. So a missing head
means two different things, and knurlogic now says which:

* on a community rung -- nothing. The `mtp` key is inherited from the
  upstream config, no publisher ships those weights, and there is no defect
  and nothing anybody can do. 8 here.
* on a VQ artifact -- it was not packed, which is a vqlab question. 23 here.

Reporting both as "no head weights, absent from the download" sent someone
looking for a fix that does not exist for most of them.

The probe in `tools/mtp_probe.py` stays as the gate: it proved the sidecar
loads through exo's registry with nothing patched, which is what makes the
port a copy rather than a rewrite.

## Measured, so it is not re-litigated

### From the first session

* **Stock mlx-lm runs the VQ artifacts.** No fork required.
* **Upstream ships the architectures** and was AHEAD of the env one was
  vendored from. Vendoring pins a KNOWN version; it never holds a stale one.
* **Architecture drift was mostly version skew**, not files mutating (VQLab
  F127). The argument is narrower than first pitched and still holds: a
  grafted file inherits its install's version, so nothing answers "which
  arithmetic am I running".
* **`model_file` is a per-artifact runtime boundary.** Engine migration is
  per-rung, not global. This turned out to be the reusable idea of the
  project -- see name aliasing below.

### The memory knobs

* **The prefill spike is the dense-expert transient, not the KV cache**:
  3.35 MB/token where KV theory predicts 0.059, a 57x gap.
  `transient = chunk * out * in * 2`.
* **Smaller is also faster**: 128 -> 32 is 1.37x on every rung. There is no
  tradeoff on this knob, so nothing may raise it above 32.
* **RTILE=64 is 0.75-0.97x and never faster** (F25/F33). The one "win" was an
  env-ordering bug that benchmarked 32 twice.
* **The spike is a FAMILY effect, not a box effect** -- same spike on M4 and
  M3, DeepSeek V4 far more dramatic than Qwen3.5. `out`/`in` are the model's
  (gate_up is `[2 * moe_intermediate_size, hidden_size]`); the machine does
  not enter. `resolve()` reads the shape off the artifact:

        VQ_DECODE_CHUNK, artifact 60 GiB, by headroom left on the box
        family                      1GiB      2GiB      4GiB      8GiB     16GiB
        deepseek-v4-ish                4         4         9        18        32
        the frozen constant            8        16        32        32        32
        qwen3.5-ish                    8        16        32        32        32

  TIGHTEN ONLY for now: the same formula loosens the knob for small-expert
  families, that direction is unmeasured, and being wrong there is an OOM.
  `DECODE_CHUNK_SHAPE_MAY_LOOSEN` flips it in one line, by a run.

### Where the rest of the provenance lives

The per-flag findings -- F54 (+5.1-6.6% prefill, bit-exact), F56 (the +11.9%
stack), F103/F105 (numerics-active, up to +0.97% ppl), F124 (+20.9% prefill
at d4-K2048) -- live in `settings.py` NEXT TO the defaults they justify, and
in `KNOB_DOC` next to what the UI shows. That is the right home: a constant
whose justification is in another file is a constant nobody can argue with.

Two more that live in code and are easy to lose:

* **Version skew is `engine.py`'s job.** mlx-lm 0.31.3 executes `model_file`
  unconditionally; 0.32.0 put it behind `trust_remote_code=` and raises
  without it. Passing the kwarg blindly is a TypeError on one and omitting it
  a ValueError on the other, so a VQ artifact cannot load on both unless
  something inspects the signature. Nobody downstream should learn that.
* **A pin is written only after a model generated a token** with clean
  provenance (`smoke --pin` -> `architectures/PINS.json`), which is why
  `OK / UNPINNED / DRIFTED / MISSING` means "it ran and came from here"
  rather than "it imports". That vocabulary is the model for the soft gate.

### The wired limit

`iogpu.wired_limit_mb: 86016` -> 84.0 GiB, and the framework reports a
working set of exactly 84.0 GiB of 96 installed. **The sysctl is what the
working set follows.** A rung that "does not fit" often fits fine on a machine
that was never told it may use its own memory. Knurlogic computes the number
and prints the command; it never runs it. The reserve left for macOS is a
JUDGEMENT and is labelled as one.

### What crosses a process boundary

exo's runner is `mp.Process` under start method "spawn". A spawned child is a
fresh interpreter: it inherits the ENVIRONMENT and nothing of `sys.modules`.

    without overlay:  child saw parent's sys.modules edit: False | overlay: False
    with overlay:     child saw parent's sys.modules edit: False | overlay: True

So overrides install through a `sitecustomize.py` on PYTHONPATH, which python
imports at interpreter startup in the master, the API and every runner. It is
a meta-path FINDER, not a preload; preloading would drag mlx into every
python process on the box. Stdlib only, because exo runs in its own env
(`/opt/anaconda3/envs/exo`) where knurlogic is not installed.

Verified on the real target: a real exo module, exo's own interpreter,
control arm and override arm, parent and spawned child, four distinct pids.

### What is live and what needs a restart

Read out of a real bundled runtime, 4523 lines:

* **`VQ_DECODE_CHUNK` is LIVE.** `_DECODE_CHUNK` resolves lazily on first
  prefill and is then read as a module global inside the expert loop, so
  rebinding it lands on the next prefill.
* **`VQLAB_CACHE_LIMIT_GB` is LIVE**, through the framework's own setter.
* The eight GEMM/numerics flags need a restart: read at import and compiled
  into Metal source.
* **`VQLAB_PREFILL_CHUNK` is read by NONE of the 37 bundled runtimes here.**
  Knurlogic emitted it for every artifact. A resolved setting that does
  nothing is the exact failure this package exists to prevent, committed by
  the package.

### The environment is the artifact's, not ours

`VQLAB_CACHE_LIMIT_GB` is read by 24 of 37 bundled runtimes, and those files
are published. Renaming it in the resolver would emit a name nobody reads and
silently stop bounding the cache on every artifact already shipped. So a knob
has a LOGICAL name and a list of env names, preferred first, and `resolve()`
emits whichever one the target actually reads. A new rung can bundle a
runtime reading `KNURLOGIC_CACHE_LIMIT_GB`; every published rung keeps its
own. With no runtime to ask it falls back to the LEGACY name -- a guess
should fail towards the 24 artifacts that exist.

Those runtimes read ~38 env vars. Knurlogic resolves 11. `VQ_FUSED_MAX_N` is
read by 37/37 and knurlogic has never heard of it. That is not a to-do list;
each one needs a measurement before it gets a default.

### Tool calling is not one format

mlx-lm picks a parser by INFERRING it from the chat template and returns None
when the inference misses -- at which point tool calls arrive as prose and a
harness looks like a model that keeps describing the function it would call.

Flash-Next asks for `<tool_call><function=NAME><parameter=P>value</parameter>`,
the Qwen3-Coder / agentic-harness dialect, not the JSON plain Qwen3 emits. It
keeps its template in a SEPARATE `chat_template.jinja` with the tokenizer
config's field empty; mlx-lm does load it (8952 chars) and infers
`qwen3_coder`, which is correct.

Swept all 54: 40 qwen3_coder, 7 glm47, 3 gemma4, 1 json_tools, 3 with no
tools in the template. **None silently unparsed.**

### What is on this machine

    54 found, 5112 GiB on disk. 54 in a format this engine loads,
    23 that fit one box.

Four tools keep four stores and none looks at the others. The store location
is per-tool config, so `knurlogic models` asks a RUNNING exo/ollama where its
models are -- 37 artifacts here live on an external volume named only in the
environment of a process started hours earlier.

**Finding a model is not being able to run it.** `mlx_lm.gguf` exposes
`convert_to_gguf` and no loader, and the server never mentions the format, so
GGUF is reported FOUND and NOT SERVABLE with the reason.

## Design decisions that are load-bearing

* **`resolve()` takes a byte count or NODES**, and returns a `Resolution` or
  a `ClusterResolution` with one per node. Not itself a Resolution: a cluster
  has no single env dict, and inventing one puts the wrong knobs on the wrong
  box. Undeclared shards split proportionally to working set and are LABELLED
  as an assumption.
* **`status.aggregate()` sums and reports `nodes_reachable`**, because a sum
  over nodes that did not answer is a smaller number that reads as good news.
  Memory carries `scope`: a number covering the whole machine is never
  printed under a label that says "weights".
* **Settings apply per knob**, not per page. Some now, some at restart, some
  never on this artifact.
* **The tuning axis is capped by measurements**: `fast` may not raise the
  decode chunk, nothing reaches RTILE=64, and a profile that cannot have what
  it asked for SAYS SO. The difference between a knob and a wish is whether
  it tells you it did not happen.
* **Knobs are tiered by who would reach for one**: 2 reach, 8 measured flags,
  23 kernel internals named and never defaulted. Inventing defaults for
  unmeasured knobs is how the frozen `2048*4096*2` got into the resolver.
* **Kernel knobs belong to whoever packs the kernels.** `Artifact.
  declared_knobs()` reads a `knobs` block from config.json and it outranks
  anything scanned or hard coded. Empty today: a hand-off point for vqLab.
* **`connect` offers a project `.claude/settings.json`, never the global
  one.** Routing every session on the machine at a local model is a config
  change nobody can see.

## The page

One column by default: what loaded, how full the box is, a place to type.
Settings and Connect are a MENU -- a drawer opened from the masthead, one
panel at a time, Escape closes, choice remembered per browser. (If that is
wrong: drop the localStorage read and it always starts closed.)

exo's visual language in teal: mono throughout, panels as bordered boxes with
the label in the border, and a topology strip where **the device IS the
gauge** -- each unit fills from the bottom with what that box is holding. A
picture of a machine next to a separate number is decoration; a picture that
IS the number is not.

The knobs are rotary dials because a knurl is the crosshatch cut into metal
so a hand can grip a machined part. Positions are discrete because the
measurements are, and the rim stops where the evidence stops: 32 on the
decode chunk, and at whatever this box's headroom allows for the cache, with
ticks past the cap drawn dead.

Every knob carries its sentence -- what it does and the run that established
it. A settings UI that lists names and values is a config file with a
stylesheet; the provenance is the feature.

## Discipline that cost this project real time

* **Every probe needs a channel PROVING the arms differ**, independent of the
  quantity measured. An A/B of a `vq_switch.py` edit ran identical code in
  both arms because the artifact executes its OWN bundled model.py; it was
  caught only because the probe counted syncs as well as time.
* **A version check run with bare `python3` inside a loop over env paths**
  answered for the SYSTEM interpreter every iteration. The same shape bit
  again this session: `EXO_MODELS_DIR=/Volumes/Models/Models`
  parsed with `split()` becomes two paths that do not exist.
* **Speed claims**: n>=3 per arm, alternating, one process per arm, quote the
  RATIO never an absolute, and refuse to quote when within-arm spread exceeds
  the between-arm difference.
* **An index is a summary, not the disk.** The MTP survey read
  `model.safetensors.index.json` and reported zero heads; eleven were sitting
  right there, deliberately kept out of the index so no loader picks them up.
  When a thing is designed to be invisible to the normal path, the normal
  path is the wrong instrument.
* **Tests must not depend on the developer's disk.** The discovery tests
  scanned the real machine and one asserted against 54 actual models until
  `include_defaults=False` existed.
* **Four bugs this session were found only by DRIVING the thing**, not
  reading it: a drag handler that re-rendered the element holding its own
  pointer capture (the knob moved and nothing happened underneath); an empty
  `<details>` rendering a row that says nothing; the cluster page never
  showing the artifact; a rollup dropping `scope` after the text renderer had
  been fixed for exactly that. No test would have caught any of them.

## Not done

* **No model picker, no load/unload.** The server holds one artifact chosen at
  startup. This is the piece that would actually replace what exo is used for,
  and it needs real lifecycle machinery.
* **The soft gate is not built**: a run record (artifact + settings + working
  set + outcome, including OOMs), `resolve()` preferring a measurement over
  its own arithmetic, and `doctor` labelling MEASURED vs PREDICTED. A run
  record can hold a NEGATIVE result, which a model card structurally cannot.
  It is also what would turn `DECODE_CHUNK_SHAPE_MAY_LOOSEN` from a global
  flag into per-family evidence.
* **Not on PyPI.** Name reserved; the package works.
* **glm5_next unpinned.** Do not claim GLM support until it has generated a
  token somewhere.
* **The ollama reader is unverified** -- written from the on-disk layout, and
  this machine's ollama store is empty.
* **Whether a VQ artifact satisfies mlx-lm's `is_batchable`** is unchecked; it
  needs a loaded model. It decides whether parallel agents batch or queue.
