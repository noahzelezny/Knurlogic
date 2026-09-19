# Where Knurlogic is, and where it goes next

*2026-09-18. Written at the end of the session that built it, so the next one
does not start by re-deriving what was already measured.*

## What it is

An artifact you cannot load is worth nothing. Knurlogic resolves the settings
and verifies the environment between a downloaded model and a working one,
then hands off to an engine that already knows how to serve.

    pip install knurlogic && knurlogic serve <artifact>   ->  http://host:port/v1

Verified from a clean venv on stock PyPI mlx-lm 0.31.3.

## What works

* `serve` -- OpenAI endpoint (adapter over mlx-lm's server), settings resolved
  and env set BEFORE the model loads, which is load-bearing: a VQ artifact's
  bundled runtime reads its knobs at import.
* `doctor` -- separates the three failure modes that look identical from
  inside a stack trace: missing architecture, wrong settings, does not fit.
* `smoke` -- generates a token AND proves where the code came from; `--pin`
  records a digest only on a clean pass.
* `vendor` -- takes an architecture under version control with provenance.
* `/` and `/status[.json]` -- what loaded, and the memory split nothing else
  shows: weights vs RECLAIMABLE cache vs transient peak vs headroom.
* `engine.py` -- the one module that knows what runs a model. A test fails if
  any other module imports an engine; it caught three leaks while being
  written, which is the only reason the claim is still true.

Architectures: qwen4_exp, qwen3_5, qwen3_5_moe, gemma4_text pinned by actual
token generation. glm5_next vendored from stock mlx-vlm 0.7.1, UNPINNED
because no box here fits the smallest GLM rung (108 GiB vs 84 usable).

## What this session established, so it is not re-litigated

* **Stock mlx-lm runs the VQ artifacts.** No fork required. The eauchs/mlx-lm
  0.32.0 these were taken from is not a dependency.
* **Upstream ships the architectures**, including deepseek_v4, deepseek_v32
  and qwen4_exp in mlx-vlm -- and was AHEAD of the env one was vendored from.
  Vendoring is for pinning a KNOWN version, never for holding a stale one.
* **Architecture drift was mostly version skew**, not files mutating on their
  own (VQLab F127). The argument for vendoring is narrower than first pitched
  and still holds: a grafted file inherits its install's version, so nothing
  answers "which arithmetic am I running".
* **`model_file` is a per-artifact runtime boundary.** A new artifact can
  bundle a runtime for a new engine while every published rung keeps the one
  it shipped with. Engine migration is per-rung, not global.

## Next: clustering, and it is the point

Knurlogic has to cluster for it to mean anything. It does not have to start
there. exo already does it, already speaks OpenAI (`api/`, 7142 lines), and is
what actually gets used day to day -- so WRAP, do not rebuild.

    exo, ~57k lines
      worker/engines/   18,478   MLX inference; where the VQ kernels integrate
      api/               7,142   already OpenAI-compatible
      master/            3,837   placement
      routing/             654

**Wrap first.** `knurlogic serve --cluster` resolves settings per node,
launches or attaches to exo, aggregates `/status` across nodes. Days.

**Then replace pieces, the same way as anywhere else.** The `sys.modules`
trick that puts a vendored architecture in front of mlx-lm's works on any
Python package -- registering a replacement `exo.master.placement_utils` is
the identical move. No fork, no permanent diff, one module at a time, each
A/B'd. It requires Knurlogic to LAUNCH the process, since registration must
precede import; launching is what the cluster command does anyway.

Smallest honest first targets: `routing/` (654 lines) or `master/` placement.

## The two shape changes -- done

*2026-09-18, the session after.*

1. `resolve(artifact, budget)` takes a byte count (one box, one `Resolution`,
   the call every existing caller makes) or nodes -- a `Node`, a sequence of
   them, or `{name: working_set_bytes}` -- and returns a `ClusterResolution`
   with one `Resolution` per node. It is deliberately not itself a
   `Resolution`: there is no single env dict for a cluster, and inventing one
   puts the wrong knobs on the wrong box. What a node HOLDS is placement, so
   `Node.holds_bytes` declares it and an undeclared shard is split
   proportionally to working set and LABELLED as an assumption on every
   resolution that rides on it.
2. `status.snapshot()` still answers for its own process -- that is all it
   can honestly do -- and gained `node`/`role`/`reachable` plus a `memory_fn`
   seam so a snapshot can be built from numbers that came off another node.
   `status.aggregate()` is the shape `/status.json` now serves even for one
   box: `{schema, cluster, nodes[]}` with the old single-node keys still at
   the top level, so a client written against one box is not rewritten when
   a second appears. The rollup SUMS and reports `nodes_reachable`, because
   a sum over nodes that did not answer is a smaller number that reads as
   good news. Memory carries `scope` (`process` or `box`): a number covering
   the whole machine is never printed under a label that says "weights".

## `serve --cluster` -- wrapping exo

`src/knurlogic/cluster.py`. Attaches to a running exo by default (that is
what is actually running day to day); `--launch` starts one with this node's
resolved settings already in its environment.

* The OpenAI surface is exo's, proxied untouched. There is no second
  implementation of chat completions in this package and there should never
  be one.
* `/status`, `/status.json` and `/` are Knurlogic's, aggregated over every
  node exo reports.
* Node inventory comes from exo's `/state` (`nodeMemory`, `nodeIdentities`).
  Those are psutil SYSTEM RAM numbers, not the Metal recommended working
  set; `--node NAME:GIB` overrides them, and when it does, status shows the
  declared number, since that is what the settings were resolved against.
* The settings of a node we LAUNCH are applied. The settings of a node we
  attached to are reported and said to be reported. Nothing here can reach
  into another machine's process.

Verified against the live two-node exo on this desk (NozzleBook Pro,
Noah's Mac Studio): inventory read correctly, `/v1/models` through
Knurlogic's port answered by exo with its 153 models, `/status` aggregated
both nodes. `tests/test_cluster.py` pins the same claims against a stub exo
whose answers carry a marker string, so a proxy that silently answered by
itself would fail the test rather than pass it quietly.

## Overrides: one place, no forks

*2026-09-18, same session.* The goal moved: VQ adoption should not require a
fork of exo plus forks of mlx-lm and mlx-vlm. One place, and the improvements
upstream is too busy to merge carried without a permanent diff.

**The finding that decides the mechanism.** exo's runner -- where MLX
inference and the VQ kernels actually run -- is an `mp.Process` under start
method "spawn" (`exo/utils/async_process.py:91`, `exo/main.py:292`). A
spawned child is a fresh interpreter and inherits NOTHING from the parent's
`sys.modules`. So `register.py`'s move, done in the process that launches
exo, does not reach the process that runs the model. Measured, with a second
channel so a child that never ran could not read as a pass:

    without override:  child saw parent's sys.modules edit: False | override: False
    with override:     child saw parent's sys.modules edit: False | override: True

The environment is what crosses a spawn boundary. So the installer is a
`sitecustomize.py` on PYTHONPATH, which python imports at interpreter startup
in every process -- master, API, each spawned runner -- before exo or mlx
import anything. It installs a meta-path FINDER, not a preload: importing the
targets at startup would drag mlx into every python process on the box.

**It is standalone by requirement, not by taste.** exo runs in its own env
(`/opt/anaconda3/envs/exo`, python 3.13) where knurlogic is not installed. An
installer that imported knurlogic would pass on the box it was written on and
fail on the one that matters -- the same shape as the bare-`python3`-in-a-loop
version check. `tests/test_override.py` shadows knurlogic with a module that
raises ImportError, so that arm is tested rather than asserted.

**Verified on the real target**, not only a fixture: a real exo module,
exo's own interpreter, control arm and override arm, parent and spawn()ed
child, four distinct pids.

    control:  exo/src/exo/routing/topics.py        proof: null
    override:  <knurlogic>/.../routing/topics.py    proof: served-by-knurlogic

**Rules, which are what keep an override from becoming a fork.** Every override
records what upstream version it is against, why, and the measurement that
justifies it; it is digest-pinned and REFUSED at import if the file changed,
because serving a file that is not the pinned one makes every measurement
taken afterwards unciteable. The installer logs each activation with its pid:
a spawned runner cannot be asked what it imported, and an override that never
fired looks exactly like one that did.

    knurlogic override add <module> <file> --against <ver> --why <measurement>
    knurlogic override run -- exo          # any command, overrides applied
    knurlogic override list                # declared, and what actually fired

`serve --cluster --launch` installs them for exo and every runner it spawns.
Attach mode says it did NOT apply them, because it cannot.

**Limits, stated once.** Only processes Knurlogic launches -- the mechanism
rides on the environment. `mlx.core` is a compiled extension and is not
overrideable; kernel changes still belong in the artifact's `model_file`.
`overrides/` ships EMPTY: the first override should be one with a measurement
behind it.

## The spike is a family effect, not a box effect

*2026-09-18, from Noah's experimental data.* The same prefill spike appeared
on M4 and on M3; DeepSeek V4 was far more dramatic than Qwen3.5. So the model
FAMILY is the bigger factor and the box is close to irrelevant once headroom
is equal.

That contradicted what the resolver actually did. `Artifact` has read
`hidden_size` and `moe_intermediate_size` off config.json since it was
written, and `resolve()` **never used either one** -- it sized the
dense-expert transient from `DECODE_CHUNK_BYTES_PER_UNIT = 2048 * 4096 * 2`,
which is the real formula frozen for ONE rung (H=4096, M=1024), because the
auto-sizer it came from only ever ran there.

The formula is `chunk * out * in * 2` and out/in are the MODEL'S: gate_up is
`[2 * moe_intermediate_size, hidden_size]`. Nothing in it refers to the
machine -- which is precisely the observation. `resolve()` now reads the
shape off the artifact:

    VQ_DECODE_CHUNK, artifact 60 GiB, by headroom left on the box
    family                      1GiB      2GiB      4GiB      8GiB     16GiB
    deepseek-v4-ish                4         4         9        18        32
    the frozen constant            8        16        32        32        32
    qwen3.5-ish                    8        16        32        32        32

DeepSeek-shaped experts hold 58.7 MB per unit of chunk against the constant's
16.8, so at 2 GiB of headroom it resolves 4 where the constant said 16 -- a
4x tighter knob for the family that was measured as "more dramatic". Above
~16 GiB of headroom every family caps at the default and the distinction
stops mattering, which is why it went unnoticed.

**Tighten only, for now.** Sizing from the model also LOOSENS the knob for
small-expert families (qwen3.5-ish would take 32 at 2 GiB). That direction is
not measured, and being wrong there is an OOM -- the failure this package
exists to prevent -- so it is refused and the refusal is printed rather than
hidden. `DECODE_CHUNK_SHAPE_MAY_LOOSEN` flips it in one line, and should be
flipped by a run, not by a preference. The row above is the experiment: take
a qwen-shaped rung to ~2 GiB of headroom and see whether 32 survives a long
prompt where 16 does.

**What this says about "presets".** If the spike keys on the family and the
family is declared in the artifact's own config, then a preset per BOX is the
wrong shape -- and a preset object may not be needed at all. The artifact
already carries what decides the knob; the resolver just has to read it.
Keep that direction: prefer computing from the artifact over enumerating
boxes, and add a preset only for something the config genuinely cannot say.

## A soft gate, a tuning axis, and the wired limit

*2026-09-18.* Noah's framing: exo only shows models it has built cards for,
and measured releases are what stop people OOMing themselves. Agreed, with
one difference -- exo's card says what a model IS (declared metadata,
hand-curated); the record that prevents an OOM has to say what a model DID.
`smoke --pin` is already that primitive: it writes a digest only after a
model generated a token. Extending it from architecture files to RUNS is the
soft gate: curate settings for popular rungs, never block an unknown one,
label which answer is measured and which is computed. The vocabulary already
exists -- `OK / UNPINNED / DRIFTED / MISSING` becomes `MEASURED / PREDICTED`.

A run record can also hold a NEGATIVE result ("this rung OOMed at this
working set with these settings"), which a card describing what a model is
structurally cannot. The failure envelope is the asset, not the success list.
*Not built yet: the record format and `resolve()` preferring it.*

**The tuning axis is built.** `--tune safe|balanced|fast` on `serve` and
`doctor`. What makes it honest is the caps:

* `fast` may NOT raise VQ_DECODE_CHUNK. Smaller is faster AND smaller in
  memory (128 -> 32 is 1.37x), so there is no tradeoff to sell there.
* Nothing at any setting may reach RTILE=64 (0.75-0.97x, never faster).
* The reclaimable cache is capped, and capped again by actual headroom --
  reclaimable is not free, it is still resident.
* A profile that cannot have what it asked for SAYS SO. On a 4 GiB-headroom
  box `fast` degrades to the tight prompt chunk and prints that headroom, not
  the profile, is what capped it. The difference between a knob and a wish is
  whether it tells you it did not happen.

**The wired limit is the biggest single "it does not fit" that is not true.**
Measured here: `iogpu.wired_limit_mb: 86016` -> 84.0 GiB, and the framework
reports a working set of exactly 84.0 GiB of 96 installed. So the sysctl IS
what the working set follows, and a rung that "does not fit" often fits fine
on a machine that was never told it may use its own memory.

`doctor`, `serve` and `/status[.json]` now read it, work out the number, and
print the command. **Knurlogic does not set it** -- it needs root, it is
system-wide, and a package that quietly raises how much memory the GPU may
wire is not one anybody should install. The reserve left for macOS is a
JUDGEMENT and is labelled as one everywhere: too little does not OOM the
model, it wedges the machine.

Also: `--working-set-gib` now defaults to asking the framework rather than to
0. `resolve()` still takes headroom as an INPUT -- that stance is what keeps
it testable -- but the COMMANDS fill it in, because forgetting the flag
silently produced the roomy defaults, which is the exact footgun this package
exists to remove.

## The knobs are in the GUI, because that is the point of a GUI

*2026-09-18. Noah on exo: "there are no actual settings. What's the point of
the GUI if you don't expose the knobs available?"* Right, and a page that
shows a green light and no knobs is a status light wearing a costume.

`/settings.json` and the Settings panel on `/`. What makes it honest is that
it keeps three questions apart and never blurs them:

    what is RUNNING     what this tune WOULD give     how to get it

A slider that appeared to retune a loaded model would be a lie: the runtime
reads its environment AT IMPORT and the import already happened. So changing
the tune shows a diff -- the running value struck through, the new one beside
it -- and the banner names the flag (`--tune fast`) and says it takes a
restart. That is more useful than no control and more honest than a control
that does nothing.

Every knob carries its sentence (`settings.KNOB_DOC`): what it does and the
run that established it. A settings UI that lists names and values is a
config file with a stylesheet -- the provenance IS the feature, and it is the
thing this project has that nothing else does.

`web.py` holds the routes so `serve` and `serve --cluster` cannot drift; the
engine seam now takes a `path -> handler` mapping instead of knowing what a
status page is.

Fixed while looking at it, both caught only by opening the page:
* the cluster page never showed the artifact -- the snapshot attached it to
  the local node, and there is no local node when attaching.
* the rollup dropped `scope`, so a box-wide total rendered under "weights +
  live" on the page after the text renderer had already been fixed for
  exactly that. A rollup is only as precise as its least precise node.

## Why a reload was never actually required

*2026-09-18. Noah: "why can't we adjust runtimes live? nobody wants to
reload a model."* Correct instinct. The answer came out of READING a real
bundled runtime (4523 lines) instead of assuming, and it is per knob:

* **VQ_DECODE_CHUNK is live.** `_DECODE_CHUNK` is resolved lazily on first
  prefill and then read as a module global inside the expert loop
  (`for c0 in range(0, len(touched), _DECODE_CHUNK)`). Rebinding that global
  lands on the next prefill. No reload. This is the knob that decides whether
  a long prompt survives.
* **VQLAB_CACHE_LIMIT_GB is live**, through the framework's own setter.
* **The eight GEMM/numerics flags genuinely need a restart.** They are read
  into module globals AT IMPORT and baked into Metal kernel source that is
  compiled once. Making those live is an override's job, not a setting's.
* **VQLAB_PREFILL_CHUNK is read by NONE of the 37 bundled runtimes on this
  machine.** Knurlogic emitted it for every artifact. A resolved setting that
  does nothing is the exact failure this package exists to prevent, committed
  by the package. `Artifact.reads_knob()` now asks the artifact's own runtime
  and the panel marks it NO EFFECT.

So "restart required" was true of most knobs, wrong about the two that matter
most for not running out of memory, and irrelevant for one that never did
anything. `POST /settings.json` applies the live ones; everything else says
what it needs and why.

A second find, from the same file: the bundled runtime's own auto-sizer uses
`per_expert = 2048 * 4096 * 2` -- the identical frozen constant knurlogic
carried until this session. The artifact's own fallback has the same blind
spot for a family with bigger experts. That is the first override with a
measurement behind it, whenever it is wanted.

Also fixed: `_artifact_runtime_modules` used `hasattr` over sys.modules,
which invokes every lazy module's `__getattr__` -- it reached into
transformers' lazy-import machinery and raised from inside a package that has
nothing to do with any of this. It reads `vars(mod)` now, which asks the
question without running anybody else's code.

## The page, simplified

Noah on exo's GUI: simple is a choice about where the eye goes, and a busy
layout gives up that control. So: one column, one number that matters
(headroom), three buttons, and the knob list showing only what CHANGED or
what can be applied live -- everything else behind "all knobs". Provenance
moved from printed-under-every-row to a click on the row. Same information,
one decision at a time.

## The VQLAB_ names: a migration, not a rename

*2026-09-18. Noah: "should probably use knurlogic instead of vqlab though."*
Right instinct, wrong operation -- and the difference is the whole lesson of
the session.

Scanned all 37 bundled runtimes on this box for the environment they actually
read. `VQLAB_CACHE_LIMIT_GB` is read by **24 of 37**, and those files are
published. Renaming it in the resolver would emit a name nobody reads and
silently stop bounding the cache on every artifact already shipped -- the
exact failure this package exists to end, dressed as housekeeping.

So a knob now has a LOGICAL name and a list of env names, preferred first
(`settings.KNOB_ALIASES`), and `resolve()` emits whichever one the target
artifact actually reads. A new rung can bundle a runtime reading
`KNURLOGIC_CACHE_LIMIT_GB`; every published rung keeps the name it shipped
with. That is the `model_file` boundary again -- per artifact, never global --
applied to the interface instead of the engine. With no runtime to ask, it
falls back to the LEGACY name: a guess should fail towards the 24 artifacts
that exist, not towards the one that is planned.

`prefill_chunk` is now emitted for nobody, with a note saying so, because no
runtime reads either alias.

**What the scan also said, and it is worth sitting with.** Those runtimes
read ~38 environment variables. Knurlogic resolves 11. `VQ_FUSED_MAX_N` is
read by 37/37 and knurlogic has never heard of it; `VQ_EXPERT_SIMD`,
`VQ_DENSE_TILED`, `VQ_D8_REGBUF` and two dozen others are read by 24-25 each.
That is not a to-do list -- every one of those needs a measurement before it
gets a default, and inventing defaults for knobs nobody has measured is how
the frozen 2048*4096*2 constant happened in the first place. But it is the
honest size of the surface, and `doctor` should probably say "this artifact
reads 38 knobs; knurlogic has a measured answer for 11" rather than implying
the list it prints is the whole environment.

## Three tiers, by who would reach for a knob

*Noah: "we don't need to expose everything, just the things a reasonable
person would reach for. Maybe fair enough to have more if people look."*
That is the right cut, and it makes the tiering principled instead of
incidental. On the real 3.2bpw artifact, 33 knobs:

    reach     2    VQ_DECODE_CHUNK, VQLAB_CACHE_LIMIT_GB -- will it run,
                   will it OOM. These are the page.
    deeper    8    the measured performance and numerics flags. Real
                   findings behind them; you go looking on purpose.
    kernel   23    tile widths, register buffers, SIMD group sizes. NAMED
                   and never defaulted -- the runtime's own defaults apply.

The third tier is the honest one. Knurlogic sets none of them and says so:
"23 more this runtime reads -- knurlogic has no measured answer for these."
Inventing defaults for unmeasured knobs is exactly how the frozen
2048*4096*2 constant got into the resolver in the first place.

`doctor` now says the same thing in a line: this artifact reads 33 settings,
knurlogic has a measured answer for 10.

## Knobs, and where kernel work lives

*2026-09-18. Noah: kernel work stays with vqLab, which packs the kernels with
the artifact -- then maybe those knobs become available.*

That draws the line cleanly. vqLab authors kernels and the knobs that go with
them; knurlogic does not own them and must not invent defaults for them.
`Artifact.declared_knobs()` reads a `knobs` block from config.json --

    "knobs": {"VQ_D8_ROWS_TG": {"default": "8", "values": [4, 8, 16],
                                "doc": "rows per threadgroup"}}

-- and it OUTRANKS anything knurlogic scans or hard codes, because the
artifact's config is the record of what shipped. The block is empty in every
artifact today; it is a hand-off point waiting for a packer to write to it.
When one does, those 23 kernel knobs stop being a list of names and become
controls, without knurlogic having measured a single one of them itself.

## The control is a knob, because that is what a knurl is

A knurl is the crosshatch cut into metal so a hand can grip a machined part.
So the two settings a person reaches for are rotary knobs with a knurled rim,
not sliders -- and it is not decoration:

* **The positions are discrete** (4/8/16/32; 0.5-16 GiB), because the
  measurements are. A continuous slider would invent positions no run ever
  measured.
* **The rim stops where the evidence stops.** VQ_DECODE_CHUNK ends at 32
  because 128 -> 32 is 1.37x and nothing above was ever better. The cache
  dial stops at whatever this box's headroom can hold and prints why
  ("stops at 6 -- 12.3 GiB of headroom is all there is to hold it in").
  Ticks past the cap are drawn dead. A control that lets you choose a setting
  the resolver would refuse is a control that lies.
* Turning one applies it LIVE, through the same path as the tune buttons.

Bug found by driving it rather than reading it: the drag handler re-rendered
the dial with `outerHTML` on every step, which destroyed the element holding
the pointer capture -- so `pointerup` never fired, the knob moved on screen
and nothing happened underneath. It mutates the live node now. That is the
one behaviour a control must never have, and no test would have caught it.

## `knurlogic models`: four stores, none of which look at each other

*2026-09-18. Noah: people will use models other than the VQ ones, and it is
annoying that ollama, mlx and exo do not look at each other's folders.*

`src/knurlogic/discover.py`, `knurlogic models`. On this machine:

    54 found, 5112 GiB on disk. 54 in a format this engine loads,
    22 that fit one box.

**Finding a model is not being able to run it**, and the counts are kept
apart for that reason: is it here, can this engine read it, will it fit.
Ollama and most of LM Studio hold GGUF, and this engine does not load GGUF --
checked, not assumed: `mlx_lm.gguf` exposes `convert_to_gguf` and no loader,
and the server module never mentions the format. A GGUF model is reported
FOUND and NOT SERVABLE with the reason. A menu of entries that 500 on click
is the same "why did it fail" that `doctor` exists to end.

**It asks the running tool where its models are.** The store location is
per-tool configuration, and this box is the case in point: 37 artifacts on an
external volume, named only in the environment of an exo process started
hours earlier. Guessing at external volumes would be wrong on every other
machine; reading `EXO_MODELS_DIR` off the running process is right on all of
them. Parsed with a boundary regex rather than `split()`, because the value
is "/Volumes/Thunderbay SSD/Exo Models" and splitting on spaces turns one
real path into two that do not exist.

A model too big for one box says "needs more than this box", not "does not
fit": knurlogic serves across nodes, so that is a clustering question rather
than a wall.

Unverified: the ollama reader is written from the on-disk layout and this
machine's ollama store is empty, so it has never run against a real pull. It
says so in the source rather than being presented as tested.

Tests pass `include_defaults=False`. Without it they scanned the real machine
and one asserted against 54 actual models -- a test that depends on what is
on the developer's disk is not a test.

## `/v1/messages`: backing a coding harness locally

*2026-09-18. Noah, looking at exo's INTEGRATIONS panel: can I use the Claude
Code harness without the sub?*

Yes, and the mechanism is plain: the harness is pointed at a server with
`ANTHROPIC_BASE_URL`, so whatever answers the Anthropic Messages shape at
that address is the model. exo implements `/v1/messages` (`api/main.py:375`,
`claude_messages`). mlx-lm's server answers `/v1/chat/completions`,
`/v1/models` and `/health` and nothing else -- checked, not assumed -- so
`knurlogic serve` could not back one. That gap is a TRANSLATION, not an
inference problem.

`src/knurlogic/messages.py` adds it, as an adapter over the endpoint the
engine is already serving. It is a self-request on purpose: the engine owns
chat templating, stop sequences and tool-call parsing, and a second
implementation would give the two endpoints different behaviour for the same
model.

    knurlogic serve <artifact>
    ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_API_KEY=x \
      ANTHROPIC_DEFAULT_SONNET_MODEL=<artifact> claude

The translation that actually matters is tools, because the shapes genuinely
differ: Anthropic puts a tool RESULT inside a user turn, OpenAI makes it its
own message keyed to the call id. Streaming is where a harness breaks, so the
event ORDER is built explicitly -- message_start, content blocks, then
message_delta carrying the stop reason, then message_stop -- and tool
arguments are forwarded as raw JSON fragments rather than parsed and
re-serialised, since the client reassembles them.

**What this cannot promise, and `doctor` should say so rather than this
module implying otherwise:** a harness leans hard on tool calling, and
whether a model emits well-formed tool calls at all is a property of the
MODEL. The translation being correct is necessary and nowhere near
sufficient.

## Not done: replacing an exo module

The MECHANISM is done and proven; no override has been written, because none
has a measurement behind it yet. The honest first targets are still
`routing/` (654 lines) or `master/` placement -- and now also the mlx-lm and
exo defaults that made prefill spike and OOM, which is the thing that
actually cost days and is the best argument this package has.

## Not done

* Not on PyPI. Name reserved; the package works.
* No model picker, no load/unload -- the server holds one artifact chosen at
  startup. This is the piece that would actually replace what exo is used for,
  and it needs real lifecycle machinery, not another panel.
* The page layout is not settled. It is one static file with no build step, so
  changing it costs nothing structural.
* glm5_next unpinned (see above). Do not claim GLM support until it has
  generated a token somewhere.
