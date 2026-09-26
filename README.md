# Knurlogic

*Make local model runtimes manageable.*

An artifact you cannot load is worth nothing. The gap between a downloaded
model and a working one is not the model — it is an environment with the right
architecture files at the right versions, and a handful of settings whose
defaults are tuned for other shapes. Getting either wrong produces the same
stack trace, so nobody can tell which one bit them.

Knurlogic resolves both, and says which is wrong when something will not run.
It serves a person through a page and an agent through MCP, from the same
answers — so Claude or Codex can see what is loaded, what fits, which
settings a model needs and why, and load it without guessing.

```
$ knurlogic doctor ./Qwen3.8-Flash-Next-VQ-3.2bpw --working-set-gib 96

artifact   Qwen3.8-Flash-Next-VQ-3.2bpw
  type     qwen4_exp_text
  size     71.7 GiB   working set 96.0 GiB
  kernels  model.py   (d2-K256 x18, d4-K2048 x126)

architecture
  ?? qwen4_exp  present but not pinned -- 'it imports' is not 'it is the
                arithmetic we measured'

settings
  VQ_DECODE_CHUNK=32
  VQLAB_PREFILL_CHUNK=2048
  VQLAB_CACHE_LIMIT_GB=4.0
  VQ_MOE_GEMMSEG_CBDEV=auto
  VQ_MOE_GEMMSEG_RTILE=32
  ...

no blockers found
```

`--exports` prints the same settings as `export K=V` lines.

## What it is

* **A resolver.** One dict, resolved from the artifact's own `config.json`
  and a memory budget, and it is the *last word*. It does not write env files
  and hope one wins: a real experiment once set a knob in a file that was
  sourced before another file which overwrote it unconditionally, so the run
  measured the same value twice and was reported as "no difference."
* **A vendored architecture set.** The model files that get grafted into
  `mlx_lm/models/` inherit whatever version that install happens to be, so
  "which arithmetic am I running" has no answer. Measured across two envs on
  one machine: three of four files differed and one was absent from both —
  the envs were on different mlx-lm versions (0.32.0 and 0.31.9), which is
  exactly the problem. Knurlogic ships the files inside the package, loads
  them into `sys.modules` without writing to `site-packages`, and reports
  `OK` / `UNPINNED` / `DRIFTED` / `MISSING`.
* **A ledger of settings, encoded as defaults.** Every constant in
  `tuning/settings.py` carries the measurement that established it — the
  prefill chunk per model family, the cache limit, the VQ kernel flags. That
  is the actual asset: the numbers cost runs, several of them cost an
  out-of-memory, and nobody should have to rediscover them. Each one reaches
  whatever actually reads it — the engine's argv or a bundled runtime.

* **Multi-token-prediction drafting, which no stock runtime does.** mlx-lm
  has no MTP path. A drafting head packed
  beside the weights is used because it is there — on a single request and
  inside a batch alike, token-identical to decoding without it. `--no-draft`
  is the troubleshooting switch.

* **An agent interface.** `knurlogic mcp` speaks MCP on stdio: `models`,
  `fit`, `settings`, `ready`, `load`, `state`, `unload`, `drafting`, `deps`.
  `fit` and `settings` and `load` compute against one memory budget, so an
  agent is never told a model fits with room to spare and then handed the
  settings for a roomier box. `load` refuses what will not fit, with no
  override, because it is arithmetic.

* **A server.** `knurlogic serve <artifact>` is knurlogic's own
  OpenAI-compatible server (plus `/v1/messages` for Claude-style harnesses),
  with the settings resolved and set *before* the model loads, which is
  load-bearing: a VQ artifact's bundled runtime reads its knobs at import.
  It answers ordinary clients and its own page; a web page in your browser
  is refused unless you allow it (`--allow-origin`), and a DNS name other
  than localhost, `.local` or the hostname needs `--allow-host`. Requests are
  capped at `--max-request-mib`; a request's images must fit the image store
  together (`--image-store-gib`) -- each image is downscaled to what the
  model takes, never refused for size unless it is too big to decode
  safely. docs/SERVER.md has the design. Several machines are knurlogic's own (`cluster/`: peers,
  Bonjour discovery, `--host cluster`); knurlogic does not drive exo.

* **A GUI that exposes the knobs.** `/` shows what loaded, the memory split
  nothing else shows, and a Settings panel with every resolved knob, the
  measurement behind it, and what another `--tune` would give you — as a diff
  against what is running, because the runtime reads its settings at import
  and a control that pretended otherwise would be lying.

* **An Anthropic-Messages endpoint**, so a coding harness pointed at
  `ANTHROPIC_BASE_URL` can run against a local model. It is a translation over
  the engine's own OpenAI endpoint, not a second inference path.

## What it is not

It does not build or score models — vqlab builds, knurlogic runs what it
built. It never sets the wired limit or deletes a model; it tells you the
command. It does not drive exo: knurlogic replaces exo's orchestration on
mlx's own distributed backends (docs/PLAN.md). A running exo is shown as one
more runtime holding memory, and `~/.exo/models` is one more model store.

The page's look is inspired by exo's dashboard; its code is knurlogic's
own. The server's prompt segmentation (system / conversation / thinking
tail, for prompt-cache checkpoints) and the shape of its scheduling loop
follow mlx-lm's server, rewritten here (engine/runtime/prompt.py,
scheduler.py); the MTP drafting began in the owner's exo fork and vqlab;
image-feature caching follows mlx-vlm's idea with a byte bound instead of a
count.

## What it stands on

    mlx, mlx-lm (>= 0.31.3). mlx-vlm for multimodal and GLM-5.3.

Some of these have forks that carry fixes upstream does not, and nothing
else says which build is installed. `knurlogic deps` reads every verdict off
the fix itself rather than a version string:

```
$ knurlogic deps
knurlogic  /opt/anaconda3/bin/python3  (python 3.12.2)
  mlx      0.31.2                           jaccl self-heal: no -- stock ring
  mlx-lm   0.31.3                           stock
  mlx-vlm  0.5.0                            knurlogic's vendored glm5_next ...
                                            cannot load here: missing ...
```

What each fork carries, and why it is or is not ported, has one home:
`PIECES` in `src/knurlogic/machine/deps.py`.

## Status

The resolver, `doctor`, `smoke`, `vendor`, `serve`, `ui` and `mcp` work.
Four architectures are pinned by actual token generation; glm5_next is
vendored and unpinned, because no box here fits the smallest GLM rung.

Drafting runs on a single request and in a batch; the batch path is gated on
token identity against mlx-lm's own generator, and has not yet been timed on
real weights through knurlogic (the exo fork measured the same loop at 23.1
vs 22 tok/s, identical tokens). An agent has loaded, used and unloaded a
model through the MCP, across two sessions.

`docs/PLAN.md` holds what is measured and what is next; `CONTEXT.md` is the
map.

---

*knurl* — the crosshatch cut into metal so a human hand can grip a machined
part. The machinery is the easy half.

---

**Note:** this repository previously reserved the name for a personal media
and memory pipeline. That scope has been replaced by the above.
