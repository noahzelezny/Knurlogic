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

## Requirements

* A Mac with Apple Silicon, macOS. The weights have to fit in its unified
  memory; `knurlogic models` and `doctor` say whether they do.
* Python 3.10 or newer.
* mlx 0.31.2 and mlx-lm 0.31.3, pinned exactly (pip installs them). The
  engine subclasses mlx-lm's internals, so another version is not assumed
  to work.

## Install

    pip install knurlogic

From source:

    git clone https://github.com/noahzelezny/Knurlogic
    cd Knurlogic && pip install -e .

## Quickstart

A model is a directory with a `config.json` and its weights
(`.safetensors`) beside it, in MLX format. Get one from Hugging Face with
the `hf` CLI (not a knurlogic dependency) -- a small Gemma 4 to start:

    pip install -U huggingface_hub
    hf download mlx-community/gemma-4-e4b-it-8bit \
      --local-dir ~/Knurlogic/Models/gemma-4-e4b-it-8bit

Any MLX-format repo of a supported family below works, sized for your
memory; without `--local-dir` it lands in the Hugging Face cache, which is
searched too.
`knurlogic models` finds what is already on the machine, in every tool's
store -- `~/Knurlogic/Models` (`KNURLOGIC_MODELS`), exo's model dirs
(`EXO_MODELS_DIR`, `EXO_MODELS_DIRS`), the Hugging Face cache
(`HF_HUB_CACHE`, `HF_HOME`), Ollama and LM Studio -- and `--servable` keeps
only what this engine can load. `--path DIR` scans one more directory.

Check it, then serve it:

    knurlogic doctor ~/Knurlogic/Models/gemma-4-e4b-it-8bit
    knurlogic serve  ~/Knurlogic/Models/gemma-4-e4b-it-8bit

On first run nothing is downloaded and nothing is written to
`site-packages`: the vendored architecture is registered in-process, the
settings are resolved and printed, then the weights load (seconds to
minutes; the page shows the phase). `serve` listens on `127.0.0.1:8080` by
default (`--host`, `--port`) and prints how to reach it; Ctrl-C stops it. Then:

    curl http://127.0.0.1:8080/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"model": "local", "messages": [{"role": "user", "content": "Hello"}]}'

`http://127.0.0.1:8080/` is the page for that server: what loaded, the
memory split, a chat, and the Settings panel. `/status.json` and
`/settings.json` are the same without a browser.

`knurlogic ui` serves the page at http://127.0.0.1:8899/ (open it in your
browser) without loading anything: every model on the disk, every runtime
holding memory, and a Launch button per model (served on `--serve-port`).
A model launched from the page, or by the MCP `load` tool, stops with the
page's Stop button or the MCP `unload` tool.

## Point a harness at it

`serve` prints these lines; `knurlogic connect --port 8080` prints them again
for a running server.

Claude Code (or any Claude-Messages harness), in one terminal:

    ANTHROPIC_BASE_URL=http://127.0.0.1:8080 \
    ANTHROPIC_API_KEY=x \
    ANTHROPIC_DEFAULT_OPUS_MODEL=local \
    ANTHROPIC_DEFAULT_SONNET_MODEL=local \
    ANTHROPIC_DEFAULT_HAIKU_MODEL=local \
    API_TIMEOUT_MS=3000000 \
    claude

The same as a project's `.claude/settings.json` `env` block scopes it to one
directory. Not the global `~/.claude/settings.json`: that routes every
session on the machine to the local model.

Anything that speaks OpenAI (Zed, Cline, Continue, OpenWebUI):
base URL `http://127.0.0.1:8080/v1`, any API key, model `local`. The server
answers the one model it loaded whatever name is sent.

An agent that should manage models rather than talk to one gets the MCP
server, on stdio:

    claude mcp add knurlogic -- knurlogic mcp

Its tools: `models` (what is on disk, whether it fits), `fit`, `settings`,
`drafting`, `ready` (is it safe to load now), `load`, `state`, `unload`,
`deps`. `knurlogic mcp --list` prints each with its description. `load`
refuses what will not fit; nothing is evicted to make room.

## Two Macs

One model can be split across two Macs, by tensor or by pipeline (layers).
A cluster job needs:

* the same knurlogic build, and the model, on both machines;
* a link between them: Thunderbolt (TCP over the bridge, or RDMA on a
  Thunderbolt 5 cable with RDMA enabled);
* the page running on each: `knurlogic ui --host cluster` (answers on the
  Thunderbolt link and loopback only). Macs find each other over Bonjour;
  `--peer HOST` names one directly. `knurlogic doctor --cluster` on each
  says what stops them seeing each other.

Then load with machines named -- the page's Launch, or the MCP `load` tool
with `machines` (and `split`). Placement, leader and cable are chosen for
you; a share that does not fit is refused before anything starts.
docs/DISCOVERY.md and docs/SERVER.md have the detail.

## Settings

Every knob is resolved from the model's `config.json` and the memory
budget, and shown with the measurement behind it in the page's Settings
panel and `/settings.json`. A launch strategy picks a bundle: `balanced`
(the default, the measured values), `fast` (wider prompt chunk where there
is room, dynamic MTP), `stable` (512-token chunks, MTP every step,
conservative memory), `lean` (8-bit KV, MTP off), `safe` (lowest peak
memory) -- `serve --tune`, or per model in Settings -> Models
(`KNURLOGIC_PRESET`); `--set KEY=VALUE` beats all of them. `--kv-bits 8`
stores the KV cache at 8 bits, about half the memory, taken by every family
but DeepSeek-V4. Compaction (`KNURLOGIC_COMPACT_AUTO`, off by default)
summarizes older turns when a harness asks for it or, when on, once a
prompt passes the trigger share of the window; it can be changed on a
running server. Long context: `KNURLOGIC_LONG_CONTEXT=yarn` applies Qwen's
documented YaRN (factor 4 over 262,144, ~1M tokens), Qwen families only,
opt-in because it can slightly hurt short prompts; a box that cannot hold
the chosen context's KV is refused with the numbers.

## Supported models

The families resolved from `config.json`'s `model_type`, each with its
architecture vendored in `engine/families/`:

* **Qwen** -- `qwen3_5` (dense), `qwen3_5_moe` (e.g. Qwen3.6-35B-A3B),
  `qwen4_exp` (Qwen3.8, including VQ artifacts). Vision, MTP drafting where
  the artifact packs a head, 8-bit KV, YaRN.
* **Gemma 4** -- `gemma4`, `gemma4_text`. Vision; 8-bit KV. Earlier reasoning
  is stripped by design, so each user turn re-prefills.
* **GLM-5** -- `glm5_next`. Pinned on a VQ artifact (GLM-5.3 Flash VQ).
* **DeepSeek-V4** -- `deepseek_v4`. Needs the pair: DeepSeek-V4-Flash has run
  split across two Macs (about 145 GiB of weights between them); no 8-bit KV.
  Not pinned yet (`knurlogic smoke --pin` has not been run on it).

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
  the envs were on different mlx-lm versions (0.32.0 and 0.31.9 -- neither
  the 0.31.3 knurlogic pins), which is exactly the problem. Knurlogic ships the files inside the package, loads
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
  Bonjour discovery, `--host cluster`).

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
command.

The server's prompt segmentation (system / conversation / thinking
tail, for prompt-cache checkpoints) and the shape of its scheduling loop
follow mlx-lm's server, rewritten here (engine/runtime/prompt.py,
scheduler.py); the MTP drafting began in vqlab;
image-feature caching follows mlx-vlm's idea with a byte bound instead of a
count.

## What it stands on

    mlx 0.31.2 and mlx-lm 0.31.3 (pinned), numpy, Pillow. No mlx-vlm:
    every family's architecture and vision tower is vendored
    (engine/families/), GLM-5.3 included.

Some of these have forks that carry fixes upstream does not, and nothing
else says which build is installed. `knurlogic deps` reads every verdict off
the fix itself rather than a version string (from a development machine
that also has mlx-vlm installed; a fresh install shows no mlx-vlm line):

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

Alpha (0.1.0). The resolver, `doctor`, `smoke`, `vendor`, `serve`, `ui` and
`mcp` work. Every vendored architecture is pinned
(`pins.json` beside it: qwen3_5, qwen3_5_moe, qwen4_exp, gemma4,
gemma4_text, glm5_next) except deepseek_v4, whose `smoke --pin` has not been
run yet.

Proven live on real weights: MTP drafting (Qwen3.8 Flash VQ, 30.2 vs 20.0
decode tok/s), images (Qwen 27B), compaction and 8-bit KV (Qwen3.6-35B-A3B),
a YaRN needle at 442,578 tokens (1M not run), a pipeline split across two
Macs with images and MTP drafting (Qwen3.8 Flash), and DeepSeek-V4-Flash
across two Macs with tool calls and compaction.
An agent has loaded, used and unloaded a model through the MCP.

Not yet measured live: pipeline prefill overlap. Known limit: a cluster
job's placement does not count the vision tower yet. A cluster job's leader
answers on loopback and on its link address (`load` and `state` report the
`url`).

`docs/dev/PLAN.md` holds what is measured and what is next; `CONTEXT.md` is the
map.
