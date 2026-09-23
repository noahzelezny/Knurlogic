# knurlogic — start here

*Routing layer. It says what lives where and points at the one home for
each fact; it does not restate them. "Every piece of information has one
home, other files point there" is the rule this repo is organised by, from
the Interpretable Context Methodology (Van Clief, arXiv:2603.16021).*

## What this is

Local models, run on your own machines, managed equally well by a person and
by an agent. knurlogic resolves the settings that decide whether a model
runs — the ones a person otherwise learns by running out of memory — says
what is true about the machine, and drafts with multi-token-prediction heads
that no stock runtime uses.

**The direction: knurlogic replaces exo.** `pip install knurlogic` is the
whole install -- one Mac or a cluster. Today it drives exo for clustering
(`place`); the replacement is built on what pip already ships: mlx's ring
and jaccl backends and launcher, and mlx-lm's `sharded_load`. What exo adds
on top -- discovery, coordination, placement, per-node downloads -- is what
knurlogic rebuilds. It still never rebuilds an ENGINE: mlx serves and
shards; knurlogic orchestrates, resolves settings, and drafts.

## If you are an agent

Use the MCP (`knurlogic mcp`, stdio). Eleven tools; `tools/list` describes
each. The loop that answers "can I run X, how, and is it safe now":

    one box      models -> fit -> settings -> ready -> load  -> state -> unload
    the cluster  models -> ready -> place -> state (poll) -> unplace

Never place or load while `ready` is false, and never wait on silence:
every server and exo instance in `state` has a phase -- downloading,
loading (layers), warming, serving, failed, or stalled, which means stop
waiting and read the advice.

Every answer says how it was measured; a refusal is an answer, an error sets
`isError`. `deps` answers "why does this work in exo and not here".

## Where things live

The folder is the rule, and each folder's `__init__.py` says what it holds,
what it may depend on, and what it enforces. Dependencies run one way, down
this list; nothing depends on a folder below it.

    src/knurlogic/
      engine/       what runs a model. The ONLY folder that may import mlx --
                    a test fails otherwise. The seam (seam.py), drafting
                    (mtp/), vendored architectures, overrides.
      machine/      what is true about this box, read rather than assumed:
                    artifacts on disk, what is loaded, memory, the wired
                    limit and the one load budget, installed dependencies.
      tuning/       what the settings should be, each beside its evidence.
      interfaces/   how a person or an agent talks to it: MCP, the page, the
                    CLI, the OpenAI and Anthropic endpoints, the exo front.

    tests/          tripwires are named in test docstrings, not here.
    tools/          probes that gate work. mtp_probe.py gates the drafting port.
    docs/PLAN.md    state, not log: what is true, what was measured so it is
                    not re-derived, what is next.

## Reading order

| If you want | Read |
|---|---|
| what is true now, and what to do next | `docs/PLAN.md` |
| why a default is what it is | `src/knurlogic/tuning/settings.py`, beside the constant |
| what knurlogic stands on, and which forks | `knurlogic deps`; `machine/deps.py` `PIECES` |
| what runs a model | `src/knurlogic/engine/seam.py` |
| how drafting works | `engine/mtp/` — `batch_loop.py` and `batch_generator.py` |
| what an agent gets | `src/knurlogic/interfaces/mcp.py` |
| what a command does | `knurlogic <cmd> --help`, then `interfaces/cli.py` `COMMANDS` |

## Contracts

* **The MCP and `/status.json` are the two wire contracts**, one per
  audience, built on the same functions. A capability on one side only is a
  bug.
* **One load budget.** `machine/wired.load_budget()` — the smaller of the GPU
  working set and memory available now — is what `fit`, `settings`, `load`
  and `serve` are all computed against. They cannot disagree.
* **A setting nobody reads is a bug.** Every resolved knob reaches its
  consumer: an artifact's bundled runtime, the engine's argv, or exo under
  exo's own names. This was violated three times before it was a rule.
* **A measurement outranks an assumption.** Numbers carry where they came
  from. If you cannot say how you know, say that instead.

## What this repo will not do

* Build models. vqlab builds; knurlogic runs what it built. Head builders
  (`mtp-graft`, `mtp-pack`) stay there.
* Set the wired limit. It reads it and hands over the exact `sudo` line.
* Delete an artifact. Reading state and starting a server are reversible.
* Implement a second inference path. If knurlogic generates a token, it is
  through an engine that already knew how.
