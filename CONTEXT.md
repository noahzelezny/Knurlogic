# knurlogic — start here

*Routing layer. This file says what lives where and points at the one home
for each fact. It does not restate them: "every piece of information has one
home, other files point there" is the rule this repo is organised by, taken
from the Interpretable Context Methodology (Van Clief, arXiv:2603.16021).*

## What this is

A tool for taking control of your own machine: the settings that decide
whether a local model runs, the ones nobody exposed, and a way to replace a
module inside somebody else's package without forking it.

It wraps rather than rebuilds. exo already places and shards; mlx-lm already
serves. knurlogic is the layer that resolves settings before the engine
imports anything, carries work that would otherwise be trapped in a fork, and
answers — to a person or an agent — what is actually true about this machine.

## Reading order

| If you want | Read |
|---|---|
| what is true now, and what to do next | `docs/PLAN.md` |
| what a command does | `knurlogic <cmd> --help`, then its module |
| why a default is what it is | `src/knurlogic/settings.py`, beside the constant |
| why an architecture file is vendored | `src/knurlogic/arch.py` docstring |
| what runs a model | `src/knurlogic/engine.py` — the only module that knows |
| the agent-facing interface | `src/knurlogic/mcp.py` |

## The seams, which are the real structure

Folder boundaries here are enforcement, not filing.

    src/knurlogic/
      engine.py        THE SEAM. The only module that imports mlx. A test
                       fails if any other one does; it has caught five leaks.
      mtp/             Drafting. Engine-side by definition — every line is
                       arithmetic on an mlx model. Its FRONT DOOR is stdlib
                       only, so asking whether an artifact has a head costs
                       nothing, and a test asserts that.
      architectures/   Vendored mlx-lm model files, pinned by digest, with
                       PROVENANCE.md saying which mlx-lm each came from.
      overrides/       A meta-path finder that crosses a spawn boundary.
                       stdlib only, because nothing else survives that trip.
      web/             The page. One file.

    tools/             Probes that gate work. mtp_probe.py is the one the
                       drafting port has to keep passing.
    tests/             115 of them. The tripwires are named in the test
                       docstrings, not here.
    docs/PLAN.md       State, not log: what is true, what was measured so it
                       is not re-derived, what is next.

## Contracts

* **`/status.json` is the wire contract.** It carries the cluster shape even
  for one box. Everything the page shows comes from it.
* **`knurlogic mcp` is the agent contract.** Every tool answers
  deterministically, reports what it looked at, and refuses rather than
  gambles. Its docstring lists each rule and the failure that bought it.
* **A measurement outranks an assumption.** Numbers in this repo carry where
  they came from. If you cannot say how you know, say that instead.

## What this repo will not do

* Build models. vqlab builds; knurlogic coalesces. Head builders
  (`mtp-graft`, `mtp-pack`) stay there and no button here invokes them.
* Set the wired limit. It reads it, works out the ceiling, and hands over the
  exact `sudo` line. Running it is a human's action.
* Delete an artifact. Reading state and starting a server are reversible.
* Implement a second inference path. If knurlogic generates a token, it is
  through an engine that already knew how.
