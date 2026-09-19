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

Verified against the live two-node exo on this desk (Laptop B,
Studio A): inventory read correctly, `/v1/models` through
Knurlogic's port answered by exo with its 153 models, `/status` aggregated
both nodes. `tests/test_cluster.py` pins the same claims against a stub exo
whose answers carry a marker string, so a proxy that silently answered by
itself would fail the test rather than pass it quietly.

## Not done: replacing an exo module

Untouched on purpose. It is the `sys.modules` move `register.py` already
makes for architectures, it needs Knurlogic to launch the process (which
`--launch` now does), and the honest first targets are still `routing/`
(654 lines) or `master/` placement. There is no MEASURED reason to do it
yet, and doing it without one is how you acquire a permanent diff.

## Not done

* Not on PyPI. Name reserved; the package works.
* No model picker, no load/unload -- the server holds one artifact chosen at
  startup. This is the piece that would actually replace what exo is used for,
  and it needs real lifecycle machinery, not another panel.
* The page layout is not settled. It is one static file with no build step, so
  changing it costs nothing structural.
* glm5_next unpinned (see above). Do not claim GLM support until it has
  generated a token somewhere.
