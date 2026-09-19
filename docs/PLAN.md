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

## Two shape changes that get cheaper if done before clustering

1. `resolve()` takes ONE memory budget. Clustering resolves per node against
   different boxes; a list now is small, later it is invasive.
2. `status` knows only its own process. Aggregation across nodes is the
   natural shape and `/status.json` is where it should arrive.

## Not done

* Not on PyPI. Name reserved; the package works.
* No model picker, no load/unload -- the server holds one artifact chosen at
  startup. This is the piece that would actually replace what exo is used for,
  and it needs real lifecycle machinery, not another panel.
* The page layout is not settled. It is one static file with no build step, so
  changing it costs nothing structural.
* glm5_next unpinned (see above). Do not claim GLM support until it has
  generated a token somewhere.
