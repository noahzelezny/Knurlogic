# Knurlogic

*Make local model runtimes manageable.*

An artifact you cannot load is worth nothing. The gap between a downloaded
model and a working one is not the model — it is an environment with the right
architecture files at the right versions, and a handful of settings whose
defaults are tuned for other shapes. Getting either wrong produces the same
stack trace, so nobody can tell which one bit them.

Knurlogic resolves both, and says which is wrong when something will not run.

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
  `settings.py` carries the measurement that established it. That is the
  actual asset — the numbers cost runs, and nobody should have to rediscover
  them.

* **A server, and a cluster front end.** `knurlogic serve <artifact>` is an
  adapter over mlx-lm's OpenAI endpoint with the settings resolved and set
  *before* the model loads, which is load-bearing: a VQ artifact's bundled
  runtime reads its knobs at import. `--cluster` does the same across several
  boxes by wrapping exo, which already places and shards — the OpenAI surface
  is exo's, proxied untouched, and Knurlogic adds the per-node resolution and
  one `/status` that covers every node.

* **An override mechanism, so none of this needs a fork.** A module inside
  `mlx_lm`, `mlx_vlm` or `exo` can be replaced from a versioned, digest-pinned
  copy in this package — no fork, no writes to site-packages, no permanent
  diff. It is installed through `sitecustomize.py` on `PYTHONPATH` rather than
  `sys.modules`, because exo's runner is a spawned process and a spawned
  process inherits the environment and nothing else; that is measured, with a
  control arm, on exo's own interpreter. `knurlogic override run -- exo`
  applies them to any command and every process it spawns.

## What it is not

It does not detect machines, fit models, or score them. Memory budget is an
input. It does not implement distributed inference; it wraps something that
does. Scope stays narrow on purpose.

## Status

The resolver, the architecture check, `doctor`, `smoke`, `vendor` and
`serve` work, verified from a clean venv on stock PyPI mlx-lm 0.31.3. Four
architectures are pinned by actual token generation; glm5_next is vendored
and unpinned, because no box here fits the smallest GLM rung.

`serve --cluster` resolves per node and aggregates status across a real
two-node exo. What it applies is the environment of a node it launches —
a node it merely attaches to gets its settings *reported*, because nothing
here can reach into another machine's process, and printing settings that
did not take effect is how a run ends up measuring the same value twice.

---

*knurl* — the crosshatch cut into metal so a human hand can grip a machined
part. The machinery is the easy half.

---

**Note:** this repository previously reserved the name for a personal media
and memory pipeline. That scope has been replaced by the above.
