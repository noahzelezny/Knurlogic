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

## What it is not

It does not detect machines, fit models, or score them. Memory budget is an
input. Scope stays narrow on purpose.

## Status

Early sketch. The resolver, the architecture check and `doctor` work; the
pin table is deliberately empty (populate it against a validated env, never
by trusting whatever happens to be installed). Serving and multi-agent
support come later.

---

*knurl* — the crosshatch cut into metal so a human hand can grip a machined
part. The machinery is the easy half.

---

**Note:** this repository previously reserved the name for a personal media
and memory pipeline. That scope has been replaced by the above.
