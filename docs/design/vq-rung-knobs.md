# VQ rung knobs: what each released rung's PUBLISHED runtime ships

*2026-09-23. Read from the Hub (`hf download TheDrainFlorist/<repo> model.py
config.json`, text files only), never from `~/.exo` copies. The machine
record is `src/knurlogic/engine/vq/rungs.json`, and `tools/vq_gate.py knobs
<dir> --markdown` regenerates this table from it. Design: `vision.md` D1.*

"knobs on HEAD" is the environment knurlogic's vendored runtime (vqlab
`42df84f`) needs to reproduce what the rung publishes. HEAD's defaults are
`VQ_GEMMSEG_BF16IO=0`, `VQ_DECODE_BF16IO=0` and `VQ_DENSE_SS=1`. An arc6-era
bundle has no bf16-I/O flags at all. For those rungs rungs.json records both
flags as `0` (inferred, since the arc6 arithmetic is the flag off). HEAD
already defaults them to 0, so the table shows no knob for them.

| repo | generation | model.py lines | sha256 (12) | GEMMSEG_BF16IO | DECODE_BF16IO | DENSE_SS | knobs on HEAD | verified |
|---|---|---|---|---|---|---|---|---|
| GLM-5.3-Flash-VQ-2.7bpw | arc6-no-flags | 4159 | `cc6dbf41fe5e` | -- | -- | 0 | VQ_DENSE_SS=0 | no |
| GLM-5.3-Flash-VQ-3.1bpw | arc6-no-flags | 4159 | `cc6dbf41fe5e` | -- | -- | 0 | VQ_DENSE_SS=0 | no |
| GLM-5.3-Flash-VQ-3.6bpw | arc6-no-flags | 4159 | `cc6dbf41fe5e` | -- | -- | 0 | VQ_DENSE_SS=0 | no |
| Qwen3.5-397B-A17B-VQ-2.2bpw | v1.5 | 4684 | `62b9eac33440` | 0 | 0 | 0 | VQ_DENSE_SS=0 | no |
| Qwen3.5-397B-A17B-VQ-2.4bpw | arc6-no-flags | 4229 | `6a3008ee72a0` | -- | -- | 0 | VQ_DENSE_SS=0 | no |
| Qwen3.5-397B-A17B-VQ-2.6bpw | arc6-no-flags | 4229 | `6a3008ee72a0` | -- | -- | 0 | VQ_DENSE_SS=0 | no |
| Qwen3.5-397B-A17B-VQ-3.1bpw | arc6-no-flags | 4229 | `6a3008ee72a0` | -- | -- | 0 | VQ_DENSE_SS=0 | no |
| Qwen3.6-35B-A3B-VQ-3.4bpw | v1.5 | 4684 | `62b9eac33440` | 0 | 0 | 0 | VQ_DENSE_SS=0 | no |
| Qwen3.6-35B-A3B-VQ-3.8bpw | v2 | 4684 | `e7ac3e04f450` | 1 | 1 | 0 | VQ_DECODE_BF16IO=1, VQ_DENSE_SS=0, VQ_GEMMSEG_BF16IO=1 | no |
| Qwen3.6-35B-A3B-VQ-4.6bpw | v2 | 4684 | `e7ac3e04f450` | 1 | 1 | 0 | VQ_DECODE_BF16IO=1, VQ_DENSE_SS=0, VQ_GEMMSEG_BF16IO=1 | no |
| Qwen3.6-35B-A3B-VQ-5.4bpw | v2 | 4684 | `e7ac3e04f450` | 1 | 1 | 0 | VQ_DECODE_BF16IO=1, VQ_DENSE_SS=0, VQ_GEMMSEG_BF16IO=1 | no |
| Qwen3.8-27B-VQ-3.9bpw | v1.5 | 5194 | `8f470e550dd7` | 0 | 0 | 1 | none | no |
| Qwen3.8-27B-VQ-4.5bpw | v1.5 | 5194 | `8f470e550dd7` | 0 | 0 | 1 | none | no |
| Qwen3.8-27B-VQ-4.8bpw | v1.5 | 5194 | `8f470e550dd7` | 0 | 0 | 1 | none | no |
| Qwen3.8-Flash-Next-VQ-2.1bpw | v2 | 4738 | `36de8d6ba21f` | 1 | 1 | 0 | VQ_DECODE_BF16IO=1, VQ_DENSE_SS=0, VQ_GEMMSEG_BF16IO=1 | no |
| Qwen3.8-Flash-Next-VQ-3.2bpw | v1.5 | 4738 | `e2474b4fafff` | 0 | 0 | 0 | VQ_DENSE_SS=0 | no |
| Qwen3.8-Flash-Next-VQ-4.4bpw | v1.5 | 4738 | `e2474b4fafff` | 0 | 0 | 0 | VQ_DENSE_SS=0 | no |
| Qwen3.8-Flash-Next-VQ-5.5bpw | v1.5 | 4738 | `e2474b4fafff` | 0 | 0 | 0 | VQ_DENSE_SS=0 | no |
| gemma-4-26b-a4b-it-VQ-6.2bpw | v1.5 | 4647 | `de30d29f2f8a` | 0 | 0 | 0 | VQ_DENSE_SS=0 | no |
| gemma-4-e4b-it-VQ-PLE | v1.5 | 5157 | `2e819b08083b` | 0 | 0 | 0 | VQ_DENSE_SS=0 | no |

## What the table says

* **There are 9 distinct published runtimes, not the 10 the design counted.**
  The design left GLM 2.7 "to be re-read". On the Hub its model.py is
  byte-identical to GLM 3.1 and 3.6. The republished 397B 2.2 (4684 lines)
  is byte-identical to 35B 3.4.
* **The runtime code barely varies.** Lines that differ from HEAD's
  `vq_switch.py` (plus `vq_dense.py` for the dense rungs):
  - 27B: 0.
  - e4b and the v1.5 MoE rungs: 1 (`VQ_DENSE_SS`).
  - v2 rungs: 3 (`VQ_DENSE_SS` and both bf16-I/O flags).
  - arc6 rungs: 506. That is an older runtime, and its flags do not
    describe all of the difference. G-VQ has to pass before HEAD serves
    these rungs.
* **The v2 rungs are Flash-Next 2.1 and Qwen3.6-35B-A3B 3.8, 4.6 and 5.4.**
  These are the rungs the old `RUNTIME_PROFILES["v1.5"]` default overrode.
  `tuning/resolve.numerics_for` now fixes that.
* **16 of the 20 rungs ship `VQ_DENSE_SS=0`.** HEAD turned the flag on in
  `ef4e8dc` (F148) but rebundled only the three 27B rungs. On MoE rungs the
  flag only reaches vq_switch's dense kernels. It is still reproduced as
  shipped: knurlogic reproduces what is published, not what was planned.
* **Some flags are missing from the arc6 bundles.** These are `VQ_D4_WALK`,
  `VQ_GEMMSEG_OTILE64`, `VQ_GEMMSEG_PH2V`, `VQ_GEMMSEG_PIPE`,
  `VQ_GEMMSEG_DSTORE`, `VQ_GEMMSEG_XT_PAD` and `VQ_IDX_MEMO`. They stay at
  HEAD's defaults because vqlab records them as bit-exact (F54, F56). G-VQ
  tests that claim. This table does not make it.

## Verified

No rung is verified yet. A rung runs on knurlogic's runtime only after
`python tools/vq_gate.py gate <artifact> --record` passes. The gate runs each
runtime in its own process with no VQ_* environment variables, and passes
when logits over the prompt agree within atol 1e-5 and 40 greedy tokens are
identical. Until then every rung loads the `model.py` it ships.
