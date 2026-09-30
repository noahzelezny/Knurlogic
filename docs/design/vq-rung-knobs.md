# VQ rung knobs: what each released rung's published runtime ships

A *rung* is one published quantization level of a model (one Hub repo). Every
rung ships its own `model.py`; knurlogic serves them all from one vendored VQ
runtime (`src/knurlogic/engine/vq/`, provenance in
`src/knurlogic/engine/vq/PROVENANCE.md`) plus per-rung *knobs*: the
environment flags that make the vendored runtime reproduce what that rung
publishes. Design: [vision.md](vision.md), "knurlogic owns the VQ runtime".

The knobs are read from the Hub (`hf download TheDrainFlorist/<repo> model.py
config.json`, text files only), never from local copies, which drift. The
machine record is `src/knurlogic/engine/vq/rungs.json`;
`tools/vq_gate.py knobs <dir> --markdown` regenerates the table below from it.

"HEAD" in the table is the vendored runtime. Its defaults are
`VQ_GEMMSEG_BF16IO=0`, `VQ_DECODE_BF16IO=0` and `VQ_DENSE_SS=1`. An
arc6-generation bundle has no bf16-I/O flags at all; rungs.json records both
as `0` for those rungs (inferred: the arc6 arithmetic is the flag off), which
HEAD already defaults to, so the table shows no knob for them.

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

* **There are 9 distinct published runtimes across 20 rungs.** GLM 2.7's
  model.py is byte-identical to GLM 3.1 and 3.6; the 397B 2.2 bundle (4684
  lines) is byte-identical to 35B 3.4.
* **The runtime code barely varies.** Lines that differ from HEAD's
  `vq_switch.py` (plus `vq_dense.py` for the dense rungs):
  - 27B: 0.
  - e4b and the v1.5 MoE rungs: 1 (`VQ_DENSE_SS`).
  - v2 rungs: 3 (`VQ_DENSE_SS` and both bf16-I/O flags).
  - arc6 rungs: 506. That is an older runtime and its flags do not describe
    all of the difference; the identity gate has to pass before HEAD serves
    these rungs.
* **The v2 rungs are Flash-Next 2.1 and Qwen3.6-35B-A3B 3.8, 4.6 and 5.4.**
  Their bf16-I/O flags are on and numerics-active, so a rung's numerics come
  from the rung: `tuning/resolve.numerics_for` applies the declared knobs,
  and a runtime profile applies only when a person asks for it.
* **16 of the 20 rungs ship `VQ_DENSE_SS=0`**, although HEAD defaults it to
  1 (only the three 27B rungs were bundled after that change). On MoE rungs
  the flag only reaches vq_switch's dense kernels. It is reproduced as
  shipped: knurlogic reproduces what is published.
* **Some flags are absent from the arc6 bundles**: `VQ_D4_WALK`,
  `VQ_GEMMSEG_OTILE64`, `VQ_GEMMSEG_PH2V`, `VQ_GEMMSEG_PIPE`,
  `VQ_GEMMSEG_DSTORE`, `VQ_GEMMSEG_XT_PAD` and `VQ_IDX_MEMO`. They stay at
  HEAD's defaults on the claim that they are bit-exact; the identity gate
  tests that claim, this table does not.

## Verified

A rung runs on knurlogic's runtime only after
`python tools/vq_gate.py gate <artifact> --record` passes (the identity gate).
The gate runs each runtime in its own process with no VQ_* environment
variables, and passes when logits over the prompt agree within atol 1e-5 and
40 greedy tokens are identical. Until a rung passes, it loads the `model.py`
it ships; the `verified` column records which have.

## Module notes

### src/knurlogic/engine/vq/rungs.py

THE RECORD IS THE PUBLISHED BUNDLE. Each released rung ships a `model.py`
whose flag defaults ARE its numerics. `rungs.json` holds, per Hub repo, the
defaults read out of that rung's PUBLISHED model.py -- never out of a local
copy, which may have drifted from the Hub. Measured against the Hub:
Flash-Next 2.1's whole runtime body differs from HEAD by exactly its three
flag defaults; the 27B's by nothing at all. Everything starts
`verified: False`.

### src/knurlogic/engine/vq/runtime.py

WHAT A PUBLISHED BUNDLE IS. Every released rung's `model.py` is three texts
concatenated by vqlab's bundlers: `vq_switch.py` (MoE expert + PLE kernels),
`vq_dense.py` on the dense rungs (VQLinear / VQEmbedding), and a loader shim
that builds the registry architecture and swaps each VQ-coded module for its
drop-in before weights load. The rungs differ from each other (and from
vqlab HEAD) in the DEFAULTS of a few env flags baked into that text --
measured against the Hub: 27B == HEAD byte for byte; Flash-Next 2.1 == HEAD
but for three default lines.

So the runtime reproduces a bundle without its text: the two runtime files
are vendored VERBATIM (PROVENANCE.md pins commit and digest), executed into
one fresh namespace per knob set -- concatenated, exactly as a bundle is, so
vq_dense finds vq_switch's kernels in its own globals the way it does inside
a model.py -- with the rung's knobs in the environment while the flags are
read. The shim's job (attach) is done by `model_classes`.

WHY A FRESH NAMESPACE PER KNOB SET, NOT ONE IMPORT. The flags are module
globals read ONCE at import (`_GEMMSEG_BF16IO = os.environ.get(...)` and
~30 more). One shared import would freeze the first rung's numerics into
every rung loaded after it in the process.

ENV PRECEDENCE: a flag already set in the process environment wins over the
rung's knob. The resolver emits the rung's own values by default, so they
agree; a value differing from the rung's is a person asking (a runtime
profile, a debugging override), and the person wins.

Text only. The vision path belongs to the family packages: they build the
multimodal model and call `attach_vq` on its language model.
