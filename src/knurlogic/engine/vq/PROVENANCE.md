# engine/vq provenance

Vendored from vqlab at commit **d271035** (42df84f plus five comment-only
lines; the last functional change to `vq_switch.py` is `ef4e8dc`,
"VQ_DENSE_SS on by default").

| file | source | lines | sha256 |
|---|---|---|---|
| `vq_switch.py` | `src/vqlab/vq_switch.py` | 4453 | `31e56dfb0e1c2138286a6f9cd84e90f0b6cecb0611f9cf4a3bb08fc6ddd38aeb` |
| `vq_dense.py` | `src/vqlab/vq_dense.py` | 513 | `5066de6e71ccbacaed7b29cea031ea2977369043fcc05d7888b7eddc761b8995` |

Both are **verbatim** (`git show d271035:<path>`); no line is changed. They
are not imported as modules: `runtime.py` executes the two texts, joined, into
one fresh namespace per knob set -- the way a published `model.py` carries
them -- so `vq_dense._resolve_kernel` finds vq_switch's kernels in its own
globals (its first lookup), and each rung's import-time flag reads see that
rung's knobs. That is why vendoring needs no edit to load inside knurlogic.

Why `vq_dense.py` too (the design names only `vq_switch.py`): the three
Qwen3.8-27B rungs and gemma e4b are DENSE VQ (`vq_linear` / `vq_embed`); their
published bundles are `vq_switch.py + vq_dense.py + shim`, and `VQLinear` /
`VQEmbedding` exist only in `vq_dense.py`. Without it four of the 20 rungs
cannot be served.

The loader shims (vqlab `add_model_file.py`, `dense_shim.py`,
`arch_resolve.py`) are NOT vendored as text: `runtime.attach_vq` does their
job as code, mlx-lm branch only (text; the vision families attach on their
own language model). Shapes follow each shim, including the MoE shim's
round-up of packed rows vs the dense shim's floor.

## Per-rung notes (published bundles, `hf download <repo> model.py`, 2026-09-23)

The full table is `docs/design/vq-rung-knobs.md`; the machine record is
`rungs.json`. Against HEAD:

* **27B 3.9 / 4.5 / 4.8** -- the published runtime body IS HEAD
  (vq_switch + vq_dense, zero changed lines). No knobs.
* **gemma e4b** -- HEAD but `VQ_DENSE_SS=0` (bundled before ef4e8dc).
* **v1.5 MoE** (397B 2.2, 35B 3.4, Flash-Next 3.2 / 4.4 / 5.5, gemma 26b)
  -- HEAD but `VQ_DENSE_SS=0`: one line.
* **v2** (Flash-Next 2.1, 35B 3.8 / 4.6 / 5.4) -- HEAD but three lines:
  both bf16-I/O flags `1`, `VQ_DENSE_SS=0`.
* **arc6-era** (GLM 2.7 / 3.1 / 3.6, 397B 2.4 / 2.6 / 3.1) -- a 4159 /
  4229-line runtime, 506 HEAD lines differ; the numerics flags do not exist
  in it (knobs set them `0`, recorded as inferred), and neither do
  `VQ_D4_WALK`, `VQ_GEMMSEG_OTILE64`, `VQ_GEMMSEG_PH2V` and four more (left at
  HEAD: vqlab measured them bit-exact). Serving these on HEAD is a runtime
  change: only a pass of the identity gate (tools/vq_gate.py) may mark them
  verified.

Distinct published runtimes: **9**, not the 10 the design counted -- GLM 2.7
was the one "to be re-read", and on the Hub it is byte-identical to 3.1 / 3.6.
397B 2.2 is byte-identical to 35B 3.4.

## Re-vendoring

Copy the two files from a new vqlab commit, update the digests in
`runtime.py` (`RUNTIME_FILES`, `VENDORED_COMMIT`) and here, run
`tools/vq_gate.py knobs <dir-of-published-bundles> --write` (knobs are
relative to HEAD, so they change with it -- and every `verified` flag must
be re-earned, since the runtime it described is gone), then the suite.
