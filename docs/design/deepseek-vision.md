# DeepSeek-V4-Flash-Vision-Exp

Scope for running deepseek-ai/DeepSeek-V4-Flash-Vision-Exp with images.
Reference: the repo's own `inference/` (model.py, vision.py,
image_processor.py) and `encoding/encoding_dsv4.py`, MIT. Source on disk:
`/Volumes/Thunderbay HDD/Teacher Models/deepseek-ai--DeepSeek-V4-Flash-Vision-Exp`
(156 GB, HF layout, FP8 attention + FP4 experts).

## What the artifact on disk is not

`Exo Models/deepseek-ai--DeepSeek-V4-Flash-Vision-Exp-mlx` (151 GB) is VQ
Lab's `teacher-prep` output: the 43 trunk layers and the head only. It has
no `vision.*`, `aligner.*`, `image_{start,end,newline,pad}`, no `mtp.*` and
no `ffn.gate.bias_vl`. It runs as a text model on today's deepseek_v4 code;
it can never take an image. Vision needs a conversion that keeps those keys.

## Differences from DeepSeek-V4-Flash

| | Flash | Vision-Exp |
|---|---|---|
| trunk | 43 layers | same shapes |
| `rms_norm_eps` | 1e-6 | 1e-20 (config; read, not assumed) |
| gate | `bias` (hash layers: none) | `bias` on every layer incl. hash, plus `bias_vl` |
| MTP | 1 MTP layer (our head) | 3 DSpark layers, block 5 |
| vision | none | 32-layer ViT + aligner |

### Image tokens
Image positions carry ids `>= vocab_size` (`vocab_size + IMAGE_START/PAD/
IMAGE/NEW_LINE/END`, 0..4). The embedding is not looked up for them:
`merge_image_embeddings` writes `image_start/pad/newline/end` (learned
vectors) and the aligner's output (`IMAGE`) into those rows of `h`.

### Routing (`bias_vl`)
For a token with id `>= vocab_size`:
* score layers: `scores + bias_vl` picks the top-k instead of `+ bias`;
* hash layers (0-2): top-k of `scores + bias_vl` replaces `tid2eid`
  (text tokens keep `tid2eid`; the image id is clamped to 0 for the lookup).
Routing weights still come from the unbiased scores. The trunk needs the
ids (it already gets `input_ids` for hash routing), so this is a small,
local edit to the vendored `MoEGate`.

### Attention inside an image span
Text attention is a 128-token sliding window plus compressed KV. Inside an
`[IMAGE_START, IMAGE_END]` span every token also sees forward to the span's
end (bidirectional within the image), up to `vision_max_n_token` = 384
(`get_image_visible`, `get_window_topk_idxs_visible`). Only the window part
changes; the compressor and indexer are as before.
Constraint from the reference: an image span must be prefilled in one
chunk. knurlogic's prefill chunking must not split a span (bump the chunk
boundary past `IMAGE_END`).
This is the riskiest piece: our vendored attention builds its window mask
in its own (Metal) path, so the visible-right extension has to be added
there and checked against the reference on a prompt with two images.

### Vision tower (vision.py, 118 lines)
ViT: patch 14, dim 1024, 16 heads, 32 blocks, 2D RoPE (theta 1e4), RMSNorm,
SwiGLU MLP (inter 2816), full bidirectional attention over one image.
Aligner: 3x3 unfold (downsample 3) -> Linear(9216, 4096) -> GELU ->
Linear(4096, 4096). About 0.45 B params, ~0.9 GB bf16. Small; ports
straight to MLX as a `Family` (vision-contracts.md), like GLM-5's.
Image processor: aspect-preserving resize to at most 384 LLM tokens
(`solve_resize_ratio`/`safe_resize`), a row-newline layout and a `perm`
reordering (`image_processor.py`, 184 lines): port as is, test against
the reference's token counts.

### Prompt
`encoding_dsv4.py` here is the Vision-Exp encoder. Diff it against the
Flash one our `deepseek_v4.jinja` ports; image parts become
`IMAGE_START ... IMAGE_END` id runs that the template cannot emit, so they
are spliced at tokenization like other families' image placeholders.

### DSpark (later)
Not our MTP head: 3 layers, `main_proj` over the HC-mean of trunk layers
40-42, a draft block of 5 noise tokens (128799) decoded in one pass, a
Markov head (rank 256) adding logits per position, and a confidence head.
A different drafting loop from `engine/mtp`. Separate phase, after images
work; the trunk exposes the layer 40-42 hidden states for it.

## Memory and placement
Trunk ~150 GB at mxfp4 experts: over either Mac alone (96 / 128 GB), so it
runs as a two-Mac split. Pipeline keeps images (cluster scope for 0.1.0);
the tower and aligner sit on rank 0 with the embedding. Tensor with images
is not in scope.
A lower-bit conversion (VQ Lab's) may bring it under 128 GB later; the
`bias_vl` and image keys must survive that conversion too.

## Phases
1. Conversion: extend the vendored `sanitize` (and Model) to keep
   `bias_vl`, the hash-layer `bias`, and the image vectors; the tower and
   aligner load standalone like GLM-5's. Convert from the HDD source. Text
   output checked against VQ Lab's text-only copy (same trunk) on a fixed
   prompt.
2. Vision: tower + aligner + processor port, image-id splicing, `bias_vl`
   routing, the image-span window, the prefill-chunk rule. Checked against
   the reference's own example (`examples/example_vl.txt`) for token ids,
   then by asking about known images.
3. Cluster: pipeline with images over the two Macs.
4. DSpark drafting.

Phases 1-2 in code (2026-10-02): the trunk's edits 14-15
(`families/deepseek/architecture/PROVENANCE.md`), the family
(`families/deepseek/vision/`, its PROVENANCE.md says what is verified and
what waits for a conversion), and the Vision-Exp template variant
(`engine/templates/PROVENANCE.md`). Not yet run on a converted artifact.
