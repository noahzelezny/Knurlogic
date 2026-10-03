# deepseek_v4 vision -- provenance

DeepSeek-V4-Flash-Vision-Exp's images. Source: the artifact's own
`inference/` (deepseek-ai/DeepSeek-V4-Flash-Vision-Exp, MIT License,
Copyright (c) 2023 DeepSeek), read from the local copy of 2026-10-02.
Design: `docs/design/deepseek-vision.md`. Contracts:
`docs/design/vision-contracts.md`. The trunk's half (bias_vl routing, the
image rows, the image-span window) is the vendored `deepseek_v4.py`'s
edits 14-15 (`../architecture/PROVENANCE.md`).

## Ported

- `tower.py`: `inference/vision.py` (ViT, Aligner, its RMSNorm, 2D RoPE)
  in MLX. Parameter names unchanged, so the artifact's `vision.*` /
  `aligner.*` load with their prefix kept (`Tower.vision`,
  `Tower.aligner`). The aligner's `F.pad` + `F.unfold(r, stride=r)` is a
  reshape: blocks row-major, each block's features channel-outermost.
- `processor.py`: `inference/image_processor.py`. `grid_tokens`,
  `solve_resize_ratio`, `safe_resize` verbatim; `load_image` takes a
  decoded PIL image (engine/vision/images does the decoding and refuses
  URLs and paths, so the reference's `load_image_bytes` is not taken) and
  uses numpy; the bf16 cast of the normalized pixels is a round-to-
  nearest-even on the float32 bits. `build_image_block` is split into
  `image_block` (IMAGE_START .. IMAGE_END) and `compress_pad` (the
  IMAGE_PADs before it), and kept whole for the tests.
- Not taken: `prepare_vl_inputs` (the serve path's key expansion plus
  `frame_key` do its work), the DSpark head.

## How an image reaches the trunk

- The prompt: `<｜deepseek_image｜>` (id 129264, read from tokenizer.json)
  is the placeholder; a message with an image has its parts joined with
  "\n\n" (`part_separator`), as `encoding_dsv4.py` joins content blocks.
- The key: the generic expansion widens the placeholder to the block's
  length; `frame_key` then puts `vocab_size + IMAGE_PAD` ids before each
  block so its IMAGE_START sits at a position == 3 (mod 4), counted in the
  full key as the reference counts `len(tokens)`.
- `encode`: the block's rows -- learned rows by type, the aligner's rows
  (in `perm` order) at the IMAGE positions. The learned rows are read
  with the tower (the trunk keeps its own copy).
- `embed`: the trunk's `embed` over the placeholder ids (an IMAGE_PAD id
  takes `image_pad` there), the block's rows merged in by sentinel, and
  `vl_ids` -- the ids with each block's types -- for the trunk's routing
  and image-span window. The types follow from `ref.grid_thw = (1,
  n_llm_h, n_llm_w)`, so a follower rank with refs and no store computes
  the same.
- `chunk_boundaries`: every block whole.

## Verified (tests/engine/test_vision_deepseek.py)

Against `tests/support/goldens/deepseek_v4_vision.npz`, made by the
reference under torch 2.11 (`build_deepseek_v4_vision.py`):
- the processor on eight aspect ratios: grid, patches bit for bit, types
  at four start positions, perm;
- the tower on the real weights (only `vision.*` / `aligner.*` and the
  four image rows read, one tensor at a time): float32 max |diff| 2.4e-5
  against the reference's float32 (max |out| 0.66; the committed golden
  is float16, so the test's own bound is that rounding). In bfloat16 a
  few rows move far from float32 (worst row cosine ~0.4 on the test
  image) -- the reference's own bfloat16 run moves the same rows as far
  (its worst 0.39); the test holds ours to the reference's spread;
- `bias_vl` routing and the image-span window (edits 14-15) on small
  inputs, and a tiny random model end to end.

## Open

1. Not run on a converted artifact: the trunk with the real weights
   needs a conversion that keeps `bias_vl`, the hash-layer `bias`, the
   image rows and the tower (phase 1 of the design). `load_weights` reads
   the HF names and `model.image_*`, quantized tower layers through
   `quantize_like`, untested on a real conversion.
2. The resolver's memory plan (`tuning/resolve.vision_budget`) keys on a
   `vision_config` and so reserves nothing for this tower or its store.
3. A text-only list content message is still joined with "" (mlx-lm's
   join); the reference joins every list content with "\n\n".
