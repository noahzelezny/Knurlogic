# engine/runtime provenance

## tensor.py -- `shard`

- written for knurlogic, 2026-09-27; no exo code.
- follows the layout of mlx-lm's Qwen3_5 `Model.shard` (mlx-lm 0.32.0,
  MIT, Copyright (c) Apple Inc.; vendored unchanged in
  engine/families/qwen/architecture/qwen3_5.py): which modules split which
  way (in_proj_qkv/conv1d in [kd, 2kd] segments, out_proj/o_proj/down_proj
  sharded-to-all, the KV-head repeat when heads < ranks, the head-count
  divisions).
- changed: switch_mlp and shared_expert parameters split through
  `tensor.predicate`, which never splits a VQ `codebook`; the split itself
  (`split_params`) is our own so it is testable in one process. mlx's
  `shard_linear` is used as-is for attention and dense MLPs.
