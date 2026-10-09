# engine/runtime provenance

## tensor.py -- `shard`

- written for knurlogic; no exo code.
- follows the layout of mlx-lm's Qwen3_5 `Model.shard` (mlx-lm 0.32.0,
  MIT, Copyright (c) Apple Inc.; vendored unchanged in
  engine/families/qwen/architecture/qwen3_5.py): which modules split which
  way (in_proj_qkv/conv1d in [kd, 2kd] segments, out_proj/o_proj/down_proj
  sharded-to-all, the KV-head repeat when heads < ranks, the head-count
  divisions).
- changed: switch_mlp and shared_expert parameters split through
  `tensor.predicate`, which never splits a VQ `codebook`; the split itself
  (`split_params`) is knurlogic's own so it is testable in one process; every
  split layer is wrapped in `Reduce`, knurlogic's own float32 all_sum, in place of
  mlx's `shard_linear` classes and the models' in-dtype `sharding_group`
  sums (left unset).

## pipeline.py

- written for knurlogic; no exo code (upstream exo's
  auto_parallel pipeline layers were not used or read for this).
- keeps the attribute contract of mlx-lm's `PipelineMixin`
  (mlx_lm/models/pipeline.py, MIT, Copyright (c) Apple Inc.; imported by
  the vendored qwen3_5.py from the INSTALLED mlx-lm): `start_idx`,
  `end_idx`, `pipeline_rank`, `pipeline_size` and the `pipeline_layers`
  property. Its `pipeline()` (a uniform split, rank 0 last) and the
  vendored forward's all_gather are not used: `split` sets start/end to
  the whole kept slice and pipeline_size to 1, so the vendored forward
  does no collective of its own.
- the per-family index fixups: glm5_next's fa_idx/ssm_idx on the slice,
  qwen4_exp's ple_layers and a sliced make_cache; the receive is in the
  receiving rank's own dtype.
- `Coord` (B0/B1/B2): the head on the last-layers rank, two fixed per-step
  broadcasts, never a verdict-dependent collective count; B0 carries the
  admitted row's first token.
