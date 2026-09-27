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
  (`split_params`) is our own so it is testable in one process; every
  split layer is wrapped in `Reduce`, our own float32 all_sum, in place of
  mlx's `shard_linear` classes and the models' in-dtype `sharding_group`
  sums (left unset).

## pipeline.py

- written for knurlogic, 2026-09-27; no exo code (upstream exo's
  auto_parallel pipeline layers were not used or read for this).
- keeps the attribute contract of mlx-lm's `PipelineMixin`
  (mlx_lm/models/pipeline.py, MIT, Copyright (c) Apple Inc.; imported by
  the vendored qwen3_5.py from the INSTALLED mlx-lm): `start_idx`,
  `end_idx`, `pipeline_rank`, `pipeline_size` and the `pipeline_layers`
  property. Its `pipeline()` (a uniform split, rank 0 last) and the
  vendored forward's all_gather are not used: `split` sets start/end to
  the whole kept slice and pipeline_size to 1, so the vendored forward
  does no collective of its own.
- the per-family index fixups follow Noah's own exo fork commits (after
  merge-base 90f24bef): f3ab3a83 (glm5_next fa_idx/ssm_idx on the slice),
  dd946407 (qwen4_exp ple_layers and a sliced make_cache); the receive
  in the receiving rank's own dtype follows 574a7bd7 / 12038d1b / 15170e22.
- `Coord` (B1/B2) follows the design of Noah's exo fork
  src/exo/worker/engines/mlx/mtp/pipeline.py (0b544af3): head on the
  last-layers rank, two fixed per-step broadcasts, never a verdict-dependent
  collective count; B0 (the admitted row's first token) is new here.
