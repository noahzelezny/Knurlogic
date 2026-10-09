# The tensor and pipeline splits

One model served by several ranks, usually one per Mac. The design, the
step plan's field-level format and why rank 0 owns all sampling:
[server](../design/server.md) ("Cluster: tensor split", "Cluster:
pipeline split", step plan) and [tensor-mtp](../design/tensor-mtp.md).
How the ranks get started on each Mac is [cluster](cluster.md).

## Where the code is

In `src/knurlogic/engine/runtime/`:

| file | what |
|---|---|
| `plan.py` | the step plan: `OPS`, `_FIELDS`, `_OPTIONAL`, `SETS`, `check`, `encode` / `decode` (JSON, never pickle), the control vector (`CONTROL_LEN`: over, step, length, active, peak). Pure Python |
| `tensor.py` | `shard` (cut every layer N ways), `Link` (the per-step exchange), `Journal` (rank 0's ops since the last exchange), `Ring` (rank 0's side, kept across executors), `TensorExecutor` (rank 0), `follow` (ranks >= 1: apply the plan, step, never sample), `init` (join the ring), `serve_follower` (a follower from start to stop), the bell (`bell_answer`, `bell_dial`) a parked rank sleeps on |
| `tensor_rules.py` | `RULES`: which arrays of a layer are cut, on which axis, in which segments; `refusals`, `unverified`. No mlx, so `tuning/` checks safetensors headers against the same table |
| `viability.py` | `refusals`: runs a module no rule knows, whole and split, and compares before the ring starts |
| `pipeline.py` | `split` (contiguous runs of layers), `Send` / `Recv` wrappers, `Silent` (a follower's lm_head never runs), `restage`, `Coord` / `coordinate` (rank 0's drafts and verdicts broadcast), `agree` |

Outside the runtime:

- `engine/prompt_cache/ring.py`: `JournalPromptCache` and `apply_cache_op`
  (see [prompt-cache](prompt-cache.md)).
- `engine/runtime/scheduler.py`: `_executor` builds a `TensorExecutor`
  when the scheduler has a ring (`self.tensor`), for either split;
  `_ring_fatal` and `RingFailed` handle a dead ring.
- `engine/families/<family>/pipeline_stage.py` and the manifest's
  `pipeline` / `tensor` / `tensor_split` entries.
- `interfaces/serve.py`: a rank's launch (`_ring_env`, `_ring_refusals`,
  `pipeline_share_bytes`); a rank >= 1 calls `tensor.serve_follower`.
- `interfaces/http/__init__.py` (`watch_ring`): rank 0's server watches
  the ring.
- `tuning/tensor_split.py`: `tensor_refusals`, `tensor_sharded`,
  `tensor_placement`, `tensor_split_refusals`; `tuning/pipeline_split.py`:
  `pipeline_refusals`, `pipeline_layer_bytes`, `pipeline_shares`,
  `leader_bytes`; `tuning/rank_order.py`: `rank_order`;
  `tuning/resolve.py`: `resolve_cluster`.
- `cluster/launch.py`: `placement`, `shape_of`, `viability_refusals`,
  `rank_argv`, `rank_env`.

## Rules that keep it correct

- **Rank 0 decides; the others follow.** Rank 0 owns the scheduler, HTTP
  and all sampling. Followers apply the plan's ops in order, take rank 0's
  sampled tokens and run the same step. They never sample a live row and
  never decide a prompt-cache hit or an eviction.
- **Every op is in the schema.** A journaled op that `plan.check` does not
  know fails the ring. A new op needs its fields in `OPS`, `_FIELDS` and
  (for extra fields) `_OPTIONAL`, and a case on the follower side
  (`tensor.follow`, or `ring.apply_cache_op` for cache ops).
- **Ranks run the same collectives.** A row prefilled in different chunk
  counts deadlocks the ring, so `admit` carries the chunk rank 0 fitted
  (`memory_guard._make_room`) and `chunk` refits a row (`_fit_next`).
- **Everything between ranks goes through `Link.exchange`**, on the
  scheduler thread.
- **A VQ codebook is replicated, never sliced** (`check_codebooks`).
  Slicing it decodes against half a codebook and emits fluent garbage.
- **The split rule table is one.** `tensor.shard` applies
  `tensor_rules.RULES` to loaded arrays and `tuning/tensor_split` checks the
  same table against headers, so the refusal and the loader agree.
- **Pipeline: rank 0 holds the last layers.** The logits are born on the
  rank that samples; the MTP head lives on rank 0 alone.

## Extending

- A family's tensor split: rules in `tensor_rules.py`, `tensor` (and
  `tensor_split` for blocked inputs) in its manifest, then the two-rank
  tests.
- A family's pipeline split: a `pipeline` entry naming its trunk core,
  and a `restage` if the trunk froze per-layer indices at `__init__`.
- A new plan op: see the rules above and the op list in
  [prompt-cache](prompt-cache.md).

## Notes

The splits span packages: `engine/runtime/` (tensor, pipeline, plan,
rules, viability), `engine/prompt_cache/ring.py`, `engine/mtp/` (the
pipeline `Coord` is used by the batch loop), `tuning/` (`fit.py`,
`tensor_split.py`, `pipeline_split.py`: fit and refusals), `interfaces/serve.py` (rank launch), `cluster/launch.py`
(placement and argv). Rank progress (steps, prefill
chunks, loaded) goes through `engine/runtime/marker.py`, so engine never
imports cluster; `tests/integration/test_layers.py` enforces it.

## Tests

`tests/engine/test_tensor.py` and `test_pipeline.py` (two-process rings
through `tests/support/tensor_ring_worker.py` and
`pipeline_ring_worker.py`), `test_ring_align.py`, `test_ring_fatal.py`,
`tests/cluster/test_fresh_placement.py`.
