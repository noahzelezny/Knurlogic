# Building on drafting (MTP)

Many artifacts ship a multi-token-prediction head that mlx-lm never runs.
knurlogic drafts with it: the head proposes, the trunk verifies, inside
the same batch engine that serves every request. The why, the batched
loop and the registry: [drafting](../design/drafting.md); drafting on a
split model: [tensor-mtp](../design/tensor-mtp.md).

## Where the code is

`src/knurlogic/engine/mtp/`:

| file | what |
|---|---|
| `__init__.py` | re-exports `_artifacts`; importing the package imports no mlx |
| `_artifacts.py` | what an artifact has: `find_head`, `Head`, `status`, `graft_weights`, `head_layouts`. Stdlib only (reads safetensors headers) |
| `registry.py` | `FamilySpec` (head, capture, draft_cache, cache_semantics, block), `register`, `resolve`, `load_head`; built from the manifests' `head` entries |
| `batch_generator.py` | `MTPBatchGenerator`: mlx-lm's `BatchGenerator` backed by `MTPBatch`, the engine with or without a head; `PrefillCancelled` |
| `batch_loop.py` | `MTPBatch`: draft one token, verify in a 2-token trunk forward, for B rows; `admit`; per-row regime chosen by measured cost (`KNURLOGIC_MTP_DYNAMIC`) |
| `block_loop.py` | `BlockBatch`: a head that drafts K tokens in one pass (DeepSeek-V4's DSpark), verified in one K+1-wide forward |
| `caches.py` | snapshot and rollback for a speculative step: `snapshot`, `restore`, `rollback`, `release`, `check_snapshot_semantics` |
| `capture.py` | capture the pre-lm_head activation (`capture_input`) |
| `sampling.py` | the same distribution mlx-lm samples from, exact rejection sampling (`rejection_correct`), `NonFiniteLogits` |
| `seed.py` | seed a head over a prompt in prefill-sized chunks (`seed_head`) |

Elsewhere:

- `engine/families/<family>/heads/`: the head classes; the manifest's
  `head` entry names them.
- `engine/serve/drafting.py`: `load_head` binds an artifact's head to the
  loaded model; `drafting_status`. State lives in `engine/serve/state.DRAFT`.
- `engine/runtime/scheduler.py` (`_executor_local`, `_executor`): builds
  the `MTPBatchGenerator` with the head when one is bound.
- `engine/runtime/pipeline.py` (`Coord`): rank 0's drafts and verdicts
  broadcast to followers.
- `interfaces/drafting.py`: `knurlogic mtp`, the machine-wide survey;
  `interfaces/mcp.py`: the `drafting` tool.
- `tuning/presets.py`: `MTP_MODE`, `MTP_MODES`; `tuning/knobs.py`:
  `mtp_of`, the `KNURLOGIC_MTP*` knobs; `tuning/fit.py`: `mtp_head_bytes`.

## Rules that keep it correct

- **Output is exact.** Sampling uses rejection with residual correction,
  so drafting changes speed, never the distribution; at temperature 0 a
  draft is accepted only if it equals the argmax.
- **Every row advances two positions per step**, accepted or not, so the
  batch never trims per row: a rejection is one whole-batch rollback.
  Caches that cannot roll forward (recurrent state) restore and replay.
- **`cache_semantics` is measured.** A new family starts at `"copy"`; use
  `"reassign"` only after `caches.check_snapshot_semantics` passes on a
  loaded model. `"copy"` on recurrent caches is correct but very slow.
- **No measured acceptance, no registration.**
- **Nothing in `engine/mtp/` names an architecture.** Family facts are in
  the manifest and the head class.
- **On a split, only rank 0 holds the head.**

## Adding a head

A `head` entry in the family's manifest (names, head `"module:Class"`
with `from_sidecar` and `draft_logits`, capture point, draft cache,
cache semantics, sidecar name, layout) and the class in `heads/`. A block
drafter adds `block`. The checklist item is
[new-model](../design/new-model.md), "Drafting".

## Notes

Drafting spans `engine/mtp/` (the engine and registry),
`engine/families/*/heads/` (by design), `engine/serve/drafting.py` and
`state.py` (binding), `interfaces/drafting.py` (the survey) and
`tuning/knobs.py` (the knobs).

## Tests

`tests/engine/test_mtp.py`, `test_batch_drafting.py`,
`test_mtp_cache_rollback.py`, `test_mtp_capture.py`,
`test_mtp_regime_backoff.py`, `test_deepseek_v4_mtp.py`,
`test_deepseek_v4_dspark.py`, `test_deepseek_v4_rollback.py`,
`test_glm5_side_state.py`.
