# MTP drafting on a tensor split

Status: plan, nothing built. Goal: a tensor-split cluster job drafts with
the model's MTP head the way a pipeline job does today. The existing MTP
on/off launch option (`launch["mtp"]`, `--no-draft`, `KNURLOGIC_MTP`) and
the dynamic switch (`mtp_dynamic`) apply unchanged. No new settings, no UI.

Measured (M3 Ultra + M4 Max, TCP over TB4, Qwen3.5-397B VQ-2.4bpw):
pipeline + MTP 33-36 tok/s; tensor without MTP ~15-22 tok/s. Everything
else below marked *guess* is not measured.

## Correction to the proposed shape

The proposal said "rank 0 drafts K tokens". The batch loop drafts **one**
token per step and verifies it in a **2-wide** forward
(`engine/mtp/batch_loop.py`, `MTPBatch._draft_step`): every row commits
exactly two tokens per step (t1, then the draft or the trunk's correction),
and a rejection costs a cache `restore` plus a second 2-wide replay forward.
That holds for pipeline too. K > 1 is a batch-loop change for both splits
and is out of scope here. The rest of the proposed shape holds, with one
change: drafts and verdicts travel in the Coord broadcasts the pipeline
already uses (B1/B2), not in the plan. See section 2.

## 1. Why tensor refuses MTP today

Nothing in the engine stops it. The refusal is plain gating in five places,
all written as "pipeline only":

| Where | What |
|---|---|
| `interfaces/serve.py` ~l.290, tensor branch | `draft = False` (hence the rank-0 log line "drafting head present, off", l.662) |
| `interfaces/http/__init__.py` l.168/175 | `agree = T.agree_head(link)` only `if pipe`; `ModelHost(draft=draft and (not ring or pipe))` |
| `engine/runtime/tensor.py` `serve_follower` l.860 | `heads = agree_head(link) if split == "pipeline" else None` |
| `engine/runtime/scheduler.py` `_executor` l.661-678 | `head = ... if (pipe and DRAFT["on"])`; `coordinate(gen, group)` only `if pipe` |
| `engine/runtime/tensor.py` `TensorExecutor.__init__` and `follow()` | `_admission_coord` installs `Coord(group)` with `head=False` (B0 only) instead of `pipeline.coordinate` |

The scheduler comment gives the original reason: "a drafting head on a
pipeline only (rank 0 holds the last layers, so the true final hidden
state)". Checked against the code, that reason does not hold for tensor:

- **Hidden state on rank 0.** `tensor.shard` wraps every layer's attention
  and MLP in `Reduce` (fp32 `all_sum`), so each layer's output, and
  therefore the final hidden state the head reads (`get_h`, `h_source`
  pre/post norm), is whole and identical on every rank. Rank 0 has it.
- **The head's own layer.** `MTPHeadQwen35` builds its own
  `arch.DecoderLayer(args, fa_idx)` from `args`; `shard()` only walks
  `model.layers` and mutates per-instance head counts, never `args`. The head
  block stays whole and runs on rank 0 with no collectives. `embed_tokens`,
  the final norm and `lm_head` are not in `RULES`, so they are replicated
  whole on every rank, which is what the head reuses. *To verify on the
  real artifact:* the q6 head sidecar's loader (`mtp/registry.load_head`)
  must not pick up `load_config`'s `vq_skipzero.shard` (rank 0 of 2) and
  build only half its rows. The tiny test cannot show this.
- **Rollback on both ranks.** Already done by the batch loop on every rank:
  `snapshot` / `restore` (`mtp/caches.py`) plus the replay forward, keyed on
  B2's flags. See section 3.
- **Dynamic MTP.** `drafting_pays` runs on rank 0 only (a follower has no
  head and gets False). B1 carries rank 0's decision, so the follower
  follows it. Nothing new.
- **Several rows.** B1 carries `[drafting] + d2[B]`, B2 carries
  `ok[B] + t2[B]`, with `live` per row coming from BA at admission. All of it
  is per-row already.

Real blockers found: none. Risks are the SKIPZERO head-load check above and
fit (section 4).

## 2. Data flow per step

Under tensor every rank computes full logits and samples, and rank 0's
tokens overwrite the follower's through the plan (`follow()`: "token(s)
differ from rank 0's; rank 0's are used"). With drafting on, one decode
step on both ranks is:

1. Plan exchange (unchanged): `TensorExecutor.step` sends `ops` +
   `tokens` (each row's t1). The follower overwrites `b.t1`.
2. **B1** (`Coord.b1`, a CPU all_gather): rank 0's `drafting_pays` verdict
   and `d2[B]`, drafted at the end of the previous step from `draft_row`.
3. Both ranks run the 2-wide verify `model([t1, d2])` through the
   tensor ring. Each layer's two all_sums now carry 2 positions instead of 1.
4. Rank 0 judges (`judge = coord.leader`). The follower's logits are real
   but unused.
5. **B2**: `ok[B] + t2[B]`. If any row rejected, both ranks `restore` and
   replay `[t1, t2]`, which is a second ring forward.
6. Rank 0 runs the head on the new hidden state to make the next `draft_row`.
   Rank 1 waits for that at the next B1.

**Plan-op changes: none.** One could fold d2 into the plan, since it is
known before the step, and save the B1 all_gather. Verdicts cannot travel
in the plan. They are needed mid-step to decide the replay, and the
follower must not judge for itself, because its fp32-reduced logits are
not guaranteed bit-equal to rank 0's. A different verdict would mean a
different replay, then a collective-count mismatch, then a hang. Reusing
B1/B2 exactly as pipeline does is the smaller change. Folding d2 into the
plan is a later optimisation, only worth doing if section 8 shows B1 costs
something measurable.

At admission, **BA** (`Coord.ba`, rank 0's `hit` and `drafts`) is made on
every rank once `Coord.head` is True. It covers the case the pipeline test
`test_a_prefix_only_the_follower_could_use_is_prefilled_on_every_rank`
covers: rank 0 drops a cache hit that has no head cache.

## 3. Cache rollback on rank 1

Nothing new is needed. The follower runs the same `MTPBatch._draft_step`
as rank 0: `snapshot(self.cache)` before the verify, then `restore` and the
replay when `not all(ok_flags)`, with `ok_flags` and `t2` taken from B2.
Its sharded KV slices and its linear-attention (GatedDeltaNet) state
slices are restored just like the pipeline follower's layer run. Because
the replay is itself a ring forward, both ranks must agree on whether to
replay. B2 guarantees that, and it is the reason the follower never
decides on its own. The prompt-cache `insert` ops in the plan carry the
committed caches, so they are unaffected.

## 4. Memory and fit

- Rank 0 binds the head: the `mtp-head*.safetensors` sidecars, 5.41 GiB for
  `mtp-head-q6` on Qwen3.5-397B, plus a per-row head KV cache (one
  full-attention layer, small). Rank 1 binds nothing new (`ModelHost(draft=False)`
  stays).
- The verify forward is 2 tokens wide. Activation transients double, which is
  small next to the step margin (*guess*). The `snapshot` copy is the same
  as on pipeline.
- Fit today ignores the head under tensor: `resolve.trunk_headers` drops
  `mtp*` files, and `cluster/launch.py` `artifact_shape` returns no
  `leader_bytes` for tensor. Changes:
  - `launch.artifact_shape` (tensor branch): add
    `"leader_bytes": R.pipeline_leader_bytes(a, vision=vision)` when MTP is on.
  - `launch.placement` (tensor branch): rank 0's `per` becomes
    `per + lead` in both the fit check and `shares[0]["bytes"]`. The reason
    string names the head.
  - `launch.py` ~l.816 (per-rank refusal): add `leader_bytes` for rank 0
    under tensor, as the pipeline branch does.
  - `pipeline_leader_bytes` probably wants a neutral name (`leader_bytes`),
    since it now serves both splits. That is a rename only.
  - Under tensor the rank sizes are equal, so the head does not move work
    between ranks the way it shifts layers on a pipeline. It only raises
    rank 0's floor. With MTP off nothing changes.

## 5. Reuse versus new code

Reused as-is: `pipeline.Coord` (B0/BA/B1/B2), `pipeline.coordinate`,
`tensor.agree_head`, `ModelHost.head_agree`, the whole of
`MTPBatch.step` / `_draft_step` / `drafting_pays`, `MTPBatchGenerator`
admission with `coord.ba`, and the follower's `follow()` loop and plan.

What changes is gating and wiring, roughly 30-60 lines (*guess*):
- `tensor.follow`: in the `else` branch call
  `PL.coordinate(gen, link.group, drafting=drafting)` instead of
  `_admission_coord`. No `silence`: the follower's logits are real.
- `TensorExecutor.__init__`: leave a Coord the scheduler already installed
  in place (`_admission_coord` already skips one that exists).
- `scheduler._executor`: drop `pipe and` from the head condition, and call
  `coordinate()` for both splits.
- `serve_follower` and `http/__init__`: call `agree_head` for both splits;
  `ModelHost(draft=draft and bool(ring) ...)` on rank 0.
- `interfaces/serve.py`: remove the tensor `draft = False`.
- Docstrings: `agree_head`, `Coord`, and `_admission_coord` say "pipeline".
- Fit: section 4.

## 6. Failure modes

- **Collective-count desync.** Rank 0 and the follower make different
  numbers of B1/B2 calls only if they disagree on `Coord.head`.
  `agree_head` all_gathers rank 0's "bound a head" after load, and a head
  that fails to bind on rank 0 turns drafting off on every rank. The
  `calls` counters let the ring test assert equality, as the pipeline test
  does.
- **Logit disagreement between ranks.** This is harmless by design: only
  rank 0 judges, and B2 tells the follower. Today's token mismatch warning
  in `follow()` still covers t1.
- **A row failing on one rank** (`ForwardFailed`, admission failure). This
  takes the existing B0 / `Coord.diverged` path, already exercised with
  drafting on pipeline. The tensor parametrisation of
  `test_a_row_failing_on_rank_0_only_fails_that_row` runs without a head
  today. Extend it (section 7).
- **A rank dying mid-verify.** An `all_sum` inside `Reduce` or a Coord
  all_gather blocks or raises on the survivor, exactly as in a plain tensor
  step today. Recovery is the job-level path (`cluster/recovery.py`).
  Nothing new: a verify step holds no state outside the rank's own
  caches.

## 7. Test plan (existing two-process ring harness)

All of these go in `tests/support/pipeline_ring_worker.py`, driven from
`tests/engine/test_pipeline.py` `_ring(...)`. The model is
`_tiny_with_head` (tiny qwen3_5 in float32, 4 attention / 2 KV heads,
splits 2 ways under `RULES`), sharded with `T.shard` after the head is
built.

1. `mtp(link, out_path, always, split_kind)`: add `split_kind="tensor"`.
   The baseline is the same tensor split with **no head on either rank**
   (`Coord(head=False)`). The drafted run has rank 0 holding the head and
   `coordinate(gen, group, drafting=True)` on both ranks, with no `silence`.
   New test `test_mtp_drafting_on_a_tensor_split_is_the_undrafted_split`,
   parametrised over `always`:
   - `split == baseline` token for token (greedy);
   - the follower's `[b0, b1, b2, steps, ba]` equals rank 0's;
   - `0 < b2 <= b1 <= steps` and `drafted > 0`;
   - with `always`, `b2 == b1` and `accepted > 0`.

   Also assert against the unsplit, undrafted engine on rank 0, as the
   pipeline test does. If fp32-reduced logits flip a greedy tie on the tiny
   model, keep only the split-vs-split equality and say so in the test.
2. `engine(..., split_kind="tensor")` with drafting: give `_tensor_engine`
   the head on rank 0 and `drafting=True` on the follower's `T.follow`.
   This exercises the real `TensorExecutor` plan path with segments and
   checkpoints. New test
   `test_the_serving_path_follows_rank_0s_plan_on_a_drafting_tensor_split`:
   rank 0's tokens must equal the unsplit `LocalExecutor` with the head.
3. `hit(...)` with `split="tensor"`: the headless-checkpoint prefix case,
   with the same assertion as the pipeline test.
4. `test_a_row_failing_on_rank_0_only_fails_that_row`: add a drafting
   variant for tensor.
5. Unit (no ring), in `tests/engine/test_tensor.py`: `placement` puts
   `leader_bytes` on rank 0 only under tensor, and a head that does not fit
   rank 0 is refused with the arithmetic. `artifact_shape` returns
   `leader_bytes` for tensor.

Run only these focused tests per step. Run the full suite once per batch,
before deploy.

## 8. Measurement plan

Same pair (M4 Max rank 0, the leader; M3 Ultra rank 1; TCP/TB4), Qwen3.5-397B VQ-2.4bpw, the
same 4 prompts, batch 1, `max_tokens` capped at 256, greedy and default
sampling:

| Run | Setting |
|---|---|
| A | tensor, MTP off (the baseline, rerun the same night) |
| B | tensor, MTP on, dynamic |
| C | tensor, MTP on, `KNURLOGIC_MTP_DYNAMIC=off` (always draft) |
| D | pipeline + MTP (reference) |

For each run, record tok/s, acceptance (`state.DRAFT` accepted/steps), the
`Coord.calls` counts, and the median step wall time split into: 1-wide
forward, 2-wide verify, replay, head draft on rank 0, and the B1/B2
all_gathers. The split is cheap timers on rank 0 behind a debug env var,
removed after.

The claim to check: with A's step time T, C should approach
`2 / (2 - a) * T`-normalised tok/s. Each step commits 2 tokens and costs
1 + (1 - a) ring forwards plus the head and two small all_gathers.
At a = 0.85 (*guess*: GLM's single-machine figure, not Qwen on a cluster),
that is about 1.7x, so ~25-37 tok/s from 15-22 (*guess*).

The thesis is that a 2-wide forward costs about the same as a 1-wide one
when latency-bound (~2 all_sums × 60 layers per forward). Check it
directly as the A vs C forward-time ratio. If the replay forward or the
rank-0-only head draft (rank 1 idles) eat the margin, B's dynamic switch
should fall back to plain steps, and B ≈ A shows that.

## 9. Effort, in independently landable steps

1. **Engine wiring plus ring test 1** (sections 5 and 7.1): follow,
   scheduler, TensorExecutor, agree_head on both splits. ~0.5 day.
   Independently testable on one machine through the 127.0.0.1 ring.
2. **Serving path** (tests 7.2-7.4): the `serve_follower` / `http`
   `agree_head` and `ModelHost(draft=...)` gating, and the serve.py
   `draft = False` removal. ~0.5 day.
3. **Fit** (section 4, test 7.5): `artifact_shape`, `placement`, the
   per-rank refusal. ~0.5 day.
4. **Live check** (section 8) plus the SKIPZERO head-load check (section 1)
   on the real artifact. One capped run per condition. ~0.5 day, done by
   Noah or with him present.

Total ~2 days (*guess*).

## 10. Open questions for Noah

1. Does the q6 head sidecar load correctly on a tensor rank 0 whose config
   carries `vq_skipzero.shard`? This needs a look at the real 397B
   artifact.
2. Fit under tensor already omits the vision tower on rank 0
   (`trunk_headers` drops it, though rank 0 encodes). Should `leader_bytes`
   fix both at once, or the head only?
3. Should d2 be folded into the plan to save B1's all_gather, or should we
   keep B1 for parity with pipeline? The proposal is to keep it unless
   section 8 shows it matters.
4. 3+ rank tensor jobs: the same code applies (Coord is N-rank), but only
   2 ranks will be tested. Is that acceptable for now?
