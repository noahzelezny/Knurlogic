> **Parked (2026-09-28).** Not built on main. 41dd038 (guard = startup budget) was reverted; the guard is the working set minus other processes' live GPU use. See docs/dev/PLAN.md OPEN THREADS and memory-ledger.md.

# Memory pacing: a server that cannot OOM (draft 2026-09-28)

Goal: "uncrashable" -- the server paces itself so a user never OOMs,
including when they open Chrome mid-run.

## What failed (M4, Flash-Next VQ 4.4, 4 agents at 50-75k tokens)

- The guard kept under Metal's recommended working set (120 GiB of 128), not
  the 112.3 GiB budget serve printed. Fixed in 41dd038.
- Memory sat at the limit plus a step's transient for 20 minutes; the machine
  swapped 6.6 GiB. At 13:56 a step peaked at 121.6 GiB (over 120) with 8 rows
  and 483k tokens of context. The guard had just measured a 3.02 GiB transient,
  stopped two rows -- after the fact. A Metal OOM is an uncatchable abort: the
  one step the guard mispredicts kills the process.
- Two blind spots: (1) the guard sees GPU memory only (`gpu_in_use`); Chrome
  is CPU memory of the same RAM. (2) the transient is learned by running the
  step, and the margin is the largest seen, carried on a line.

## Design: predict, then fit the step to the room

Each tick, before `_step()`:

1. **Room** = min(allowance if set, live room) - OS floor, where live room =
   this process's active + what macOS can hand over now (free + inactive +
   purgeable, `host_statistics64`, the same figure `load_budget` uses),
   re-read at most every 2 s. OS floor: max(4 GiB, 5% of RAM). The allowance
   stays as the optional fixed cap.
2. **Predict** the next step's transient before it runs, from the step's
   shape: rows, prefill tokens this step, context those tokens attend over.
   Keep the measured line (`_transient`), but key it on (prefill tokens x
   context) + decode rows x context rather than context alone, and take the
   upper envelope with 25% headroom.
3. **Fit the step** if predicted peak > room, in this order, cheapest loss
   first:
   a. give back prompt cache (already done);
   b. shrink the prefill chunk (512 -> 256 -> 128) for this step;
   c. defer prefill: run decode-only this step, admit later;
   d. step a subset of rows (oldest first); the rest wait a step;
   e. only if one row at the minimum chunk does not fit: stop the newest
      request with 503 + Retry-After (never the process).
4. **Measure** the real peak after the step (as now) and correct the model;
   a miss > 10% shrinks the next step's chunk pre-emptively.
5. **Pressure**: macOS `kern.memorystatus_vm_pressure_level` warn/critical,
   or swapouts rising between readings, shrinks room by the margin until it
   clears -- knurlogic backs off when the user opens something, speeds up
   when it passes.

Admission uses the same room: a request is admitted only if its cache plus the
predicted peak of its first prefill chunk at the minimum size fits.

## What remains possible

Another app taking more than the OS floor within one 2 s reading. The floor
is the insurance; the reading interval is the knob.

## Rings

Each rank computes its own room; the ring takes the tightest (the existing
`peers_over_now` exchange) and every rank applies the same step shape, so the
collectives stay in lockstep.

## Proof

The M4 harness (`a local test harness`): 4 streams opening at 100k tokens, growing
to 150k; plus a round that allocates 20 GiB in another process mid-run
(Chrome stand-in). Pass = 3/3 rounds with no METAL abort and no swap growth;
waits and 503s are fine.

## Open

- Whether chunk shrinking alone covers MoE decode transients at 8 rows.
- Cost: pacing trades throughput for safety; measure tokens/s at the limit.

## design review (2026-09-28): build with changes -- adopted

- **A step is a whole admission.** `MTPBatchGenerator._next` runs every
  prefill chunk, checkpoint copy and `seed_head` of one admission inside one
  `ex.step()`. The chunk can shrink per admission (`admit` reads
  `prefill_step_size` at call time; checkpoint cuts and `seed_head` are
  chunk-agnostic), not mid-prompt. The "2 s reading" is really "one step",
  and a 100k admission on 397B is one step of tens of seconds.
- **Drop "step a subset of rows"** (no engine support). Instead **pause the
  newest row**: `extract_cache` -> prompt cache, re-queue the job with its
  tokens so far; it re-admits as a full cache hit. Pause, not kill.
- **Room keeps the wired ceiling**: limit = min(working-set budget - others'
  GPU, allowance, own active + own mlx cache + available RAM - OS floor).
  Metal aborts against `iogpu.wired_limit`, not free RAM. Available RAM via
  `host_statistics64` (ctypes), the MIN of recent readings; pressure level via
  sysctl.
- **Two transient lines**: prefill (chunk x context, steps with an admission)
  and decode (rows x context, steps without). `_measure(admitted=)`.
- **Ladder**: cache -> chunk 512/256/128 (search inside `_room_for`) -> defer
  admission (`_Wait`) -> pause newest row, checked BEFORE `_step` -> 503.
- **Ring**: rank 0's chunk ships in the `admit` journal op (differing chunk
  counts deadlock the collectives); follower `Mark.limit()` computes the same
  RAM-based room.
- **Phase 2**: resumable admission -- K chunks per `next()`, the row stays in
  `_unprocessed_sequences`; room re-read between chunks, decode rows stop
  stalling behind long prompts. Design after phase 1 passes the harness.
- **Harness adds** a 100k admission while another process allocates 20 GiB.

## Phase 1 built (2026-09-28)

- Room: `Scheduler._limit` = min(working set - others' GPU, allowance,
  `RamRoom`) - margin, and a margin more under pressure; RAM from
  `machine/ram.py` (ctypes `host_statistics64`, sysctl pressure level).
  Followers: `tensor.Mark` takes the same `RamRoom`.
- Two lines: `_tx["prefill"]` (x = context x chunk / 512) and
  `_tx["decode"]` (x = rows' contexts summed); `_measure(admitted=, chunk=)`.
- Chunk search in `_make_room` (512 -> 256 -> 128, full before lean), else
  `_Wait`; the chunk rides the `admit` op (`chunk`, plan.py) and is set on
  the engine before the step that prefills that row, on every rank.
- `_pace` before `_step`: pause the newest row (`pause` op on a ring; its
  cache to the prompt cache, the job re-queued with `Job.resume`, re-admitted
  by `_resume` as a hit). The last row is left to `_guard_memory` (503).
- M4 14:22-14:30 (Flash-Next VQ 4.4, 4 x ~106k prompts at once, Metal abort
  at ~116 GiB over a ~112.2 limit, no guard line): (1) all four were
  inserted in ONE tick and each priced against active memory that did not
  yet hold the others' KV (a queued row's KV lands only in its admission
  step); (2) the guard reads memory between steps, and the last climb was
  inside one admission step. Fixed by `_reserve` / `_pending_bytes`: queued
  rows' priced KV is charged in `_room_for`, `_fits`, `_room_to_admit` and
  added to `_pace`'s prediction for the step that admits them.
- What a pause costs (design review): a paused row resumes as ONE segment,
  so segment checkpoints it had not yet stored are lost; its logits
  processors' penalty windows restart from the resumed prompt; the seed, if
  one was set, is folded forward by the tokens made (`seed + made`) so the
  resumed draw continues rather than replays its first tokens (the stream
  is not bit-identical to an unpaused run). A hit -- a resumed row always
  is one -- is priced and reserved past the hit only: its KV is resident.
- M4 (b557e63, 107.1 GiB limit): a lone 106k row prefilled ~8 min and
  `_guard_memory` stopped it at once, every request. The guard stopped rows
  on the margin-reduced limit, a prediction. Now: prompt cache, then other
  rows paused (a queued one deferred), and a lone row is stopped only if
  active + its measured transient passes the hard ceiling (`_ceiling`: no
  margin). Its step's measured context also no longer counts prompts queued
  behind it ("208545 tokens" was 106k + a queued 102k). Why the lone row
  landed over the limit is not settled without the box: the guard never
  read reservations (no double count there); candidates are growth the
  lean price omits (MTP head cache, the hybrid's state priced from config
  before a measurement) and the limit tightening over 8 min (the others'
  largest reading, RAM's least).
- M4 (2584c3a, 1h45, no crash; 8 of 14 requests 503): (1) `_hold` refused
  a request once the rows it found had finished -- with one ~130k row fitting
  at a time, nearly everyone. Now a held request waits first in line and
  blocks newer admissions (anti-starvation); only `_make_room` on an idle
  box refuses. (2) A 136k row stopped right after a 12-min prefill: it had
  made no token yet, so `_pause` treated it as unpaid, and a lone row over
  the ceiling is stopped. Now only rows still queued in the engine are
  deferred; a prefilled row pauses with its cache; `_victim` picks a row
  not yet prefilled first. Under-pricing (~1.3x, 5.8 priced vs ~7.5 grown):
  a config price starts at x1.3 (`CONFIG_SCALE`), and each admission step's
  growth against its price raises the scale (`_learn_growth`, never down,
  capped at 3) -- a lean admission stores no checkpoint, so `_learn` never
  measured those rows.
