# Memory ledger (design, 2026-09-28) -- replaces measured-memory pacing

Goal: knurlogic never crashes from memory. Whatever agents send, however many
at once, whatever else the user opens, the worst outcome is a wait.

Done when: (1) a simulator proves the ledger's invariants; (2) a few-minute
calibration per model family shows the ledger matches real memory; (3) one
time-capped soak on the M4 (big model, long agents, another app grabbing
memory) passes with no crash.

## Why the measured-memory guard failed

It controlled on mlx's active memory against limits and margins. That signal
is noisy (in-step transients, buffer cache, lazy allocation), lags (a 2-minute
prefill is one step) and only reports a mistake after it is made. Every live
bug in memory-pacing phase 1 came from it; each fix added a heuristic
(learned prices, x1.3, growth ratio, min-of-readings, reservations).

review also found what the "unexplained" growth was: `BatchKVCache` is one
dense `[B, H, Lmax, D]` tensor per layer. A 1k row beside a 130k row costs
130k of full-attention KV, and growth by `concatenate` in 256-token steps
briefly holds old + new -- a transient the size of the whole batch KV.

## The ledger

- Budget = min(wired ceiling - other processes' GPU, allowance, own + available
  RAM - OS floor). Inputs from machine/ram.py; macOS pressure shrinks it.
- Booked once: weights, image store.
- The batch is the unit: full-attention bytes/token x B x Lmax (rounded up to
  256), plus each row's fixed state (linear/SSM, sliding window, MTP head).
- Prices are exact, derived from the model's own caches (`make_cache()` for
  one row, `nbytes` at 256 and 512 positions), not learned. Covers qwen3_5 /
  moe hybrids, qwen4_exp (+ indexer keys, MTP), gemma4 sliding, glm5_next
  MLA, 8-bit KV.
- Incremental booking: a row books prompt + the next 256 of growth; before
  each step the next boundary is booked. Booking to max_tokens would admit
  one agent at a time (max_tokens 32k-128k).
- Step transient, booked per step: max(prefill = k_model x chunk x context,
  growth = one copy of the batch KV at a 256 boundary; merge copy at
  admission). k_model calibrated once per family at chunk 512, 32k context.
- Admission is arithmetic: book or wait (FIFO, a held request blocks newer
  ones). Refuse up front only what an empty ledger could not book.
- Budget shrinks (Chrome, Gemma): evict prompt cache, then pause the newest
  unpaid row, then the newest row. A paused row's cache stays on the row
  (booked, resident) -- never a round trip through the prompt cache, which
  deep-copies.
- Measured active memory is a drift alarm only: log active - ledger per step;
  persistent drift tightens the budget and is a bug to fix.
- Ring: rank 0 owns the ledger; ranks' shares follow from their shard (tensor:
  equal; pipeline: per-stage layers). Followers report only budget changes.
  The chosen chunk rides the admit op (kept from phase 1).

## Kept from memory-pacing / deleted

Kept: machine/ram.py (RamRoom, pressure, OS floor), chunk in the admit op,
pause/resume mechanics (cache kept on the row), victim choice, FIFO hold.
Deleted: `_tx` lines, `_spike`, `_learn`, `_learn_growth`, CONFIG_SCALE,
reservations, the measured-room search in `_fits`/`_room_for`/`_make_room`,
`_guard_memory` as a controller.

## Testing

1. Pure-Python ledger + fake engine with exact batch-shaped nbytes; property
   tests: booked <= budget always; bookings == fake nbytes after every op
   (admit, boundary, pause, resume, finish, cancel, evict, shrink); FIFO, no
   starvation under random arrivals; no pause/resume cycle without a budget
   change. Seconds, no model.
2. Per-family calibration on the M4, minutes each: cache nbytes match the
   formula; one 32k prefill for k_model; measured peak <= ledger.
3. One soak on the M4, time-capped, pass criteria written first.

## Open

- Dense padding makes mixed-length batches expensive (a short agent beside a
  long one pays the long one's KV). A ragged or paged batch cache fixes
  throughput; the ledger is correct without it. Decide after the ledger.
- k_model per family: only a run settles it.
