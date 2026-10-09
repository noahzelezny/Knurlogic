"""usage.knurlogic.timing: what one request took (`rates`: queue, TTFT,
the prefill and decode rates, the telemetry contract's *_ms and *_tps)
and where its time went (`Spans`: a partition of its wall time).

`usage.knurlogic.timing` already gives the rates (TTFT, prefill tok/s,
decode tok/s). A rate says how fast; it does not say where the rest of the
time went. When a served prefill runs at well under the rate of a plain
forward of the same model, the missing time is somewhere between the HTTP
handler and the first token, and a rate cannot point at it.

This is a CURSOR, not a set of timers. Each boundary the request crosses
charges the time since the previous boundary to one named bucket and moves
the cursor, so the buckets sum to the time from the first mark to the last
BY CONSTRUCTION -- nothing is double-counted and nothing falls between two
timers. What a partition cannot do is split time two requests share: a
step that decodes three rows is charged in full to each of them, because
each of them waited for all of it. Batching shows up as its own bucket
(`decode_forward_shared`), not as a discount.

Host clocks only: no mx.eval is added and nothing is synchronized, so
turning this on does not change what the GPU runs. Off with
KNURLOGIC_TIMING_SPANS=off.
"""
from __future__ import annotations

import os
import time


def enabled() -> bool:
    """Read live, so a running server can be switched without a restart."""
    return os.environ.get("KNURLOGIC_TIMING_SPANS", "on") != "off"


class Spans:
    """A cursor over one request's wall time, charged bucket by bucket."""

    __slots__ = ("start", "last", "buckets")

    def __init__(self, start: float | None = None):
        self.start = time.perf_counter() if start is None else start
        self.last = self.start
        self.buckets: dict[str, float] = {}

    def to(self, bucket: str, now: float | None = None) -> float:
        """Charge the time since the last mark to `bucket`; move the mark.
        A clock that went backwards charges 0 rather than a negative."""
        now = time.perf_counter() if now is None else now
        dt = max(now - self.last, 0.0)
        self.buckets[bucket] = self.buckets.get(bucket, 0.0) + dt
        self.last = max(now, self.last)
        return dt

    def report(self, end: float | None = None) -> dict:
        """The buckets, the whole, and what the buckets do not cover
        (0 unless a mark was skipped: a check on the instrument, not on
        the request)."""
        end = time.perf_counter() if end is None else end
        total = sum(self.buckets.values())
        whole = max(end - self.start, 0.0)
        return {"spans_s": {k: round(v, 4) for k, v in
                            sorted(self.buckets.items(), key=lambda kv: -kv[1])},
                "spans_whole_s": round(whole, 4),
                "spans_unaccounted_s": round(whole - total, 4)}


def step_bucket(prefilling: bool, part: str, shared: bool = False) -> str:
    """The name a step's time is charged under for one row.

    part: "gap" (scheduler work between steps: memory guard, chunk refit,
    admitting others), "forward" (the executor's step: the model, sampling
    and, on a ring, the collectives) or "host" (after the step: detokenize,
    stop strings, handing deltas to the HTTP thread). `shared` marks a step
    that admitted ANOTHER request: for a decoding row, a step slowed by
    someone else's prompt; for a row still waiting its own turn (the
    executor admits one row per step), a step it spent in line."""
    phase = "prefill" if prefilling else "decode"
    if shared and part == "forward":
        return "prefill_waiting" if prefilling else "decode_forward_shared"
    return f"{phase}_{part}"


# Under this much fresh prefill a rate is noise (the page's own floor):
# nothing is sent, so nothing is shown.
PREFILL_MIN_TOKENS = 256
PREFILL_MIN_S = 0.05


def rates(row, done: float, completion: int, prefilled,
          cached: int | None = 0, chunk: int | None = None) -> dict:
    """What the request took, measured here where the steps run: time in
    the queue, time to first token (from submit, as a client feels it), and
    the rates of the two phases. Prefill is from admission to the first
    token, over the tokens actually prefilled (not the ones the prompt
    cache supplied); decode is over the tokens after the first."""
    first = row.first or done
    out = {"queue_s": round(max(row.admitted - row.job.submitted, 0), 4),
           "ttft_s": round(max(first - row.job.submitted, 0), 4),
           "prompt_cached_tokens": int(cached or 0),
           "prompt_computed_tokens": int(prefilled or 0)}
    if chunk:
        out["prefill_chunk"] = int(chunk)
    if cached and not prefilled:
        out["prefill"] = "cached"    # nothing was computed: no rate exists
    # the compute only: from the step that began the prefill, not from the
    # admission (which also counts waiting for the scheduler's turn)
    pre = first - (getattr(row, "began", 0.0) or row.admitted)
    if (prefilled or 0) >= PREFILL_MIN_TOKENS and pre >= PREFILL_MIN_S:
        out["prefill_tok_s"] = round(prefilled / pre, 1)
    dec = done - first
    if completion > 1 and dec > 0:
        out["decode_tok_s"] = round((completion - 1) / dec, 1)
    # the telemetry contract's names (docs/design/telemetry.md): submitted
    # -> admitted, admitted -> first token, first token -> finish
    out["queue_ms"] = round(out["queue_s"] * 1000, 1)
    out["prefill_ms"] = round(max(first - row.admitted, 0) * 1000, 1)
    out["decode_ms"] = round(max(dec, 0) * 1000, 1)
    if "prefill_tok_s" in out:
        out["prefill_tps"] = out["prefill_tok_s"]
    if "decode_tok_s" in out:
        out["decode_tps"] = out["decode_tok_s"]
    return out
