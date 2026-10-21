"""The stop: a logits processor that closes the think block once the probe
says the answer is ready.

It runs in the row's processor list, so the batch engine applies it where
it applies every processor (plain decode, and the verify of a drafting
step). After each decode forward the loop hands it the probe's score for
the token that step commits (`feed`); at a paragraph break inside the
thinking, a score at or over the threshold makes the next tokens the
template's think-end, after which the model writes its answer as it would
have. Onboarding uses the same object to keep the state at every break
(`keep_states`) instead of stopping.
"""

from __future__ import annotations

import threading

import mlx.core as mx
import numpy as np

#: tokens whose text holds a paragraph break, per tokenizer (a vocabulary
#: decoded once: about a second for 250k ids)
_BREAKS: dict = {}
_lock = threading.Lock()


def break_ids(tokenizer) -> tuple[frozenset, frozenset, frozenset]:
    """(ids holding "\\n\\n", ids ending in "\\n", ids starting with "\\n"):
    a break is one token holding both newlines, or two tokens meeting at
    them -- the cut the probe was fitted at (docs/design/auto-effort.md)."""
    key = id(tokenizer)
    with _lock:
        hit = _BREAKS.get(key)
        if hit is not None and hit[0] is tokenizer:
            return hit[1]
        both, ends, starts = set(), set(), set()
        for i in range(len(tokenizer.vocab) if hasattr(tokenizer, "vocab")
                       else tokenizer.vocab_size):
            t = tokenizer.decode([i])
            if "\n\n" in t:
                both.add(i)
            if t.endswith("\n"):
                ends.add(i)
            if t.startswith("\n"):
                starts.add(i)
        sets = (frozenset(both), frozenset(ends), frozenset(starts))
        _BREAKS[key] = (tokenizer, sets)
        return sets


class AutoStop:
    """One request's stop. `threshold` None: never stop (onboarding)."""

    def __init__(self, tokenizer, *, reasoning: bool,
                 threshold: float | None, report: dict | None = None,
                 keep_states: bool = False):
        self.breaks, self.nl_end, self.nl_start = break_ids(tokenizer)
        self.start = tuple(getattr(tokenizer, "think_start_tokens", ()) or ())
        self.end = tuple(getattr(tokenizer, "think_end_tokens", ()) or ())
        self.inside = reasoning
        self.threshold = threshold
        self.report = report if report is not None else {}
        self.keep_states = keep_states
        #: (history length at the break, states [L, D]) -- onboarding's
        self.states: list[tuple[int, np.ndarray]] = []
        #: every token the row has emitted, as the processor saw them
        self.tokens: list[int] = []
        #: history length -> (score, state) read by the loop, lazy
        self._reads: dict[int, tuple] = {}
        self._checked = -1
        self._forced_from: int | None = None
        self.report.update(stopped=False, breaks_read=0, max_confidence=None)

    def feed(self, at: int, score: mx.array | None,
             state: mx.array | None) -> None:
        """From the loop, after a decode forward: the probe's score (and
        the states, when onboarding) for the token that makes the history
        `at` long. A drafting step feeds two."""
        self._reads = {k: v for k, v in self._reads.items() if k >= at - 1}
        self._reads[at] = (score, state)

    def __call__(self, hist: mx.array, logits: mx.array) -> mx.array:
        n = int(hist.size)
        if n > len(self.tokens):
            self._track(hist[len(self.tokens):].tolist())
        if self._forced_from is not None:
            k = n - self._forced_from
            return _only(logits, self.end[k]) if k < len(self.end) else logits
        if self.inside and n in self._reads and n != self._checked \
                and self._at_break():
            self._checked = n
            self._read(n)
            if self._forced_from is not None:
                return _only(logits, self.end[0])
        return logits

    def _track(self, new: list[int]) -> None:
        for t in new:
            self.tokens.append(t)
            if self.end and tuple(self.tokens[-len(self.end):]) == self.end:
                self.inside = False
            elif self.start and tuple(self.tokens[-len(self.start):]) == self.start:
                self.inside = True

    def _at_break(self) -> bool:
        last = self.tokens[-1]
        return last in self.breaks or (
            len(self.tokens) > 1 and last in self.nl_start
            and self.tokens[-2] in self.nl_end)

    def _read(self, n: int) -> None:
        score, state = self._reads.pop(n)
        if self.keep_states and state is not None:
            self.states.append((n, np.array(state)))
        if score is None:
            return
        p = float(score.item())
        r = self.report
        r["breaks_read"] += 1
        r["max_confidence"] = max(p, r["max_confidence"] or 0.0)
        if self.threshold is not None and self.end and p >= self.threshold:
            self._forced_from = n
            r.update(stopped=True, confidence=round(p, 4),
                     stopped_at_token=n)


def _only(logits: mx.array, token: int) -> mx.array:
    """`logits` with every token but `token` impossible."""
    keep = mx.arange(logits.shape[-1]) == token
    return mx.where(keep, logits, mx.array(-float("inf"), logits.dtype))
