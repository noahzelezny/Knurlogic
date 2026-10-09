"""The executor: the seam between knurlogic's scheduler and whatever runs
the model's steps (docs/design/server.md).

A scheduler hands the executor admissions and asks it for steps; the
executor answers with events. The local batch engine (MTPBatchGenerator,
drafting or not, with vision) is the executor; a tensor split runs the
same executor on every rank. Nothing here assumes the layers run in this
process. Rules the executor keeps:

  * A token event carries the token and ITS logprob (plus top-k when
    asked), never a [V] row: what crosses a process boundary stays small.
  * A failure is a `RowFailure` event for that row, not an exception; the
    executor has already dropped the row.
  * Admission carries the request's sampling parameters and the object
    that receives its cache report -- no thread-local, no tagged sampler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Admission:
    """One request, tokenized, as the executor admits it."""

    #: prompt tokens by segment (system / user / ...); the prompt cache
    #: stores a checkpoint at the end of each segment but the last
    segments: list[list[int]]
    max_tokens: int
    #: the prompt cache entry restored for this request and the tokens it
    #: covers (a prefix of the prompt); empty for a fresh prefill
    cache: list | None = None
    prefix: list[int] = field(default_factory=list)
    #: engine/mtp/sampling.make_distribution's kwargs (temp, top_p,
    #: top_k, min_p); empty = greedy
    sampling: dict = field(default_factory=dict)
    processors: list = field(default_factory=list)
    #: the control-token machine (runtime/control_tokens.ControlMachine), or None
    #: for the engine's default
    state_machine: Any = None
    top_logprobs: int = 0
    #: receives the cache report (engine/prompt_cache/report.attach)
    report: Any = None
    #: on a tensor ring, what the other ranks need to rebuild what is not
    #: data here (processors, the state machine): {"penalties": the
    #: make_logits_processors kwargs, "initial": the machine's start state}
    wire: dict | None = None
    #: the prefill chunk the scheduler fitted this admission at (0 = the
    #: engine's own); on a ring it rides the admit op, as ranks prefilling
    #: in different chunk counts deadlock in the collectives
    chunk: int = 0
    #: (uid, done, total) -> stop?, after every prefill chunk of this
    #: admission (Scheduler._prefill_hook): progress, and True to stop the
    #: prefill -- honoured on a single machine only (MTPBatchGenerator)
    on_chunk: Any = None


@dataclass
class Progress:
    uid: int
    done: int
    total: int


@dataclass
class Checkpoint:
    """A segment ended inside the prompt: its cache, for the prompt cache."""
    uid: int
    tokens: list[int]
    cache: list


@dataclass
class Token:
    uid: int
    token: int
    logprob: float
    finish: str | None = None
    #: the control-token state machine's state and any sequence it matched
    state: str | None = None
    match: list[int] | None = None
    top_logprobs: list[tuple] | None = None


@dataclass
class Finished:
    """After a row's finishing Token: its cache, keyed by every token it
    holds, for the prompt cache."""
    uid: int
    tokens: list[int]
    cache: list


@dataclass
class RowFailure:
    uid: int
    error: BaseException


Event = Progress | Checkpoint | Token | Finished | RowFailure


class LocalExecutor:
    """The batch engine in this process."""

    def __init__(self, generator):
        self.gen = generator
        self._top: dict = {}

    def insert(self, a: Admission) -> int:
        (uid,) = self.gen.insert_segments(
            segments=[a.segments], max_tokens=[a.max_tokens],
            caches=[a.cache], all_tokens=[list(a.prefix)],
            samplers=[dict(a.sampling)], logits_processors=[list(a.processors)],
            control=[a.state_machine] if a.state_machine else None,
            reports=[a.report])
        if a.on_chunk is not None and \
                hasattr(self.gen, "prefill_hooks"):
            self.gen.prefill_hooks[uid] = a.on_chunk
        if a.top_logprobs:
            self._top[uid] = a.top_logprobs
        return uid

    def step(self) -> list[Event]:
        prompt, gen = self.gen.next()
        out: list[Event] = []
        failed, ends = [], []
        for r in prompt:
            if isinstance(r.progress, BaseException):
                out.append(RowFailure(r.uid, r.progress))
                failed.append(r.uid)
                continue
            if r.uid in failed:
                continue
            out.append(Progress(r.uid, *r.progress))
            if r.end_of_segment and not r.end_of_prompt:
                ends.append(r.uid)
        for uid, (cache, key) in self.gen.extract_cache(ends).items():
            out.append(Checkpoint(uid, list(key), cache))
        if failed:
            self.remove(failed)
        for r in gen:
            k = self._top.get(r.uid, 0)
            out.append(Token(r.uid, r.token, r.logprobs[r.token].item(),
                             r.finish_reason, r.current_state,
                             r.match_sequence, _top(r.logprobs, k)))
            if r.finish_reason is not None:
                out.append(Finished(r.uid, list(r.all_tokens or []),
                                    r.prompt_cache))
                self._top.pop(r.uid, None)
        return out

    def remove(self, uids: list[int]) -> None:
        for u in uids:
            self._top.pop(u, None)
        self.gen.remove(list(uids))

    def next_admission(self) -> int | None:
        """The row the next step() admits (prefills, whole), or None: the
        engine admits the head of its queue, one per step, while the batch
        has room (MTPBatchGenerator._next)."""
        g = self.gen
        q = getattr(g, "_unprocessed_sequences", None)
        if not q:
            return None
        batch = getattr(g, "_batch", None)
        if batch is not None and len(batch) >= g.completion_batch_size:
            return None
        return q[0][0]

    def queued(self) -> set:
        """Rows inserted but not yet prefilled."""
        return {s[0] for s in getattr(self.gen, "_unprocessed_sequences",
                                      None) or ()}

    def set_chunk(self, n: int) -> None:
        """The prefill chunk for the next admission: the engine reads
        prefill_step_size when it admits, inside step()."""
        self.gen.prefill_step_size = int(n)

    @property
    def cache_nbytes(self) -> int:
        return self.gen.prompt_cache_nbytes

    def cost_per_token(self, rows: int):
        return self.gen.cost_per_token(rows)

    def close(self) -> None:
        self.gen.close()


def _top(logprobs, k: int) -> list[tuple] | None:
    if not k:
        return None
    import mlx.core as mx
    idx = mx.argpartition(-logprobs, kth=k - 1)[:k]
    vals = logprobs[idx]
    order = mx.argsort(-vals)
    return list(zip(idx[order].tolist(), vals[order].tolist()))
