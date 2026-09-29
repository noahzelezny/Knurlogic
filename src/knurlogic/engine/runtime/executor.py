"""The executor: the seam between knurlogic's scheduler and whatever runs
the model's steps (docs/SERVER.md, build step 1).

A scheduler hands the executor admissions and asks it for steps; the
executor answers with events. Today there is one executor, the local batch
engine (MTPBatchGenerator, drafting or not, with vision). A
tensor split (engine/runtime/tensor.py) runs the same executor on every
rank, rank 0's journaling each admission for the others; a pipeline
executor comes later behind the same protocol: nothing here
assumes the layers run in this process.

Rules the protocol keeps:

  * A token event carries the token and ITS logprob (plus top-k when
    asked), never a [V] row: what crosses a process boundary stays small
    (every rank of a ring computes the logits; only rank 0 samples).
  * A failure is a `RowFailure` event for that row, not an exception passed
    along as progress; the executor has already dropped the row.
  * Admission carries the request's sampling parameters and the object
    that receives its cache report -- no thread-local, no tagged sampler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, Union


@dataclass
class Admission:
    """One request, tokenized, as the executor admits it."""

    #: prompt tokens by segment (system / user / ...); the prompt cache
    #: stores a checkpoint at the end of each segment but the last
    segments: List[List[int]]
    max_tokens: int
    #: the prompt cache entry restored for this request and the tokens it
    #: covers (a prefix of the prompt); empty for a fresh prefill
    cache: Optional[list] = None
    prefix: List[int] = field(default_factory=list)
    #: engine/mtp/sampling.make_distribution's kwargs (temp, top_p,
    #: top_k, min_p); empty = greedy
    sampling: dict = field(default_factory=dict)
    processors: list = field(default_factory=list)
    #: mlx-lm's SequenceStateMachine for the control-token stops, or None
    #: for the engine's default
    state_machine: Any = None
    top_logprobs: int = 0
    #: receives the cache report (engine/serve/cache_report.attach)
    report: Any = None
    #: on a tensor ring, what the other ranks need to rebuild what is not
    #: data here (processors, the state machine): {"penalties": the
    #: make_logits_processors kwargs, "initial": the machine's start state}
    wire: Optional[dict] = None


@dataclass
class Progress:
    uid: int
    done: int
    total: int


@dataclass
class Checkpoint:
    """A segment ended inside the prompt: its cache, for the prompt cache."""
    uid: int
    tokens: List[int]
    cache: list


@dataclass
class Token:
    uid: int
    token: int
    logprob: float
    finish: Optional[str] = None
    #: the control-token state machine's state and any sequence it matched
    state: Optional[str] = None
    match: Optional[List[int]] = None
    top_logprobs: Optional[List[tuple]] = None


@dataclass
class Finished:
    """After a row's finishing Token: its cache, keyed by every token it
    holds, for the prompt cache."""
    uid: int
    tokens: List[int]
    cache: list


@dataclass
class RowFailure:
    uid: int
    error: BaseException


Event = Union[Progress, Checkpoint, Token, Finished, RowFailure]


class Executor(Protocol):
    def insert(self, admission: Admission) -> int: ...

    def step(self) -> List[Event]:
        """One admission and/or one decode step; [] when idle."""
        ...

    def remove(self, uids: List[int]) -> None: ...

    @property
    def cache_nbytes(self) -> int: ...

    def cost_per_token(self, rows: int) -> Optional[float]:
        """Measured seconds per token at this batch width, or None."""
        ...

    def close(self) -> None: ...


class LocalExecutor:
    """The batch engine in this process, behind the protocol."""

    def __init__(self, generator):
        self.gen = generator
        self._top: dict = {}

    def insert(self, a: Admission) -> int:
        (uid,) = self.gen.insert_segments(
            segments=[a.segments], max_tokens=[a.max_tokens],
            caches=[a.cache], all_tokens=[list(a.prefix)],
            samplers=[dict(a.sampling)], logits_processors=[list(a.processors)],
            state_machines=[a.state_machine] if a.state_machine else None,
            reports=[a.report])
        if a.top_logprobs:
            self._top[uid] = a.top_logprobs
        return uid

    def step(self) -> List[Event]:
        prompt, gen = self.gen.next()
        out: List[Event] = []
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

    def remove(self, uids: List[int]) -> None:
        for u in uids:
            self._top.pop(u, None)
        self.gen.remove(list(uids))

    @property
    def cache_nbytes(self) -> int:
        return self.gen.prompt_cache_nbytes

    def cost_per_token(self, rows: int):
        return self.gen.cost_per_token(rows)

    def close(self) -> None:
        self.gen.close()


def _top(logprobs, k: int) -> Optional[List[tuple]]:
    if not k:
        return None
    import mlx.core as mx
    idx = mx.argpartition(-logprobs, kth=k - 1)[:k]
    vals = logprobs[idx]
    order = mx.argsort(-vals)
    return list(zip(idx[order].tolist(), vals[order].tolist()))
