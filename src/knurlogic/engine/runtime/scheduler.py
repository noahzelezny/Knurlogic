"""The scheduler: ONE thread that owns the MLX stream (docs/SERVER.md).

    HTTP threads --submit(Job)--> queue --> tokenize --> prompt cache
      --> executor.insert --> executor.step --> events --> per-request text
      (runtime/request.py) --> the Job's outbox --> HTTP threads

Everything that touches the model happens here, in order: loads and
unloads (commands), tokenizing (vision's image work included -- its pins
are thread-local), prompt-cache lookups and inserts, admission and steps.
A failure is per request: an exception while tokenizing or admitting fails
that request, a failed row (RowFailure) fails its own, and a step that
raises fails the rows in it and rebuilds the executor. The thread lives.

A request waits for the host to be ready; while nothing is running the
loop blocks on the queue instead of spinning.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import prompt as P
from .executor import (Admission, Checkpoint, Finished, LocalExecutor,
                       Progress, RowFailure, Token)
from .request import Delta, Request, control_machine

logger = logging.getLogger(__name__)


@dataclass
class Job:
    """One request as the scheduler takes it."""
    request: P.ChatRequest
    args: P.PromptArgs
    max_tokens: int = 512
    sampling: dict = field(default_factory=dict)      # temp, top_p, ..., seed
    penalties: dict = field(default_factory=dict)     # make_logits_processors
    stops: List[str] = field(default_factory=list)
    logprobs: bool = False
    top_logprobs: int = 0
    #: filled by the scheduler: ("progress", (done, total)) / ("delta",
    #: Delta) / ("error", exc) / ("done", usage)
    outbox: "queue.Queue" = field(default_factory=queue.Queue)
    cancelled: bool = False
    #: set once tokenized
    prompt_tokens: int = 0

    def cancel(self) -> None:
        """From the HTTP thread: the client went away; free the row."""
        self.cancelled = True


class PromptCache:
    """mlx-lm's LRUPromptCache, with the one rule it lacks: an exact hit is
    returned one token short (trimmed; a cache that cannot trim is a miss),
    so there is always a token to process -- an exact hit otherwise left
    nothing and killed the generation thread (GLM none-then-low, M4)."""

    def __init__(self, max_size: int = 10, max_bytes: Optional[int] = None):
        from mlx_lm.models.cache import LRUPromptCache
        self.lru = LRUPromptCache(max_size=max_size,
                                  **({"max_bytes": max_bytes}
                                     if max_bytes else {}))

    def fetch(self, key, tokens):
        from mlx_lm.models import cache as C
        cache, rest = self.lru.fetch_nearest_cache(key, tokens)
        if cache is None or rest or not tokens:
            return cache, rest
        if C.can_trim_prompt_cache(cache):
            C.trim_prompt_cache(cache, 1)
            return cache, list(tokens[-1:])
        return None, list(tokens)

    def insert(self, key, tokens, cache, kind: str) -> None:
        self.lru.insert_cache(key, list(tokens), cache, cache_type=kind)

    def trim_to(self, n_bytes: int) -> None:
        self.lru.trim_to(n_bytes=n_bytes)

    @property
    def nbytes(self) -> int:
        return int(self.lru.nbytes)


@dataclass
class _Row:
    job: Job
    text: Request
    types: List[str]          # segment types still to label checkpoints


class Scheduler:
    def __init__(self, host, *, completion_batch_size: int = 32,
                 prefill_step_size: int = 2048, prompt_cache_size: int = 10,
                 prompt_cache_bytes: Optional[int] = None,
                 stats: Optional[dict] = None):
        self.host = host
        self.completion_batch_size = completion_batch_size
        self.prefill_step_size = prefill_step_size
        self.cache_bytes = prompt_cache_bytes
        self.cache = PromptCache(prompt_cache_size)
        self.stats = stats if stats is not None else {}
        self._jobs: "queue.Queue" = queue.Queue()
        self._commands: "queue.Queue" = queue.Queue()
        self._waiting: List[Job] = []
        self._rows: Dict[int, _Row] = {}
        self._ex: Optional[LocalExecutor] = None
        self._stop = False
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="knurlogic-scheduler")

    # ------------------------------------------------------ any thread

    def start(self) -> "Scheduler":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop = True
        self._wake.set()
        self._thread.join(10)

    def submit(self, job: Job) -> Job:
        self._jobs.put(job)
        self._wake.set()
        return job

    def load(self, path: str) -> threading.Event:
        """Queue a load; the event is set when it has finished (either
        way -- read host.state)."""
        done = threading.Event()
        self.host.expect(str(path))
        self._commands.put(("load", str(path), done))
        self._wake.set()
        return done

    def unload(self) -> threading.Event:
        done = threading.Event()
        self._commands.put(("unload", None, done))
        self._wake.set()
        return done

    @property
    def width(self) -> int:
        """Rows admitted and decoding right now."""
        return len(self._rows)

    def more_helps(self, rows: int):
        """Would one more concurrent row raise throughput? True/False from
        the engine's timings at `rows` and `rows + 1` (seconds per token
        across the batch), None until both are measured."""
        ex = self._ex
        if ex is None or rows < 1:
            return None
        now, more = ex.cost_per_token(rows), ex.cost_per_token(rows + 1)
        if now is None or more is None:
            return None
        return more < now

    @property
    def busy(self) -> bool:
        return bool(self._rows) or bool(self._waiting) or \
            not self._jobs.empty()

    # --------------------------------------------------- scheduler thread

    def _run(self) -> None:
        import mlx.core as mx
        # this thread's own default stream: MLX streams are per thread
        self._stream = mx.default_stream(mx.default_device())
        try:
            while not self._stop:
                self._loop_once()
        finally:
            # Arrays made on this thread's stream are released on this
            # thread: freed after it has gone, MLX crashes (measured: the
            # process segfaults when the model is freed on the main thread
            # after a stopped scheduler).
            self._fail_all(RuntimeError("the server is stopping"))
            self._close_executor()
            self.cache = None
            # the model too: its lazily built arrays (rope tables, caches)
            # were made on this thread
            if hasattr(self.host, "unload"):
                self.host.unload()
            import gc
            gc.collect()
            mx.synchronize()

    def _loop_once(self) -> None:
        try:
            self._tick()
        except Exception:
            # Nothing here should raise; if it does, fail what is in
            # flight rather than the thread.
            logger.exception("scheduler tick failed")
            self._fail_all(RuntimeError("the scheduler hit an internal "
                                        "error; see the server log"))
            self._close_executor()

    def _tick(self) -> None:
        self._do_commands()
        self._take_jobs()
        if self.host.state == "ready":
            self._admit_waiting()
        elif self.host.state in ("empty", "failed") and self._waiting \
                and self._commands.empty():
            why = self.host.error or "no model is loaded"
            for j in self._waiting:
                self._error(j, RuntimeError(f"no model to serve: {why}"))
            self._waiting.clear()
        if self._ex is not None and self._rows:
            self._step()
            return
        # idle: block until something arrives
        self._wake.wait(0.5 if self._waiting else None)
        self._wake.clear()

    def _do_commands(self) -> None:
        while True:
            try:
                cmd, arg, done = self._commands.get_nowait()
            except queue.Empty:
                return
            try:
                if self._rows:
                    self._fail_all(RuntimeError(
                        "the model was switched while this request was "
                        "running"))
                self._close_executor()
                self.cache = PromptCache(self.cache.lru.max_size)
                if cmd == "load":
                    self.host.load(arg)
                else:
                    self.host.unload()
                if self.host.state != "ready":
                    for j in self._waiting:
                        self._error(j, RuntimeError(
                            f"no model is loaded: {self.host.error or 'unloaded'}"))
                    self._waiting.clear()
            finally:
                done.set()

    def _take_jobs(self) -> None:
        while True:
            try:
                self._waiting.append(self._jobs.get_nowait())
            except queue.Empty:
                return

    def _executor(self) -> LocalExecutor:
        if self._ex is None:
            from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
            from knurlogic.engine.serve import state
            head = state.DRAFT.get("head") if state.DRAFT.get("on") else None
            vision = state.VISION.get("serve")
            gen = MTPBatchGenerator(
                self.host.model, head,
                stats=state.DRAFT if head is not None else state.VISION_STATS,
                vision=vision,
                completion_batch_size=self.completion_batch_size,
                prefill_step_size=self.prefill_step_size,
                stream=self._stream)
            if head is not None:
                state.DRAFT["batch_installed"] = True
            self._ex = LocalExecutor(gen)
            if vision is not None:
                from knurlogic.engine.vision import cachehook
                cachehook.install(self.cache.lru, lambda: (
                    state.VISION["serve"].store
                    if state.VISION.get("serve") else None))
        return self._ex

    def _close_executor(self) -> None:
        if self._ex is not None:
            try:
                self._ex.close()
            except Exception:
                logger.exception("closing the executor")
            self._ex = None

    def _admit_waiting(self) -> None:
        # One admission per tick into the executor's queue; the executor
        # itself admits one row per step, so decoding rows are not held.
        while self._waiting:
            job = self._waiting.pop(0)
            if job.cancelled:
                continue
            try:
                self._insert(job)
            except P.PromptError as e:
                self._error(job, e)
            except Exception as e:
                logger.exception("could not admit a request")
                self._error(job, e)

    def _insert(self, job: Job) -> None:
        from knurlogic.engine.vision import cachehook
        from knurlogic.engine.vision import request as vreq
        from knurlogic.engine.serve import state
        tok = self.host.tokenizer
        ex = self._executor()
        cachehook.sweep()
        with cachehook.admit_guard():
            if vreq.has_images(job.request.messages):
                v = state.VISION.get("serve")
                if v is None:
                    raise P.PromptError("this request has images but the "
                                        "served model has no vision")
                prompt, segs, types, initial = v.tokenize(
                    P.tokenize, self, tok, job.request, job.args)
            else:
                prompt, segs, types, initial = P.tokenize(
                    self, tok, job.request, job.args)
            job.prompt_tokens = len(prompt)
            cache, rest = self.cache.fetch(self.host.model_key, prompt)
            n = len(prompt) - len(rest)
            segs, types = [list(s) for s in segs], list(types)
            while n > 0 and segs:
                if n >= len(segs[0]):
                    n -= len(segs.pop(0))
                    types.pop(0)
                else:
                    segs[0] = segs[0][n:]
                    n = 0
            sm, seqs = control_machine(tok, initial)
            procs = []
            if job.penalties:
                from mlx_lm.sample_utils import make_logits_processors
                procs = make_logits_processors(**job.penalties)
            uid = ex.insert(Admission(
                segments=segs, max_tokens=job.max_tokens, cache=cache,
                prefix=prompt[:len(prompt) - len(rest)],
                sampling=job.sampling, processors=procs, state_machine=sm,
                top_logprobs=job.top_logprobs, report=job.request))
        text = Request(tok.detokenizer, sequences=seqs, stops=job.stops,
                       tool_parser=getattr(tok, "tool_parser", None),
                       tools=job.request.tools,
                       logprobs=job.logprobs or bool(job.top_logprobs),
                       prompt_tokens=len(prompt))
        self._rows[uid] = _Row(job, text, types)
        if self.cache_bytes is not None:
            self.cache.trim_to(self.cache_bytes - ex.cache_nbytes)

    def _step(self) -> None:
        ex = self._ex
        try:
            events = ex.step()          # on the executor's own stream
        except Exception as e:
            logger.exception("a step failed; failing its rows")
            self._fail_all(e)
            self._close_executor()
            return
        drop = []
        for e in events:
            row = self._rows.get(e.uid)
            if row is None:
                continue
            if isinstance(e, Progress):
                row.job.outbox.put(("progress", (e.done, e.total)))
            elif isinstance(e, Checkpoint):
                if row.types:
                    self.cache.insert(self.host.model_key, e.tokens, e.cache,
                                      row.types.pop(0))
            elif isinstance(e, Token):
                d = row.text.feed(e)
                if d:
                    row.job.outbox.put(("delta", d))
                if row.text.finished and e.finish is None:
                    # a stop string ended the answer; the engine's row is
                    # still going
                    drop.append(e.uid)
                    self._done(e.uid)
            elif isinstance(e, Finished):
                self.cache.insert(self.host.model_key, e.tokens, e.cache,
                                  "assistant")
                self._done(e.uid)
            elif isinstance(e, RowFailure):
                self._rows.pop(e.uid, None)
                self._error(row.job, e.error)
        for uid, row in list(self._rows.items()):
            if row.job.cancelled:
                drop.append(uid)
                self._rows.pop(uid, None)
        if drop:
            ex.remove(sorted(set(drop)))

    def _done(self, uid: int) -> None:
        row = self._rows.pop(uid, None)
        if row is None:
            return
        tail = row.text.finish("stop")
        if tail:
            row.job.outbox.put(("delta", tail))
        from knurlogic.engine.serve import cache_report
        row.job.outbox.put(("done", row.text.usage(
            cache_report.of(row.job.request))))

    def _error(self, job: Job, err: BaseException) -> None:
        job.outbox.put(("error", err))

    def _fail_all(self, err: BaseException) -> None:
        for uid, row in list(self._rows.items()):
            self._error(row.job, err)
        self._rows.clear()
