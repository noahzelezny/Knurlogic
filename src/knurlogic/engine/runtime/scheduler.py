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

Memory is guarded here too, because a step that outgrows the GPU working
set is not an exception: Metal aborts the process (measured on the M4:
Flash 4.4 at 97 GiB, four long reviews and a full prompt cache climbed to
118 of 120 GiB over two hours, then "Insufficient Memory" killed the server
and every request in it). Before each step, when active memory is past the
limit (the working set less a margin for one step's temporaries), the
prompt cache gives up entries first -- they are a convenience -- and then
the newest rows are stopped with `OutOfMemory` (a 503: retry), the least
work lost. While memory is past the admission mark, new requests wait.

A step cannot be interrupted, and admitting a row prefills its whole
prompt inside one step (plus a deep copy per segment checkpoint), so the
per-step check alone lets one long prompt jump past the limit (measured:
an agent's 50k-token turn took the same server from 114 to 118 GiB within
a minute, no step between). So admission estimates too, from what this
model's caches measured (fixed state per row plus bytes per token: hybrid
models carry linear-attention state whatever the length). A prompt wants
its KV twice -- the row, and the checkpoint copies the prompt cache keeps;
the prompt cache gives way for it; if only one copy fits, the row is
admitted LEAN, without checkpoints (the request beats the cache); failing
that it waits for running rows, or with none running is refused -- never
admitted to abort the process.
"""

from __future__ import annotations

import logging
import queue
from pathlib import Path
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import prompt as P
from .executor import (Admission, Checkpoint, Finished, LocalExecutor,
                       Progress, RowFailure, Token)
from .request import Delta, Request, control_machine

logger = logging.getLogger(__name__)
GIB = 1 << 30


class OutOfMemory(RuntimeError):
    """This row was stopped so the server and the other rows keep running."""


def _kv_from_config(path) -> Optional[tuple]:
    """(0, bytes per token) from the artifact's config -- its full-attention
    layers' K and V -- so the first prompt after a load is costed before a
    cache has been measured; the measurements replace it. None if the
    config does not say."""
    try:
        from knurlogic.machine.artifact import Artifact
        from knurlogic.tuning.resolve import kv_bytes_per_token
        cfg = Artifact.load(path).raw_config
        per, _why = kv_bytes_per_token(cfg.get("text_config", cfg))
        return (0.0, float(per)) if per > 0 else None
    except Exception:
        return None


class _Wait(Exception):
    """Not admitted yet: the request goes back to the front of the queue."""


def _cache_nbytes(cache) -> int:
    n = 0
    for c in cache or ():
        try:
            n += int(getattr(c, "nbytes", 0) or 0)
        except Exception:
            pass
    return n


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
class Command:
    """A load or unload for the scheduler thread; `done` is set when it has
    run (or was refused: `error`)."""
    kind: str
    path: Optional[str] = None
    force: bool = True
    executes: bool = False
    #: set when the scheduler has taken it on (or refused it: then `done`)
    started: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    error: str = ""


@dataclass
class _Row:
    job: Job
    text: Request
    types: List[str]          # segment types still to label checkpoints


class Scheduler:
    def __init__(self, host, *, completion_batch_size: int = 32,
                 prefill_step_size: int = 2048, prompt_cache_size: int = 10,
                 prompt_cache_bytes: Optional[int] = None,
                 working_set_bytes: Optional[int] = None,
                 stats: Optional[dict] = None):
        self.host = host
        self.completion_batch_size = completion_batch_size
        self.prefill_step_size = prefill_step_size
        self.cache_bytes = prompt_cache_bytes
        #: the GPU working set the guard keeps under; None = asked of the
        #: framework on first use; 0 = unguarded
        self.working_set = working_set_bytes
        #: the largest transient a step has been measured to add (bytes
        #: above the active memory it started at), 0 until measured
        self._spike = 0
        #: (fixed bytes, bytes per token) of one row's cache, measured
        self._kv = None
        self._samples: dict = {}
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

    def load(self, path: str, *, executes_artifact_code: bool = False,
             force: bool = True) -> "Command":
        """Queue a load. `force=False` refuses (Command.error) if requests
        are running or queued when the switch would happen -- decided on
        the scheduler thread, so it cannot race them. Wait on
        Command.done; read Command.error, then host.state."""
        if self.host.state == "empty":
            # the first load: requests arriving now wait for it
            self.host.expect(str(path))
        return self._command(Command("load", str(path), force,
                                     executes_artifact_code))

    def unload(self, *, force: bool = True) -> "Command":
        return self._command(Command("unload", None, force))

    def _command(self, cmd: "Command") -> "Command":
        self._commands.put(cmd)
        self._wake.set()
        return cmd

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
            self._fail_queued(RuntimeError("the server is stopping"))
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
        if self.host.state == "ready" and self._room_to_admit():
            self._admit_waiting()
        elif self.host.state in ("empty", "failed") and self._waiting \
                and self._commands.empty():
            why = self.host.error or "no model is loaded"
            for j in self._waiting:
                self._error(j, RuntimeError(f"no model to serve: {why}"))
            self._waiting.clear()
        if self._ex is not None and self._rows:
            self._guard_memory()
        if self._ex is not None and self._rows:
            self._step()
            return
        # idle: block until something arrives
        self._wake.wait(0.5 if self._waiting else None)
        self._wake.clear()

    def _do_commands(self) -> None:
        while True:
            try:
                c = self._commands.get_nowait()
            except queue.Empty:
                return
            if c.kind == "load" and self.host.state == "ready" \
                    and self.host.path == c.path:
                c.started.set()
                c.done.set()      # already served: nothing to fail or drop
                continue
            self._take_jobs()
            busy = len(self._rows) + len(self._waiting)
            if busy and not c.force:
                c.error = (f"{busy} request(s) are running or queued on "
                           f"{Path(self.host.path or '').name}; switching "
                           f"would fail them. Retry when idle, or force.")
                c.started.set()
                c.done.set()
                continue
            c.started.set()
            try:
                if self._rows:
                    self._fail_all(RuntimeError(
                        "the model was switched while this request was "
                        "running"))
                self._close_executor()
                self.cache = PromptCache(self.cache.lru.max_size)
                # what was measured belongs to the model it was measured on
                # (a 35B's slope admitting a 397B's prompt is the abort the
                # guard exists for)
                self._kv, self._samples, self._spike = None, {}, 0
                if c.kind == "load":
                    self.host.load(c.path,
                                   executes_artifact_code=c.executes)
                    self._kv = _kv_from_config(c.path)
                else:
                    self.host.unload()
                if self.host.state != "ready":
                    for j in self._waiting:
                        self._error(j, RuntimeError(
                            f"no model is loaded: {self.host.error or 'unloaded'}"))
                    self._waiting.clear()
            finally:
                c.done.set()

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
        # Text requests are cheap to tokenize and all go in; the executor
        # admits one row per step, so decoding rows are not held. An image
        # request is encoded HERE (the tower runs at tokenize), so at most
        # one per tick: a burst of images is interleaved with decode steps
        # instead of stalling every running row until all are encoded.
        from knurlogic.engine.vision import VisionError
        from knurlogic.engine.vision import request as vreq
        held = []
        try:
            self._admit_from_queue(held, VisionError, vreq)
        finally:
            # requests still waiting for memory go back in their order
            self._waiting[:0] = held

    def _admit_from_queue(self, held, VisionError, vreq) -> None:
        while self._waiting:
            job = self._waiting.pop(0)
            if job.cancelled:
                continue
            if job.prompt_tokens and self._rows \
                    and not self._fits(job.prompt_tokens):
                # waited before and still would not fit: not tokenized
                # again, and not in the way of smaller prompts behind it
                held.append(job)
                continue
            images = vreq.has_images(job.request.messages)
            try:
                self._insert(job)
            except _Wait:
                held.append(job)
                continue
            except (P.PromptError, VisionError, OutOfMemory) as e:
                # the client's request, not a fault here: no traceback
                logger.info("refused a request: %s", e)
                self._error(job, e)
            except Exception as e:
                logger.exception("could not admit a request")
                self._error(job, e)
            if images and self._rows:
                return            # decode a step before the next encode

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
                # the pins tokenize took belong to the guard until the row
                # is admitted: a failure before then releases them
                from knurlogic.engine.vision import key as K
                cachehook.pending(v, K.images_in(prompt))
            else:
                prompt, segs, types, initial = P.tokenize(
                    self, tok, job.request, job.args)
            job.prompt_tokens = len(prompt)
            lean = self._make_room(len(prompt)) == "lean"
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
            if lean and len(segs) > 1:
                # one segment: no checkpoint copies of this prompt
                segs, types = [[t for s in segs for t in s]], []
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
        try:
            text = Request(tok.detokenizer, sequences=seqs, stops=job.stops,
                           tool_parser=getattr(tok, "tool_parser", None),
                           tools=job.request.tools,
                           logprobs=job.logprobs or bool(job.top_logprobs),
                           prompt_tokens=len(prompt))
        except BaseException:
            ex.remove([uid])      # admitted but unobserved: never orphaned
            raise
        self._rows[uid] = _Row(job, text, types)
        if self.cache_bytes is not None:
            self.cache.trim_to(self.cache_bytes - ex.cache_nbytes)

    # ------------------------------------------------------------- memory

    def _working_set(self) -> int:
        if self.working_set is None:
            import importlib   # engine.serve exports a load() function
            load = importlib.import_module("knurlogic.engine.serve.load")
            self.working_set = int(load.memory().get("working_set_bytes")
                                   or 0)
        return self.working_set

    def _margin(self) -> int:
        """Room one step's temporaries need. Measured: the largest spike a
        step of THIS model has made (its prefill chunk, its attention, its
        batch), with a quarter again of headroom. Until a step has been
        measured, a guess -- 5% of the working set, at least 4 GiB -- which
        the first steps replace."""
        if self._spike:
            return max(GIB, int(self._spike * 1.25))
        return max(4 * GIB, self._working_set() // 20)

    def _limit(self) -> int:
        """Active bytes a step may start at: the working set less a step's
        temporaries. 0 = unguarded."""
        ws = self._working_set()
        return ws - self._margin() if ws else 0

    def _measure(self, before: int) -> None:
        """The step's TRANSIENT: its peak above the larger of where it
        started and where it ended. What it kept (an admitted row's KV,
        checkpoint copies) is growth, not transient -- counted as spike, one
        50k-token admission ratcheted the margin up for the life of the
        load (Fable 5.1)."""
        import mlx.core as mx
        spike = int(mx.get_peak_memory()) - max(before, self._active())
        if spike > self._spike * 1.25 and spike > GIB // 4:
            logger.info("a step's transient measured at %.2f GiB: the "
                        "memory margin is now %.2f GiB", spike / GIB,
                        max(GIB, int(spike * 1.25)) / GIB)
        self._spike = max(self._spike, spike)

    def _reset_peak(self) -> int:
        import mlx.core as mx
        mx.reset_peak_memory()
        return self._active()

    def _active(self) -> int:
        import mlx.core as mx
        return int(mx.get_active_memory())

    def _release(self) -> None:
        import mlx.core as mx
        mx.clear_cache()

    def _room_to_admit(self) -> bool:
        """A request is considered when nothing is running (it could not
        wait for memory anyone else would free) or memory is under the
        limit; whether ITS prompt fits is _make_room's question. The limit
        already holds a step's measured transient back -- holding a margin
        back again here (and in _room_for) left 397B, whose weights leave 8
        GiB, refusing 20k-token prompts with nothing else running."""
        limit = self._limit()
        return (not limit or not self._rows or self._active() < limit)

    def _learn(self, tokens, cache) -> None:
        """A row's cache as fixed + per-token bytes, from the shortest and
        longest caches seen. One ratio on a short cache charged a hybrid
        model's fixed linear-attention state to every token (Flash 4.4:
        long agent turns refused that fit). Until two lengths 2048 apart
        are known, the ratio at the longest -- an overestimate, the safe
        side."""
        n = len(tokens)
        if n < 256:
            return
        b = _cache_nbytes(cache)
        s = self._samples
        if "lo" not in s or n < s["lo"][0]:
            s["lo"] = (n, b)
        if "hi" not in s or n > s["hi"][0]:
            s["hi"] = (n, b)
        (n0, b0), (n1, b1) = s["lo"], s["hi"]
        if n1 - n0 >= 2048 and b1 > b0:
            slope = (b1 - b0) / (n1 - n0)
            self._kv = (max(b0 - slope * n0, 0.0), slope)
        else:
            self._kv = (0.0, b1 / n1)

    def _cost(self, n_tokens: int, copies: int) -> int:
        fixed, per = self._kv
        return int(copies * (fixed + per * n_tokens))

    def _fits(self, n_tokens: int) -> bool:
        """Could a prompt of n_tokens fit once, counting what the prompt
        cache would give up? No side effects: a request that waits must not
        empty the shared prompt cache on every tick it waits."""
        limit = self._limit()
        if not limit or not self._kv:
            return True
        room = limit - self._active() + (self.cache.nbytes
                                         if self.cache is not None else 0)
        return self._cost(n_tokens, 1) <= room

    def _room_for(self, n_tokens: int, copies: int = 2):
        """(fits, need, room) for a prompt of n_tokens held `copies` times,
        the prompt cache giving way if that is what it takes."""
        limit = self._limit()
        if not limit or not self._kv:
            return True, 0, 0
        need = self._cost(n_tokens, copies)
        room = limit - self._active()
        if need > room and self.cache is not None and self.cache.nbytes:
            before = self.cache.nbytes
            self.cache.trim_to(max(before - (need - room), 0))
            self._release()
            room = limit - self._active()
            logger.info("the prompt cache gave up %.1f GiB for a %d-token "
                        "prompt", (before - self.cache.nbytes) / GIB,
                        n_tokens)
        return need <= room, need, room

    def _make_room(self, n_tokens: int) -> str:
        """"full" if a prompt of n_tokens fits with its checkpoints, "lean"
        if only without; else _Wait (rows are running and will free memory)
        or OutOfMemory (none are)."""
        fits, need, room = self._room_for(n_tokens, copies=2)
        if fits:
            return "full"
        fits, need, room = self._room_for(n_tokens, copies=1)
        if fits:
            logger.info("a %d-token prompt admitted without checkpoints: "
                        "%.1f GiB free, its cache twice would be %.1f",
                        n_tokens, room / GIB, 2 * need / GIB)
            return "lean"
        limit = self._limit()
        if self._rows:
            raise _Wait()
        raise OutOfMemory(
            f"this prompt ({n_tokens} tokens) needs about {need / GIB:.1f} "
            f"GiB for its cache; {max(room, 0) / GIB:.1f} GiB is free under "
            f"the server's limit ({limit / GIB:.1f} GiB). Send a shorter "
            f"conversation, or serve a smaller model")

    def _guard_memory(self) -> None:
        limit = self._limit()
        if not limit:
            return
        over = self._active() - limit
        if over <= 0:
            return
        if self.cache is not None and self.cache.nbytes:
            before = self.cache.nbytes
            self.cache.trim_to(max(before - over - self._margin(), 0))
            self._release()
            over = self._active() - limit
            logger.warning("memory past the limit (%.1f GiB): the prompt "
                           "cache gave up %.1f GiB", limit / GIB,
                           (before - self.cache.nbytes) / GIB)
        while over > 0 and self._rows:
            uid = max(self._rows)             # the newest: least work lost
            row = self._rows.pop(uid)
            self._ex.remove([uid])
            self._release()
            logger.warning("memory past the limit (%.1f GiB): stopped the "
                           "newest request (%d prompt tokens)", limit / GIB,
                           row.job.prompt_tokens)
            self._error(row.job, OutOfMemory(
                f"stopped to keep the server within its memory "
                f"({limit / GIB:.1f} GiB usable, {len(self._rows)} other "
                f"request(s) running); retry, or ask for fewer tokens"))
            over = self._active() - limit

    def _step(self) -> None:
        ex = self._ex
        before = self._reset_peak()
        try:
            events = ex.step()          # on the executor's own stream
        except Exception as e:
            logger.exception("a step failed; failing its rows")
            self._fail_all(e)
            self._close_executor()
            return
        self._measure(before)
        drop = []
        for e in events:
            row = self._rows.get(e.uid)
            if row is None:
                continue
            if isinstance(e, Progress):
                row.job.outbox.put(("progress", (e.done, e.total)))
            elif isinstance(e, Checkpoint):
                self._learn(e.tokens, e.cache)
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
                self._learn(e.tokens, e.cache)
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

    def _fail_queued(self, err: BaseException) -> None:
        self._take_jobs()
        for j in self._waiting:
            self._error(j, err)
        self._waiting.clear()
