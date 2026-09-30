"""The scheduler: ONE thread that owns the MLX stream (docs/design/server.md).

    HTTP threads --submit(Job)--> queue --> tokenize --> prompt cache
      --> executor.insert --> executor.step --> events --> per-request text
      (runtime/request.py) --> the Job's outbox --> HTTP threads

Everything that touches the model happens here, in order. A failure is per
request (a failed tokenize/admit, a RowFailure, or a raising step fails
only its rows; the executor is rebuilt); the thread lives.

Memory is guarded here, because outgrowing the GPU working set is not an
exception: Metal aborts the process. Before each step, past the limit, the
prompt cache gives up entries first and then the newest rows are stopped
with `OutOfMemory` (a 503). Admission estimates too, from what this model's
caches measured: a row is admitted with checkpoints, LEAN without them,
waits, or with nothing running is refused -- never admitted to abort.
Design: docs/design/server.md (memory guard).
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


class RingFailed(RuntimeError):
    """A rank of this split model is gone or stalled: the job is being torn
    down, and every request in flight is answered with a 503."""


def _context_cap() -> int:
    """KNURLOGIC_CONTEXT_LENGTH: the longest prompt + answer a request may
    use, read at every admission so the setting applies live. 0 = none."""
    import os
    try:
        return max(int(os.environ.get("KNURLOGIC_CONTEXT_LENGTH") or 0), 0)
    except ValueError:
        return 0


def _kv_from_config(path, kv_bits=None) -> Optional[tuple]:
    """(0, bytes per token) from the artifact's config -- its full-attention
    layers' K and V -- so the first prompt after a load is costed before a
    cache has been measured; the measurements replace it. None if the
    config does not say."""
    try:
        from knurlogic.machine.artifact import Artifact
        from knurlogic.tuning.resolve import kv_bytes_per_token
        cfg = Artifact.load(path).raw_config
        per, _why = kv_bytes_per_token(cfg.get("text_config", cfg), kv_bits)
        return (0.0, float(per)) if per > 0 else None
    except Exception:
        return None


def _timing(row, done: float, completion: int, prefilled) -> dict:
    """What the request took, measured here where the steps run: time in
    the queue, time to first token (from submit, as a client feels it), and
    the rates of the two phases. Prefill is from admission to the first
    token, over the tokens actually prefilled (not the ones the prompt
    cache supplied); decode is over the tokens after the first."""
    first = row.first or done
    out = {"queue_s": round(max(row.admitted - row.job.submitted, 0), 4),
           "ttft_s": round(max(first - row.job.submitted, 0), 4)}
    pre = first - row.admitted
    if prefilled and pre > 0:
        out["prefill_tok_s"] = round(prefilled / pre, 1)
    dec = done - first
    if completion > 1 and dec > 0:
        out["decode_tok_s"] = round((completion - 1) / dec, 1)
    return out


class _Wait(Exception):
    """Not admitted yet: the request goes back to the front of the queue."""


class _RingWait(_Wait):
    """The prompt cache gave way on a ring; wait one exchange for the
    peers' memory before deciding (Scheduler._make_room)."""


def _cache_nbytes(cache) -> int:
    n = 0
    for c in cache or ():
        try:
            n += int(getattr(c, "nbytes", 0) or 0)
        except Exception:
            pass    # an estimate: a layer that cannot report its size counts 0
    return n


def _fixed_nbytes(cache) -> int:
    """The bytes of a row's cache that do not grow with its length: the
    layers that cannot trim (a hybrid's linear-attention / recurrent state).
    A layer that does not say is counted as growing -- the safe side."""
    n = 0
    for c in cache or ():
        try:
            t = getattr(c, "is_trimmable", None)
            if callable(t) and not t():
                n += int(getattr(c, "nbytes", 0) or 0)
        except Exception:
            pass    # an estimate: a layer that cannot report its size counts 0
    return n


@dataclass
class Job:
    """One request as the scheduler takes it."""
    request: P.ChatRequest
    args: P.PromptArgs
    max_tokens: Optional[int] = None    # None: the rest of the context window
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
    #: the rows it waits for, once it has had to wait for memory
    waiting_on: Optional[set] = None
    #: perf_counter at submit, for usage.knurlogic.timing
    submitted: float = 0.0

    def cancel(self) -> None:
        """From the HTTP thread: the client went away; free the row."""
        self.cancelled = True


class PromptCache:
    """mlx-lm's LRUPromptCache, with the one rule it lacks: an exact hit is
    returned one token short (trimmed; a cache that cannot trim is a miss),
    so there is always a token to process -- an exact hit otherwise leaves
    nothing and kills the generation thread (seen with GLM, none then
    low effort)."""

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

    def hit_length(self, key, tokens) -> int:
        """Tokens fetch() would hand back cached, without the copy fetch
        makes (the admission prices the checkpoints past the hit before it
        fetches). 0 if the trie cannot be asked: every boundary priced, the
        safe side."""
        try:
            r = self.lru._trie.search(key, tokens)
        except Exception:
            return 0
        if r.exact is not None:
            return max(len(tokens) - 1, 0)
        short = len(r.shorter) if r.shorter is not None else 0
        if r.longer is not None and r.common_prefix > short:
            return min(len(tokens) - 1, r.common_prefix)
        return short

    def insert(self, key, tokens, cache, kind: str, origin=None) -> None:
        """`origin`: (event, uid) the cache came from -- what a ring's
        journal names (engine/runtime/tensor.JournalPromptCache)."""
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


def _checkpoints(segs, n: int, hit: int) -> List[int]:
    """The lengths the engine copies the row's cache at: every segment's
    end but the last (the prompt's own end is stored when the row
    finishes), past the prompt cache's hit."""
    out, at = [], 0
    for seg in segs[:-1]:
        at += len(seg)
        if hit < at < n:
            out.append(at)
    return out


@dataclass
class _Row:
    job: Job
    text: Request
    types: List[str]          # segment types still to label checkpoints
    admitted: float = 0.0     # perf_counter when its prefill was queued
    first: float = 0.0        # ... when its first token came out
    made: int = 0             # tokens it has generated (its context grows)


class Scheduler:
    def __init__(self, host, *, completion_batch_size: int = 32,
                 prefill_step_size: int = 2048, prompt_cache_size: int = 10,
                 prompt_cache_bytes: Optional[int] = None,
                 working_set_bytes: Optional[int] = None,
                 stats: Optional[dict] = None, tensor=None,
                 gpu_in_use=None):
        """`tensor`: rank 0's engine/runtime/tensor.Ring when this model is
        split across ranks; the prompt cache is then count-based only.
        `gpu_in_use`: () -> bytes of GPU memory every process on this
        machine holds (serve.load.gpu_in_use), or None; what the others
        hold comes off the working set."""
        if tensor is not None and prompt_cache_bytes:
            raise ValueError("a tensor-split server's prompt cache is "
                             "count-based: every rank must evict the same "
                             "entries, and bytes are each rank's own. Drop "
                             "the prompt cache byte cap")
        self.tensor = tensor
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
        #: the transient against the context a step spans: "lo" and "hi"
        #: are (context tokens, bytes) at the shortest context seen and
        #: (longest context, largest transient) -- _transient's line
        self._tx: dict = {}
        self._gpu_in_use = gpu_in_use
        #: other processes' GPU bytes, the largest of the recent readings
        self._others: List[int] = []
        self._others_at = 0.0
        #: (fixed bytes, bytes per token) of one row's cache, measured
        self._kv = None
        self._samples: dict = {}
        self.cache = self._new_cache(prompt_cache_size)
        self.stats = stats if stats is not None else {}
        self._jobs: "queue.Queue" = queue.Queue()
        self._commands: "queue.Queue" = queue.Queue()
        self._waiting: List[Job] = []
        self._rows: Dict[int, _Row] = {}
        #: why the last tick left requests waiting (requests()), or None
        self._holding: Optional[str] = None
        #: the prompt cache gave way on a ring and the peers have not yet
        #: said what that freed there (_make_room)
        self._ring_trimmed = False
        self._ex: Optional[LocalExecutor] = None
        self._stop = False
        #: set by abort(): every request from then on gets this error
        self._aborted: Optional[BaseException] = None
        self._wake = threading.Event()
        #: live knobs rank 0 applied, for the other ranks (share_live)
        self._sets: "queue.Queue" = queue.Queue()
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
        job.submitted = time.perf_counter()
        if self._aborted is not None:
            job.outbox.put(("error", self._aborted))
            return job
        self._jobs.put(job)
        self._wake.set()
        return job

    def abort(self, err: BaseException) -> None:
        """From ANY thread, including a signal handler while the scheduler
        thread is blocked in a collective that will never return: answer
        every admitted, waiting and queued request with `err`, and every
        later one too. Only thread-safe queues are written; the scheduler's
        own structures are read, never changed."""
        self._aborted = err
        for row in list(self._rows.values()):
            row.job.outbox.put(("error", err))
        for job in list(self._waiting):
            job.outbox.put(("error", err))
        while True:
            try:
                self._jobs.get_nowait().outbox.put(("error", err))
            except queue.Empty:
                break

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

    def share_live(self, applied: dict) -> None:
        """Knobs this server just applied live (engine/serve.apply_live):
        on a ring, the ones that act on a rank's own engine (plan.SETS) go
        to every other rank as `set` ops. Journaled on the scheduler thread
        (the journal is that thread's); an idle, parked ring is rung to
        take them (Ring.park)."""
        if self.tensor is None:
            return
        from .plan import SETS
        for k, v in applied.items():
            if k in SETS:
                self._sets.put((k, str(v)))
        self._wake.set()

    def _journal_sets(self) -> None:
        while True:
            try:
                k, v = self._sets.get_nowait()
            except queue.Empty:
                return
            self.tensor.journal.add("set", name=k, value=v)

    def _command(self, cmd: "Command") -> "Command":
        self._commands.put(cmd)
        self._wake.set()
        return cmd

    @property
    def width(self) -> int:
        """Rows admitted and decoding right now."""
        return len(self._rows)

    def requests(self) -> dict:
        """What is running and what is waiting, from any thread, without a
        lock: only copies of the scheduler's structures are read (each copy
        is one bytecode under the GIL), never the engine.

        in_flight  rows admitted and generating (at most `capacity`)
        pending    requests waiting: queued, held for memory, or admitted
                   past the batch and waiting for a slot in it
        capacity   the most rows decoded together (decode_concurrency)
        oldest_pending_s  how long the oldest pending one has waited
        holding    why they wait: loading | memory | batch_full | None"""
        cap = int(self.completion_batch_size)
        rows = [r for _, r in sorted(dict(self._rows).items())]
        waiting = list(self._waiting) + list(self._jobs.queue)
        waiting = [j for j in waiting if not j.cancelled]
        past = [r.job for r in rows[cap:]]
        pend = waiting + past
        now = time.perf_counter()
        oldest = max((now - j.submitted for j in pend if j.submitted),
                     default=0.0)
        holding = self._holding if pend else None
        if pend and holding is None:
            holding = "batch_full" if past else "queued"
        return {"in_flight": min(len(rows), cap), "pending": len(pend),
                "capacity": cap, "oldest_pending_s": round(oldest, 1),
                "holding": holding}

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
            if self.tensor is not None:
                try:
                    self.tensor.stop()        # the other ranks leave too
                except Exception:
                    logger.exception("stopping the other ranks")
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
        if self.tensor is not None:
            self._journal_sets()
        self._take_jobs()
        room = self.host.state == "ready" and self._room_to_admit()
        if room:
            self._admit_waiting()
        elif self.host.state in ("empty", "failed") and self._waiting \
                and self._commands.empty():
            why = self.host.error or "no model is loaded"
            for j in self._waiting:
                self._error(j, RuntimeError(f"no model to serve: {why}"))
            self._waiting.clear()
        self._holding = self._why_waiting(room)
        if self._ex is not None and self._rows:
            self._guard_memory()
        if self._ex is not None and self._rows:
            self._step()
            return
        # idle: block until something arrives; on a ring the other ranks
        # sleep too, rather than spin in the next collective
        if self.tensor is not None and self.host.state == "ready":
            self.tensor.park()
        self._wake.wait(0.5 if self._waiting else None)
        self._wake.clear()

    def _why_waiting(self, room: bool) -> Optional[str]:
        if not self._waiting:
            return None
        if self.host.state != "ready":
            return "loading"
        if not room or any(j.waiting_on for j in self._waiting):
            return "memory"
        return None

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
            if self.tensor is not None and (
                    c.kind == "unload"
                    or getattr(self.host, "model", None) is not None):
                # the first load is the ring's; nothing after it
                c.error = ("this server's model is split across "
                           f"{self.tensor.world} ranks; it serves "
                           f"{Path(self.host.path or '').name} until it "
                           "stops. Restart the ring to serve another model")
                c.started.set()
                c.done.set()
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
                self.cache = self._new_cache(self.cache.lru.max_size)
                # what was measured belongs to the model it was measured on
                # (a 35B's slope admitting a 397B's prompt is the abort the
                # guard exists for)
                self._kv, self._samples, self._spike = None, {}, 0
                self._tx = {}
                if c.kind == "load":
                    self.host.load(c.path,
                                   executes_artifact_code=c.executes)
                    self._kv = _kv_from_config(
                        c.path, getattr(self.host, "kv_bits", None))
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

    def _new_cache(self, size: int):
        c = PromptCache(size)
        if self.tensor is None:
            return c
        from .tensor import JournalPromptCache
        return JournalPromptCache(c, self.tensor.journal)

    def _executor(self) -> LocalExecutor:
        if self._ex is not None or self.tensor is None:
            return self._executor_local()
        from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
        from knurlogic.engine.serve import state
        from .tensor import TensorExecutor
        # a drafting head on a pipeline only (rank 0 holds the last layers,
        # so the true final hidden state); vision on either split: rank 0
        # encodes, and every rank embeds the rows it ships (tensor.py)
        pipe = getattr(self.tensor, "split", "tensor") == "pipeline"
        head = state.DRAFT.get("head") if (pipe and state.DRAFT.get("on")) \
            else None
        vision = state.VISION.get("serve")
        gen = MTPBatchGenerator(
            self.host.model, head,
            stats=state.DRAFT if head is not None else state.VISION_STATS,
            vision=vision, why=str(state.DRAFT.get("why") or ""),
            completion_batch_size=self.completion_batch_size,
            prefill_step_size=self.prefill_step_size, stream=self._stream)
        if pipe:
            from .pipeline import coordinate
            coordinate(gen, self.tensor.link.group)
            if head is not None:
                state.DRAFT["batch_installed"] = True
        self._ex = TensorExecutor(gen, self.tensor, over=self._over_local)
        if vision is not None:
            from knurlogic.engine.vision import cachehook
            cachehook.install(self.cache.lru, lambda: (
                state.VISION["serve"].store
                if state.VISION.get("serve") else None))
        return self._ex

    def _over_local(self) -> int:
        """This rank's active memory past its limit (signed; 0 unguarded)."""
        limit = self._limit()
        return self._local_active() - limit if limit else 0

    def _executor_local(self) -> LocalExecutor:
        if self._ex is None:
            from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
            from knurlogic.engine.serve import state
            head = state.DRAFT.get("head") if state.DRAFT.get("on") else None
            vision = state.VISION.get("serve")
            gen = MTPBatchGenerator(
                self.host.model, head,
                stats=state.DRAFT if head is not None else state.VISION_STATS,
                vision=vision, why=str(state.DRAFT.get("why") or ""),
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
                self._hold(job, held)
                continue
            images = vreq.has_images(job.request.messages)
            try:
                self._insert(job)
            except _RingWait:
                held.append(job)        # retried after the next exchange
                continue
            except _Wait:
                self._hold(job, held)
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

    def _hold(self, job: Job, held: list) -> None:
        """A request that does not fit waits for the rows running when it
        was first held -- not for whatever arrives after: under a steady
        stream of small prompts those never all finish, and it waited until
        its client gave up. Once they have all finished and it
        still does not fit, it is refused, as it would be on an idle box."""
        if job.waiting_on is None:
            job.waiting_on = set(self._rows)
        if job.waiting_on & set(self._rows):
            held.append(job)
            return
        self._error(job, OutOfMemory(
            f"this prompt ({job.prompt_tokens} tokens) waited for the "
            f"requests running when it arrived; they have finished and it "
            f"still does not fit beside the ones running now. Retry, send a "
            f"shorter conversation, or serve a smaller model"))

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
            from knurlogic.machine.artifact import context_length
            cap = _context_cap()
            window = cap or context_length(self.host.path or "")
            if window and len(prompt) >= window:
                whose = ("this server's context length is "
                         f"{cap} (KNURLOGIC_CONTEXT_LENGTH)" if cap else
                         f"this model's context length is {window}")
                raise P.PromptError(
                    f"this prompt is {len(prompt)} tokens; {whose}, "
                    f"which leaves no room for an answer")
            if job.max_tokens is None:
                job.max_tokens = window - len(prompt) if window else 1 << 20
            elif cap:
                job.max_tokens = min(job.max_tokens, cap - len(prompt))
            hit = getattr(self.cache, "hit_length", lambda k, t: 0)(
                self.host.model_key, prompt)
            lean = self._make_room(len(prompt),
                                   _checkpoints(segs, len(prompt), hit)) \
                == "lean"
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
            sampling, wire = job.sampling, None
            if self.tensor is not None:
                from .tensor import assign_seed
                sampling = assign_seed(sampling)
                wire = {"penalties": dict(job.penalties or {}),
                        "initial": initial}
            uid = ex.insert(Admission(
                segments=segs, max_tokens=job.max_tokens, cache=cache,
                prefix=prompt[:len(prompt) - len(rest)],
                sampling=sampling, processors=procs, state_machine=sm,
                top_logprobs=job.top_logprobs, report=job.request,
                wire=wire))
        try:
            text = Request(tok.detokenizer, sequences=seqs, stops=job.stops,
                           tool_parser=getattr(tok, "tool_parser", None),
                           tools=job.request.tools,
                           logprobs=job.logprobs or bool(job.top_logprobs),
                           prompt_tokens=len(prompt))
        except BaseException:
            ex.remove([uid])      # admitted but unobserved: never orphaned
            raise
        self._rows[uid] = _Row(job, text, types,
                               admitted=time.perf_counter())
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

    def _margin(self, extra: int = 0) -> int:
        """Room one step's temporaries need: the transient a step of THIS
        model is predicted to make at the context it is about to span (the
        running rows' plus `extra`, a prompt being admitted), with a
        quarter again -- but never below 5% of the working set (at least 4
        GiB). GLM-5.3 on an M4 Max (128 GB) learned 2.6 GiB at 8k-token
        prompts, ran at 116 of a 116.8 GiB limit, and a step at 16k aborted
        Metal: the largest transient seen so far under-reads a longer
        context's."""
        floor = max(4 * GIB, self._working_set() // 20)
        return max(floor, int(self._transient(self._context() + extra)
                              * 1.25))

    def _context(self) -> int:
        """Tokens of context the running rows' next step spans."""
        return sum(r.job.prompt_tokens + r.made for r in self._rows.values())

    def _transient(self, ctx: int) -> int:
        """A step's transient at `ctx` tokens of context: the largest
        measured, or, past the longest context measured, that line carried
        on. A prefill chunk attends over every token before it, so its
        temporaries grow with the context: the 27B on an M3 Ultra (96 GB)
        measured 1.58, 2.40 then 3.39 GiB as four agents' prompts grew to
        98k tokens, and the step that first ran past the margin those left
        aborted Metal. Until two contexts 8192 apart are known, the
        transient is taken as proportional to the context -- an
        overestimate, the safe side."""
        lo, hi = self._tx.get("lo"), self._tx.get("hi")
        if hi is None or ctx <= hi[0]:
            return self._spike
        if hi[0] - lo[0] >= 8192 and hi[1] > lo[1]:
            slope = (hi[1] - lo[1]) / (hi[0] - lo[0])
        else:
            slope = hi[1] / hi[0]
        return max(self._spike, int(hi[1] + slope * (ctx - hi[0])))

    def _limit(self, extra: int = 0) -> int:
        """Active bytes a step may start at: the working set less what
        other processes hold of the GPU and a step's temporaries. 0 =
        unguarded."""
        ws = self._working_set()
        return ws - self._others_bytes() - self._margin(extra) if ws else 0

    def _others_bytes(self) -> int:
        """GPU memory the other processes on this machine hold, the most
        of the last ten readings (at most one every two seconds): the
        wired limit caps everyone's total, and a window server or another
        agent's model grows between readings."""
        if self._gpu_in_use is None:
            return 0
        now = time.monotonic()
        if now - self._others_at >= 2.0:
            self._others_at = now
            total = self._gpu_in_use()
            if total is not None:
                mine = self._local_active() + self._cached()
                self._others = (self._others + [max(total - mine, 0)])[-10:]
        return max(self._others, default=0)

    def _measure(self, before: int, ctx: int = 0) -> None:
        """The step's TRANSIENT: its peak above the larger of where it
        started and where it ended. What it kept (an admitted row's KV,
        checkpoint copies) is growth, not transient -- counted as spike, one
        50k-token admission ratcheted the margin up for the life of the
        load. `ctx`: the context the step spanned."""
        import mlx.core as mx
        spike = int(mx.get_peak_memory()) - max(before, self._here())
        grew = spike > self._transient(ctx) * 1.25 and spike > GIB // 4
        self._spike = max(self._spike, spike)
        if ctx >= 1024 and spike > 0:
            lo, hi = self._tx.get("lo"), self._tx.get("hi")
            if lo is None or ctx < lo[0]:
                self._tx["lo"] = (ctx, spike)
            self._tx["hi"] = (max(ctx, hi[0] if hi else 0),
                              max(spike, hi[1] if hi else 0))
        if grew:
            logger.info("a step's transient measured at %.2f GiB over %d "
                        "tokens of context: the memory margin is now %.2f "
                        "GiB", spike / GIB, ctx, self._margin() / GIB)

    def _reset_peak(self) -> int:
        import mlx.core as mx
        mx.reset_peak_memory()
        return self._here()

    def _here(self) -> int:
        """This process's active memory (a step's transient is measured
        here, never against the peers')."""
        return self._active() if self.tensor is None else \
            self._local_active()

    def _active(self) -> int:
        """Active memory as the guard counts it. On a ring, the tightest
        rank rules: the peers' over-limit from the last exchange
        (tensor.Ring.peers_over_now) is read as if it were here."""
        a = self._local_active()
        peers = self.tensor.peers_over_now() if self.tensor is not None \
            else None
        if peers is not None:
            limit = self._limit()
            if limit:
                a = max(a, limit + peers)
        return a

    def _local_active(self) -> int:
        import mlx.core as mx
        return int(mx.get_active_memory())

    def _cached(self) -> int:
        import mlx.core as mx
        return int(mx.get_cache_memory())

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
        s["fixed"] = max(s.get("fixed", 0), min(_fixed_nbytes(cache), b))
        if "lo" not in s or n < s["lo"][0]:
            s["lo"] = (n, b)
        if "hi" not in s or n > s["hi"][0]:
            s["hi"] = (n, b)
        (n0, b0), (n1, b1) = s["lo"], s["hi"]
        if n1 - n0 >= 2048 and b1 > b0:
            slope = (b1 - b0) / (n1 - n0)
            self._kv = (max(b0 - slope * n0, 0.0), slope)
        else:
            # one length: what cannot grow (a hybrid's recurrent state) is
            # fixed, the rest per token. Charging a Flash's 34 deltanet
            # layers to every token of short runs prices a 24k prompt at
            # 3.4 GiB (0.85 measured) and refuses it
            f = float(s.get("fixed", 0))
            self._kv = (f, max(b1 - f, 0) / n1)

    def _cost(self, n_tokens: int, copies: int) -> int:
        fixed, per = self._kv
        return int(copies * (fixed + per * n_tokens))

    def _need(self, n_tokens: int, checkpoints) -> int:
        """Bytes admitting a prompt of n_tokens adds, measured on the 27B
        (an M3 Ultra) to within 2%: the row's cache; a copy of it at
        each checkpoint (a segment boundary past the prompt cache's hit --
        a chat's trailing one-token segments are two nearly whole copies);
        and, with rows running, the copy the running batch concatenates
        it into, while the row's own is still held. Priced as one row and
        one checkpoint, a 57k-token prompt beside two running rows was
        charged 7.3 GiB, grew memory 21.3, and the next step aborted
        Metal."""
        def one(n):
            return self._cost(n, 1)
        need = one(n_tokens) + sum(one(c) for c in checkpoints)
        if self._rows:
            need += one(n_tokens)
        return need

    def _fits(self, n_tokens: int) -> bool:
        """Could a prompt of n_tokens fit once, counting what the prompt
        cache would give up? No side effects: a request that waits must not
        empty the shared prompt cache on every tick it waits."""
        limit = self._limit(n_tokens)
        if not limit or not self._kv:
            return True
        room = limit - self._active() + (self.cache.nbytes
                                         if self.cache is not None else 0)
        return self._need(n_tokens, ()) <= room

    def _room_for(self, n_tokens: int, checkpoints=()):
        """(fits, need, room) for a prompt of n_tokens with checkpoints at
        these lengths, the prompt cache giving way if that is what it
        takes."""
        limit = self._limit(n_tokens)
        if not limit or not self._kv:
            return True, 0, 0
        need = self._need(n_tokens, checkpoints)
        room = limit - self._active()
        held = self.cache.nbytes if self.cache is not None else 0
        if need > room + held:
            # not even an empty prompt cache would make it fit: evicting
            # for it only threw away the entries a lean admission (or the
            # next request) could have hit
            return False, need, room
        if need > room and held:
            before = self.cache.nbytes
            self.cache.trim_to(max(before - (need - room), 0))
            self._ring_trimmed = self.cache.nbytes < before
            self._release()
            room = limit - self._active()
            logger.info("the prompt cache gave up %.1f GiB for a %d-token "
                        "prompt", (before - self.cache.nbytes) / GIB,
                        n_tokens)
        return need <= room, need, room

    def _make_room(self, n_tokens: int, checkpoints=None) -> str:
        """"full" if a prompt of n_tokens fits with its checkpoints (at
        these lengths; None = one, at its end), "lean" if only without;
        else _Wait (rows are running and will free memory) or OutOfMemory
        (none are)."""
        if checkpoints is None:
            checkpoints = [n_tokens]
        fits, full, room = self._room_for(n_tokens, checkpoints)
        if fits:
            return "full"
        fits, need, room = self._room_for(n_tokens)
        if fits:
            logger.info("a %d-token prompt admitted without checkpoints: "
                        "%.1f GiB free, with them it would take %.1f",
                        n_tokens, room / GIB, full / GIB)
            return "lean"
        limit = self._limit(n_tokens)
        if self._rows:
            raise _Wait()
        # On a ring the peers' number is as of the last exchange: what the
        # prompt cache just gave up here is given up there only when the
        # pops travel (the next exchange; an idle rank 0 sends them in
        # park()). Wait for that rather than refuse on a stale number --
        # once nothing is left to give up, the refusal below stands.
        if self.tensor is not None and self.cache is not None and \
                self._ring_trimmed:
            self._ring_trimmed = False
            raise _RingWait()
        raise OutOfMemory(
            f"this prompt ({n_tokens} tokens) needs about {need / GIB:.1f} "
            f"GiB for its cache; {max(room, 0) / GIB:.1f} GiB is free under "
            f"the server's limit ({limit / GIB:.1f} GiB). Send a shorter "
            f"conversation, or serve a smaller model")

    def _guard_memory(self) -> None:
        limit = self._limit()
        if not limit:
            return
        # mlx's freed buffers stay wired until cleared, and the step's
        # transient lands on top of them: past the limit, they go first
        if self._cached() and self._local_active() + self._cached() > limit:
            self._release()
        over = self._active() - limit
        if over <= 0:
            return
        if self.cache is not None and self.cache.nbytes:
            before = self.cache.nbytes
            # the limit already leaves a step's margin; trimming another
            # would evict a margin's worth of prompt cache for nothing
            self.cache.trim_to(max(before - over, 0))
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
        ctx = self._context()
        before = self._reset_peak()
        try:
            events = ex.step()          # on the executor's own stream
        except Exception as e:
            logger.exception("a step failed; failing its rows")
            self._fail_all(e)
            self._close_executor()
            return
        self._measure(before, ctx)
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
                                      row.types.pop(0),
                                      origin=("checkpoint", e.uid))
            elif isinstance(e, Token):
                row.made += 1
                if not row.first:
                    row.first = time.perf_counter()
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
                                  "assistant", origin=("finished", e.uid))
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
        report = cache_report.of(row.job.request)
        usage = row.text.usage(report)
        usage.setdefault("knurlogic", {})["timing"] = _timing(
            row, time.perf_counter(), usage.get("completion_tokens", 0),
            (report or {}).get("prefilled"))
        row.job.outbox.put(("done", usage))

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
