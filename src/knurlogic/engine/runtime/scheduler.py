"""The scheduler: ONE thread that owns the MLX stream (docs/design/server.md).

    HTTP threads --submit(Job)--> queue --> tokenize --> prompt cache
      --> executor.insert --> executor.step --> events --> per-request text
      (runtime/request.py) --> the Job's outbox --> HTTP threads

Everything that touches the model happens here, in order. A failure is per
request (a failed tokenize/admit, a RowFailure, or a raising step fails
only its rows; the executor is rebuilt); the thread lives.

Memory is guarded on this thread too, before each step and at each
admission: the guard's methods are MemoryGuard's (runtime/memory_guard.py),
the prompt cache's commands PromptCacheCommands' (engine/prompt_cache/
commands.py); Scheduler inherits both.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from knurlogic.engine.prompt_cache.commands import (
    Command,
    PromptCacheCommands,
)
from knurlogic.engine.prompt_cache.memory import PromptCache, _owner
from knurlogic.machine.memory.pressure import (
    own_compressed_bytes,
    system_pressure_level,
)

from . import prompt as P
from .executor import (
    Admission,
    Checkpoint,
    Finished,
    LocalExecutor,
    Progress,
    RowFailure,
    Token,
)
from .memory_guard import MemoryGuard, OutOfMemory, _RingWait, _Wait
from .request import Request, control_machine
from .spans import Spans, step_bucket
from .spans import enabled as spans_enabled

logger = logging.getLogger(__name__)

#: a running prefill's progress line: every this many chunks or seconds
PREFILL_LOG_CHUNKS = 8
PREFILL_LOG_S = 10.0


class RingFailed(RuntimeError):
    """A rank of this split model is gone or stalled: the job is being torn
    down, and every request in flight is answered with a 503."""


def ring_error(exc: BaseException) -> bool:
    """A failure of the link between the ranks, not of one request: the
    ranks out of step (tensor.Desync), a forward a rank could not finish
    (mtp ForwardFailed), or a collective MLX's distributed backends
    raised ("[jaccl] Send failed with error code -12", "[ring] ...")."""
    from .tensor import Desync
    if isinstance(exc, (Desync, ConnectionError)):
        # ConnectionError: the bell (TCP) between the ranks closed, broke or
        # timed out lining up (Link.align)
        return True
    if type(exc).__name__ == "ForwardFailed":
        return True
    msg = str(exc)
    return isinstance(exc, RuntimeError) and (
        "[jaccl]" in msg or "[ring]" in msg)


#: tokens of a context window the warm-up prompt leaves free (chat
#: template, the 2-token answer, slack)
_WARM_MARGIN = 128


def _context_cap() -> int:
    """KNURLOGIC_CONTEXT_LENGTH: the longest prompt + answer a request may
    use, read at every admission so the setting applies live. 0 = none."""
    import os
    try:
        return max(int(os.environ.get("KNURLOGIC_CONTEXT_LENGTH") or 0), 0)
    except ValueError:
        return 0


def _kv_from_config(path, kv_bits=None) -> tuple | None:
    """(0, bytes per token) from the artifact's config -- its full-attention
    layers' K and V -- so the first prompt after a load is costed before a
    cache has been measured; the measurements replace it. None if the
    config does not say."""
    try:
        from knurlogic.machine.artifact import Artifact
        from knurlogic.tuning.fit import kv_bytes_per_token
        cfg = Artifact.load(path).raw_config
        per, _why = kv_bytes_per_token(cfg.get("text_config", cfg), kv_bits)
        return (0.0, float(per)) if per > 0 else None
    except (OSError, ValueError, KeyError, AttributeError, TypeError):
        return None


# Under this much fresh prefill a rate is noise (the page's own floor):
# nothing is sent, so nothing is shown.
PREFILL_MIN_TOKENS = 256
PREFILL_MIN_S = 0.05


def _timing(row, done: float, completion: int, prefilled,
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


@dataclass
class Job:
    """One request as the scheduler takes it."""
    request: P.ChatRequest
    args: P.PromptArgs
    max_tokens: int | None = None    # None: the rest of the context window
    sampling: dict = field(default_factory=dict)      # temp, top_p, ..., seed
    penalties: dict = field(default_factory=dict)     # make_logits_processors
    stops: list[str] = field(default_factory=list)
    logprobs: bool = False
    top_logprobs: int = 0
    #: filled by the scheduler: ("progress", (done, total)) / ("delta",
    #: Delta) / ("error", exc) / ("done", usage)
    outbox: queue.Queue = field(default_factory=queue.Queue)
    cancelled: bool = False
    #: set once tokenized
    prompt_tokens: int = 0
    #: the rows it waits for, once it has had to wait for memory
    waiting_on: set | None = None
    #: perf_counter at submit, for usage.knurlogic.timing
    submitted: float = 0.0
    #: perf_counter when the HTTP layer began building it (0: not stamped)
    received: float = 0.0
    #: where its wall time went, bucket by bucket (spans.py); None when off
    spans: Spans | None = None
    #: the prompt cache entry it was served came from disk, unused since
    #: its restore: {"tokens", "read_ms"} (usage.knurlogic.cache.disk)
    disk: dict | None = None
    #: the request's id (the client's X-Request-Id, else a ULID), as usage.knurlogic.request_id
    request_id: str | None = None
    #: who made it (X-Client-Session / -Role / -Run; telemetry.md): the
    #: owner of the prompt-cache entries it makes. No session: owns nothing
    session: str | None = None
    role: str | None = None
    run: str | None = None
    #: X-Cache-Retain: pin -- its session's entries are never auto-deleted
    pin: bool = False
    #: X-Cache-Keep: latest -- its entries replace its session's earlier
    #: ones (the client never resumes from an earlier step)
    keep_latest: bool = False
    #: where this prompt left its session's longest entry, when it did
    #: (usage.knurlogic.cache.diverged; Scheduler._diverged)
    diverged: dict | None = None

    def cancel(self) -> None:
        """From the HTTP thread: the client went away; free the row."""
        self.cancelled = True


def _checkpoints(segs, n: int, hit: int) -> list[int]:
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
    types: list[str]          # segment types still to label checkpoints
    admitted: float = 0.0     # perf_counter when its prefill was queued
    first: float = 0.0        # ... when its first token came out
    began: float = 0.0        # ... when the first step that computes it began
    made: int = 0             # tokens it has generated (its context grows)
    chunk: int = 0            # the prefill chunk it was fitted at
    prefilling: bool = True   # no token out before the step now running


class Scheduler(MemoryGuard, PromptCacheCommands):
    def __init__(self, host, *, completion_batch_size: int = 32,
                 prefill_step_size: int = 2048, prompt_cache_size: int = 10,
                 prompt_cache_bytes: int | None = None,
                 working_set_bytes: int | None = None,
                 stats: dict | None = None, tensor=None,
                 gpu_in_use=None, compressed=own_compressed_bytes,
                 system_pressure=system_pressure_level):
        """`compressed`: () -> bytes of this process macOS has compressed
        or swapped (own_compressed_bytes), or None not to watch.
        `system_pressure`: () -> macOS's pressure level
        (system_pressure_level); compressed bytes warn only above normal.
        `tensor`: rank 0's engine/runtime/tensor.Ring when this model is
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
        # the load-time warm-up row is not a user's request (requests())
        self._warming = False
        if hasattr(host, "warm"):
            host.warm = self._warm_up
        self.completion_batch_size = completion_batch_size
        self.prefill_step_size = prefill_step_size
        self.cache_bytes = prompt_cache_bytes
        #: the GPU working set the guard keeps under; None = asked of the
        #: framework on first use; 0 = unguarded
        self.working_set = working_set_bytes
        #: measured step transients (bytes above the active memory the
        #: step started and ended at), two lines (_transient): "decode"
        #: keyed by the context the step spans, "prefill" by context x
        #: chunk / UNIT_CHUNK; each {bucket: (x, bytes)}
        self._tx: dict = {"decode": {}, "prefill": {}}
        #: the chunk _make_room fitted the prompt being admitted at, and
        #: each row not yet prefilled's (set on the engine before its step)
        self._chunk_pick: int | None = None
        self._chunk_of: dict[int, int] = {}
        self._gpu_in_use = gpu_in_use
        #: other processes' GPU bytes, the largest of the recent readings
        self._others: list[int] = []
        self._others_at = 0.0
        #: (fixed bytes, bytes per token) of one row's cache, measured
        self._kv: tuple | None = None
        self._samples: dict = {}
        self.cache = self._new_cache(prompt_cache_size)
        #: prompt-cache entries restored from disk at this load and not yet
        #: served: {tuple(tokens): {"tokens", "read_ms"}}
        self._restored: dict = {}
        #: the loaded model's disk key, and {tuple(tokens): {"file",
        #: "owner"}} of its key directory's files: what an admission reads
        #: back on demand (_read_back), kept current at each save and drop
        self._disk_key: dict | None = None
        self._disk_index: dict = {}
        #: the last measured prefill rate (tok/s) of this model: the
        #: save's break-even rule (prompt_disk.not_worth); None: not yet
        self._prefill_tps: float | None = None
        if hasattr(host, "after_bind"):
            host.after_bind = self._restore_disk
        self.stats = stats if stats is not None else {}
        self._jobs: queue.Queue = queue.Queue()
        self._commands: queue.Queue = queue.Queue()
        self._waiting: list[Job] = []
        self._rows: dict[int, _Row] = {}
        #: why the last tick left requests waiting (requests()), or None
        self._holding: str | None = None
        #: this process's compressed/swapped bytes, last read, and when
        self._compressed = compressed
        self._system_pressure = system_pressure
        self._pressure = 0
        self._pressure_level = 1
        self._pressure_at = 0.0
        #: the prompt cache gave way on a ring and the peers have not yet
        #: said what that freed there (_make_room)
        self._ring_trimmed = False
        self._ex: LocalExecutor | None = None
        self._stop = False
        #: set by abort(): every request from then on gets this error
        self._aborted: BaseException | None = None
        #: rank 0 of a split model: the collective error that ended the
        #: ring (_ring_fatal), and who is told so the process can leave
        #: and the job be relaunched (interfaces/http watch_ring)
        self.ring_failed: BaseException | None = None
        self.on_ring_failed = None
        self._wake = threading.Event()
        #: live knobs rank 0 applied, for the other ranks (share_live)
        self._sets: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="knurlogic-scheduler")

    # ------------------------------------------------------ any thread

    def start(self) -> Scheduler:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop = True
        self._wake.set()
        self._thread.join(10)

    def stop_ring(self, timeout: float) -> bool:
        """From any thread: stop at the next step boundary (the scheduler
        thread's own exit sends the other ranks `stop`). True when it got
        there within `timeout`; False when a step never ended -- a peer is
        gone and its collective will not return -- and the caller fails
        what is in flight and exits regardless."""
        self._stop = True
        self._wake.set()
        if self._thread.ident is None:   # never ran: nothing answers its rows
            return False
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def wait_stopped(self, timeout: float) -> bool:
        """From another thread: True once the scheduler thread has finished
        its cleanup, False when it is still in it after `timeout`."""
        if self._thread.ident is None:
            return True
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def submit(self, job: Job) -> Job:
        job.submitted = time.perf_counter()
        if spans_enabled():
            # getattr: a scheduler is also handed bare job-like objects
            # (a ring's control jobs, test doubles) with no HTTP stamp
            received = getattr(job, "received", 0.0)
            job.spans = Spans(received or job.submitted)
            if received:
                job.spans.to("http_build", job.submitted)
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
             force: bool = True) -> Command:
        """Queue a load. `force=False` refuses (Command.error) if requests
        are running or queued when the switch would happen -- decided on
        the scheduler thread, so it cannot race them. Wait on
        Command.done; read Command.error, then host.state."""
        if self.host.state == "empty":
            # the first load: requests arriving now wait for it
            self.host.expect(str(path))
        return self._command(Command("load", str(path), force,
                                     executes_artifact_code))

    def unload(self, *, force: bool = True) -> Command:
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

    def _command(self, cmd: Command) -> Command:
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
        holding    why they wait: loading | memory | batch_full | None
        memory_short  why a minimal prompt would be refused now, or None
                   (memory_short())"""
        cap = int(self.completion_batch_size)
        rows = [r for _, r in sorted(dict(self._rows).items())]
        if self._warming:
            rows = []   # the warm-up is the server's, not a request: the
            # page showed it as one while the model was still arriving
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
                "holding": holding,
                "memory_short": None if self._warming
                else self.memory_short(),
                # a warning, not a refusal: macOS compresses idle pages
                # routinely, and a guard that refuses on it refuses with
                # memory to spare
                "memory_pressure": self._pressure_short()}

    def ahead(self, job) -> int:
        """How many requests wait in front of `job` for admission (0 once
        it is admitted or gone); any thread, copies only, as requests()."""
        waiting = [j for j in list(self._waiting) + list(self._jobs.queue)
                   if not j.cancelled]
        try:
            return waiting.index(job)
        except ValueError:
            return 0

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
            # a broken ring takes no `stop`: it is a collective, and one
            # with a gone peer never returns
            if self.tensor is not None and self.ring_failed is None:
                try:
                    self.tensor.stop()        # the other ranks leave too
                except Exception:  # shutdown must finish whatever the ranks do (logged)
                    logger.exception("stopping the other ranks")
            # nothing is saved on the way out: a prompt cache reaches disk
            # only when a client asks (POST /v1/prompt-cache/save, park)
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
        # the scheduler thread must outlive one bad tick (logged, fails the rows)
        except Exception as exc:
            if self.tensor is not None and ring_error(exc):
                # on a ring, a tick whose collective or bell failed (a Desync
                # in park, a gone peer) leaves the ranks out of step for good
                self._ring_fatal(exc)
                return
            # Nothing here should raise; if it does, fail what is in
            # flight rather than the thread.
            logger.exception("scheduler tick failed")
            self._fail_all(RuntimeError("the scheduler hit an internal "
                                        "error; see the server log"))
            self._close_executor()

    def _tick(self) -> None:
        self._sample_pressure()
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
            if self._rows:
                self._fit_next()
        if self._ex is not None and self._rows:
            self._step()
            return
        # idle: block until something arrives; on a ring the other ranks
        # sleep too, rather than spin in the next collective
        if self.tensor is not None and self.host.state == "ready":
            self.tensor.park()
        self._wake.wait(0.5 if self._waiting else None)
        self._wake.clear()

    def _why_waiting(self, room: bool) -> str | None:
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
            if c.kind == "save":
                c.started.set()
                try:
                    c.result = self._save_session(c.session) if c.session \
                        else self._save_disk()
                    if self.tensor is not None:
                        self.tensor.journal.add(
                            "save_cache", **({"session": c.session}
                                             if c.session else {}))
                finally:
                    c.done.set()
                continue
            if c.kind in ("drop", "pin", "list", "park"):
                c.started.set()
                try:
                    c.result = getattr(self, "_cmd_" + c.kind)(c)
                except Exception as e:  # answered as the command's error (logged)
                    logger.exception("prompt cache %s failed", c.kind)
                    c.error = f"{type(e).__name__}: {e}"
                finally:
                    c.done.set()
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
                # the model going: nothing saved unless a client asked
                self._restored = {}
                self._disk_key, self._disk_index = None, {}
                self._prefill_tps = None
                self.cache = self._new_cache(self.cache.lru.max_size)
                # what was measured belongs to the model it was measured on
                # (a 35B's slope admitting a 397B's prompt is the abort the
                # guard exists for)
                self._kv, self._samples = None, {}
                self._tx = {"decode": {}, "prefill": {}}
                self._chunk_of = {}
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

    def _window(self) -> tuple:
        """(KNURLOGIC_CONTEXT_LENGTH or 0, the context window _insert
        checks a prompt against: that cap, else the model's own; 0 = none)."""
        from knurlogic.machine.artifact import context_length
        cap = _context_cap()
        return cap, cap or context_length(
            getattr(self.host, "path", "") or "")

    def _warm_up(self) -> None:
        """One tiny generation through the real path -- the same insert and
        step a request takes, on every rank of a ring (the other ranks
        follow rank 0's admission) -- run on this thread while the host is
        "warming", so the first user request finds kernels compiled and
        buffers filled. The answer is discarded."""
        # A prompt a little wider than one prompt chunk, so the prefill
        # kernels for the chunk's real shape are compiled here and an MoE
        # model's experts are routed to (and read) by more than a token's
        # worth of rows; a "Hello, world." warm-up left both to the first
        # request (100+ s on a 108 GiB model). Bounded at ~4300 tokens.
        n = min(max(int(self.prefill_step_size) // 3, 1), 1400) + 1
        # ...but under the context window _insert checks, or it raises
        # PromptError and the warm-up never runs: each repeat is at most
        # ~5 tokens, with room kept for the chat template and the answer.
        _, window = self._window()
        if window:
            n = max(min(n, (window - _WARM_MARGIN) // 5), 1)
        job = Job(request=P.ChatRequest(request_type="text",
                                        prompt="Hello, world. " * n),
                  args=P.PromptArgs(), max_tokens=2,
                  sampling={"temp": 0.0}, submitted=time.perf_counter())
        self._warming = True
        try:
            self._insert(job)
            for _ in range(64):     # prefill chunks + 2 tokens; bounded
                if not self._rows:
                    break
                self._step()
        finally:
            self._warming = False

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
        from knurlogic.engine.prompt_cache.ring import JournalPromptCache
        return JournalPromptCache(c, self.tensor.journal)

    def _executor(self) -> LocalExecutor:
        if self._ex is not None or self.tensor is None:
            return self._executor_local()
        from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
        from knurlogic.engine.serve import state

        from .tensor import TensorExecutor
        # a drafting head on either split: rank 0 has the true final hidden
        # state (a pipeline's last layers; a tensor split's all_sums make
        # every layer's output whole on every rank) and drafts; vision on
        # either split: rank 0 encodes, and every rank embeds the rows it
        # ships (tensor.py)
        head = state.DRAFT.get("head") if state.DRAFT.get("on") else None
        vision = state.VISION.get("serve")
        gen = MTPBatchGenerator(
            self.host.model, head,
            stats=state.DRAFT if head is not None else state.VISION_STATS,
            vision=vision, why=str(state.DRAFT.get("why") or ""),
            completion_batch_size=self.completion_batch_size,
            prefill_step_size=self.prefill_step_size, stream=self._stream)
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

    def _ring_fatal(self, exc: BaseException) -> None:
        """Rank 0 of a split model, on the scheduler thread: a collective
        failed or the ranks fell out of step, and no later step can put
        them back. Answer everything with RingFailed, stop at this step
        boundary (no `stop` to the others: _run), and tell whoever leaves
        the process so the page's recovery relaunches the job. Once."""
        if self.ring_failed is not None:
            return
        logger.error("the ring between the ranks failed; the job stops "
                     "here so it can be relaunched", exc_info=exc)
        self.ring_failed = exc
        link = getattr(self.tensor, "link", None)
        if link is not None:
            link.dead = True    # the executor's closing reset must not wait
        self.abort(RingFailed(
            "this model is split across machines and the link between "
            f"them failed ({exc}); the job is restarting -- retry once "
            "it is loaded again"))
        self._stop = True
        self._wake.set()
        cb = self.on_ring_failed
        if cb is not None:
            try:
                cb(exc)
            except Exception:  # leaving must not depend on the callback (logged)
                logger.exception("on_ring_failed")

    def _close_executor(self) -> None:
        if self._ex is not None:
            try:
                self._ex.close()
            except Exception:  # shutdown must finish whatever close raises (logged)
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
        held: list = []
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
            # one request's failure goes to that request; the loop goes on (logged)
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
        from knurlogic.engine.serve import state
        from knurlogic.engine.vision import cachehook
        from knurlogic.engine.vision import request as vreq
        tok = self.host.tokenizer
        sp = getattr(job, "spans", None)
        if sp is not None:
            sp.to("queue")
        ex = self._executor()
        cachehook.sweep()
        with cachehook.admit_guard():
            if vreq.has_images(job.request.messages):
                v = state.VISION.get("serve")
                if v is None:
                    from knurlogic.engine.serve.vision import no_vision_why
                    raise P.PromptError(no_vision_why())
                prompt, segs, types, initial = v.tokenize(
                    P.tokenize, self, tok, job.request, job.args)
                # the pins tokenize took belong to the guard until the row
                # is admitted: a failure before then releases them
                from knurlogic.engine.vision import key as K
                cachehook.pending(v, K.images_in(prompt))
            else:
                prompt, segs, types, initial = P.tokenize(
                    self, tok, job.request, job.args)
            if sp is not None:
                sp.to("tokenize")
            job.prompt_tokens = len(prompt)
            cap, window = self._window()
            if window and len(prompt) >= window:
                whose = ("this server's context length is "
                         f"{cap} (KNURLOGIC_CONTEXT_LENGTH)" if cap else
                         f"this model's context length is {window}")
                raise P.PromptError(
                    f"this prompt is {len(prompt)} tokens; {whose}, "
                    f"which leaves no room for an answer")
            if job.max_tokens is None:
                job.max_tokens = window - len(prompt) if window else 1 << 20
            elif window:
                job.max_tokens = min(job.max_tokens, window - len(prompt))
            s = getattr(job, "session", None)
            if s and getattr(job, "pin", False) and \
                    s not in self.cache.pinned:
                self._pin(s, True)          # sticky for the session
            if s:
                self._read_back(job, prompt)
            hit = getattr(self.cache, "hit_length", lambda k, t: 0)(
                self.host.model_key, prompt)
            if s:
                self._diverged(job, prompt, hit)
            lean = self._make_room(len(prompt),
                                   _checkpoints(segs, len(prompt), hit)) \
                == "lean"
            if sp is not None:
                sp.to("admit_memory")
            chunk = int(self._chunk_pick or self.prefill_step_size)
            cache, rest = self.cache.fetch(self.host.model_key, prompt)
            if self._restored and len(rest) < len(prompt):
                self._disk_hit(job, prompt, len(prompt) - len(rest))
            if sp is not None:
                sp.to("cache_fetch")
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
                wire=wire, chunk=chunk,
                on_chunk=self._prefill_hook(job, len(prompt),
                                            len(prompt) - len(rest))))
        try:
            text = Request(tok.detokenizer, sequences=seqs, stops=job.stops,
                           tool_parser=getattr(tok, "tool_parser", None),
                           tools=job.request.tools,
                           no_tools=job.request.tool_choice == "none",
                           logprobs=job.logprobs or bool(job.top_logprobs),
                           prompt_tokens=len(prompt))
        except BaseException:
            ex.remove([uid])      # admitted but unobserved: never orphaned
            raise
        self._rows[uid] = _Row(job, text, types,
                               admitted=time.perf_counter(), chunk=chunk)
        self._chunk_of[uid] = chunk
        if self.cache_bytes is not None:
            self.cache.trim_to(self.cache_bytes - ex.cache_nbytes)
        if sp is not None:
            sp.to("admit_other")

    def _prefill_hook(self, job: Job, total: int, hit: int):
        """(uid, done, total) -> stop?, called by the engine after every
        prefill chunk of this request's admission: its progress goes to the
        client (a stream writes a keepalive, so a client that hung up is
        noticed), a line goes to the log every PREFILL_LOG_CHUNKS chunks or
        PREFILL_LOG_S seconds, and True asks the engine to stop the prefill
        (the client is gone). The engine honours a stop on a single machine
        only: on a ring every rank must run the same forwards, so there a
        cancelled row is dropped at the next step boundary (_step)."""
        st = {"n": 0, "t0": time.perf_counter()}
        st["logged"] = st["t0"]

        def hook(uid: int, done: int, n: int) -> bool:
            st["n"] += 1
            now = time.perf_counter()
            if st["n"] % PREFILL_LOG_CHUNKS == 0 or \
                    now - st["logged"] >= PREFILL_LOG_S:
                st["logged"] = now
                rate = (done - hit) / max(now - st["t0"], 1e-9)
                logger.info("prefill %d: %d/%d tokens, chunk %d, %.0f tok/s",
                            uid, done, n, st["n"], rate)
            job.outbox.put(("progress", (done, n)))
            return job.cancelled

        return hook

    def _queued(self) -> set:
        f = getattr(self._ex, "queued", None)
        return f() if f is not None else set()

    def _next_admission(self) -> int | None:
        f = getattr(self._ex, "next_admission", None)
        return f() if f is not None else None

    def _requeue(self, uid: int, why: str) -> None:
        """Row `uid`, inserted but not yet prefilled, goes back to the head
        of the queue: admitted again (tokenized, fitted) when there is
        room. It has computed nothing, so nothing is lost."""
        row = self._rows.pop(uid)
        self._chunk_of.pop(uid, None)
        assert self._ex is not None
        self._ex.remove([uid])
        self._release()
        row.job.waiting_on = None
        self._waiting.insert(0, row.job)
        logger.info("%s: a %d-token prompt not yet prefilled waits for the "
                    "rows running", why, row.job.prompt_tokens)

    def _step(self) -> None:
        ex = self._ex
        assert ex is not None
        ctx = self._context()
        nxt = self._next_admission()
        chunk = None
        if nxt is not None:
            chunk = int(self._chunk_of.get(nxt) or self.prefill_step_size)
            ex.set_chunk(chunk)
        before = self._reset_peak()
        now = time.perf_counter()
        for r in self._rows.values():
            if not r.began:
                r.began = now
            r.prefilling = not r.first
            if getattr(r.job, "spans", None) is not None:
                r.job.spans.to(step_bucket(r.prefilling, "gap"), now)
        try:
            events = ex.step()          # on the executor's own stream
        # a failed step fails its rows; the scheduler thread lives on (logged)
        except Exception as exc:
            if self.tensor is not None and ring_error(exc):
                self._ring_fatal(exc)
                return
            logger.exception("a step failed; failing its rows")
            self._fail_all(exc)
            self._close_executor()
            return
        stepped = time.perf_counter()
        for uid, r in self._rows.items():
            if getattr(r.job, "spans", None) is not None:
                r.job.spans.to(step_bucket(
                    r.prefilling, "forward",
                    shared=nxt is not None and uid != nxt), stepped)
        self._measure(before, ctx, chunk)
        self._chunk_of.pop(nxt, None)
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
                    kind = row.types.pop(0)
                    self._superseded(self.cache.insert(
                        self.host.model_key, e.tokens, e.cache, kind,
                        origin=("checkpoint", e.uid),
                        owner=_owner(row.job, e.uid, kind)))
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
                self._superseded(self.cache.insert(
                    self.host.model_key, e.tokens, e.cache, "assistant",
                    origin=("finished", e.uid),
                    owner=_owner(row.job, e.uid, "assistant")))
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
        for uid in [u for u in self._chunk_of if u not in self._rows]:
            del self._chunk_of[uid]
        now = time.perf_counter()
        for r in self._rows.values():
            if getattr(r.job, "spans", None) is not None:
                r.job.spans.to(step_bucket(r.prefilling, "host"), now)

    def _done(self, uid: int) -> None:
        row = self._rows.pop(uid, None)
        if row is None:
            return
        tail = row.text.finish("stop")
        if tail:
            row.job.outbox.put(("delta", tail))
        from knurlogic.engine.prompt_cache import report as cache_report
        report = cache_report.of(row.job.request)
        if report is not None:
            d = row.job.disk or {}
            used = int(report.get("used", 0) or 0)
            report["disk"] = {"tokens": min(int(d.get("tokens", 0)), used),
                              "read_ms": d.get("read_ms", 0.0)
                              if d and used else 0.0}
            if row.job.diverged:
                report["diverged"] = row.job.diverged
        usage = row.text.usage(report)
        done = time.perf_counter()
        timing = _timing(
            row, done, usage.get("completion_tokens", 0),
            (report or {}).get("prefilled"),
            (report or {}).get("used"), row.chunk or self.prefill_step_size)
        sp = getattr(row.job, "spans", None)
        if sp is not None:
            sp.to(step_bucket(row.prefilling, "host"), done)
            timing.update(sp.report(done))
        if timing.get("prefill_tok_s"):
            self._prefill_tps = float(timing["prefill_tok_s"])
        kn = usage.setdefault("knurlogic", {})
        kn["timing"] = timing
        rid = getattr(row.job, "request_id", None)
        if rid:
            kn["request_id"] = rid
        row.job.outbox.put(("done", usage))

    def _error(self, job: Job, err: BaseException) -> None:
        job.outbox.put(("error", err))

    def _fail_all(self, err: BaseException) -> None:
        for row in list(self._rows.values()):
            self._error(row.job, err)
        self._rows.clear()

    def _fail_queued(self, err: BaseException) -> None:
        self._take_jobs()
        for j in self._waiting:
            self._error(j, err)
        self._waiting.clear()
