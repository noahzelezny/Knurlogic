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
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

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
from .request import Request, control_machine
from .spans import Spans, step_bucket
from .spans import enabled as spans_enabled

logger = logging.getLogger(__name__)
GIB = 1 << 30

#: this server's own memory macOS has compressed or swapped out, past which
#: -- while macOS itself reports memory pressure -- the server warns
#: (Scheduler._pressure_short): the model's pages are being paged back in
#: by every step, and a prefill that took 98 s ran 17+ min with both GPUs
#: at 100% (2026-10-05, another process's leak). Compressed pages alone
#: are not pressure: 11.3 GiB sat compressed with 80 GiB unused and no
#: swap (left from an earlier squeeze; a page stays compressed until it is
#: touched), and the card said "slowed" about nothing.
PRESSURE_BYTES = 1 << 30
#: how often that is read (a task_info call; on the scheduler thread)
PRESSURE_EVERY_S = 5.0
#: a running prefill's progress line: every this many chunks or seconds
PREFILL_LOG_CHUNKS = 8
PREFILL_LOG_S = 10.0


def system_pressure_level() -> int:
    """macOS's own memory pressure: kern.memorystatus_vm_pressure_level,
    1 normal, 2 warn, 4 critical. 1 where it cannot be read."""
    import ctypes
    import ctypes.util
    try:
        lib = ctypes.CDLL(ctypes.util.find_library("c"))
        v = ctypes.c_int(0)
        n = ctypes.c_size_t(ctypes.sizeof(v))
        if lib.sysctlbyname(b"kern.memorystatus_vm_pressure_level",
                            ctypes.byref(v), ctypes.byref(n), None, 0):
            return 1
        return int(v.value) or 1
    except (OSError, AttributeError, ValueError, TypeError):
        return 1


def own_compressed_bytes() -> int:
    """Bytes of this process the macOS compressor holds (compressed in RAM
    or swapped out with its segment): task_info(TASK_VM_INFO).compressed.
    0 where it cannot be read."""
    import ctypes
    import ctypes.util
    try:
        lib = ctypes.CDLL(ctypes.util.find_library("System"))
        buf = (ctypes.c_uint64 * 64)()
        cnt = ctypes.c_uint32(ctypes.sizeof(buf) // 4)
        task = ctypes.c_uint32.in_dll(lib, "mach_task_self_")
        if lib.task_info(task, 22, ctypes.byref(buf), ctypes.byref(cnt)):
            return 0
        # task_vm_info: virtual_size, region_count+page_size, resident,
        # resident_peak, device, device_peak, internal, internal_peak,
        # external, external_peak, reusable, reusable_peak, purgeable x3,
        # compressed (u64 index 15)
        return int(buf[15])
    except (OSError, AttributeError, ValueError, TypeError):
        return 0


#: a prefill step's transient is read off a line in context x chunk, in
#: units of this chunk (so x reads in tokens of context at chunk 512)
UNIT_CHUNK = 512
#: the smallest prefill chunk an admission is shrunk to before it waits
MIN_CHUNK = 128


class OutOfMemory(RuntimeError):
    """This row was stopped so the server and the other rows keep running.
    `memory`: the guard's terms when it refused (Scheduler._memory), for
    the 503's body, or None."""

    def __init__(self, message: str = "", memory: dict | None = None):
        super().__init__(message)
        self.memory = memory


def _terms(mem: dict) -> str:
    """Scheduler._memory on one log line, in GiB."""
    def g(v):
        return "-" if v is None else f"{v / GIB:.1f}"
    ranks = mem.get("ranks")
    pairs = [(r.get("rank"), g(r.get("over_limit_bytes")))
             for r in ranks or ()]
    per = f" per rank {pairs}" if ranks else ""
    return (f"rank {mem['rank']} working_set {g(mem['working_set'])} "
            f"others {g(mem['others'])} (readings "
            f"{[g(x) for x in mem['others_readings']]}) margin "
            f"{g(mem['margin'])} (transient {g(mem['transient'])} at chunk "
            f"{mem['chunk']}, samples {mem['transient_samples']}, context "
            f"{mem['context']}+{mem['tokens']}) local_active "
            f"{g(mem['local_active'])} active {g(mem['active'])} cached "
            f"{g(mem['cached'])} prompt_cache {g(mem['prompt_cache'])} "
            f"peers_over {g(mem['peers_over'])}{per} limit "
            f"{g(mem['limit'])} room {g(mem.get('room'))} need "
            f"{g(mem.get('need'))}")


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
        from knurlogic.tuning.resolve import kv_bytes_per_token
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
        except (AttributeError, TypeError, ValueError, RuntimeError):
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
        except (AttributeError, TypeError, ValueError, RuntimeError):
            pass    # an estimate: a layer that cannot report its size counts 0
    return n


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

    def cancel(self) -> None:
        """From the HTTP thread: the client went away; free the row."""
        self.cancelled = True


class PromptCache:
    """mlx-lm's LRUPromptCache, with the one rule it lacks: an exact hit is
    returned one token short (trimmed; a cache that cannot trim is a miss),
    so there is always a token to process -- an exact hit otherwise leaves
    nothing and kills the generation thread (seen with GLM, none then
    low effort)."""

    def __init__(self, max_size: int = 10, max_bytes: int | None = None):
        from mlx_lm.models.cache import LRUPromptCache
        self.lru = LRUPromptCache(max_size=max_size,
                                  **({"max_bytes": max_bytes}
                                     if max_bytes else {}))
        #: the side map: {tuple(tokens): {session, role, run, pinned, file,
        #: saved_at}} of the entries a session made. mlx-lm's LRU evicts
        #: silently (and drops prefixes on insert): pruned against it lazily
        self.owners: dict = {}
        #: sessions pinned (X-Cache-Retain: pin, or POST .../pin): sticky
        #: on this server, for their later entries too
        self.pinned: set = set()
        #: {session: time.time() of its last entry}: a pinned session idle
        #: longer than PARK_IDLE_S is parked (Scheduler._park_idle)
        self.seen: dict = {}

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
        except (AttributeError, TypeError, ValueError, KeyError):
            return 0
        if r.exact is not None:
            return max(len(tokens) - 1, 0)
        short = len(r.shorter) if r.shorter is not None else 0
        if r.longer is not None and r.common_prefix > short:
            return min(len(tokens) - 1, r.common_prefix)
        return short

    def insert(self, key, tokens, cache, kind: str, origin=None,
               owner: dict | None = None) -> None:
        """`origin`: (event, uid) the cache came from -- what a ring's
        journal names (engine/runtime/tensor.JournalPromptCache). `owner`:
        {session, role, run} of the request that made it."""
        self.lru.insert_cache(key, list(tokens), cache, cache_type=kind)
        self.own(tokens, owner)

    def own(self, tokens, owner: dict | None) -> None:
        """Record who an entry (just inserted) belongs to. A new insert of
        the same tokens is a new entry: not on disk until saved again."""
        t = tuple(tokens)
        s = (owner or {}).get("session")
        if not s:
            self.owners.pop(t, None)
            return
        self.owners[t] = {"session": s, "role": owner.get("role"),
                          "run": owner.get("run"),
                          "pinned": s in self.pinned, "file": None,
                          "saved_at": None}
        self.seen[s] = time.time()

    def live(self) -> list:
        """(model, tokens, CacheEntry, meta or None) of every entry in the
        LRU, least recent first; prunes the side map of what is gone."""
        from knurlogic.engine.serve import prompt_disk
        out = [(m, t, e, self.owners.get(tuple(t)))
               for m, t, e in prompt_disk._lru_entries(self.lru)]
        here = {tuple(t) for _, t, _, _ in out}
        for t in [t for t in self.owners if t not in here]:
            del self.owners[t]
        return out

    def of_session(self, session: str) -> list:
        """(model, tokens) of the live entries `session` owns: the one
        selection a drop, a park and a save of a session share."""
        return [(m, t) for m, t, _, meta in self.live()
                if meta is not None and meta["session"] == session]

    def remove(self, model, tokens) -> bool:
        from knurlogic.engine.serve import prompt_disk
        self.owners.pop(tuple(tokens), None)
        return prompt_disk.remove_entry(self.lru, model, tokens)

    def drop(self, session: str) -> int:
        """Every entry of `session` out of memory; forgets its pin."""
        n = sum(self.remove(m, t) for m, t in self.of_session(session))
        self.pinned.discard(session)
        self.seen.pop(session, None)
        return n

    def set_pinned(self, session: str, pinned: bool) -> int:
        """Pin or unpin `session` here (its live entries and its later
        ones). Returns its live entries."""
        (self.pinned.add if pinned else self.pinned.discard)(session)
        n = 0
        for meta in self.owners.values():
            if meta["session"] == session:
                meta["pinned"] = bool(pinned)
                n += 1
        return n

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
    path: str | None = None
    force: bool = True
    executes: bool = False
    #: set when the scheduler has taken it on (or refused it: then `done`)
    started: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    error: str = ""
    #: what a "save" did (engine/serve/prompt_disk.save's counts)
    result: dict | None = None
    #: a "drop" or "pin"'s session, and a "pin"'s word
    session: str | None = None
    pinned: bool = True


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


#: a pinned session with no new entry this long has its entries moved to
#: disk and freed from memory (read back on its next request)
PARK_IDLE_S = 600.0
#: how often idle pinned sessions are looked for
PARK_CHECK_S = 30.0


def _owner(job) -> dict | None:
    """{session, role, run} of the request a cache entry came from, or
    None when it named no session (it owns nothing)."""
    s = getattr(job, "session", None)
    if not s:
        return None
    return {"session": s, "role": getattr(job, "role", None),
            "run": getattr(job, "run", None)}


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


class Scheduler:
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
        #: when idle pinned sessions were last looked for (monotonic)
        self._park_checked = time.monotonic()
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

    def save_prompt_cache(self, session: str | None = None) -> Command:
        """Queue a save of the prompt cache to disk (prompt_disk.save), on
        the scheduler thread between steps; on a ring every rank saves its
        own part. `session`: only that session's newest entry (its longest),
        if not on disk yet -- what a client asks for right after its context
        compacts. Command.result has the counts."""
        return self._command(Command("save", None, session=session))

    def drop_prompt_cache(self, session: str) -> Command:
        """Queue a drop of `session`'s prompt-cache entries: out of memory
        (on a ring every rank's part, a `drop` op) and its files off disk
        under every model's key. Command.result: {"memory", "disk"}."""
        return self._command(Command("drop", None, session=session))

    def pin_prompt_cache(self, session: str, pinned: bool) -> Command:
        """Queue a pin (or unpin) of `session`: its files are then never
        swept, only dropped. Command.result: {"session", "pinned",
        "entries"}."""
        return self._command(Command("pin", None, session=session,
                                     pinned=bool(pinned)))

    def list_prompt_cache(self) -> Command:
        """Queue a listing of the in-memory entries (read between steps).
        Command.result: {"entries": [...], "key_id", "model"}."""
        return self._command(Command("list", None))

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
            # kept across a restart (engine/serve/prompt_disk)
            if self.ring_failed is None:
                self._save_disk()
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
        self._park_idle()
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
            if c.kind in ("drop", "pin", "list"):
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
                # the model going: its prompt cache to disk, restored when
                # it loads again (engine/serve/prompt_disk)
                if getattr(self.host, "model", None) is not None:
                    self._save_disk()
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

    # ------------------------------------------- the prompt cache on disk

    def _save_disk(self, only_new: bool = False,
                   select=None) -> dict | None:
        """The sessions' prompt-cache entries to disk under the loaded
        model's key (engine/serve/prompt_disk); anonymous entries are not
        saved, nor one quicker to recompute than to read back (the
        break-even rule; on a single server only: a ring's ranks must all
        keep the same entries, and each has its own bytes). `only_new`:
        only what is not on disk yet; `select`: these token tuples only.
        Never raises: a save that fails is logged and the unload goes
        on."""
        if self.cache is None:
            return None
        from knurlogic.engine.serve import prompt_disk
        try:
            key = prompt_disk.host_key(self.host)
            if key is None:
                return None
            self.cache.live()               # prunes the side map
            got = prompt_disk.save(
                self.cache.lru, key, owners=self.cache.owners,
                only_new=only_new, select=select,
                prefill_tps=self._prefill_tps if self.tensor is None
                else None,
                model=Path(self.host.path or "").name or None)
            for t, meta in self.cache.owners.items():
                if meta.get("file"):
                    self._disk_index[t] = {
                        "file": Path(meta["file"]),
                        "owner": {k: meta[k] for k in
                                  ("session", "role", "run")}}
            return got
        except Exception:  # never fails an unload or a stop (logged)
            logger.exception("saving the prompt cache to disk failed")
            return None

    def _save_session(self, session: str) -> dict | None:
        """POST /v1/prompt-cache/save {"session"}: that session's newest
        entry -- its longest -- to disk if it is not there yet."""
        mine = [tuple(t) for _, t in self.cache.of_session(session)] \
            if self.cache is not None else []
        if not mine:
            return {"saved": 0, "kept": 0, "skipped": 0, "bytes": 0,
                    "why": [], "entries": 0, "not_worth": 0}
        return self._save_disk(only_new=True, select={max(mine, key=len)})

    def _park_idle(self) -> None:
        """Between steps, at most every PARK_CHECK_S: a pinned session with
        no new entry for PARK_IDLE_S is parked -- its entries saved and
        freed from memory -- on a single server (a ring has no read-back
        yet: parked, it would only miss)."""
        now = time.monotonic()
        if self.tensor is not None or self.cache is None or \
                not self.cache.pinned or self.host.state != "ready" or \
                now - self._park_checked < PARK_CHECK_S:
            return
        self._park_checked = now
        busy = {getattr(r.job, "session", None)
                for r in self._rows.values()} | \
            {getattr(j, "session", None) for j in self._waiting}
        for s in sorted(self.cache.pinned - busy):
            if time.time() - self.cache.seen.get(s, 0.0) > PARK_IDLE_S:
                self._park(s)

    def _park(self, session: str) -> int:
        """`session`'s entries to disk (what is not there yet), then out of
        memory -- one the save did not write (not worth it, unsaveable)
        stays. Its next request reads them back (_read_back). Returns how
        many were freed."""
        mine = {tuple(t) for _, t in self.cache.of_session(session)}
        if mine:
            self._save_disk(only_new=True, select=mine)
        n = 0
        for m, t in self.cache.of_session(session):
            meta = self.cache.owners.get(tuple(t)) or {}
            if meta.get("file") and Path(meta["file"]).exists():
                n += self.cache.remove(m, t)
        if n:
            logger.info("prompt cache: pinned session %s idle; %d entr%s "
                        "parked on disk", session, n,
                        "y" if n == 1 else "ies")
        return n

    def _read_back(self, job: Job, prompt: list) -> None:
        """Before the admission's hit: when an entry of this model on disk
        is a prefix of `prompt` and longer than the best memory hit, read
        it back into the prompt cache (here, on the scheduler thread), so
        fetch() serves it -- and usage.knurlogic.cache.disk says so, like a
        restored hit. Any on-disk entry, pinned or not. A single server
        only: a ring's ranks would each have to read theirs in step."""
        if not self._disk_index or self.tensor is not None or \
                self._disk_key is None:
            return
        hit = self.cache.hit_length(self.host.model_key, prompt)
        best = None
        for t in self._disk_index:
            n = len(t)
            if n > hit + 1 and n <= len(prompt) and \
                    (best is None or n > len(best)) and \
                    tuple(prompt[:n]) == t:
                best = t
        if best is None:
            return
        from knurlogic.engine.serve import prompt_disk
        f = self._disk_index[best]["file"]
        got = prompt_disk.read(self._disk_key, [f])
        if not got:
            self._disk_index.pop(best, None)    # gone or corrupt: a miss
            return
        back = prompt_disk.insert(self.cache.lru, self.host.model_key, got)
        prompt_disk.adopt(self.cache.owners, self.cache.pinned, back)
        self._restored.update(back)

    def _restore_disk(self) -> None:
        """ModelHost.after_bind: what was saved for this model, into the
        new prompt cache, before the warm-up. On a ring a collective (every
        rank restores the same entries, or none)."""
        from knurlogic.engine.serve import prompt_disk
        try:
            key = prompt_disk.host_key(self.host)
        except Exception:  # a key that cannot be made is a miss (logged)
            logger.exception("prompt cache: no key for the loaded model")
            key = None
        self._restored = prompt_disk.restore(
            self.cache.lru, self.host.model_key, key,
            max_bytes=self.cache_bytes,
            link=self.tensor.link if self.tensor is not None else None)
        prompt_disk.adopt(self.cache.owners, self.cache.pinned,
                          self._restored)
        self._disk_key = key
        self._disk_index = {}
        if key is not None and self.tensor is None:
            try:
                self._disk_index = prompt_disk.index(
                    prompt_disk.root() / prompt_disk.key_id(key))
            except OSError:
                logger.exception("prompt cache: indexing the saved entries")

    # ------------------------------------------ sessions' entries (commands)

    def _cmd_drop(self, c: Command) -> dict:
        from knurlogic.engine.serve import prompt_disk
        s = c.session
        mem = self.cache.drop(s) if self.cache is not None else 0
        disk = prompt_disk.drop_files(s)
        for t in [t for t, v in self._disk_index.items()
                  if (v.get("owner") or {}).get("session") == s]:
            del self._disk_index[t]
        for t in [t for t, v in self._restored.items()
                  if (v.get("owner") or {}).get("session") == s]:
            del self._restored[t]
        logger.info("prompt cache: session %s dropped (%d in memory, %d on "
                    "disk)", s, mem, disk)
        return {"session": s, "memory": mem, "disk": disk}

    def _cmd_pin(self, c: Command) -> dict:
        n = self._pin(c.session, c.pinned)
        return {"session": c.session, "pinned": c.pinned, "entries": n}

    def _pin(self, session: str, pinned: bool) -> int:
        from knurlogic.engine.serve import prompt_disk
        n = self.cache.set_pinned(session, pinned) \
            if self.cache is not None else 0
        prompt_disk.set_pin(session, pinned)
        return n

    def _cmd_list(self, c: Command) -> dict:
        from knurlogic.engine.serve import prompt_disk
        key = self._disk_key
        out = []
        for _m, t, e, meta in (self.cache.live() if self.cache else []):
            meta = meta or {}
            ints = all(isinstance(x, int) for x in t)
            f = meta.get("file")
            out.append({
                "session": meta.get("session"), "role": meta.get("role"),
                "run": meta.get("run"), "tokens": len(t),
                "bytes": int(e.nbytes), "in_memory": True,
                "on_disk": bool(f) and Path(f).exists(),
                "saved_at": meta.get("saved_at"),
                "pinned": bool(meta.get("pinned")),
                "hash": prompt_disk.tokens_hash(t) if ints else None})
        return {"entries": out,
                "key_id": prompt_disk.key_id(key) if key else None,
                "model": Path(self.host.path or "").name or None}

    def _disk_hit(self, job: Job, prompt: list, used: int) -> None:
        """The entry fetch() handed `job` is one restored from disk: say so
        in its usage (usage.knurlogic.cache.disk). Once the session's next
        answer is cached, the entry it hits is that one, made in memory --
        a memory hit."""
        from knurlogic.engine.serve import prompt_disk
        src = prompt_disk.source_of(self.cache.lru, self.host.model_key,
                                    prompt, used)
        d = self._restored.get(src) if src is not None else None
        if d is not None:
            job.disk = {"tokens": min(int(d["tokens"]), used),
                        "read_ms": d["read_ms"]}

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

    # ------------------------------------------------------------- memory

    def _memory(self, n_tokens: int = 0, need: int | None = None,
                room: int | None = None, chunk: int | None = None) -> dict:
        """The guard's terms for a prompt of n_tokens prefilled `chunk`
        tokens at a time (None: the configured chunk), in bytes: limit =
        working_set - others - margin; room = limit - active, on a ring the
        tighter of that and the peers' (_room). Which term made a refusal
        is read off this (the log line, the 503's `memory`)."""
        peers = self.tensor.peers_over_now() if self.tensor is not None \
            else None
        if n_tokens and chunk is None:
            chunk = self.prefill_step_size
        out = {"rank": int(getattr(getattr(self.tensor, "link", None),
                                   "rank", 0) or 0),
               "tokens": int(n_tokens),
               "chunk": int(chunk) if chunk else None,
               "working_set": self._working_set(),
               "others": self._others_bytes(),
               "others_readings": list(self._others),
               "margin": self._margin(n_tokens, chunk),
               "transient": self._transient(self._context() + n_tokens,
                                            chunk),
               "transient_samples": {k: sorted(v.values())
                                     for k, v in self._tx.items()},
               "context": self._context(),
               "local_active": self._local_active(),
               "active": self._active(),
               "cached": self._cached(),
               "prompt_cache": (self.cache.nbytes
                                if self.cache is not None else 0),
               "peers_over": peers,
               "limit": self._limit(n_tokens, chunk)}
        if peers is not None:
            from knurlogic.engine.serve import state
            out["ranks"] = [dict(r) for r in
                            state.SERVED.get("ranks") or []]
        if need is not None:
            out["need"] = int(need)
        if room is not None:
            out["room"] = int(room)
        return out

    def memory_short(self, n_tokens: int = 1024) -> str | None:
        """Why a minimal prompt would be refused now, or None: a loaded
        server that cannot admit one is not ready (/v1/models, the page's
        instance card). Only with nothing running -- then a refusal is
        final, not a wait. Read at the smallest chunk, the one a refusal
        is made at, and with no reserve for a prompt nobody sent: the
        margin is the step about to run's (_margin). Side-effect free
        (_fits). Memory pressure is reported apart (requests()'s
        memory_pressure), never as a refusal."""
        try:
            if self._rows or not self._kv or not self._limit() or \
                    self._fits(n_tokens):
                return None
            m = self._memory(n_tokens, chunk=MIN_CHUNK)
        except Exception:   # a status read must never fail the server
            return None
        peers = m["peers_over"]
        if peers is not None and \
                self._peer_room(peers, n_tokens, MIN_CHUNK) < \
                m["limit"] - m["local_active"]:
            over = [r for r in m.get("ranks", [])
                    if r.get("rank") and r.get("over_limit_bytes", 0) > 0]
            who = (f"rank {over[0]['rank']}" if len(over) == 1
                   else "a peer rank")
            why = (f"{peers / GIB:.1f} GiB over its limit" if peers > 0
                   else f"{-peers / GIB:.1f} GiB under its limit, short of "
                   f"a {n_tokens}-token step's "
                   f"{m['transient'] / GIB:.1f} GiB transient")
        else:
            who = f"rank {m['rank']}"
            why = (f"limit {m['limit'] / GIB:.1f} GiB (working set "
                   f"{m['working_set'] / GIB:.1f} - others "
                   f"{m['others'] / GIB:.1f} - margin "
                   f"{m['margin'] / GIB:.1f}), active "
                   f"{m['local_active'] / GIB:.1f}")
        return f"loaded, no memory for requests: {who}, {why}"

    # ---------------------------------------------------- pressure, prefill

    def _sample_pressure(self, now: float | None = None) -> None:
        """Read this process's compressed/swapped bytes and macOS's pressure
        level every PRESSURE_EVERY_S; one WARNING when both say so (past
        PRESSURE_BYTES, level above normal), one INFO when either drops
        back. Compressed pages with the system at normal are not pressure
        (PRESSURE_BYTES)."""
        if self._compressed is None:
            return
        now = time.monotonic() if now is None else now
        if self._pressure_at and now - self._pressure_at < PRESSURE_EVERY_S:
            return
        self._pressure_at = now
        try:
            b = int(self._compressed() or 0)
            level = int(self._system_pressure() or 1) \
                if self._system_pressure is not None else 1
        except Exception:   # a reading must never fail the server
            b, level = 0, 1
        was = self._pressure_short() is not None
        self._pressure, self._pressure_level = b, level
        now_short = self._pressure_short()
        if now_short and not was:
            logger.warning("%s; requests are still admitted", now_short)
        elif was and not now_short:
            logger.info("memory pressure over: %.1f GiB of this server "
                        "compressed, macOS pressure level %d", b / GIB, level)

    def _pressure_short(self) -> str | None:
        """Why the server may run slow for memory pressure, or None."""
        b = self._pressure
        if b <= PRESSURE_BYTES or self._pressure_level < 2:
            return None
        rank = int(getattr(getattr(self.tensor, "link", None), "rank", 0)
                   or 0)
        what = "critical" if self._pressure_level >= 4 else "warn"
        return (f"rank {rank}: macOS reports memory pressure ({what}) and "
                f"has compressed {b / GIB:.1f} GiB of the model")

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

    def _working_set(self) -> int:
        if self.working_set is None:
            import importlib  # engine.serve exports a load() function
            load = importlib.import_module("knurlogic.engine.serve.load")
            self.working_set = int(load.memory().get("working_set_bytes")
                                   or 0)
        return self.working_set

    def _margin(self, extra: int = 0, chunk: int | None = None) -> int:
        """Room the next step's temporaries need: the transient THIS model
        is predicted to make in the step about to run (_transient: the
        running rows' context plus `extra`, a prompt being admitted and
        prefilled `chunk` tokens at a time -- None with no prompt: a decode
        step), with a quarter again -- but never below 5% of the working
        set (at least 4 GiB). Never the largest transient ever seen: one
        59k-token prefill's 17 GiB (GLM-5.3-Flash, 2026-10-05) stayed the
        margin of every later step, idle included, and every prompt was
        refused with 30-60 GiB unused across the ring."""
        floor = max(4 * GIB, self._working_set() // 20)
        if extra and chunk is None:
            chunk = self.prefill_step_size
        return max(floor, int(self._transient(self._context() + extra,
                                              chunk) * 1.25))

    def _context(self) -> int:
        """Tokens of context the running rows' next step spans."""
        return sum(r.job.prompt_tokens + r.made for r in self._rows.values())

    def _transient(self, ctx: int, chunk: int | None = None) -> int:
        """A step's predicted transient at `ctx` tokens of context, read
        off two measured lines. A step that admits a prompt prefills it
        `chunk` tokens at a time (the whole admission is one step), each
        chunk's temporaries spanning the chunk against the context before
        it: the prefill line is keyed by x = ctx x chunk / UNIT_CHUNK, so
        a smaller chunk reads lower (GLM-5.3-Flash at chunk 2048, its DSA
        indexer scoring chunk x context: 3.2 GiB at 2.7k tokens, 10.4 at
        33k, 17.0 at 59k). A decode step (`chunk` None) is keyed by the
        context alone. A prefill step decodes the running rows too: never
        under the decode line's reading."""
        dec = self._line("decode", ctx)
        if chunk is None:
            return dec
        return max(dec, self._line("prefill", self._prefill_x(ctx, chunk)))

    def _prefill_x(self, ctx: int, chunk: int) -> int:
        """The prefill line's key: the context a chunk's temporaries span
        (the model's `prefill_span`, else all of it) x chunk / UNIT_CHUNK.
        GLM-5.3's latent prefill reads at most index_topk keys a query: keyed
        by the whole context, one 2.7k-token warm-up carried in proportion
        refused a 339k-token prompt even at chunk 128."""
        span = self._span_fn()
        try:
            c = int(span(ctx)) if span is not None else ctx
        except Exception:  # a model's hint must never fail the guard
            c = ctx
        return c * int(chunk) // UNIT_CHUNK

    def _span_fn(self):
        """The served model's prefill_span, wherever the wrappers put it
        (the model, its language_model, its trunk), or None."""
        m = getattr(self.host, "model", None)
        if getattr(self, "_span_for", self) is not m:   # read per model
            self._span_for, self._span = m, None
            for o in (m, getattr(m, "language_model", None),
                      getattr(m, "model", None)):
                f = getattr(o, "prefill_span", None)
                if callable(f):
                    self._span = f
                    break
        return self._span

    def _line(self, kind: str, x: int) -> int:
        """The measured `kind` transients read at x: between two samples,
        the line between them; past the longest, the line through the
        shortest and longest carried on (once they are 8192 apart and
        rising, else in proportion to x); below the shortest, the larger
        of that first segment and the proportion. With one sample, in
        proportion to x -- the safe side; a shrinking chunk is what makes a
        long prompt fit then, not a guessed slope. Never under a sample
        at or below x. 0 until measured."""
        pts = sorted(self._tx.get(kind, {}).values())
        if not pts or x <= 0:
            return 0
        (x0, y0), (x1, y1) = pts[0], pts[-1]
        if len(pts) == 1:
            est = y0 * x / x0
        elif x <= x0:
            xb, yb = pts[1]
            est = max(y0 * x / x0, y0 + (yb - y0) * (x - x0) / (xb - x0))
        elif x >= x1:
            slope = ((y1 - y0) / (x1 - x0) if x1 - x0 >= 8192 and y1 > y0
                     else y1 / x1)
            est = y1 + slope * (x - x1)
        else:
            k = next(i for i, (px, _) in enumerate(pts) if px >= x)
            (xa, ya), (xb, yb) = pts[k - 1], pts[k]
            est = ya + (yb - ya) * (x - xa) / (xb - xa)
        seen = max((py for px, py in pts if px <= x), default=0)
        return max(int(est), seen, 0)

    def _limit(self, extra: int = 0, chunk: int | None = None) -> int:
        """Active bytes a step may start at: the working set less what
        other processes hold of the GPU and the next step's temporaries
        (_margin). 0 = unguarded."""
        ws = self._working_set()
        return ws - self._others_bytes() - self._margin(extra, chunk) \
            if ws else 0

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

    def _measure(self, before: int, ctx: int = 0,
                 chunk: int | None = None) -> None:
        """The step's TRANSIENT: its peak above the larger of where it
        started and where it ended. What it kept (an admitted row's KV,
        checkpoint copies) is growth, not transient -- counted as spike, one
        50k-token admission ratcheted the margin up for the life of the
        load. `ctx`: the context the step spanned; `chunk`: the step
        prefilled a row at that chunk (the prefill line), None: a decode
        step (the decode line)."""
        import mlx.core as mx
        spike = int(mx.get_peak_memory()) - max(before, self._here())
        kind = "decode" if chunk is None else "prefill"
        x = ctx if chunk is None else self._prefill_x(ctx, chunk)
        grew = spike > self._line(kind, x) * 1.25 and spike > GIB // 4
        if ctx >= 1024 and spike > 0 and x > 0:
            # one sample per eighth of a doubling of x, the largest kept
            pts = self._tx.setdefault(kind, {})
            b = int(math.log2(x) * 8)
            if b not in pts or spike > pts[b][1]:
                pts[b] = (x, spike)
        if grew:
            logger.info("a %s step's transient measured at %.2f GiB over "
                        "%d tokens of context%s", kind, spike / GIB, ctx,
                        f" at chunk {chunk}" if chunk else "")

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
        assert self._kv is not None     # _cost is asked once one is known
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
        """Could a prompt of n_tokens fit once, at the smallest chunk,
        counting what the prompt cache would give up? No side effects: a
        request that waits must not empty the shared prompt cache on every
        tick it waits."""
        if not self._limit() or not self._kv:
            return True
        room = self._room(n_tokens, MIN_CHUNK) + (
            self.cache.nbytes if self.cache is not None else 0)
        return self._need(n_tokens, ()) <= room

    def _chunks(self) -> list[int]:
        """The prefill chunks an admission may take, largest first: the
        launch chunk, halved down to MIN_CHUNK (2048 -> 1024 -> 512 -> 256
        -> 128)."""
        c = max(int(self.prefill_step_size), 1)
        out = [c]
        while c // 2 >= MIN_CHUNK:
            c //= 2
            out.append(c)
        return out

    def _peer_room(self, peers: int, n_tokens: int, chunk: int | None) -> int:
        """Room on the tightest peer rank for the step admitting n_tokens
        at `chunk`. A peer reports its active memory less its own limit,
        which already holds its floor margin (at least 4 GiB) free; the
        step's transient past that floor is taken from its room -- the
        same transient as here (equal shards under tensor; a pipeline's
        stages run one layer's temporaries at a time, near enough)."""
        t = self._transient(self._context() + n_tokens, chunk)
        return -peers - max(0, int(t * 1.25) - 4 * GIB)

    def _room(self, n_tokens: int = 0, chunk: int | None = None) -> int:
        """Bytes free for a prompt of n_tokens' cache in the step that
        prefills it at `chunk`: under this rank's limit for that step, and
        on a ring under every peer's (_peer_room)."""
        limit = self._limit(n_tokens, chunk)
        if self.tensor is None:
            return limit - self._active()
        room = limit - self._local_active()
        peers = self.tensor.peers_over_now()
        if peers is not None:
            room = min(room, self._peer_room(peers, n_tokens, chunk))
        return room

    def _room_for(self, n_tokens: int, checkpoints=(),
                  chunk: int | None = None):
        """(fits, need, room) for a prompt of n_tokens with checkpoints at
        these lengths, prefilled `chunk` tokens at a time (None: the
        launch chunk), the prompt cache giving way if that is what it
        takes."""
        chunk = chunk or self.prefill_step_size
        if not self._limit(n_tokens, chunk) or not self._kv:
            return True, 0, 0
        need = self._need(n_tokens, checkpoints)
        room = self._room(n_tokens, chunk)
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
            room = self._room(n_tokens, chunk)
            logger.info("the prompt cache gave up %.1f GiB for a %d-token "
                        "prompt", (before - self.cache.nbytes) / GIB,
                        n_tokens)
        return need <= room, need, room

    def _make_room(self, n_tokens: int, checkpoints=None) -> str:
        """"full" if a prompt of n_tokens fits with its checkpoints (at
        these lengths; None = one, at its end), "lean" if only without;
        else _Wait (rows are running and will free memory) or OutOfMemory
        (none are). The chunk it fits at is left in _chunk_pick.

        Cheapest loss first: the prompt cache gives way; then the prefill
        chunk shrinks (2048 -> ... -> 128: a chunk's temporaries span it
        against the whole context before it, and a whole admission is ONE
        step, so the chunk is chosen here, for the whole prompt -- a
        little slower, instead of a refusal); then the checkpoints go;
        then it waits, or with nothing running is refused."""
        if checkpoints is None:
            checkpoints = [n_tokens]
        chunks = self._chunks()
        self._chunk_pick = None
        full = 0
        for c in chunks:
            fits, full, room = self._room_for(n_tokens, checkpoints, c)
            if fits:
                self._picked(c, n_tokens, room)
                return "full"
        for c in chunks:
            fits, need, room = self._room_for(n_tokens, (), c)
            if fits:
                self._picked(c, n_tokens, room)
                logger.info("a %d-token prompt admitted without checkpoints: "
                            "%.1f GiB free, with them it would take %.1f; %s",
                            n_tokens, room / GIB, full / GIB,
                            _terms(self._memory(n_tokens, need, room, c)))
                return "lean"
        limit = self._limit(n_tokens, chunks[-1])
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
        mem = self._memory(n_tokens, need, room, chunks[-1])
        logger.warning("refused a %d-token prompt: %s", n_tokens, _terms(mem))
        if room <= 0 and not (self.cache is not None and self.cache.nbytes):
            # nothing runs and nothing is left to give up, yet no memory is
            # free: whatever holds it is out of this server's reach
            raise OutOfMemory(
                f"no memory is free under the server's limit "
                f"({limit / GIB:.1f} GiB) with nothing running and nothing "
                f"cached; it is holding memory it cannot release, so this "
                f"server needs a restart", mem)
        raise OutOfMemory(
            f"this prompt ({n_tokens} tokens) needs about {need / GIB:.1f} "
            f"GiB for its cache; {max(room, 0) / GIB:.1f} GiB is free under "
            f"the server's limit ({limit / GIB:.1f} GiB) even prefilled "
            f"{chunks[-1]} tokens at a time. Send a shorter conversation, "
            f"or serve a smaller model", mem)

    def _picked(self, chunk: int, n_tokens: int, room: int) -> None:
        self._chunk_pick = chunk
        if chunk < self.prefill_step_size:
            logger.info("a %d-token prompt is prefilled %d tokens at a time "
                        "(not %d) to fit the memory left: a %.1f GiB "
                        "transient predicted, %.1f GiB free for its cache",
                        n_tokens, chunk, self.prefill_step_size,
                        self._transient(self._context() + n_tokens, chunk)
                        / GIB, room / GIB)

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
            if len(self._rows) > 1 and uid in self._queued():
                # not prefilled yet: nothing is lost by queueing it again
                self._requeue(uid, "memory past the limit (%.1f GiB)"
                              % (limit / GIB))
                over = self._active() - limit
                continue
            row = self._rows.pop(uid)
            assert self._ex is not None     # there are rows, so an executor
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

    def _fit_next(self) -> None:
        """Before a step that prefills a row: refit its chunk to the room
        now. It was fitted when inserted; rows inserted in the same tick
        were each fitted against the same free memory, and the rows
        running have grown since. The largest chunk whose step fits on
        every rank is set (on a ring, the `chunk` op carries it to the
        other ranks before the step). If not even MIN_CHUNK fits: with
        other rows running it goes back to the queue (they will free
        memory); alone it is stopped with OutOfMemory. A row once
        prefilled has no chunk left to shrink: its admission is one
        step."""
        uid = self._next_admission()
        row = self._rows.get(uid) if uid is not None else None
        if row is None or not self._kv or not self._limit():
            return
        n = row.job.prompt_tokens
        need = self._cost(n, 2 if len(self._rows) > 1 else 1)
        cur = self._chunk_of.get(uid, self.prefill_step_size)
        for c in self._chunks():
            room = self._room(0, c)
            if need <= room:
                if c != cur:
                    self._chunk_of[uid] = row.chunk = c
                    refit = getattr(self._ex, "refit", None)
                    if refit is not None:
                        refit(uid, c)
                    logger.info("a %d-token prompt about to prefill is "
                                "refitted to chunk %d (was %d): a %.1f GiB "
                                "transient predicted, %.1f GiB free after "
                                "its cache", n, c, cur,
                                self._transient(self._context(), c) / GIB,
                                (room - need) / GIB)
                return
        if len(self._rows) > 1:
            self._requeue(uid, f"no room to prefill at chunk {MIN_CHUNK}")
            return
        self._rows.pop(uid)
        self._chunk_of.pop(uid, None)
        assert self._ex is not None
        self._ex.remove([uid])
        self._release()
        mem = self._memory(0, need, self._room(0, MIN_CHUNK), MIN_CHUNK)
        logger.warning("stopped a %d-token prompt before its prefill: %s", n,
                       _terms(mem))
        self._error(row.job, OutOfMemory(
            f"this prompt ({n} tokens) no longer fits: about "
            f"{need / GIB:.1f} GiB for its cache and a step prefilling it "
            f"{MIN_CHUNK} tokens at a time; retry, or send a shorter "
            f"conversation", mem))

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
                    self.cache.insert(self.host.model_key, e.tokens, e.cache,
                                      row.types.pop(0),
                                      origin=("checkpoint", e.uid),
                                      owner=_owner(row.job))
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
                                  "assistant", origin=("finished", e.uid),
                                  owner=_owner(row.job))
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
        from knurlogic.engine.serve import cache_report
        report = cache_report.of(row.job.request)
        if report is not None:
            d = row.job.disk or {}
            used = int(report.get("used", 0) or 0)
            report["disk"] = {"tokens": min(int(d.get("tokens", 0)), used),
                              "read_ms": d.get("read_ms", 0.0)
                              if d and used else 0.0}
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
