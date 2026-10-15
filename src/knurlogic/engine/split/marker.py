"""The rank side of a cluster job's progress marker, as the engine sees it:
calls the engine makes from inside a rank (a step taken, a prefill chunk
done, the model loaded), and LOAD_STOP, the flag that stops a rank's
weight read. Stdlib only at import; a no-op outside a job.

The marker itself (cluster/jobs.Marker) is set here by whoever starts the
rank (interfaces/serve.py); engine never imports cluster.
"""
from __future__ import annotations

import faulthandler
import logging
import os
import sys
import threading
import time

logger = logging.getLogger(__name__)

#: busy with no step or prefill chunk for this long: stalled -- the page's
#: watchdog (cluster/jobs.Watch) stops the job; a follower's own
#: (StepWatch) logs where every thread is
STALL_S = 120.0
#: how often a follower's StepWatch looks
WATCH_EVERY_S = 5.0

#: set from a signal handler to stop a load at the next batch boundary
#: (engine/runtime/model_host.evaluate_everything, split/weights, the bell)
LOAD_STOP = threading.Event()

#: this process's marker, when it is a rank of a job; its StepWatch, when
#: it is a follower; whether its prefill chunks are logged (a pipeline
#: follower: where a hung send was is in the log)
CURRENT: dict = {"marker": None, "watch": None, "trace_chunks": False}


def progress(**fields) -> None:
    """From anywhere in a rank (engine included): update its marker; a
    no-op outside a job."""
    m = CURRENT["marker"]
    if m is not None:
        m.set(**fields)


def chunk_start(start: int, end: int, total: int) -> None:
    """A prefill chunk [start, end) of `total` tokens is about to run its
    forward (and, on a pipeline rank, send its activations on). Logged on
    a pipeline follower only: the last line before a hang names the chunk
    whose send never completed."""
    if CURRENT["trace_chunks"]:
        logger.info("prefill chunk %d-%d of %d: forward, then send",
                    start, end, total)


def chunk_done() -> None:
    """A prefill chunk finished (engine/mtp/batch_loop.admit). One step
    admits a whole prompt, so a 30k-token prefill of the 397B over two Macs
    is one step of ~2 minutes: the chunk count is what shows it moving."""
    m = CURRENT["marker"]
    if m is not None:
        m.bump("chunk")
    w = CURRENT["watch"]
    if w is not None:
        w.moved()


def dump_threads() -> None:
    """Every thread's stack, to stderr (the rank's log)."""
    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)


class StepWatch:
    """A follower's own step watchdog: inside a step (`begin` .. `end`)
    with no step or prefill chunk for `stall_s`, it logs every thread's
    traceback once -- where a hung collective or send is -- and kills
    nothing (the page decides). Its begin/end also tell the marker the
    rank is busy, and its step count, so the page never kills it mid-step
    (cluster/jobs.mid_step)."""

    def __init__(self, stall_s: float = STALL_S, dump=dump_threads,
                 clock=time.monotonic):
        self.stall_s = stall_s
        self.dump = dump
        self.clock = clock
        self.since: float | None = None
        self.dumped = False

    def begin(self) -> None:
        self.since, self.dumped = self.clock(), False
        progress(busy=True)

    def moved(self) -> None:
        if self.since is not None:
            self.since, self.dumped = self.clock(), False

    def end(self, step: int) -> None:
        self.since = None
        progress(busy=False, step=step)

    def check(self) -> bool:
        """Dump once when stalled. -> whether it dumped now."""
        since = self.since
        if since is None or self.dumped or \
                self.clock() - since <= self.stall_s:
            return False
        self.dumped = True
        logger.error("no step or prefill chunk in %.0f s inside a step; "
                     "every thread's stack follows (nothing is killed)",
                     self.clock() - since)
        self.dump()
        return True

    def start(self, every: float = WATCH_EVERY_S) -> StepWatch:
        """Look every `every` s on a daemon thread, in a rank of a job
        (a marker set); a ring run in a test's threads starts none."""
        if CURRENT["marker"] is None:
            return self

        def loop():
            while True:
                time.sleep(every)
                self.check()
        threading.Thread(target=loop, name="knurlogic-step-watch",
                         daemon=True).start()
        return self


def after_load() -> None:
    """The model is in memory: arm the jaccl self-heal deadline, if the
    page asked for one (it does only when the fork is installed and the
    link is jaccl). Read live by libjaccl on every collective."""
    from knurlogic.tuning import knobs
    ms = knobs.env_int("KNURLOGIC_JACCL_TIMEOUT_MS")
    if ms is not None and ms >= 0:
        os.environ["JACCL_COLLECTIVE_TIMEOUT_MS"] = str(ms)
    progress(phase="ready")
