"""A cluster job's files, its ranks' progress markers, and the verdict on
whether it is still healthy. Stdlib only: the page reads these, and the
page never imports mlx.

Each rank writes ~/.cache/knurlogic/jobs/<job>/rank<r>.json (phase, step
counter, work in flight, time) every MARK_S seconds and at every phase
change; the page that started it tears the whole job down on a gone pid, a
rank that never joined, or rank 0 busy with a step counter unmoved for
STALL_S. Idle is not stalled.

Design: docs/design/cluster.md (jobs).
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from pathlib import Path

from knurlogic.cluster import every
from knurlogic.machine import files, paths, processes

#: how often a rank rewrites its marker
MARK_S = 2.0
#: rank 0 busy and its step counter still for this long: stalled
STALL_S = 120.0
#: a rank still joining the ring after this long never will
JOIN_S = 300.0
#: SIGTERM, then this long, then SIGKILL: past rank 0's own 15 s to stop
#: the ring between steps (http.watch_ring), warm-up included
GRACE_S = 25.0
#: after the SIGKILL, this long for the process to be gone: a rank holding
#: 50 GiB of Metal memory takes seconds to give it back, and a job is not
#: stopped -- nor its memory free -- until its pid is
REAP_S = 30.0
#: a peer page of a job unreachable (or not running its rank) this long:
#: the job is gone, and this page's ranks go too
PEER_GONE_S = 20.0
#: the self-heal deadline set after load, when the fork is present
JACCL_TIMEOUT_MS = 60000
#: a job id: what the coordinator mints; nothing else is a path part
JOB_RX = re.compile(r"[0-9a-f]{8,32}")


def root() -> Path:
    base = paths.cache_dir() / "jobs"
    base.mkdir(parents=True, exist_ok=True)
    return base


def log_tail(path, n: int = 64 << 10) -> str:
    """The last `n` bytes of a rank's or server's log, as text ("" when it
    cannot be read)."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(fh.tell() - n, 0))
            return fh.read().decode("utf-8", "replace")
    except (OSError, TypeError):
        return ""


def job_dir(job: str) -> Path:
    if not JOB_RX.fullmatch(str(job or "")):
        raise ValueError(f"not a job id: {str(job)[:40]!r}")
    d = root() / job
    d.mkdir(parents=True, exist_ok=True)
    return d


def marker_path(job: str, rank: int) -> Path:
    return job_dir(job) / f"rank{int(rank)}.json"


def read_marker(job: str, rank: int) -> dict | None:
    d = files.read_json(marker_path(job, rank))
    return d if isinstance(d, dict) else None


# ------------------------------------------------------------- rank side

class Marker:
    """One rank's progress marker. `set(**fields)` changes it (written at
    once on a phase change); a daemon thread rewrites it every MARK_S with
    whatever `probe()` adds, so the time moves while the rank is healthy."""

    def __init__(self, job: str, rank: int, probe=None):
        self.path = marker_path(job, rank)
        self.doc: dict = {"job": job, "rank": int(rank), "pid": os.getpid(),
                    "phase": "joining", "step": 0, "chunk": 0, "busy": None,
                    "t": time.time(), "since": time.time()}
        self.probe = probe
        self._lock = threading.Lock()
        self._last = 0.0

    def set(self, **fields) -> None:
        with self._lock:
            phase = fields.get("phase")
            if phase and phase != self.doc["phase"]:
                self.doc["since"] = time.time()
                self._last = 0.0          # a phase change is written now
            turn = fields.get("turn")
            if turn and turn != self.doc.get("turn"):
                self._last = 0.0          # so is a load-order turn
            self.doc.update({k: v for k, v in fields.items()
                             if v is not None})
            now = time.time()
            if now - self._last >= MARK_S:
                self._flush(now)

    def bump(self, field: str) -> None:
        """Count one more unit of work in `field` (a prefill chunk)."""
        with self._lock:
            self.doc[field] = int(self.doc.get(field) or 0) + 1
            now = time.time()
            if now - self._last >= MARK_S:
                self._flush(now)

    def _flush(self, now: float) -> None:
        self.doc["t"] = now
        self._last = now
        try:
            files.write_json(self.path, self.doc)
        except OSError:
            pass

    def beat(self) -> None:
        extra: dict = {}
        if self.probe is not None:
            try:
                extra = self.probe() or {}
            except Exception:  # the heartbeat thread must go on without a probe
                logging.getLogger(__name__).debug(
                    "rank probe failed", exc_info=True)
                extra = {}
        with self._lock:
            self.doc.update({k: v for k, v in extra.items() if v is not None})
            self._flush(time.time())

    def start(self) -> Marker:
        self.beat()
        every(MARK_S, self.beat, "knurlogic-marker")
        return self


# ------------------------------------------------------------- page side

class Watch:
    """The page's memory of each rank's step counter, so a stall is
    measured on the page's own clock -- a rank whose marker thread cannot
    run (GIL held across a blocked eval) is judged the same way."""

    def __init__(self):
        self.seen: dict = {}              # (job, rank) -> (step, since)

    def verdict(self, job: str, ranks: list, alive, now: float | None = None,
                read=read_marker) -> str:
        """"" when healthy, else why the job has failed (check's reason)."""
        return self.check(job, ranks, alive, now, read)[0]

    def check(self, job: str, ranks: list, alive, now: float | None = None,
              read=read_marker) -> tuple[str, str | None]:
        """("", None) when healthy, else (why the job has failed, its
        recovery kind when the watchdog knows it: "stall"). `ranks`: the
        ranks this page started, [{"rank", "pid", "machine"}]; `alive(pid)`
        says whether that rank's process is still there."""
        now = time.time() if now is None else now
        for r in ranks:
            if r.get("stopping"):
                continue                  # being torn down already
            rank = int(r["rank"])
            ms = r.get("machines") or []
            who = r.get("machine") or (ms[rank] if rank < len(ms)
                                       else None) or "this machine"
            if not alive(int(r["pid"])):
                return f"rank {rank} on {who} (pid {r['pid']}) exited", None
            m = read(job, rank) or {}
            phase = m.get("phase") or "joining"
            since = next((v for v in (m.get("since"), r.get("t"))
                          if v is not None), now)
            if phase == "joining" and now - float(since) > JOIN_S:
                return (f"rank {rank} on {who} has not joined the ring in "
                        f"{JOIN_S:.0f} s"), None
            step = int(m.get("step") or 0)
            # a step, or a prefill chunk inside one (a long prompt is one
            # step of many chunks), is progress
            work = (step, int(m.get("chunk") or 0))
            key = (job, rank)
            was = self.seen.get(key)
            # the clock runs only while work is in flight: a ring idle for
            # an hour and then given a request has not been stalled an hour
            # and only on a loaded ring: a request that arrives while the
            # ranks read their weights (137 GiB over SMB) waits on the load
            if was is None or was[0] != work or not m.get("busy") \
                    or phase != "ready":
                self.seen[key] = (work, now)
                continue
            if m.get("busy") and now - was[1] > STALL_S:
                return (f"stopped: {who} stalled mid-step; its GPU may be "
                        f"hung -- check it, then load again (rank {rank} had "
                        f"work in flight and took no step in "
                        f"{now - was[1]:.0f} s, step {step})"), "stall"
        return "", None

    def forget(self, job: str) -> None:
        for k in [k for k in self.seen if k[0] == job]:
            self.seen.pop(k)


def phase_of(job: str, ranks: list, read=read_marker) -> str:
    """The job's phase from its markers: the least advanced rank's."""
    order = ("joining", "loading", "ready")
    got = []
    for r in ranks:
        p = (read(job, int(r["rank"])) or {}).get("phase") or "joining"
        got.append(order.index(p) if p in order else 0)
    return order[min(got)] if got else "joining"


# ------------------------------------------------------------- registry
# Ranks this machine started, keyed "<job>/<rank>": a follower binds no
# port, so the port-keyed servers.json cannot hold it. Rank 0 also goes in
# servers.json under its port, so chat, the relay and residency find it.

def registry_path() -> Path:
    return root() / "jobs.json"


def registry() -> dict:
    return {k: v for k, v in files.read_json(registry_path(), {}).items()
            if isinstance(v, dict) and isinstance(v.get("pid"), int)}


def save_registry(reg: dict) -> None:
    files.write_json(registry_path(), reg)


def by_job(reg: dict | None = None) -> dict:
    """{job: [rank records]} from the registry."""
    out: dict = {}
    for key, rec in sorted((reg if reg is not None else registry()).items()):
        out.setdefault(rec.get("job") or key.split("/")[0], []).append(rec)
    return out


def is_rank(pid: int, job: str) -> bool:
    """Alive, and still this job's process (a pid is reused once free). A
    recorded pid whose command line cannot be read (AccessDenied, a zombie)
    is alive while it exists: taking it for gone let a relaunch start
    beside a rank still holding its GPU."""
    out = processes.command(pid)
    if not out:
        return processes.exists(pid)
    return f"--job {job}" in out or f"--job={job}" in out


def pids_of_job(job: str) -> list:
    """Every process on this machine running a rank of `job` (by its
    command line), records or not."""
    if not JOB_RX.fullmatch(str(job or "")):
        return []
    return processes.matching(rf"[-]-job[ =]{re.escape(job)}( |$)")


def mid_step(job: str, rank: int, pid: int, read=read_marker) -> bool:
    """Rank `rank` of `job`, process `pid`, says by its marker that it is
    inside a step (work in flight) or stuck in one. Killed there, a rank
    left its GPU's command queue waiting on a collective that never
    completed: the Mac held the job's memory at 100% GPU until a reboot."""
    m = read(job, int(rank)) or {}
    return m.get("pid") == int(pid) and (
        bool(m.get("busy")) or m.get("phase") == "stuck")


def terminate(pids: list, grace: float = GRACE_S, alive=None,
              reap: float = 0.0, spare=None) -> list:
    """SIGTERM every pid, wait up to `grace`, SIGKILL what is left, then
    up to `reap` for the killed to be gone (wait_gone says which are not).
    -> the pids that needed the SIGKILL. A pid this user may not signal
    is not waited on: `alive` counts one it may not inspect as running
    (machine/processes.running), so it would hold the whole grace.
    `spare(pid)` true (a rank mid-step, `mid_step`): never SIGKILLed; it
    is left running and logged as stuck."""
    import signal
    alive = alive or _reaped_alive
    ours = []
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except PermissionError:
            continue
        except OSError:
            pass
        ours.append(pid)
    pids = ours
    end = time.time() + grace
    while time.time() < end and any(alive(p) for p in pids):
        time.sleep(0.1)
    killed = []
    for pid in pids:
        if alive(pid) and spare is not None and spare(pid):
            logging.getLogger(__name__).error(
                "pid %d is mid-step and did not stop in %.0f s: not killed "
                "(killing it there can hang its GPU until a reboot); it is "
                "stuck -- check this machine", pid, grace)
        elif alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
            except OSError:
                pass
    if killed and reap > 0:
        wait_gone(killed, reap, alive)
    return killed


def wait_gone(pids: list, timeout: float, alive=None,
              every: float = 0.1) -> list:
    """Wait up to `timeout` for every pid to be gone. -> the ones still
    there."""
    alive = alive or _reaped_alive
    end = time.time() + timeout
    left = [p for p in pids if alive(p)]
    while left and time.time() < end:
        time.sleep(every)
        left = [p for p in left if alive(p)]
    return left


def _reaped_alive(pid: int) -> bool:
    """Running, a child of ours that has exited reaped on the way."""
    return processes.running(pid, reap=True)
