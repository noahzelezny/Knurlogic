"""A cluster job's files, its ranks' progress markers, and the verdict on
whether it is still healthy. Stdlib only: the page reads these, and the
page never imports mlx.

Stock mlx has no collective timeouts, so nothing inside a rank can
interrupt an eval blocked on a peer that died. The failure path is out of
band instead:

  each rank    writes ~/.cache/knurlogic/jobs/<job>/rank<r>.json -- its
               phase, its step counter, whether work is in flight (rank 0
               only knows), and the time -- every MARK_S seconds from a
               daemon thread, and at every phase change
  its page     watches the ranks it started (verdict()): a pid gone, a rank
               that never joined, or rank 0 busy with a step counter that
               has not moved for STALL_S. Idle is not stalled: a ring with
               nothing in flight sits in its exchange forever, correctly.
               On any of them the page tears the whole job down.

With the jaccl self-heal fork (machine/deps.py), a wedged RDMA collective
also throws inside the rank -- but only once JACCL_COLLECTIVE_TIMEOUT_MS is
set, which happens AFTER the load (0 while loading: a cold 400 GB read is
not a hang). the maintainer's pattern, fork d2e82f92 / 43dc7f56.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

#: how often a rank rewrites its marker
MARK_S = 2.0
#: rank 0 busy and its step counter still for this long: stalled
STALL_S = 120.0
#: a rank still joining the ring after this long never will
JOIN_S = 300.0
#: SIGTERM, then this long, then SIGKILL
GRACE_S = 10.0
#: the self-heal deadline set after load, when the fork is present
JACCL_TIMEOUT_MS = 60000
#: a job id: what the coordinator mints; nothing else is a path part
JOB_RX = re.compile(r"[0-9a-f]{8,32}")


def root() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME",
                               Path.home() / ".cache")) / "knurlogic" / "jobs"
    base.mkdir(parents=True, exist_ok=True)
    return base


def job_dir(job: str) -> Path:
    if not JOB_RX.fullmatch(str(job or "")):
        raise ValueError(f"not a job id: {str(job)[:40]!r}")
    d = root() / job
    d.mkdir(parents=True, exist_ok=True)
    return d


def marker_path(job: str, rank: int) -> Path:
    return job_dir(job) / f"rank{int(rank)}.json"


def _write(path: Path, doc: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc))
    tmp.replace(path)                # a reader never sees half a file


def read_marker(job: str, rank: int) -> dict | None:
    try:
        d = json.loads(marker_path(job, rank).read_text())
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------- rank side

class Marker:
    """One rank's progress marker. `set(**fields)` changes it (written at
    once on a phase change); a daemon thread rewrites it every MARK_S with
    whatever `probe()` adds, so the time moves while the rank is healthy."""

    def __init__(self, job: str, rank: int, probe=None):
        self.path = marker_path(job, rank)
        self.doc = {"job": job, "rank": int(rank), "pid": os.getpid(),
                    "phase": "joining", "step": 0, "busy": None,
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
            self.doc.update({k: v for k, v in fields.items()
                             if v is not None})
            now = time.time()
            if now - self._last >= MARK_S:
                self._flush(now)

    def _flush(self, now: float) -> None:
        self.doc["t"] = now
        self._last = now
        try:
            _write(self.path, self.doc)
        except OSError:
            pass

    def beat(self) -> None:
        extra = {}
        if self.probe is not None:
            try:
                extra = self.probe() or {}
            except Exception:
                extra = {}
        with self._lock:
            self.doc.update({k: v for k, v in extra.items() if v is not None})
            self._flush(time.time())

    def start(self) -> "Marker":
        self.beat()

        def loop():
            while True:
                time.sleep(MARK_S)
                self.beat()
        threading.Thread(target=loop, daemon=True,
                         name="knurlogic-marker").start()
        return self


#: this process's marker, when it is a rank of a job
CURRENT: dict = {"marker": None}


def progress(**fields) -> None:
    """From anywhere in a rank (engine included): update its marker; a
    no-op outside a job."""
    m = CURRENT["marker"]
    if m is not None:
        m.set(**fields)


def after_load() -> None:
    """The model is in memory: arm the jaccl self-heal deadline, if the
    page asked for one (it does only when the fork is installed and the
    link is jaccl). Read live by libjaccl on every collective."""
    ms = os.environ.get("KNURLOGIC_JACCL_TIMEOUT_MS")
    if ms and ms.isdigit():
        os.environ["JACCL_COLLECTIVE_TIMEOUT_MS"] = ms
    progress(phase="ready")


# ------------------------------------------------------------- page side

class Watch:
    """The page's memory of each rank's step counter, so a stall is
    measured on the page's own clock -- a rank whose marker thread cannot
    run (GIL held across a blocked eval) is judged the same way."""

    def __init__(self):
        self.seen: dict = {}              # (job, rank) -> (step, since)

    def verdict(self, job: str, ranks: list, alive, now: float | None = None,
                read=read_marker) -> str:
        """"" when healthy, else why the job has failed. `ranks`: the
        ranks this page started, [{"rank", "pid", "machine"}]; `alive(pid)`
        says whether that rank's process is still there."""
        now = time.time() if now is None else now
        for r in ranks:
            rank = int(r["rank"])
            ms = r.get("machines") or []
            who = r.get("machine") or (ms[rank] if rank < len(ms)
                                       else None) or "this machine"
            if not alive(int(r["pid"])):
                return f"rank {rank} on {who} (pid {r['pid']}) exited"
            m = read(job, rank) or {}
            phase = m.get("phase") or "joining"
            since = next((v for v in (m.get("since"), r.get("t"))
                          if v is not None), now)
            if phase == "joining" and now - float(since) > JOIN_S:
                return (f"rank {rank} on {who} has not joined the ring in "
                        f"{JOIN_S:.0f} s")
            step = int(m.get("step") or 0)
            key = (job, rank)
            was = self.seen.get(key)
            # the clock runs only while work is in flight: a ring idle for
            # an hour and then given a request has not been stalled an hour
            if was is None or was[0] != step or not m.get("busy"):
                self.seen[key] = (step, now)
                continue
            if m.get("busy") and now - was[1] > STALL_S:
                return (f"rank {rank} on {who} has had work in flight and "
                        f"not taken a step in {now - was[1]:.0f} s (step "
                        f"{step}); the ring is stalled")
        return ""

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
    try:
        raw = json.loads(registry_path().read_text())
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in (raw.items() if isinstance(raw, dict) else ())
            if isinstance(v, dict) and isinstance(v.get("pid"), int)}


def save_registry(reg: dict) -> None:
    _write(registry_path(), reg)


def by_job(reg: dict | None = None) -> dict:
    """{job: [rank records]} from the registry."""
    out: dict = {}
    for key, rec in sorted((reg if reg is not None else registry()).items()):
        out.setdefault(rec.get("job") or key.split("/")[0], []).append(rec)
    return out


def is_rank(pid: int, job: str) -> bool:
    """Alive, and still this job's process (a pid is reused once free)."""
    import subprocess
    try:
        out = subprocess.run(["ps", "-o", "command=", "-p", str(int(pid))],
                             capture_output=True, text=True,
                             timeout=5).stdout
    except Exception:
        return False
    return f"--job {job}" in out or f"--job={job}" in out


def terminate(pids: list, grace: float = GRACE_S, alive=None) -> list:
    """SIGTERM every pid, wait up to `grace`, SIGKILL what is left.
    -> the pids that needed the SIGKILL."""
    import signal
    alive = alive or _pid_alive
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    end = time.time() + grace
    while time.time() < end and any(alive(p) for p in pids):
        time.sleep(0.1)
    killed = []
    for pid in pids:
        if alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
            except OSError:
                pass
    return killed


def _pid_alive(pid: int) -> bool:
    try:
        done, _ = os.waitpid(pid, os.WNOHANG)     # reap a child of ours
        if done == pid:
            return False
    except ChildProcessError:
        pass
    except OSError:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
