"""This machine for a cluster job: what it offers and what it holds.

Its build, software and memory as a Prepare reads them (node_info), the
model a request names on its disk (resolve), and its own ranks: alive,
loading, exiting or on a ring slot another job needs.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from knurlogic.cluster import jobs as J
from knurlogic.cluster import links, tables
from knurlogic.machine import identity, processes, servers
from knurlogic.machine.memory.wired import available_now, gpu_working_set
from knurlogic.machine.processes import PROC_ERRORS

_INFO: dict = {"doc": None, "at": 0.0}


def _mlx_version() -> str:
    from importlib import metadata
    try:
        return metadata.version("mlx")
    except metadata.PackageNotFoundError:
        return ""


_BUILD: dict = {}


def build_fingerprint(root=None, cache=None) -> str:
    """Which build of knurlogic this is, beyond its version string (every
    dev build says 0.1.0.dev0): a hash over the installed package's .py
    files -- relative path and bytes, in a fixed order, so a wheel install
    and a PYTHONPATH source tree of the same commit agree -- and the mlx
    version, as "<12 hex>+mlx<version>". Read once per process."""
    import hashlib
    memo = _BUILD if cache is None else cache
    if root is None:
        import knurlogic
        root = Path(knurlogic.__file__).resolve().parent
    root = Path(root)
    key = str(root)
    if key not in memo:
        h = hashlib.sha256()
        for f in sorted(root.rglob("*.py"),
                        key=lambda q: q.relative_to(root).as_posix()):
            if "__pycache__" in f.parts:
                continue
            h.update(f.relative_to(root).as_posix().encode() + b"\0")
            try:
                h.update(f.read_bytes())
            except OSError:
                pass
            h.update(b"\0")
        memo[key] = h.hexdigest()[:12]
    return f"{memo[key]}+mlx{_mlx_version() or 'none'}"


def _selfheal() -> bool:
    from knurlogic.machine import deps
    m = deps.probe(sys.executable).get("mlx") or {}
    return bool(m.get("jaccl_selfheal"))


def node_info(working_set_bytes: int = 0, ttl: float = 30.0) -> dict:
    """What a coordinator needs to know of THIS machine, for its status'
    `cluster` block. Cached: it runs a few subprocesses."""
    now = time.time()
    doc = _INFO["doc"]
    if doc is None or now - _INFO["at"] > ttl:
        from knurlogic import __version__
        chip = identity.chip()
        try:
            tb = [{"iface": i["iface"], "ip": i["ip"],
                   "gbps": i.get("gbps"), "generation": i.get("generation")}
                  for i in links.thunderbolt()]
        except (*PROC_ERRORS, ValueError, KeyError):
            tb = []
        try:
            heal = _selfheal()
        except (*PROC_ERRORS, ValueError):
            heal = False
        from knurlogic.engine.matmul_padding import gpu_architecture
        doc = {"chip": chip, "gpu_architecture": gpu_architecture(),
               "bandwidth_gbs": identity.chip_bandwidth_gbs(chip),
               "thunderbolt": tb, "rdma": links.rdma(),
               "versions": {"knurlogic": __version__,
                            "mlx": _mlx_version(),
                            "build": build_fingerprint()},
               "jaccl_selfheal": heal}
        _INFO.update(doc=doc, at=now)
    from knurlogic.machine.memory import allowance
    return dict(doc, working_set_bytes=allowance.cap(
        gpu_working_set(int(working_set_bytes or 0))),
        available_bytes=available_now())


def resolve(identity: str | None, name: str = ""):
    """The local path for this identity: several copies of it are the same
    weights, and machine/artifact.resolve_identity picks one (the one
    called `name`, else one on this Mac's disk)."""
    from knurlogic.machine.artifact import resolve_identity
    return resolve_identity(identity, name=name)


def held_here(job: str = "", reg=None, sreg=None, rss=None) -> list:
    """What knurlogic ranks and servers on this machine hold now, other
    than `job`'s: [(what, pid, bytes)]. A rank being stopped is left out
    -- start waits for it to be gone -- so this is what a new load would
    have to share the working set with."""
    rss = rss or processes.rss
    reg = J.registry() if reg is None else reg
    out, seen = [], set()
    for rec in reg.values():
        j = str(rec.get("job") or "")
        pid = int(rec["pid"])
        if j == job or rec.get("stopping") or not alive(j, pid):
            continue
        seen.add(pid)
        out.append((f"job {j} rank {rec.get('rank')}", pid, rss(pid)))
    for port, rec in sorted((servers.registry() if sreg is None
                             else sreg).items()):
        pid = int(rec["pid"])
        if pid in seen or (job and rec.get("job") == job) \
                or not servers.is_our_server(pid):
            continue
        out.append((f"the server on port {port}", pid, rss(pid)))
    return out


def loading_elsewhere(job: str, reg: dict) -> str:
    """"" unless another job's rank here is still joining or loading --
    one load at a time on a machine."""
    for j, recs in J.by_job(reg).items():
        live = [r for r in recs if not r.get("stopping")]
        if j == job or not live:
            continue
        ph = J.phase_of(j, live)
        if ph != "ready":
            return (f"job {j}'s rank {live[0].get('rank')} is still "
                    f"{ph} here; one load at a time")
    return ""


def local_info() -> dict:
    """This machine's cluster block, off the page's own status."""
    try:
        snap, _ = tables.status_fn()
        own = next(n for n in snap.get("nodes") or []
                   if n.get("role") in ("local", "server"))
        return dict(own.get("cluster") or node_info(),
                    available_bytes=available_now())
    except (LookupError, StopIteration, AttributeError, TypeError, OSError, ValueError):
        return node_info()


def exiting(reg: dict) -> dict:
    """{pid: job} of the ranks here that were stopped and are not gone."""
    return {int(r["pid"]): str(r.get("job") or "") for r in reg.values()
            if r.get("stopping")}


def wait_for_exit(reg: dict, wait_s: float) -> list:
    """Wait up to `wait_s` for every rank on this machine that was stopped
    (a previous job, mid-relaunch) to be gone. -> [(job, rank, pid)] still
    alive after the wait: a job being stopped counts as alive until its
    process is reaped, and its memory -- and its ring port -- are not
    free until then."""
    going = {int(r["pid"]): (str(r.get("job") or ""), r.get("rank"))
             for r in reg.values() if r.get("stopping")}
    if not going:
        return []
    left = J.wait_gone(list(going), wait_s,
                       alive=lambda p: alive(going[p][0], p))
    return [(going[p][0], going[p][1], p) for p in left]


def slot_conflict(spec: dict, reg: dict):
    """(job, rank, pid) of an alive rank of a DIFFERENT job on this
    machine whose ring port falls in this job's slot, else None. A
    coordinator's `_slot` only scans its own registry (`_used_slots`),
    so a quick relaunch can still hand a job a slot a previous job's
    still-live rank holds on a PEER machine; this is the peer-side net
    that catches it before two jobs' ranks talk over the same port."""
    try:
        port = int(spec["hosts"][spec["rank"]].rsplit(":", 1)[1])
    except (IndexError, ValueError, TypeError, AttributeError, KeyError):
        return None
    slot = tables.slot_of_port(port)
    for rec in reg.values():
        j = str(rec.get("job") or "")
        if not j or j == spec["job"]:
            continue
        rp = rec.get("ring_port")
        if not isinstance(rp, int) or tables.slot_of_port(rp) != slot:
            continue
        pid = int(rec["pid"])
        if alive(j, pid):
            return j, rec.get("rank"), pid
    return None


def alive(job: str, pid: int) -> bool:
    proc = next((p for (j, _), p in tables.PROCS.items()
                 if j == job and p.pid == pid), None)
    if proc is not None:
        # poll() answers None while another thread waits on the process
        # (Popen's wait lock), dead or not; a zombie is dead
        return proc.poll() is None and processes.running(pid)
    return J.is_rank(pid, job)
