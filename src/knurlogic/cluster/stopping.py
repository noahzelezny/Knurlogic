"""Stop: a job's ranks on this page end, and the other pages are told.

Followers' pages are told before rank 0's, so no page sees a follower
leave unmarked; a peer's Stop is answered once its ranks are marked
stopping and they end on a thread. A stop is done when the processes are
gone, not when they were signalled.

Design: docs/design/liveness.md.
"""
from __future__ import annotations

import logging
import subprocess
import threading
import time

from knurlogic.cluster import here, recovery, tables, transport
from knurlogic.cluster import jobs as J
from knurlogic.cluster.protocol import (
    Stopped,
    link_name,
    typed,
)
from knurlogic.machine import servers

logger = logging.getLogger(__name__)


def stop(job: str, reason: str = "unloaded", propagate: bool = True,
         post=None, grace: float = J.GRACE_S,
         reap: float = J.REAP_S, kind: str | None = None,
         wait: bool = True) -> dict:
    """Stop every rank of `job` on this machine (SIGTERM, then SIGKILL
    after `grace`), forget it, and -- when `propagate` -- tell every other
    page of the job to do the same.

    `wait` False (another page's Stop) answers once the ranks here are
    marked stopping and ends them on a thread: a follower leaves only on
    rank 0's `stop`, and rank 0's page sends its own SIGTERM only after
    every page has answered -- waiting here held an idle job's unload for
    the whole grace, and then SIGKILLed the follower."""
    job = str(job or "")
    cleared: list = []
    if recovery.kind(reason, kind) == "requested":
        cleared = recovery.cancel_job(job)   # asked for: never recovered
    # an unload carries no kind of its own: say "requested" from the start,
    # so a page polling while the ranks exit never shows it as a failure
    stop_kind = kind or ("requested" if recovery.kind(reason) == "requested"
                         else None)
    with tables.LOCK:
        tables.PREPARED.pop(job, None)
        spec = tables.SPECS.pop(job, None)
        ending = job in tables.STOPPING       # another stop is ending it already
        tables.STOPPING.add(job)
        mine: dict = {}

        def mark(reg: dict) -> None:
            # marked first: a start on this page waits for these to be
            # gone, and a prepare does not count them as staying
            for k, v in reg.items():
                if v.get("job") == job:
                    reg[k] = mine[k] = dict(
                        v, stopping=v.get("stopping") or time.time(),
                        stop_reason=v.get("stop_reason") or reason,
                        stop_kind=v.get("stop_kind") or stop_kind)
        if any(v.get("job") == job for v in J.registry().values()):
            J.update_registry(mark)
    # the job's machines: its spec, or -- after a page restart, SPECS being
    # in memory -- the ids its rank records here keep
    nodes: list = (spec or {}).get("nodes") or next(
        (v["nodes"] for v in mine.values() if v.get("nodes")), [])
    if propagate and not nodes and not mine and not spec:
        # nothing of it here (a failed job, its ranks gone and recovery's
        # record all that is left): every peer page hears the stop, so the
        # record goes on whichever Mac holds it
        nodes = [{"id": i} for i in peer_pages()]
    if not wait:
        if not ending:
            threading.Thread(target=_end_stop, daemon=True,
                             name=f"knurlogic-stop-{job}",
                             args=(job, mine, spec, reason, stop_kind,
                                   grace, reap)).start()
        return typed(Stopped(
            job=job, ranks_here=sorted(v["rank"] for v in mine.values()),
            exiting=[int(v["pid"]) for v in mine.values()
                     if here.alive(job, int(v["pid"]))],
            reason=reason, cleared=cleared))
    # the other pages first: a rank stopped here before its peers hear of
    # it fails their ranks' ring ("[ring] connection to a peer was lost"),
    # and their pages would record a crash -- and recover it -- where the
    # job was asked to stop. Each answers once its ranks are marked
    # (wait=False), so this waits for no rank to leave
    told: list = []
    if propagate and nodes:
        told, more = _tell_peers(job, nodes, reason, kind, post)
        cleared += [c for c in more if c not in cleared]
    killed, left = _end_stop(job, mine, spec, reason, stop_kind, grace,
                             reap)
    return typed(Stopped(job=job, ranks_here=sorted(v["rank"] for v in
                                                    mine.values()),
                         killed=killed, exiting=left, told=told,
                         reason=reason, cleared=cleared))


def _end_stop(job: str, mine: dict, spec: dict | None, reason: str,
              stop_kind: str | None, grace: float, reap: float) -> tuple:
    """`stop`'s second half, after the ranks here are marked: end them
    (J.terminate) and forget the job once they are gone. (killed pids,
    pids still exiting). The job stays in `tables.STOPPING` until its records
    are gone too: the watcher, which skips it while there, would else
    start another stop of it from its "all stopping" records."""
    try:
        killed, left = _end_ranks(job, mine, grace, reap)
        with tables.LOCK:
            _forget_job(job, mine, spec, left, reason, stop_kind)
    finally:
        with tables.LOCK:
            tables.STOPPING.discard(job)
    return killed, left


def _end_ranks(job: str, mine: dict, grace: float, reap: float) -> tuple:
    """SIGTERM this page's ranks of `job`, SIGKILL what outlives `grace`
    (one mid-step is spared): (killed pids, pids still exiting)."""
    pids = [int(v["pid"]) for v in mine.values()
            if here.alive(job, int(v["pid"]))]

    def alive(p):
        return here.alive(job, p)
    rank_of = {int(v["pid"]): int(v["rank"]) for v in mine.values()}

    def spare(p):
        return J.mid_step(job, rank_of[p], p)

    killed = J.terminate(pids, grace=grace, alive=alive,
                         reap=reap, spare=spare) if pids else []
    return killed, [p for p in pids if alive(p)]


def _forget_job(job: str, mine: dict, spec: dict | None, left: list,
                reason: str, stop_kind: str | None) -> None:
    """Drop what this page holds of `job` but the ranks in `left`, still
    exiting (their records stay, phase "stopping", for the watcher to
    finish); once none is left, record how it ended (`ENDED`). Under
    tables.LOCK."""
    for (j, r) in [k for k in tables.PROCS if k[0] == job]:
        if tables.PROCS[(j, r)].pid in left:
            continue
        try:
            tables.PROCS.pop((j, r)).wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass                        # not gone yet; the watcher reaps it
    gone = [k for k, v in mine.items() if int(v["pid"]) not in left]
    if gone:
        J.update_registry(lambda reg: _drop(reg, gone))
    if any(v.get("job") == job for v in servers.registry().values()):
        servers.update_registry(lambda sreg: _drop(sreg, [
            p for p, v in sreg.items()
            if v.get("job") == job and int(v["pid"]) not in left]))
    tables.WATCH.forget(job)
    for state in (tables.PEER_BOOT, tables.PEER_SILENT):
        for k in [k for k in state if k[0] == job]:
            state.pop(k)
    tables.PEER_SLOW.difference_update({k for k in tables.PEER_SLOW if k[0] == job})
    # stopped means gone: a rank still exiting keeps its record
    # (phase "stopping"), and the watcher finishes the stop
    if (mine or spec) and not left:
        any_rec: dict = next(iter(mine.values()), {})
        port = next((int(v["port"]) for v in mine.values()
                     if v.get("port")), None) or (
            int((spec or {}).get("port") or 0) or None)
        # a failover that already moved the job keeps saying where it
        # went: the failing rank's stop can land after it
        moved = tables.ENDED.get(job) if (tables.ENDED.get(job) or {}).get(
            "relaunched") else None
        # an unload says so, so the page drops its card instead of
        # showing "failed" (picker.js followLaunch)
        tables.ENDED[job] = {"reason": reason, "kind": stop_kind,
                      "t": time.time(), "port": port,
                      "split": any_rec.get("split")
                      or (spec or {}).get("split"),
                      "link": link_name(any_rec.get("link")
                                        or (spec or {}).get("link")),
                      "machines": any_rec.get("machines")
                      or [n.get("name") for n in (spec or {}).get(
                          "nodes") or []]}
        if moved:
            tables.ENDED[job].update(reason=moved["reason"],
                              relaunched=moved["relaunched"])


def _drop(reg: dict, keys: list) -> None:
    for k in keys:
        reg.pop(k, None)


def _tell_peers(job: str, nodes: list, reason: str, kind: str | None,
                post=None) -> tuple[list, list]:
    """Stop to every other page of the job (each answers once its ranks
    are marked stopping): (who was told, what their recovery cleared).

    Followers' pages first, all at once, then rank 0's: its rank 0 is what
    makes the followers leave, so every page has marked its follower by
    then, and none reads its follower's exit as a crash -- wherever the
    stop began (a follower's page, with three machines or more, once told
    rank 0's page first)."""
    from knurlogic.machine import identity
    me = identity.identity().get("id")
    known = peer_pages()
    # the address is this page's own record of that peer, never the spec's:
    # a prepare body cannot aim this page's stop elsewhere
    targets = [(n, known[str(n.get("id") or "")]) for n in nodes
               if n.get("id") != me and known.get(str(n.get("id") or ""))]
    body = {"job": job, "reason": reason, **({"kind": kind} if kind else {})}

    def tell(target):
        n, page = target
        return n, (post or transport.send)(page, "Stop", body)
    told: list = []
    cleared: list = []
    for wave in ([t for t in targets if t[0].get("rank") != 0],
                 [t for t in targets if t[0].get("rank") == 0]):
        for got in transport.parallel(tell, wave):
            if not isinstance(got, tuple):
                logger.debug("could not tell a page to stop job %s: %s",
                             job, got)
                continue
            n, ans = got
            told.append(n.get("name") or n.get("id"))
            # what that Mac's recovery cleared (a failed job's record lives
            # on whichever page started it)
            if isinstance(ans, dict):
                cleared += [c for c in ans.get("cleared") or []
                            if c not in cleared]
    return told, cleared


def peer_pages() -> dict:
    """{peer id: page address} from this page's PEERS store (an answering
    record first)."""
    out = {}
    peers: list = tables.peers_fn()
    for p in sorted(peers, key=lambda p: p.state == "answering"):
        if getattr(p, "id", ""):
            out[p.id] = p.key
    return out
