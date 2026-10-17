"""The watch: is each job this page has ranks in still healthy?

Every WATCH_S the page reads its ranks' markers and asks the other pages
what they hold; a job stops on evidence -- a rank gone, stalled or never
joined, a machine silent -- never on a page's silence alone. Also what
the page shows of its jobs (jobs_document, job_state).

Design: docs/design/liveness.md.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from knurlogic.cluster import (
    every,
    here,
    recovery,
    stopping,
    tables,
    transport,
)
from knurlogic.cluster import jobs as J
from knurlogic.cluster.protocol import (
    JobState,
    link_name,
    typed,
)
from knurlogic.machine import servers

logger = logging.getLogger(__name__)
#: how often the page checks its ranks
WATCH_S = 2.0


def watch_once(now: float | None = None) -> list:
    """Check every job with a rank on this machine; stop the failed ones.
    -> [(job, reason)] stopped."""
    out = []
    for job, recs in J.by_job().items():
        if job in tables.STOPPING:
            continue
        if all(r.get("stopping") for r in recs):
            live = [r for r in recs if here.alive(job, int(r["pid"]))]
            if live and all(J.mid_step(job, int(r["rank"]), int(r["pid"]))
                            for r in live):
                continue                  # spared mid-step: nothing to do
            # stopped, and a rank was still exiting: finish it
            stopping.stop(job, reason=recs[0].get("stop_reason") or "stopped",
                 propagate=False)
            continue
        why, fkind = tables.WATCH.check(job, recs,
                                  lambda pid, j=job: here.alive(j, pid), now=now)
        if not why:
            why, fkind = peer_verdict(job, recs)
        elif " exited" in why:
            why, fkind = exit_reason(why, recs, fkind)
        if why:
            stopping.stop(job, reason=why, kind=fkind)
            out.append((job, why))
    return out


def exit_reason(why: str, recs: list, kind: str | None = None
                ) -> tuple[str, str | None]:
    """`why` (a rank "exited") with what the ranks' logs say, and its
    recovery kind: a link init failure, a refusal to start (the job fails
    with its words), out of memory -- else `why` and `kind` as given."""
    line = next((x for x in (tables.link_init_failure(J.log_tail(r.get("log")))
                             for r in recs) if x), "")
    if line:
        return f"{why}: link init failed: {line}"[:300], kind
    tails = [J.log_tail(r.get("log")) for r in recs]
    ref = next((x for x in map(recovery.refusal_line, tails) if x), "")
    if ref:
        return f"{why}: {ref}"[:700], "refusal"
    mem = next((x for x in map(recovery.memory_line, tails) if x), "")
    if mem:
        return f"{why}: out of memory: {mem}"[:300], "memory"
    if any(tables.RING_LOST in t for t in tails):
        # its ring lost a peer: a machine of the job went away (or its
        # rank did) -- recovery waits until every machine answers again
        return f"{why}: its ring lost a peer ({tables.RING_LOST})"[:300], "machine"
    return why, kind


def job_state(job: str) -> dict:
    """JobState: whether this page still runs its ranks of `job`; a rank
    whose process exited before this page's watcher stopped the job is
    `exited`, with that watcher's own reason."""
    # one snapshot: start() saves a record, then unprepares, and stop()
    # drops the records and writes ENDED, each under tables.LOCK -- read apart, a
    # peer could see neither and stop a healthy job
    with tables.LOCK:
        prepared = job in tables.PREPARED
        recs = J.by_job().get(job, [])
        ended = dict(tables.ENDED.get(job) or {})
    live = sorted(int(r["rank"]) for r in recs if not r.get("stopping")
                  and here.alive(job, int(r["pid"])))
    dead = [r for r in recs if not r.get("stopping")
            and int(r["rank"]) not in live]
    exited, exited_kind = (exit_reason(
        f"rank {dead[0]['rank']} on {dead[0].get('machine') or 'its machine'}"
        f" (pid {dead[0]['pid']}) exited", dead) if dead else (None, None))
    stopping = [r for r in recs if r.get("stopping")]
    return typed(JobState(
        job=job, ranks_here=live, prepared=prepared,
        exited=exited, exited_kind=exited_kind,
        stopping=bool(stopping),
        phase=J.phase_of(job, [r for r in recs if not r.get("stopping")])
        if live else None,
        # by process, records or not: a relaunch waits for none left
        processes=J.pids_of_job(job),
        ended=ended.get("reason") or (
            stopping[0].get("stop_reason") if stopping else None),
        ended_kind=ended.get("kind") or (
            stopping[0].get("stop_kind") if stopping else None)))


def _ask_job(page: str, job: str) -> dict:
    return transport.send(page, "JobState", {"job": job})


def _record_of(nid: str):
    """This page's peer record of machine `nid` (an answering one first)."""
    recs = [p for p in tables.peers_fn() if getattr(p, "id", "") == nid]
    return sorted(recs, key=lambda p: p.state != "answering")[0] \
        if recs else None


def peer_verdict(job: str, recs: list, ask=None, pages=None,
                 me=None, probe=None,
                 now: float | None = None) -> tuple[str, str | None]:
    """("", None) while every other machine of the job still runs its rank,
    else (why not, recovery kind), from what that machine's page says: the
    job ended there; a rank of it exited there (with that page's reason);
    it restarted (a new boot_id) holding no rank; it holds no rank and is
    not preparing one. A page that does not answer says nothing: a page
    is not its rank (ranks outlive a page restart), so its silence never
    stops a job. A peer MACHINE that goes silent is caught by the ranks'
    own bell watchers (engine/split/liveness, docs/design/liveness.md) --
    once they are armed; while the job is still joining, by its kernel
    (`_silent_while_joining`)."""
    nodes = next((r.get("nodes") for r in recs if r.get("nodes")), None) \
        or (tables.SPECS.get(job) or {}).get("nodes") or []
    if not nodes:
        return "", None
    if me is None:
        from knurlogic.machine import identity
        me = identity.identity().get("id")
    pages = stopping.peer_pages() if pages is None else pages
    ask = ask or _ask_job
    for n in nodes:
        nid = str(n.get("id") or "")
        if not nid or nid == me:
            continue
        name = n.get("name") or nid
        key = (job, nid)
        page = pages.get(nid)
        try:
            if not page:
                raise ConnectionError("not a peer this page knows")
            doc = ask(page, job)
        except (*transport.NO_ANSWER, AttributeError) as e:
            if key not in tables.PEER_SLOW:
                tables.PEER_SLOW.add(key)
                logger.warning("cluster job %s: %s's page has not answered "
                               "(%s); a page's silence is not evidence, the "
                               "job is kept", job, name, type(e).__name__)
            why = _silent_while_joining(job, recs, key, name, page, probe,
                                        now)
            if why:
                return why, "machine"
            continue
        tables.PEER_SILENT.pop(key, None)         # the machine answered
        if not transport.is_answer(doc):
            # its handler failed or refused: an answer about nothing, so
            # no evidence either way (transport.answer)
            if key not in tables.PEER_SLOW:
                tables.PEER_SLOW.add(key)
                logger.warning("cluster job %s: %s's page could not say "
                               "(%s); the job is kept", job, name,
                               (doc or {}).get("error"))
            continue
        tables.PEER_SLOW.discard(key)
        why, kind = _page_says(key, name, doc)
        if why:
            return why, kind
    return "", None


def _page_says(key: tuple, name: str, doc: dict) -> tuple[str, str | None]:
    """`peer_verdict` for one machine whose page answered JobState with
    `doc`: ("", None) while it still runs its rank of the job (key: (job,
    its id)), else (why not, recovery kind)."""
    if doc.get("ended"):
        return (f"{name} stopped the job: {doc['ended']}"[:300],
                doc.get("ended_kind"))
    if doc.get("exited"):
        return (f"{name}: {doc['exited']}"[:700],
                doc.get("exited_kind") or "machine")
    rec = _record_of(key[1])
    boot = getattr(rec, "boot_id", "") or ""
    if doc.get("ranks_here") or doc.get("prepared"):
        # ranks run in their own session and outlive a restarted page:
        # one still holding its rank is believed, new process or not (a
        # page redeploy used to stop a healthy job here)
        if boot:
            tables.PEER_BOOT[key] = boot
        return "", None
    if boot and tables.PEER_BOOT.get(key, boot) != boot:
        return (f"{name} restarted (a new knurlogic process); its rank of "
                f"the job is gone", "machine")
    return f"{name} no longer runs its rank of the job", "machine"


def _silent_while_joining(job: str, recs: list, key: tuple, name: str,
                          page, probe=None, now: float | None = None) -> str:
    """Why the machine behind a page that did not answer is gone, or "".

    Before the bell watchers are armed -- every rank of the job here still
    joining (its marker phase): rank 0 waiting at the bell for a rank that
    never dials, or any rank in mlx's ring init, which accepts with no
    timeout -- a machine that dies leaves no connection to watch, and the
    job would wait out the bell (link.BELL_EARLY_S) or forever. So its
    kernel is asked instead, on the page's own address (a Thunderbolt
    address: cluster peers are gated to the cable): accepted or refused
    is a machine that is there (a page restarting); only silence --
    liveness.SILENT_S of it, keepalive's own window -- is evidence."""
    from knurlogic.engine.split import liveness
    if not page or J.phase_of(job, recs) != "joining":
        tables.PEER_SILENT.pop(key, None)
        return ""
    host, _, port = str(page).rpartition(":")
    try:
        answers = (probe or liveness.kernel_answers)(host, int(port))
    except ValueError:
        return ""
    now = time.monotonic() if now is None else now
    if answers is not False:
        tables.PEER_SILENT.pop(key, None)
        return ""
    since = tables.PEER_SILENT.setdefault(key, now)
    if now - since < liveness.SILENT_S:
        return ""
    return (f"the machine holding {name}'s rank stopped answering on the "
            f"job's link while the job was joining ({liveness.SILENT_S} s "
            f"of silence)")


def ensure_watcher() -> None:
    """The page's watcher, started once per process: `watch_once` every
    WATCH_S. A test sets tables.WATCHER and drives watch_once by hand."""
    with tables.LOCK:
        if tables.WATCHER.is_set():
            return
        tables.WATCHER.set()
    every(WATCH_S, watch_once, "knurlogic-cluster-watch")


def start_watching_existing() -> None:
    """At page start: ranks from an earlier page process are watched too,
    and the models an earlier page process was recovering are again."""
    if J.registry():
        ensure_watcher()
    recovery.restore()


def jobs_document() -> list:
    """The jobs with a rank on this machine, and the ones that ended
    lately with why, for /loaded.json."""
    out = []
    for job, recs in J.by_job().items():
        r0 = min(recs, key=lambda r: r["rank"])
        phase = J.phase_of(job, recs)
        # a rank stayed inside a step rather than leave (a stop spares it):
        # neither stopping forever nor loaded
        stuck = ({"stuck": servers.stuck_reason(r0.get("machines") or [])}
                 if phase == "stuck" else {})
        if all(r.get("stopping") for r in recs):
            out.append({"job": job, "phase": "stuck" if stuck else "stopping",
                        **stuck,
                        "split": r0.get("split"),
                        "link": link_name(r0.get("link")),
                        "reason": r0.get("stop_reason"),
                        "kind": r0.get("stop_kind"),
                        "machines": r0.get("machines"),
                        "port": next((r.get("port") for r in recs
                                      if r.get("port")), None),
                        "exiting": sorted(int(r["pid"]) for r in recs)})
            continue
        out.append({"job": job, "split": r0.get("split"),
                    "link": link_name(r0.get("link")),
                    "machines": r0.get("machines"),
                    "leader": r0.get("leader"), "world": r0.get("world"),
                    "ranks_here": sorted(r["rank"] for r in recs),
                    "port": next((r.get("port") for r in recs
                                  if r.get("port")), None),
                    "artifact": Path(r0.get("artifact") or "").name,
                    "identity": r0.get("identity") or "",
                    "phase": phase, **stuck,
                    **{k: (tables.SPECS.get(job) or {}).get(k) for k in
                       ("cable", "cable_note")
                       if (tables.SPECS.get(job) or {}).get(k)},
                    **({"url": tables.leader_url(dict(tables.SPECS[job], port=next(
                        (r.get("port") for r in recs if r.get("port")),
                        None)))} if tables.SPECS.get(job) else {})})
    now = time.time()
    for job, e in list(tables.ENDED.items()):
        if now - e["t"] > 600:
            tables.ENDED.pop(job, None)
        else:
            out.append({"job": job, "phase": "stopped", **e})
    return out


def failure_of_port(port) -> str:
    """Why the cluster job whose rank 0 serves on `port` here cannot
    answer -- "" when no job of this machine's serves there. A dropped
    connection to a live job's rank 0 has the watcher look at once, so the
    answer carries the job's real stop reason, not a guess."""
    try:
        port = int(port or 0)
    except (TypeError, ValueError):
        return ""
    if not port:
        return ""
    job = job_of_port(port)
    if job:
        try:
            watch_once()
        except Exception:  # an HTTP request must still answer; logged
            logger.debug("watch_once failed while reading job %s", job, exc_info=True)
    for j, e in sorted(tables.ENDED.items(), key=lambda kv: -kv[1]["t"]):
        if (job and j == job) or (not job and e.get("port") == port):
            return e.get("reason") or "stopped"
    if job:
        return (f"rank 0 of cluster job {job} dropped the connection; the "
                f"job is failing")
    return ""


def job_of_port(port: int) -> str:
    for job, recs in J.by_job().items():
        if any(r.get("port") == port for r in recs):
            return job
    return ""
