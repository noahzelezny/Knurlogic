"""Loads and unloads from the page: POST /loaded.json (`_load_fn`) on this
machine, on one peer (`forward_launch`, a Load/Unload message) or across
machines (`cluster_launch`); the peer side of that message
(`peer_launch`); and GET /loaded.json (`_loaded_fn`): residency with its
cluster jobs, recovery state and each recent launch's progress."""

from __future__ import annotations

import http.client
import json
import time
from pathlib import Path
from urllib.parse import urlparse

from knurlogic.interfaces import spawn
from knurlogic.interfaces.page import documents, nodes, peers, router
from knurlogic.machine import identity, loaded
from knurlogic.machine.servers import registry
from knurlogic.tuning.checks import PATH_KEYS, clean_sets
from knurlogic.tuning.presets import preset_or


def tracked_load(**kw) -> dict:
    """mcp.load, for a load this page was asked for: a server it starts is
    relaunched if it dies unasked (cluster/recovery.py)."""
    from knurlogic.cluster import recovery
    from knurlogic.interfaces import mcp
    out = mcp.load(**kw)
    if isinstance(out, dict) and out.get("pid") and out.get("port"):
        recovery.track_single(int(out["port"]), {
            "artifact": out.get("starting") or kw.get("artifact"),
            "tune": kw.get("tune"), "sets": kw.get("sets") or {},
            "draft": kw.get("draft", True)},
            pid=int(out["pid"]))
    return out


def _load_fn(serve_port: int):
    def handler(_q: dict, body=None) -> dict:
        try:
            req = json.loads(body or b"{}")
        except ValueError:
            req = {}
        if not isinstance(req, dict):
            req = {}
        act = req.get("action")
        nodes = req.get("nodes")
        if act == "load" and isinstance(nodes, list) and len(nodes) >= 2:
            return _then_refresh(cluster_launch(req, serve_port))
        if act == "load" and isinstance(nodes, list) and len(nodes) == 1:
            req = {k: v for k, v in req.items()
                   if k not in ("nodes", "split", "link")}
            req["node"] = nodes[0]
        if act == "unload" and req.get("job"):
            from knurlogic.cluster import launch, recovery
            # what recovery held of it, said: a failed job has no ranks
            # left, and clearing its record is the whole of the unload
            cleared = recovery.cancel_job(str(req["job"]))
            out = launch.stop(str(req["job"]), reason="unloaded")
            both = cleared + [c for c in out.get("cleared") or []
                              if c not in cleared]
            return dict(out, cleared=both)
        node = req.get("node")
        if node and node != identity.identity().get("id"):
            return _then_refresh(forward_launch(req))
        if node:
            # this machine, picked by id: the page's own load, by identity
            from knurlogic.machine.artifact import AmbiguousIdentity, resolve_identity
            try:
                req = dict(req, target=req.get("target") or resolve_identity(
                    req.get("identity"),
                    name=str(req.get("name") or "")) or "")
            except AmbiguousIdentity as e:
                return {"loaded": False, "refused": "ambiguous identity",
                        "note": str(e)}
        target, where = req.get("target") or "", req.get("where") or ""
        try:
            # Through the MCP's own functions: the page refuses what an agent
            # is refused -- will not fit, memory still moving -- in the same
            # words. The page loading past a check the MCP enforces would be
            # a capability on one side only.
            if act == "load":
                return _then_refresh(tracked_load(artifact=target,
                                port=int(req.get("port") or 0),
                                tune=preset_or(req.get("tune"), _default_tune()),
                                sets=req.get("sets") or {},
                                force=bool(req.get("force")),
                                draft=req.get("draft") is not False))
            if act == "unload":
                # Ours to stop only if we started it. Anything else is
                # somebody's server and not this page's to kill.
                for port, rec in registry().items():
                    if rec.get("artifact") == target or str(port) == str(target):
                        return spawn.stop(port)
                return {"error": "this page did not start that; stop it "
                                 "where it was started"}
            if act == "ollama-unload":
                return loaded.ollama_unload(where, target)
        # a load action's failure is the page's answer, not a dead handler
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
        return {"error": f"unknown action {act!r}"}
    return handler


# --- load / unload on ONE peer ---------------------------------------------
# The coordinator page (this one, with a peer picked) never tells a peer a
# path and never takes the peer's address from the request: the address is
# the one its PEERS store has for an ANSWERING peer with that id, the
# artifact is named by identity (machine/artifact.identity). Running
# knurlogic on a machine is its consent; the peer still decides
# where a request may come from (its network gate, no Origin) and does its
# own fit check against its own load budget. The refusal text comes back
# as the peer wrote it.


def _default_tune() -> str:
    """The tune a launch takes when none is named: this machine's knurlogic
    strategy (tuning/strategy.py), default unless one was chosen."""
    from knurlogic.tuning import strategy
    return strategy.get()


def _peer_by_id(node: str):
    for p in (nodes.PEERS.all() if nodes.PEERS else []):
        if getattr(p, "id", "") == node and p.state == "answering":
            return p
    return None


def forward_launch(req: dict, post=None) -> dict:
    """A load or unload for the peer `req["node"]`, sent as a Load or
    Unload message. Builds the forwarded body from scratch: identity, port,
    tune, allow-listed sets, force -- nothing else of what the page sent
    goes over."""
    node = str(req.get("node") or "")
    if any(k in req for k in PATH_KEYS if k != "target") or (
            req.get("action") == "load" and req.get("target")):
        return {"error": "a model on another machine is named by its "
                         "identity, never by a path"}
    p = _peer_by_id(node)
    if p is None:
        return {"error": f"{node!r} is not a machine that is answering this "
                         f"page; it can only launch on peers it can see"}
    who = p.name or p.host
    act = req.get("action")
    if act == "load":
        sets, bad = clean_sets(req.get("sets") or {})
        if bad:
            return {"error": f"not a launch setting knurlogic passes to "
                             f"another machine: {', '.join(bad)}"}
        kind = "Load"
        doc = {"identity": str(req.get("identity") or ""),
               "name": Path(str(req.get("name") or "")).name[:255],
               "tune": preset_or(req.get("tune"), "default"),
               "sets": sets, "force": bool(req.get("force")),
               "draft": req.get("draft") is not False}
        if req.get("port"):
            try:
                doc["port"] = int(req["port"])
            except (TypeError, ValueError):
                return {"error": f"port {str(req['port'])[:20]!r} is not "
                                 f"a number"}
        if not doc["identity"]:
            return {"error": "no identity for that model; this page's "
                             "/models.json gives one per artifact"}
    elif act == "unload":
        kind = "Unload"
        try:
            doc = {"port": int(req.get("port"))}  # type: ignore[arg-type]  # None: TypeError, answered below
        except (TypeError, ValueError):
            return {"error": "unload on another machine names the port"}
    else:
        return {"error": f"unknown action {act!r}"}
    from knurlogic.cluster import transport
    from knurlogic.cluster.peers import clean
    post = post or transport.send
    try:
        out = post(p.key, kind, doc, who=who)
    except (OSError, ValueError, http.client.HTTPException) as e:
        return {"error": f"{who} did not answer: {type(e).__name__}: {e}"}
    if not isinstance(out, dict):
        return {"error": f"{who} answered something not a reply"}
    out = clean(out)
    out["machine"] = who
    return out


def cluster_launch(req: dict, serve_port: int) -> dict:
    """POST /loaded.json {action: load, identity, nodes: [>= 2 ids],
    split, link[, cable]}: this page coordinates (cluster/launch.py)."""
    from knurlogic.cluster import launch
    if any(k in req for k in PATH_KEYS if k != "target") or req.get("target"):
        return {"error": "a model across machines is named by its "
                         "identity, never by a path"}
    # tcp|rdma (older callers: ring|jaccl) -> mlx's backend, in one place
    link = launch.backend(req.get("link")) or req.get("link")
    snap, _ = nodes._status_fn()
    own: dict = next((n for n in snap.get("nodes") or []
                if n.get("role") in ("local", "server")), {})
    return launch.launch(
        dict(req, link=link), me=identity.identity(),
        peers=nodes.PEERS.all() if nodes.PEERS else [],
        local_info=own.get("cluster") or launch.node_info(),
        ui_port=spawn.SERVE_PORT["ui"], serve_port=serve_port)


def peer_launch(req: dict, load=None, stop=None, resolve=None) -> tuple:
    """A Load or Unload message, on the machine asked to load: (status,
    doc). The caller has passed the peer gate (no Origin; loopback,
    Thunderbolt or a --peer address), the size cap and the envelope
    check. Running knurlogic on a machine is its consent to load for the
    cluster. The request is a load by identity -- resolved to a path HERE,
    from this machine's own stores -- or an unload of a port this machine
    started; `action` is the kind's (load|unload)."""
    if not isinstance(req, dict):
        return 400, {"error": "the body must be a JSON object"}
    if any(k in req for k in PATH_KEYS):
        return 400, {"error": "a launch names a model by identity, never by "
                              "a path"}
    act = req.get("action")
    if act == "unload":
        port = req.get("port")
        if not isinstance(port, int) or isinstance(port, bool):
            return 400, {"error": "unload names the port"}
        return 200, (stop or spawn.stop)(port)
    if act != "load":
        return 400, {"error": f"unknown action {act!r}"}
    sets, bad = clean_sets(req.get("sets") or {})
    if bad:
        return 400, {"error": f"not a launch setting: {', '.join(bad)}"}
    tune = preset_or(req.get("tune"), _default_tune())
    from knurlogic.machine.artifact import AmbiguousIdentity, resolve_identity
    try:
        path = resolve(req.get("identity")) if resolve else resolve_identity(
            req.get("identity"), name=str(req.get("name") or ""))
    except AmbiguousIdentity as e:
        return 409, {"loaded": False, "refused": "ambiguous identity",
                     "note": str(e)}
    if not path:
        name = identity.identity().get("name") or "this machine"
        return 404, {"loaded": False, "refused": f"not on {name}",
                     "note": f"{name} has no artifact with identity "
                             f"{str(req.get('identity'))[:64]!r} in its "
                             f"model stores. Copy it there first."}
    port = req.get("port")
    port = port if isinstance(port, int) and not isinstance(port, bool) \
        and 1024 <= port < 65536 else 0
    if load is None:
        load = tracked_load
    # the fit check is this machine's, against its own load budget
    return 200, load(artifact=path, port=port, tune=tune, sets=sets,
                     force=bool(req.get("force")),
                     draft=req.get("draft") is not False)


def with_jobs(doc: dict) -> dict:
    """/loaded.json plus the cluster jobs with a rank here (`jobs`), and
    rank 0's resident row marked with its job, so the job is listed once,
    on its leader, with its machines."""
    try:
        from knurlogic.cluster import launch
        js = launch.jobs_document()
    except (OSError, ValueError, KeyError, AttributeError, TypeError):
        js = []
    # a port is reused by the next server: an ended job never claims the
    # row of the one now serving there (a single-Mac load on 8080 after a
    # stopped cluster job there was shown as that job, "stopped", twice)
    ports = {}
    for j in js:
        if j.get("port") and j.get("phase") != "stopped":
            ports[j["port"]] = j
    from knurlogic.cluster import jobs as J
    from knurlogic.cluster import recovery
    # what this machine's ranks of each job hold: a job's card sums every
    # machine's (rank 0's row alone showed its own share, or nothing)
    procs = {int(p.get("pid", 0)): int(p.get("bytes", 0)) for p in
             ((doc.get("memory") or {}).get("processes") or [])
             if isinstance(p, dict)}
    try:
        reg = J.registry()
    except (OSError, ValueError):
        reg = {}
    for j in js:
        j["bytes"] = sum(procs.get(int(v.get("pid") or 0), 0)
                         for v in reg.values()
                         if isinstance(v, dict) and v.get("job") == j.get("job"))
    for j in js:
        j["recovery"] = (recovery.for_job(j["job"]) if j.get("job")
                         else None) or (recovery.served_view(
                             j["port"], j.get("phase") == "ready")
                             if j.get("port") and j.get("phase")
                             not in ("stopped", "stopping") else None)
    rows = []
    for r in doc.get("resident") or []:
        if not isinstance(r, dict):
            rows.append(r)
            continue
        u = urlparse(r.get("where") or "")
        j = ports.get(u.port) if u.port else None
        rec = None
        if u.port and r.get("runtime") == "knurlogic" and \
                u.hostname in ("127.0.0.1", "localhost", None):
            rec = recovery.for_port(u.port) or recovery.served_view(
                u.port, r.get("state") not in ("loading", "warming"))
        if j and j.get("recovery"):
            rec = j["recovery"]
        r = dict(r, recovery=rec)
        rows.append(dict(r, cluster={**{k: j.get(k) for k in (
            "job", "split", "link", "machines", "leader", "phase")},
            **({"url": j["url"]} if j.get("url") else {})})
                    if j else r)
    # tracked here and not serving now: waiting to be relaunched, or failed
    return dict(doc, resident=rows, jobs=js, recovery=recovery.not_serving())


#: How long after its start a server's load is still reported (`loads`):
#: long enough for a cold 400 GB read, short enough that an old failure
#: does not linger on the page.
LOAD_REPORT_S = 1800


def _share_of(rec: dict) -> int:
    """What a cluster rank holds once loaded, from its marker; 0 when it is
    not a rank or has not said yet."""
    if not rec.get("job") or rec.get("rank") is None:
        return 0
    from knurlogic.cluster import jobs as J
    try:
        m = J.read_marker(rec["job"], int(rec["rank"])) or {}
    except ValueError:
        return 0
    return int(m.get("share_bytes") or 0)


def load_progress(doc: dict) -> list:
    """What this machine's recent launches are doing, for the page's load
    indicator -- from what /loaded.json already gathered (the resident rows
    and the OS memory map), plus each server's log: no extra HTTP.

    One entry per server started in the last LOAD_REPORT_S: phase (loading,
    stalled, warming, ready, exited), seconds since start, bytes its process
    holds against the artifact's size, the last log line, and for one that
    exited the tail of its log (why it failed)."""
    from knurlogic.machine.servers import is_our_server
    now = time.time()
    procs = {int(p.get("pid", 0)): int(p.get("bytes", 0)) for p in
             ((doc.get("memory") or {}).get("processes") or [])
             if isinstance(p, dict)}
    rows = {urlparse(r.get("where") or "").port: r
            for r in doc.get("resident") or []
            if isinstance(r, dict) and r.get("runtime") == "knurlogic"}
    out = []
    entries = [(port, rec) for port, rec in sorted(registry().items())]
    # a follower rank binds no port, so only jobs.json holds it; rank 0 is in
    # both registries, counted once (by pid)
    from knurlogic.cluster import jobs as J
    seen = {int(rec.get("pid") or 0) for _, rec in entries}
    jreg = J.registry()
    ranks = {v["pid"]: v for v in jreg.values()}
    # rank 0's server record, with the rank its job record adds
    entries = [(port, dict(rec, rank=ranks[rec["pid"]].get("rank"))
                if ranks.get(rec.get("pid"), {}).get("rank") is not None
                else rec)
               for port, rec in entries]
    for key, rec in sorted(jreg.items()):
        if rec["pid"] not in seen:
            seen.add(rec["pid"])
            entries.append((rec.get("port") or 0, dict(rec, rank=rec.get(
                "rank", int(key.rsplit("/", 1)[-1]) if "/" in key else 0))))
    for port, rec in entries:
        t = rec.get("t")
        if not t or now - t > LOAD_REPORT_S:
            continue
        path, pid = rec.get("artifact") or "", int(rec.get("pid") or 0)
        log = Path(rec.get("log", ""))
        try:
            with log.open("rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 65536))
                lines = [ln for ln in f.read().decode(errors="replace")
                         .splitlines() if ln.strip()]
            quiet = now - log.stat().st_mtime
        except OSError:
            lines, quiet = [], None
        e = {"port": port, "name": Path(path).name,
             **({"job": rec["job"]} if rec.get("job") else {}),
             **({"identity": rec["identity"]} if rec.get("identity") else {}),
             **({"rank": rec["rank"]} if "rank" in rec else {}),
             "seconds": round(now - t), "bytes": procs.get(pid, 0),
             # a rank of a split job holds its share, not the artifact:
             # the rank writes it to its marker (engine/runtime/marker.progress)
             "total_bytes": int(_share_of(rec) or rec.get("bytes") or 0),
             "last_log_line": lines[-1][:200] if lines else ""}
        r = rows.get(port) if port else None
        if not is_our_server(pid) and not (
                rec.get("job") and J.is_rank(pid, rec["job"])):
            e.update(phase="exited", log_tail=[ln[:200] for ln in lines[-4:]])
            from knurlogic.cluster.recovery import refusal_line
            why = refusal_line("\n".join(lines[-60:]))
            if why:
                e["refused"] = why
        elif r is not None and r.get("state") == "warming":
            e["phase"] = "warming"       # weights in; the first-request warm-up
        elif r is None or r.get("state") == "loading":
            e["phase"] = ("stalled" if quiet is not None
                          and quiet > spawn.STALL_QUIET_S else "loading")
        elif e["total_bytes"] and e["bytes"] < spawn.WARM_FRACTION * e["total_bytes"]:
            e["phase"] = "warming"
        else:
            e["phase"] = "ready"
        out.append(e)
    return out


def _loaded_fn():
    """/loaded.json as `web` answers it for this box; with ?peers=1 (what the
    page asks) it also carries `peers`: each other machine's residency."""
    local = documents.loaded_document()

    def handler(q: dict) -> dict:
        # a copy: the local document is cached and shared between requests
        doc = with_jobs(local(q))
        try:
            loads = load_progress(doc)
        except (OSError, ValueError, KeyError, AttributeError, TypeError):
            loads = []
        if loads:
            doc = dict(doc, loads=loads)
        if not (q.get("peers") or [""])[0]:
            return doc
        return dict(doc, peers=peers.peer_residency(nodes.PEERS))
    return handler


def _then_refresh(out):
    """A launch's answer, passed on; a launch that went ahead first
    refreshes what this page's chat and router can reach."""
    if isinstance(out, dict) and not out.get("error") \
            and not out.get("refused") and out.get("loaded") is not False:
        router.refresh_targets()
    return out
