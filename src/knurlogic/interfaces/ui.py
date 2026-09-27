"""`knurlogic ui` -- the page, with nothing loaded and nothing else running.

THE POINT: knurlogic must not need exo, and it must not need a model already
loaded. It sits ON TOP of whatever is there. Open this and you see every
model on the disk, everything resident in every runtime, and where the memory
went -- on a machine with no exo, no ollama and no weights in RAM, all of
which are ordinary states rather than errors.

`serve` needs no exo either; nothing in knurlogic drives it. What was
missing is a way to open the page WITHOUT loading a model, which is the
thing "one place to see all of it" actually requires. It costs no GPU memory
and imports no engine: this module never touches mlx.

Loading from here starts `knurlogic serve` as a child process, because that
is what puts a model in memory with its settings resolved first. The child
owns the model; this page owns nothing but the view.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from knurlogic.machine import identity, loaded, status, wired
from knurlogic.machine.servers import (is_our_server, registry,
                                       save_registry, serve_log)
from knurlogic.interfaces import web
# Imported here, not inside the status handler: the page fires several
# requests at once, and two threads importing a module for the first time
# race -- measured as "partially initialized module 'typing'" on a restart.
from knurlogic.cluster import exo as exo_witness  # noqa: E402

#: Children started from the page: {port: (Popen, artifact path)}.
_CHILDREN: dict = {}
#: the page serves requests on threads: check-the-port-then-spawn is one step
_SPAWN_LOCK = __import__("threading").Lock()
#: the largest body the page accepts (the API server's default cap)
MAX_BODY = 512 << 20

#: The port a knurlogic on ANOTHER node is expected to answer on, which is
#: the one this page launches models on.
_SERVE_PORT: dict = {"n": 8080, "ui": 8899}


#: Where to look for exo, only to ask WHO IS THERE: exo is one witness of
#: which other machines exist (cluster/exo.py). knurlogic never drives it.
EXO_URL = exo_witness.EXO_URL


def _local_name(nodes) -> str:
    """Which of exo's nodes is the box this is running on.

    Matched on the product name exo reports (`modelId` is "Mac Studio", not
    `Mac15,14`) against what this machine says it is. Getting this wrong
    would put the local memory map on somebody else's gauge.
    """
    me = (wired.machine().get("model") or "").lower()
    if not me:
        return ""
    for n in nodes:
        if (n.model_id or "").lower() == me:
            return n.name
    return ""


def _local_memory(n, mm) -> dict:
    """This box's own numbers, preferring what it measured itself.

    exo reports system RAM per node, which is the only thing available for a
    peer. Locally there is something better -- vm_stat, read through
    `loaded.memory_map` -- so use it and fall back to exo's figures when it
    is unavailable. `render_cluster` needs the full shape, headroom included;
    leaving a key out is a KeyError at render time and not a smaller answer.
    """
    total = (mm or {}).get("installed_bytes") or n.ram_total
    used = (mm or {}).get("used_bytes")
    if used is None:
        used = max(n.ram_total - n.ram_available, 0)
    return {
        "available": True,
        "device": "vm_stat" if mm else "exo (system RAM)",
        "working_set_bytes": total,
        "active_bytes": used,
        "cache_bytes": 0,
        "headroom_bytes": max(total - used, 0),
        "scope": "box",
    }


_MM = {"doc": None, "at": 0.0}

#: The other machines this page knows (cluster/peers.py); None until the
#: page starts, so importing this module starts nothing.
PEERS = None


def _status_fn(_n=0):
    """A status for a box that is serving nothing.

    The node, its machine and its memory map are all still real -- that is
    the whole content of this mode. `artifact` is simply absent, and the
    page already handles that: it is the same shape `serve` emits.

    If exo is up, every node it knows about is included. A machine on the
    desk is a machine on the page whether or not this process is serving it:
    the complaint that produced this was seeing the other box fill up in
    exo's window and not in knurlogic's.

    A peer's own memory map arrives only if a knurlogic there answers on the
    network, and `serve` binds loopback by default -- so the usual case is
    exo's RAM figures and a plain gauge, which is still the machine and still
    its real occupancy.
    """
    # Reused for a few seconds, as `serve` does: the map runs `top`, which
    # takes over a second on a loaded box, and the page and every peer
    # asking for it would otherwise each pay that.
    now = time.time()
    if _MM["doc"] is None or now - _MM["at"] > 4.0:
        try:
            _MM["doc"] = loaded.memory_map()
        except Exception:
            _MM["doc"] = None
        _MM["at"] = now
    mm = _MM["doc"]

    me = identity.identity()
    exo_nodes = []
    try:
        exo_nodes = exo_witness.inventory(EXO_URL)
    except Exception:
        exo_nodes = []
    local = _local_name(exo_nodes) if exo_nodes else ""
    mine = next((n for n in exo_nodes if n.name == local), None)

    # This machine, always, under its own name -- exo or no exo.
    snaps = [status.snapshot(
        node=me["name"], role="local", memory_map=mm,
        memory_fn=(lambda: _local_memory(mine, mm)) if mine else None)]

    # Peers first: a node that answers for itself is the best witness of
    # itself. Then whatever only exo knows about, drawn from exo's figures.
    peers = PEERS.all() if PEERS else []
    claimed = {me["name"]}
    for p in peers:
        if p.state in ("answering", "version_mismatch") and p.node:
            snaps.append({**p.node, "role": "remote", "address": p.key,
                          "found_by": sorted(p.found_by), "state": p.state,
                          **({"problem": p.problem} if p.problem else {})})
        else:
            snaps.append({**status.snapshot(
                node=p.name or p.host, role="remote", reachable=False,
                memory_fn=lambda: {"available": False},
                machine_fn=lambda: {}),
                "found_by": sorted(p.found_by), "state": p.state,
                "problem": p.problem, "address": p.key})
        claimed |= {p.name, p.host}
    for n in exo_nodes:
        if n.name == local or n.name in claimed or n.ip in claimed:
            continue
        snaps.append({**exo_witness._snapshot_for(
            n, local, peer_port=_SERVE_PORT["ui"]),
            "found_by": ["exo"]})
    # what a coordinator page needs of this machine to place a rank on it
    try:
        from knurlogic.interfaces import cluster_jobs
        snaps[0]["cluster"] = cluster_jobs.node_info(
            (snaps[0].get("memory") or {}).get("working_set_bytes") or 0)
    except Exception as e:
        snaps[0]["cluster"] = {"error": f"{type(e).__name__}: {e}"}
    snap = status.aggregate(snaps)
    snap["wired"] = wired.advise(0)
    # Who this machine is, and -- measured by the peers, since this machine
    # cannot see connections its own firewall drops -- whether they can
    # reach it.
    snap["me"] = {**me, "port": _SERVE_PORT["ui"]}
    if DISCOVERY is not None:
        snap["discovery"] = DISCOVERY.status()
    if PEERS:
        snap["peers"] = [p.public() for p in peers]
        seen = PEERS.seen_by_peers()
        if seen:
            snap["me"]["seen_by"] = seen
        prob = PEERS.self_problem()
        if prob:
            snap["me"]["problem"] = prob
    return snap, status.render_cluster(snap)


#: A loading server whose log has said nothing for this long is reported as
#: stalled. Not killed -- a 400 GB rung read cold can be slow and silent --
#: but named, so an agent stops waiting and a person looks at the log.
STALL_QUIET_S = 180

#: A server holding less than this share of its weights is still warming.
WARM_FRACTION = 0.9


def _artifact_bytes(path: str) -> int:
    try:
        from knurlogic.machine.artifact import Artifact
        return int(Artifact.load(path).bytes_on_disk)
    except Exception:
        return 0


def _answers(port: int) -> bool:
    import urllib.request
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1.5)
        return True
    except Exception:
        return False


def children() -> list:
    """Every server knurlogic started, from any session, with its PHASE.

    `alive` alone was the trap: loading, serving and hung all read `alive:
    true`, and an agent that cannot tell them apart either guesses or waits
    forever -- the failure that made exo hard to drive. So each server says:

      warming   the port answers but the weights are not resident yet
                (mlx maps them lazily); memory is still moving
      serving   answering, and holding its weights
      loading   alive, not answering yet; with seconds elapsed, the last log
                line, and how long the log has been quiet
      stalled   loading, and the log has been quiet for STALL_QUIET_S --
                stop waiting and read the log
      exited    gone; exit code when this process started it, log tail
    """
    now = time.time()
    try:
        pids = {r["pid"]: r["bytes"] for r in loaded.memory_map()["processes"]}
    except Exception:
        pids = {}
    out = []
    for port, rec in sorted(registry().items()):
        pid = int(rec["pid"])
        mine = _CHILDREN.get(port)
        code = mine[0].poll() if mine and mine[0].pid == pid else None
        alive = code is None and is_our_server(pid)
        log = Path(rec.get("log", ""))
        try:
            lines = [l for l in log.read_text(errors="replace").splitlines()
                     if l.strip()]
            quiet = now - log.stat().st_mtime
        except OSError:
            lines, quiet = [], None
        row = {"port": port, "artifact": rec.get("artifact"), "pid": pid,
               "log": str(log), "started": rec.get("started"),
               "seconds_since_start": (round(now - rec["t"]) if rec.get("t")
                                       else None)}
        if alive and _answers(port):
            held = pids.get(pid, 0)
            size = _artifact_bytes(rec.get("artifact", ""))
            row["bytes_resident"] = held
            # Answering is not loaded. mlx maps weights lazily: measured, a
            # 15.5 GiB model answered with 3 GiB resident and reached 15.0
            # seconds later. Until it holds its weights, memory is still
            # moving, and a fit taken now is stale.
            if size and held < WARM_FRACTION * size:
                row["phase"] = "warming"
                row["weights_resident_fraction"] = round(held / size, 2)
            else:
                row["phase"] = "serving"
        elif alive:
            stalled = quiet is not None and quiet > STALL_QUIET_S
            row["phase"] = "stalled" if stalled else "loading"
            row["log_quiet_seconds"] = round(quiet) if quiet is not None else None
            row["last_log_line"] = lines[-1][:200] if lines else ""
            row["bytes_resident"] = pids.get(pid, 0)
            if stalled:
                row["advice"] = (f"no log output for {round(quiet)}s while "
                                 f"loading. Stop waiting; read {log}. "
                                 f"unload(port={port}) if it is hung.")
        else:
            row["phase"] = "exited"
            if code is not None:
                row["exit_code"] = code
            row["log_tail"] = lines[-15:]
        out.append(row)
    return out


def loading() -> list:
    """Servers knurlogic started whose memory is still in motion."""
    return [c for c in children()
            if c["phase"] in ("loading", "warming", "stalled")]


def _spawn(*a, **k):
    """_spawn_unlocked under the lock: two loads for one port at once would
    both pass the registry check, and the second child's record would hide
    the first (still loading, holding its memory, invisible to unload)."""
    with _SPAWN_LOCK:
        return _spawn_unlocked(*a, **k)


def _spawn_unlocked(path: str, port: int, tune: str = "balanced",
           sets: dict | None = None, draft: bool = True) -> dict:
    """Start `knurlogic serve` for one artifact, on its own port.

    Deliberately a child process rather than an in-process load: the
    settings that matter are resolved and put in the ENVIRONMENT before the
    engine imports anything, and that cannot be done to a process that is
    already running. A fresh process is the only way the knobs are honoured
    in full -- which is the difference between this and switching a model
    inside a running server.
    """
    import os
    if not Path(path).exists():
        return {"error": f"no such artifact: {path}"}
    rec = registry().get(port)
    if rec and is_our_server(int(rec["pid"])):
        return {"error": f"port {port} is already serving "
                         f"{rec.get('artifact')} (pid {rec['pid']})"}
    cmd = [sys.executable, "-m", "knurlogic", "serve", path,
           "--port", str(port), "--tune", tune]
    # Settings chosen at LAUNCH, which for most of these is the only moment
    # they can be chosen: they are read at import and compiled into kernel
    # source, so a running server cannot be told about them.
    for k, v in sorted((sets or {}).items()):
        cmd += ["--set", f"{k}={v}"]
    if not draft:
        cmd.append("--no-draft")
    log = serve_log(port)
    try:
        with open(log, "w") as fh:
            # Its own session, so it is not taken down with the terminal or
            # the MCP client that asked for it.
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                                    start_new_session=True)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    _CHILDREN[port] = (proc, path)
    reg = registry()
    reg[port] = {"pid": proc.pid, "artifact": path, "log": str(log),
                 "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "t": time.time()}
    save_registry(reg)
    return {"starting": path, "port": port, "pid": proc.pid,
            "log": str(log),
            "note": "the model is loading in its own process; poll `state` "
                    "(started_here) or GET /v1/models on the port"}


def _stop(port: int) -> dict:
    import os
    import signal
    reg = registry()
    rec = reg.get(port)
    if not rec:
        return {"error": f"knurlogic has no record of a server on port "
                         f"{port}; it stops only what it started"}
    if rec.get("job"):
        # rank 0 of a cluster job: the job stops, on every machine
        from knurlogic.interfaces import cluster_jobs
        out = cluster_jobs.stop(rec["job"], reason="unloaded")
        return {**out, "stopped": rec.get("artifact"), "port": port}
    pid = int(rec["pid"])
    if not is_our_server(pid):
        reg.pop(port, None)
        save_registry(reg)
        _CHILDREN.pop(port, None)
        return {"error": f"the server on port {port} (pid {pid}) is already "
                         f"gone; record cleared", "log": rec.get("log")}
    os.kill(pid, signal.SIGTERM)
    for _ in range(40):
        time.sleep(0.25)
        if not is_our_server(pid):
            break
    else:
        os.kill(pid, signal.SIGKILL)
    mine = _CHILDREN.pop(port, None)
    if mine:
        try:
            mine[0].wait(timeout=5)       # reap, so it is not left a zombie
        except Exception:
            pass
    reg.pop(port, None)
    save_registry(reg)
    return {"stopped": rec.get("artifact"), "port": port, "pid": pid}


def _load_fn(serve_port: int):
    def handler(_q: dict, body=None) -> dict:
        try:
            req = json.loads(body or b"{}")
        except Exception:
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
            from knurlogic.interfaces import cluster_jobs
            return cluster_jobs.stop(str(req["job"]), reason="unloaded")
        node = req.get("node")
        if node and node != identity.identity().get("id"):
            return _then_refresh(forward_launch(req))
        if node:
            # this machine, picked by id: the page's own load, by identity
            from knurlogic.machine.artifact import resolve_identity
            req = dict(req, target=req.get("target") or resolve_identity(
                req.get("identity")) or "")
        target, where = req.get("target") or "", req.get("where") or ""
        try:
            # Through the MCP's own functions: the page refuses what an agent
            # is refused -- will not fit, memory still moving -- in the same
            # words. The page loading past a check the MCP enforces would be
            # a capability on one side only.
            if act == "load":
                from knurlogic.interfaces import mcp
                return _then_refresh(mcp.load(artifact=target,
                                port=int(req.get("port") or serve_port),
                                tune=req.get("tune") or "balanced",
                                sets=req.get("sets") or {},
                                force=bool(req.get("force"))))
            if act == "unload":
                # Ours to stop only if we started it. Anything else is
                # somebody's server and not this page's to kill.
                for port, rec in registry().items():
                    if rec.get("artifact") == target or str(port) == str(target):
                        return _stop(port)
                return {"error": "this page did not start that; stop it "
                                 "where it was started"}
            if act == "ollama-unload":
                return loaded.ollama_unload(where, target)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
        return {"error": f"unknown action {act!r}"}
    return handler


# --- load / unload on ONE peer ---------------------------------------------
# The coordinator page (this one, with a peer picked) never tells a peer a
# path and never takes the peer's address from the request: the address is
# the one its PEERS store has for an ANSWERING peer with that id, the
# artifact is named by identity (machine/artifact.identity). Running
# knurlogic on a machine is its consent, as with exo; the peer still decides
# where a request may come from (its network gate, no Origin) and does its
# own fit check against its own load budget. The refusal text comes back
# as the peer wrote it.

#: the peer-only route a forwarded load arrives on
PEER_LOAD_PATH = "/peer/loaded.json"
#: a forwarded request is a few fields; anything bigger is not one
PEER_LOAD_MAX = 16 << 10
#: a load answers once the fit is checked and the child started
PEER_LOAD_S = 60.0
TUNES = ("safe", "balanced", "fast")
#: request keys that would name a place on disk; refused outright, never
#: ignored, so a coordinator that sends one learns it is wrong
PATH_KEYS = ("path", "target", "artifact", "where", "dir", "directory")


def launch_knobs() -> frozenset:
    """The knob names a forwarded load may set: the ones knurlogic documents
    (tuning/settings.KNOB_DOC) and their aliases. Nothing else is passed on:
    `--set` puts it in the child's environment."""
    from knurlogic.tuning import settings as S
    return frozenset(S.KNOB_DOC) | frozenset(
        n for v in S.KNOB_ALIASES.values() for n in v) | frozenset(
        S.NUMERICS_FLAGS)


def clean_sets(sets) -> tuple:
    """(allowed {name: value}, [refused names]). Values are short plain
    tokens: digits, letters, '.', '-', '_'."""
    import re
    ok, bad = {}, []
    allowed = launch_knobs()
    for k, v in (sets.items() if isinstance(sets, dict) else ()):
        v = str(v)
        if k in allowed and len(v) <= 64 and re.fullmatch(r"[\w.\-]*", v):
            ok[k] = v
        else:
            bad.append(str(k)[:64])
    return ok, bad


def _peer_by_id(node: str):
    for p in (PEERS.all() if PEERS else []):
        if getattr(p, "id", "") == node and p.state == "answering":
            return p
    return None


def _post_json(url: str, doc: dict, headers: dict, timeout: float):
    import urllib.error
    import urllib.request
    req = urllib.request.Request(
        url, data=json.dumps(doc).encode(), method="POST",
        headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def forward_launch(req: dict, post=None) -> dict:
    """A load or unload for the peer `req["node"]`, sent to that peer's
    PEER_LOAD_PATH. Builds the forwarded request from scratch: action,
    identity, port, tune, allow-listed sets, force -- nothing else of what
    the page sent goes over."""
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
        doc = {"action": "load", "identity": str(req.get("identity") or ""),
               "tune": req.get("tune") if req.get("tune") in TUNES
               else "balanced", "sets": sets, "force": bool(req.get("force"))}
        if req.get("port"):
            doc["port"] = int(req["port"])
        if not doc["identity"]:
            return {"error": "no identity for that model; this page's "
                             "/models.json gives one per artifact"}
    elif act == "unload":
        try:
            doc = {"action": "unload", "port": int(req.get("port"))}
        except (TypeError, ValueError):
            return {"error": "unload on another machine names the port"}
    else:
        return {"error": f"unknown action {act!r}"}
    post = post or _post_json
    try:
        code, raw = post(f"http://{p.key}{PEER_LOAD_PATH}", doc,
                         {}, PEER_LOAD_S)
    except Exception as e:
        return {"error": f"{who} did not answer: {type(e).__name__}: {e}"}
    try:
        out = json.loads(raw)
    except Exception:
        out = None
    if not isinstance(out, dict):
        text = raw.decode(errors="replace")[:500] if isinstance(
            raw, bytes) else str(raw)[:500]
        return {"error": f"{who} refused ({code}): {text}"}
    out = clean(out)
    out["machine"] = who
    return out


def cluster_launch(req: dict, serve_port: int) -> dict:
    """POST /loaded.json {action: load, identity, nodes: [>= 2 ids],
    split, link}: this page coordinates (interfaces/cluster_jobs.py)."""
    from knurlogic.interfaces import cluster_jobs
    if any(k in req for k in PATH_KEYS if k != "target") or req.get("target"):
        return {"error": "a model across machines is named by its "
                         "identity, never by a path"}
    link = {"tcp": "ring", "rdma": "jaccl"}.get(req.get("link"),
                                                req.get("link"))
    snap, _ = _status_fn()
    own = next((n for n in snap.get("nodes") or []
                if n.get("role") in ("local", "server")), {})
    return cluster_jobs.launch(
        dict(req, link=link), me=identity.identity(),
        peers=PEERS.all() if PEERS else [],
        local_info=own.get("cluster") or cluster_jobs.node_info(),
        ui_port=_SERVE_PORT["ui"], serve_port=serve_port)


def clean(doc):
    from knurlogic.cluster.peers import clean as _clean
    return _clean(doc)


def peer_refusal(headers, client_ip: str, local_ip: str, gate=None,
                 manual_hosts=(), what: str = "peer requests"):
    """The gate every /peer/ route shares: (status, doc) when refused, else
    None. No Origin header (a browser never reaches a peer route), and the
    connection arrived on loopback or Thunderbolt, or from a peer address
    named with --peer -- in every mode, whatever --host says."""
    from knurlogic.cluster.links import Gate
    if headers.get("Origin") is not None:
        return 403, {"error": "a web page cannot drive another machine"}
    ip = (client_ip or "").removeprefix("::ffff:")
    g = gate or Gate()
    if not (g.allows(local_ip) or ip in set(manual_hosts)):
        return 403, {"error": f"{what} are taken over Thunderbolt or "
                              f"loopback, or from a peer named with --peer; "
                              f"this came from {ip}"}
    return None


def peer_launch(headers, client_ip: str, local_ip: str, body: bytes,
                gate=None, manual_hosts=(), load=None, stop=None,
                resolve=None) -> tuple:
    """POST /peer/loaded.json, on the machine asked to load: (status, doc).

    Refused, in this order, unless: no Origin header (a browser never
    reaches this); the connection arrived on loopback or Thunderbolt, or
    from a peer address named with --peer. Running knurlogic on a machine
    is its consent to load for the cluster, as with exo. Then the
    request is a load by identity -- resolved to a path HERE, from this
    machine's own stores -- or an unload of a port this machine started."""
    refused = peer_refusal(headers, client_ip, local_ip, gate,
                           manual_hosts, what="launches")
    if refused:
        return refused
    if len(body or b"") > PEER_LOAD_MAX:
        return 413, {"error": "a launch request is small"}
    try:
        req = json.loads(body or b"")
    except ValueError as e:
        return 400, {"error": f"the body must be a JSON object: not JSON "
                              f"({e})"}
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
        return 200, (stop or _stop)(port)
    if act != "load":
        return 400, {"error": f"unknown action {act!r}"}
    sets, bad = clean_sets(req.get("sets") or {})
    if bad:
        return 400, {"error": f"not a launch setting: {', '.join(bad)}"}
    tune = req.get("tune") if req.get("tune") in TUNES else "balanced"
    from knurlogic.machine.artifact import resolve_identity
    path = (resolve or resolve_identity)(req.get("identity"))
    if not path:
        name = identity.identity().get("name") or "this machine"
        return 404, {"loaded": False, "refused": f"not on {name}",
                     "note": f"{name} has no artifact with identity "
                             f"{str(req.get('identity'))[:64]!r} in its "
                             f"model stores. Copy it there first."}
    port = req.get("port")
    port = port if isinstance(port, int) and not isinstance(port, bool) \
        and 1024 <= port < 65536 else _SERVE_PORT["n"]
    if load is None:
        from knurlogic.interfaces import mcp
        load = mcp.load
    # the fit check is this machine's, against its own load budget
    return 200, load(artifact=path, port=port, tune=tune, sets=sets,
                     force=bool(req.get("force")))


#: How long the page waits for all peers together. A peer's /loaded.json
#: runs its memory map (about a second on a busy box); past this the local
#: answer goes out without that peer, which is listed as not answering.
PEER_LOADED_S = 2.5

#: Chat endpoints on peers, as the peers themselves last reported them:
#: {base: {"machine": name, "relay": the peer's page}}. `base` is the
#: model's address as seen from here (what the page shows and keys a chat
#: by); a peer's server listens on ITS loopback, so every request for it
#: goes to the peer's page relay (PEER_RELAY) instead. Refilled by every
#: peer survey.
_PEER_TARGETS: dict = {}
#: the peer page's relay prefix: /peer/v1/... reaches the model servers
#: that page itself started, by model name (peer_relay)
PEER_RELAY = "/peer"


def upstream(base: str, path: str) -> str:
    """The URL a request for `path` on the model at `base` goes to: the
    server itself when it is this machine's, the peer page's relay when it
    is a peer's."""
    t = _PEER_TARGETS.get(base)
    if t:
        return t["relay"] + PEER_RELAY + path
    return base + path


def _peer_where(where: str, host: str) -> str:
    """A peer reports its models at ITS loopback; seen from here the same
    port is at the peer's address. An endpoint on any OTHER host is
    dropped (""): peers are found by Bonjour, which anything on the network
    can advertise into, and what a peer reports becomes an address this
    page's chat proxy will POST to -- so a peer may only offer itself."""
    u = urlparse(where or "")
    if not u.port or u.scheme not in ("http", ""):
        return ""
    if u.hostname in ("127.0.0.1", "localhost", "::1", host):
        return f"http://{host}:{u.port}"
    return ""


def peer_residency(peers, timeout: float = PEER_LOADED_S,
                   fetch=None) -> list:
    """What every answering peer says it is holding, one entry per machine.

    Asked in parallel with one shared deadline, so a slow or dead peer costs
    at most `timeout` and never the local answer. Peers are asked plain
    /loaded.json -- never ?peers=1 -- so two pages asking each other cannot
    recurse. Each row is labelled with its machine and its address rewritten
    from the peer's loopback to the peer's address."""
    import threading
    import urllib.request
    if fetch is None:
        def fetch(url, t):
            with urllib.request.urlopen(url, timeout=t) as r:
                return json.loads(r.read())
    todo = [p for p in (peers.all() if peers else [])
            if p.state == "answering"]
    out: dict = {}

    def one(p):
        try:
            doc = fetch(f"http://{p.key}/loaded.json", timeout)
            rows = []
            for r in doc.get("resident") or []:
                if isinstance(r, dict):
                    rows.append(dict(r, machine=p.name or p.host,
                                     where=_peer_where(r.get("where"),
                                                       p.host)))
            js = [clean(j) for j in doc.get("jobs") or []
                  if isinstance(j, dict)]
            out[p.key] = {"machine": p.name or p.host, "address": p.key,
                          "id": getattr(p, "id", ""), "resident": rows,
                          "jobs": js}
        except Exception as e:
            out[p.key] = {"machine": p.name or p.host, "address": p.key,
                          "resident": [],
                          "error": f"{type(e).__name__}: {e}"}
    ts = [threading.Thread(target=one, args=(p,), daemon=True) for p in todo]
    for t in ts:
        t.start()
    end = time.time() + timeout
    for t in ts:
        t.join(max(end - time.time(), 0))
    res = []
    for p in todo:
        # a thread still running past the deadline has written nothing yet
        res.append(out.get(p.key) or {
            "machine": p.name or p.host, "address": p.key, "resident": [],
            "error": f"did not answer in {timeout:.1f} s"})
    targets = {}
    for m in res:
        for r in m["resident"]:
            if r.get("runtime") == "knurlogic" and r.get("where"):
                targets[r["where"].rstrip("/")] = {
                    "machine": m["machine"],
                    "relay": f"http://{m['address']}"}
    _PEER_TARGETS.clear()
    _PEER_TARGETS.update(targets)
    _PEER_AT[0] = time.time()
    return res


def with_jobs(doc: dict) -> dict:
    """/loaded.json plus the cluster jobs with a rank here (`jobs`), and
    rank 0's resident row marked with its job, so the job is listed once,
    on its leader, with its machines."""
    try:
        from knurlogic.interfaces import cluster_jobs
        js = cluster_jobs.jobs_document()
    except Exception:
        js = []
    ports = {j["port"]: j for j in js if j.get("port")}
    rows = []
    for r in doc.get("resident") or []:
        u = urlparse((r.get("where") or "") if isinstance(r, dict) else "")
        j = ports.get(u.port) if u.port else None
        rows.append(dict(r, cluster={k: j.get(k) for k in (
            "job", "split", "link", "machines", "leader", "phase")})
                    if j else r)
    return dict(doc, resident=rows, jobs=js)


def _loaded_fn():
    """/loaded.json as `web` answers it for this box; with ?peers=1 (what the
    page asks) it also carries `peers`: each other machine's residency."""
    local = web.loaded_document()

    def handler(q: dict) -> dict:
        # a copy: the local document is cached and shared between requests
        doc = with_jobs(local(q))
        if not (q.get("peers") or [""])[0]:
            return doc
        return dict(doc, peers=peer_residency(PEERS))
    return handler


def refresh_targets() -> None:
    """Ask the peers what they serve NOW and forget the router's cached
    table: after a launch, and once before refusing a model this page does
    not know, so a chat to a job launched a moment ago is not refused until
    the page's next survey."""
    _ROUTES["at"] = 0.0
    if PEERS is not None:
        try:
            peer_residency(PEERS)
        except Exception:
            pass


def _then_refresh(out):
    """A launch's answer, passed on; a launch that went ahead first
    refreshes what this page's chat and router can reach."""
    if isinstance(out, dict) and not out.get("error") \
            and not out.get("refused") and out.get("loaded") is not False:
        refresh_targets()
    return out


def known_target(base: str) -> bool:
    """`base` is one this page may send to -- re-surveying the peers once
    before saying no."""
    if base in chat_targets():
        return True
    refresh_targets()
    return base in chat_targets()


def chat_targets() -> set:
    """Endpoints the page may send a chat to: servers knurlogic started that
    are still ours. A fixed allow-list, so the proxy cannot be
    pointed at an arbitrary address by whatever is in the request."""
    from knurlogic.machine.servers import is_our_server
    out = set()
    for port, rec in registry().items():
        if is_our_server(int(rec["pid"])):
            out.add(f"http://127.0.0.1:{port}")
    # and knurlogic servers a peer reported in its own residency: machines
    # this page already polls, never an address taken from the request
    out.update(_PEER_TARGETS)
    return out


def _send_json(handler, code: int, doc) -> None:
    out = json.dumps(doc).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(out)))
    handler.end_headers()
    handler.wfile.write(out)


def _stream(handler, url: str, body: bytes, timeout: float = 3600) -> None:
    """POST `body` to `url` and pass the answer back as it arrives, byte for
    byte: SSE, prefill keepalives and all. The upstream's status and
    Content-Type go with it; an upstream that cannot be reached is a 502."""
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    try:
        up = urllib.request.urlopen(req, timeout=timeout)
        code, ctype = up.status, up.headers.get("Content-Type",
                                                "application/json")
    except urllib.error.HTTPError as e:
        up, code = e, e.code
        ctype = e.headers.get("Content-Type", "application/json")
    except Exception as e:
        _send_json(handler, 502, {"error": f"{type(e).__name__}: {e}"})
        return
    handler.send_response(code)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.close_connection = True
    try:
        while True:
            chunk = up.read1(8192) if hasattr(up, "read1") else up.read(8192)
            if not chunk:
                break
            handler.wfile.write(chunk)
            handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
        pass            # the client stopped listening; nothing to answer
    finally:
        up.close()


def proxy_chat(handler, where: str, body: bytes) -> None:
    """POST /chat?where=<base>: forward a chat request to a running model
    and stream its answer back as it arrives.

    The control page serves no model, so its chat has to reach the one the
    person clicked -- a server `load` started -- and a browser
    will not let a page on this port call another port directly."""
    base = (where or "").rstrip("/")
    if not known_target(base):
        _send_json(handler, 403, {"error": f"not a running model this page "
                                           f"knows: {base or '(none)'}"})
        return
    _stream(handler, upstream(base, "/v1/chat/completions"), body)


#: the largest settings change `/apply` forwards; a knob set is a few bytes
APPLY_MAX = 16 << 10
APPLY_S = 10.0


def apply_settings(where: str, body: bytes, post=None) -> tuple:
    """POST /apply?where=<base>: forward a live-knob change to a running
    model server's own POST /settings.json and hand back its per-knob report.

    Only a knurlogic model server this page knows (chat_targets) -- never a
    peer's page, never another path -- a small JSON object, a short
    deadline. The page sends it only when the person clicks Apply; which
    knobs actually move is the model server's to decide and report."""
    import urllib.error
    import urllib.request
    base = (where or "").rstrip("/")
    if not known_target(base):
        return 403, {"error": f"not a running model this page knows: "
                              f"{base or '(none)'}"}
    if len(body or b"") > APPLY_MAX:
        return 413, {"error": f"a settings change is at most {APPLY_MAX} "
                              f"bytes"}
    try:
        want = json.loads(body or b"")
    except ValueError:
        want = None
    if not isinstance(want, dict):
        return 400, {"error": "the body must be a JSON object of knobs"}
    if post is None:
        def post(url, data, t):
            req = urllib.request.Request(
                url, data=data, method="POST",
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=t) as r:
                    return r.status, r.read()
            except urllib.error.HTTPError as e:
                return e.code, e.read()
    try:
        code, raw = post(f"{base}/settings.json",
                         json.dumps(want).encode(), APPLY_S)
        return code, json.loads(raw)      # JSON only, never an HTML page
    except Exception as e:
        return 502, {"error": f"{type(e).__name__}: {e}"}


#: The paths the page's router forwards by `model`: the two chat surfaces
#: a client pointed at this page (Claude Code, an OpenAI SDK) uses, and
#: Claude Code's token count (it names the model too).
ROUTE_PATHS = ("/v1/messages", "/v1/chat/completions",
               "/v1/messages/count_tokens")
ROUTE_S = 2.0
_ROUTES: dict = {"at": 0.0, "map": {}}
#: when peers were last asked what they serve (peer_residency)
_PEER_AT = [0.0]
#: how old that may be before the router asks again itself
PEER_SURVEY_MAX_AGE_S = 30.0


def routable(fetch=None, ttl: float = 5.0) -> dict:
    """{model id: base} for every running knurlogic server this page knows
    (chat_targets: its own and the ones peers reported). Each is asked its
    /v1/models -- the id a client names is the one the server answers to --
    in parallel with one deadline; one that does not answer is left out.
    Cached for a few seconds: Claude Code sends several requests a turn."""
    import threading
    import urllib.request
    now = time.time()
    if fetch is None:
        if now - _ROUTES["at"] < ttl:
            return dict(_ROUTES["map"])

        def fetch(url, t):
            with urllib.request.urlopen(url, timeout=t) as r:
                return json.loads(r.read())
        # the page's own polling refreshes what peers serve, but a client
        # (Claude Code) may call before anyone has opened the page: the
        # router then saw only this machine
        if PEERS is not None and now - _PEER_AT[0] > PEER_SURVEY_MAX_AGE_S:
            try:
                peer_residency(PEERS)
            except Exception:
                pass
    found: dict = {}

    def one(base):
        try:
            for m in fetch(upstream(base, "/v1/models"),
                           ROUTE_S).get("data") or []:
                if isinstance(m, dict) and m.get("id"):
                    found.setdefault(str(m["id"]), base)
        except Exception:
            pass
    ts = [threading.Thread(target=one, args=(b,), daemon=True)
          for b in sorted(chat_targets())]
    for t in ts:
        t.start()
    end = time.time() + ROUTE_S
    for t in ts:
        t.join(max(end - time.time(), 0))
    out = dict(found)
    _ROUTES.update(at=now, map=out)
    return out


def route_models_document(fetch=None) -> dict:
    """GET /v1/models on the page: every model its router can reach."""
    return {"object": "list", "data": [
        {"id": m, "object": "model", "owned_by": "knurlogic", "server": b}
        for m, b in sorted(routable(fetch).items())]}


def route(handler, path: str, body: bytes, fetch=None) -> None:
    """POST /v1/messages or /v1/chat/completions on the page: forward to
    the running knurlogic server whose model id is the request's `model`.

    Claude Code takes ONE base URL and names a model per tier; each
    knurlogic server holds one model. So the page, which already knows
    every running server here and on its peers, is the one address that
    can hand each request to the server holding the model it names. Only
    to those servers -- the same allow-list the page's chat uses."""
    try:
        model = json.loads(body or b"{}").get("model")
    except Exception:
        model = None
    table = routable(fetch)
    base = table.get(model) if isinstance(model, str) else None
    if base is None:
        # launched a moment ago? ask the peers once more before refusing
        refresh_targets()
        table = routable(fetch)
        base = table.get(model) if isinstance(model, str) else None
    if base is None:
        # say it in the shape either client reads as an error
        _send_json(handler, 404, {
            "type": "error",
            "error": {"type": "not_found_error",
                      "message": f"no running model {model!r}; running: "
                                 f"{', '.join(sorted(table)) or 'none'}"},
            "models": sorted(table)})
        return
    _stream(handler, upstream(base, path), body)


def local_models(fetch=None) -> dict:
    """{model id: base} over the servers THIS machine started (never the
    ones peers reported, so two pages relaying for each other cannot
    loop). What a peer page's relay resolves a model name against."""
    import threading
    import urllib.request
    from knurlogic.machine.servers import is_our_server
    if fetch is None:
        def fetch(url, t):
            with urllib.request.urlopen(url, timeout=t) as r:
                return json.loads(r.read())
    bases = sorted(f"http://127.0.0.1:{port}" for port, rec in
                   registry().items() if is_our_server(int(rec["pid"])))
    found: dict = {}

    def one(base):
        try:
            for m in fetch(f"{base}/v1/models", ROUTE_S).get("data") or []:
                if isinstance(m, dict) and m.get("id"):
                    found.setdefault(str(m["id"]), base)
        except Exception:
            pass
    ts = [threading.Thread(target=one, args=(b,), daemon=True)
          for b in bases]
    for t in ts:
        t.start()
    end = time.time() + ROUTE_S
    for t in ts:
        t.join(max(end - time.time(), 0))
    return dict(found)


def _resolve(table: dict, model):
    """The base serving `model`: by exact id, else by its last path part
    (a page may name a model by its folder). Absent, and only one model is
    running, that one."""
    if isinstance(model, str) and model:
        if model in table:
            return table[model]
        tail = model.rstrip("/").split("/")[-1]
        hits = {b for m, b in table.items()
                if m.rstrip("/").split("/")[-1] == tail}
        return hits.pop() if len(hits) == 1 else None
    bases = set(table.values())
    return bases.pop() if len(bases) == 1 else None


def peer_relay(handler, method: str, path: str, body: bytes,
               fetch=None) -> None:
    """GET /peer/v1/models, POST /peer/v1/chat/completions, /v1/messages,
    /v1/messages/count_tokens -- on the machine holding the model, for a
    peer page whose router or chat names it. The caller has passed
    peer_refusal. Resolved by model name against the servers this machine
    started (local_models) and streamed back as it arrives."""
    table = local_models(fetch)
    if method == "GET":
        if path != "/v1/models":
            _send_json(handler, 404, {"error": "not a relayed path"})
            return
        _send_json(handler, 200, {"object": "list", "data": [
            {"id": m, "object": "model", "owned_by": "knurlogic"}
            for m in sorted(table)]})
        return
    if path not in ROUTE_PATHS:
        _send_json(handler, 404, {"error": "not a relayed path"})
        return
    try:
        doc = json.loads(body or b"{}")
    except ValueError:
        doc = None
    if not isinstance(doc, dict):
        _send_json(handler, 400, {"error": "the body must be a JSON object"})
        return
    model = doc.get("model")
    base = _resolve(table, model)
    if base is None:
        _send_json(handler, 404, {
            "type": "error",
            "error": {"type": "not_found_error",
                      "message": f"no running model {model!r} on "
                                 f"{identity.identity().get('name') or 'this machine'}"
                                 f"; running: "
                                 f"{', '.join(sorted(table)) or 'none'}"},
            "models": sorted(table)})
        return
    _stream(handler, base + path, body)


#: What `/peek` may read, and the query keys it passes along. Reads only:
#: a running model's settings and sampling defaults, a peer page's machine
#: settings. Nothing that changes anything is reachable through it.
PEEK_PATHS = ("/settings.json", "/v1/models", "/models.json")
PEEK_KEYS = ("tune", "working_set_gib", "wired_gib")
PEEK_S = 3.0


def peek_targets() -> set:
    """Addresses `/peek` may read from: the running models the chat proxy
    already allows, and the pages of peers that are answering -- machines
    this page polls anyway, never an address taken from the request."""
    out = set(chat_targets())
    for p in (PEERS.all() if PEERS else []):
        if p.state == "answering":
            out.add(f"http://{p.key}")
    return out


def peek(q: dict, fetch=None) -> tuple:
    """GET /peek?where=<base>&path=<path>: another server's read-only
    document, for a page that cannot call another port or machine itself.

    (status, body): the upstream's JSON as it came, or a JSON error. GET
    only, a fixed list of paths and targets, a short deadline: Settings on
    a peer's tab reads that peer and can never set anything on it."""
    import urllib.parse
    import urllib.request
    where = ((q.get("where") or [""])[0] or "").rstrip("/")
    path = (q.get("path") or [""])[0]
    if path not in PEEK_PATHS:
        return 403, json.dumps({"error": f"not a readable path: {path!r}"})
    if where not in peek_targets():
        return 403, json.dumps({"error": f"not a server this page knows: "
                                         f"{where or '(none)'}"})
    fwd = {k: q[k][0] for k in PEEK_KEYS if q.get(k)}
    url = upstream(where, path) + ("?" + urllib.parse.urlencode(fwd) if fwd else "")
    if fetch is None:
        def fetch(u, t):
            with urllib.request.urlopen(u, timeout=t) as r:
                return r.read()
    try:
        body = fetch(url, PEEK_S)
        json.loads(body)          # pass on JSON only, never an HTML page
        return 200, body.decode() if isinstance(body, bytes) else body
    except Exception as e:
        return 502, json.dumps({"error": f"{type(e).__name__}: {e}"})


#: Bonjour (cluster/discovery.py); None when it could not start.
DISCOVERY = None


def _start_discovery(me: dict, host: str, port: int, reachable: bool):
    """Browse always -- a loopback page can still reach peers outbound --
    and advertise only when bound where others can reach it: advertising an
    address nobody can connect to is the silence this replaces."""
    global DISCOVERY
    try:
        from knurlogic import __version__
        from knurlogic.cluster import discovery as dsd

        def found(services):
            for svc in services:
                txt = svc.get("txt") or {}
                if txt.get("id") == me["id"] or not PEERS:
                    continue
                p = PEERS.add(svc["host"], svc["port"], "bonjour")
                p.id = p.id or txt.get("id", "")
                p.name = p.name or txt.get("name", "")
        d = dsd.Discovery(on_change=found)
        txt = {"id": me["id"], "name": me["name"], "ver": __version__,
               "schema": status.SCHEMA, "role": "ui"}
        if host == "cluster":
            # Advertised on the Thunderbolt links only: a peer that can
            # only see us over the cable cannot pick Wi-Fi.
            import socket as _s
            from knurlogic.cluster import links
            for i in links.thunderbolt():
                d.if_index = _s.if_nametoindex(i["iface"])
                d.register(f"{me['name']} {me['id'][:6]}", port, txt)
            d.if_index = 0
        elif reachable:
            d.if_index = dsd.interface_of(host)
            d.register(f"{me['name']} {me['id'][:6]}", port, txt)
            d.if_index = 0
        d.browse()
        DISCOVERY = d.start()
    except Exception as e:
        print(f"bonjour unavailable ({type(e).__name__}: {e}); peers can "
              f"still be named with --peer", file=sys.stderr)


def make_handler(routes: dict, gate=None, allow_origins=(),
                 allow_hosts=(), gate_for_peers=None):
    """The page's request handler: its routes, the router, the proxies, and
    the guards in front of every one of them. `gate_for_peers`: the
    /peer/ gate's link check (default cluster/links.Gate; tests pass
    their own)."""

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _gated(self) -> bool:
            """True when this request is refused: --host cluster's gate,
            or the browser guard knurlogic's server applies too (a page
            in the user's browser must not drive this one either)."""
            if gate is not None:
                local = self.connection.getsockname()[0]
                if not gate.allows(local):
                    self._send(gate.refusal(local), "text/plain", 403)
                    return True
            from knurlogic.interfaces.http.server import browser_refusal
            why = browser_refusal(self.headers, allow_origins, allow_hosts)
            if why is not None:
                self.close_connection = True
                self._send(why.encode(), "text/plain; charset=utf-8", 403)
                return True
            return False

        def do_GET(self):
            u = urlparse(self.path)
            if u.path.startswith(PEER_RELAY + "/v1/"):
                self._peer_relay("GET", u.path)
                return
            if self._gated():
                return
            intro = self.headers.get("X-Knurlogic-Peer")
            if intro and PEERS:
                PEERS.introduce(self.client_address[0], intro)
            if u.path.rstrip("/") == "/v1/models":
                self._send(json.dumps(route_models_document()).encode(),
                           "application/json")
                return
            if u.path.rstrip("/") == "/peek":
                code, doc = peek(parse_qs(u.query))
                self._send(doc.encode(), "application/json", code)
                return
            h = routes.get(u.path.rstrip("/") or "/")
            if h is None:
                self._send(b"not found", "text/plain", 404)
                return
            body, ctype = h(parse_qs(u.query), 0)
            self._send(body, ctype)

        def do_POST(self):
            u = urlparse(self.path)
            if u.path.rstrip("/") == PEER_LOAD_PATH:
                self._peer_load()
                return
            from knurlogic.interfaces import cluster_jobs
            if u.path.rstrip("/") in cluster_jobs.PEER_PATHS:
                self._peer_cluster(u.path.rstrip("/"))
                return
            if u.path.startswith(PEER_RELAY + "/v1/"):
                self._peer_relay("POST", u.path)
                return
            if self._gated():
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = -1
            if not 0 <= n <= MAX_BODY:
                self._send(b"Content-Length must be a number of bytes up "
                           b"to %d" % MAX_BODY, "text/plain", 400)
                return
            if u.path.rstrip("/") == "/chat":
                where = (parse_qs(u.query).get("where") or [""])[0]
                proxy_chat(self, where, self.rfile.read(n) if n else b"")
                return
            if u.path.rstrip("/") in ROUTE_PATHS:
                route(self, u.path.rstrip("/"),
                      self.rfile.read(n) if n else b"")
                return
            if u.path.rstrip("/") == "/apply":
                where = (parse_qs(u.query).get("where") or [""])[0]
                if n > APPLY_MAX:
                    self._send(b"a settings change is small", "text/plain",
                               413)
                    return
                code, doc = apply_settings(where, self.rfile.read(n)
                                           if n else b"")
                self._send(json.dumps(doc).encode(), "application/json",
                           code)
                return
            h = routes.get("POST " + (u.path.rstrip("/") or "/"))
            if h is None:
                self._send(b"not found", "text/plain", 404)
                return
            body, ctype = h(parse_qs(u.query), 0, self.rfile.read(n) if n
                            else b"")
            self._send(body, ctype)

        def _peer_relay(self, method: str, path: str):
            """PEER_RELAY/v1/...: a peer page reaching a model this machine
            started. The peer gate (peer_refusal), then a plain
            Content-Length body -- never Transfer-Encoding, the framing a
            smuggled request hides behind -- then peer_relay."""
            manual = [p.host for p in (PEERS.all() if PEERS else [])
                      if "manual" in p.found_by]
            refused = peer_refusal(
                self.headers, self.client_address[0],
                self.connection.getsockname()[0], manual_hosts=manual,
                what="relayed requests")
            if refused:
                self.close_connection = True
                _send_json(self, *refused)
                return
            if self.headers.get("Transfer-Encoding"):
                self.close_connection = True
                _send_json(self, 411, {"error": "send the body with a "
                           "Content-Length and no Transfer-Encoding"})
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = -1
            if not 0 <= n <= MAX_BODY:
                self.close_connection = True
                _send_json(self, 400, {"error": f"Content-Length must be a "
                                       f"number of bytes up to {MAX_BODY}"})
                return
            body = self.rfile.read(n) if n else b""
            peer_relay(self, method,
                       path[len(PEER_RELAY):].rstrip("/"), body)

        def _peer_cluster(self, path: str):
            """/peer/cluster/*: the same gate as PEER_LOAD_PATH."""
            from knurlogic.interfaces import cluster_jobs
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = -1
            if not 0 <= n <= cluster_jobs.PEER_MAX:
                self.close_connection = True
                _send_json(self, 413, {"error": "a cluster request is small"})
                return
            manual = [p.host for p in (PEERS.all() if PEERS else [])
                      if "manual" in p.found_by]
            refused = peer_refusal(
                self.headers, self.client_address[0],
                self.connection.getsockname()[0], gate=gate_for_peers,
                manual_hosts=manual, what="cluster requests")
            body = self.rfile.read(n) if n else b""
            if refused:
                self.close_connection = True
                _send_json(self, *refused)
                return
            code, doc = cluster_jobs.peer_route(path, body)
            web._LOADED["doc"] = None
            _send_json(self, code, doc)

        def _peer_load(self):
            """PEER_LOAD_PATH: its own checks (peer_launch), not the
            browser guard's -- it refuses ANY Origin, and its gate is
            Thunderbolt/loopback or a --peer address in every mode."""
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = -1
            if not 0 <= n <= PEER_LOAD_MAX:
                self._send(b'{"error": "a launch request is small"}',
                           "application/json", 413)
                return
            manual = [p.host for p in (PEERS.all() if PEERS else [])
                      if "manual" in p.found_by]
            code, doc = peer_launch(
                self.headers, self.client_address[0],
                self.connection.getsockname()[0],
                self.rfile.read(n) if n else b"", manual_hosts=manual)
            if code == 200:
                web._LOADED["doc"] = None     # residency just changed
            self._send(json.dumps(doc).encode(), "application/json", code)
    return H


def serve_ui(host: str, port: int, serve_port: int, peers=(),
             allow_origins=(), allow_hosts=()) -> int:
    global PEERS
    _SERVE_PORT["n"] = serve_port
    _SERVE_PORT["ui"] = port
    from knurlogic.cluster import links
    from knurlogic.cluster.peers import Peers
    me = identity.identity()
    # `cluster`: every address bound, only loopback and Thunderbolt
    # answered (cluster/links.Gate), advertised on Thunderbolt only.
    gate = links.Gate() if host == "cluster" else None
    bind = "0.0.0.0" if gate else host
    reachable = host not in ("127.0.0.1", "localhost", "::1")
    PEERS = Peers(me, port, manual=peers, reachable=reachable).start()
    _start_discovery(me, host, port, reachable)
    routes = web.routes(
        status_fn=_status_fn,
        settings_fn=web.machine_settings(),
        models_fn=web.models_document(serving=""),
        loaded_fn=_loaded_fn(),
        load_fn=_load_fn(serve_port))
    # the knurlogic allowance: THIS machine's only, and set only by a POST
    # the page sends when its user applies it; a peer's is read from that
    # peer's /settings.json through /peek
    routes["/allowance.json"] = lambda _q, _n=0: web._json(
        web.allowance_doc())
    routes["POST /allowance.json"] = lambda _q, _n=0, body=None: web._json(
        web.set_allowance(body))

    H = make_handler(routes, gate, allow_origins, allow_hosts)
    from knurlogic.interfaces import cluster_jobs
    cluster_jobs.start_watching_existing()

    srv = ThreadingHTTPServer((bind, port), H)
    if gate:
        tb = [i["ip"] for i in links.thunderbolt()]
        print(f"knurlogic  cluster mode: answering on "
              f"{', '.join(f'http://{ip}:{port}' for ip in tb) or 'no Thunderbolt link yet'}"
              f" and http://127.0.0.1:{port}; Wi-Fi and Ethernet refused")
    else:
        print(f"knurlogic  http://{host}:{port}")
    print(f"  nothing loaded, nothing required -- not exo, not a model.")
    print(f"  loading from the page starts `knurlogic serve` on port "
          f"{serve_port}.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        # The page's own children end with the page. Through _stop, so the
        # registry an MCP session reads does not keep a dead entry.
        for port in list(_CHILDREN):
            _stop(port)
        # and the cluster ranks it started, on every machine of their job
        from knurlogic.interfaces import cluster_jobs
        for job in list(cluster_jobs.SPECS):
            cluster_jobs.stop(job, reason="the page that started it closed")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="knurlogic ui",
        description="the page, without loading anything: every model on the "
                    "disk, everything resident in every runtime, and where "
                    "the memory went")
    p.add_argument("--host", default="127.0.0.1",
                   help="127.0.0.1 (default: this machine only), an "
                        "address, or `cluster`: answer on the Thunderbolt "
                        "link(s) and loopback only, advertise there, refuse "
                        "Wi-Fi and Ethernet")
    p.add_argument("--port", type=int, default=8899)
    p.add_argument("--serve-port", type=int, default=8080,
                   help="the port a model loaded from this page is served on")
    p.add_argument("--peer", action="append", default=[],
                   metavar="HOST[:PORT]",
                   help="another machine's knurlogic page (repeatable). "
                        "Naming it on one side is enough: it learns this "
                        "machine from the request. Remembered once it "
                        "answers.")
    p.add_argument("--allow-origin", action="append", default=[],
                   metavar="URL", help="a web page origin allowed to call "
                   "this page's API from a browser (repeatable)")
    p.add_argument("--allow-host", action="append", default=[],
                   metavar="NAME", help="a DNS name this machine is reached "
                   "by, beyond localhost, IPs, .local and its hostname")
    a = p.parse_args(argv)
    peers = []
    for spec in a.peer:
        host, _, port = spec.rpartition(":") if ":" in spec else (spec, "", "")
        peers.append((host, int(port) if port.isdigit() else a.port))
    return serve_ui(a.host, a.port, a.serve_port, peers,
                    allow_origins=a.allow_origin, allow_hosts=a.allow_host)
