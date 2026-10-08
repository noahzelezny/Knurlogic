"""`knurlogic ui` -- the page, with nothing loaded and nothing else running.

It needs no model loaded and no other runtime: open it and you see every
model on the disk, everything resident in every runtime, and where the
memory went. It costs no GPU memory and never imports mlx.

Loading from here starts `knurlogic serve` as a child process, because that
is what puts a model in memory with its settings resolved first. The child
owns the model; this page owns nothing but the view.
"""

from __future__ import annotations

import argparse
import http.client
import json
import logging
import subprocess
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, quote, urlparse

from knurlogic.interfaces.page import documents
from knurlogic.machine import identity, loaded, status, wired
from knurlogic.machine.servers import is_our_server, registry, save_registry, serve_log

# the launch facts cluster jobs share: tuning/settings owns them
from knurlogic.tuning.settings import PATH_KEYS, clean_sets, preset_or

if TYPE_CHECKING:
    from knurlogic.cluster.peers import Peers

logger = logging.getLogger(__name__)

#: Children started from the page: {port: (Popen, artifact path)}.
_CHILDREN: dict = {}
#: the page serves requests on threads: check-the-port-then-spawn is one step
_SPAWN_LOCK = __import__("threading").Lock()
#: the largest body the page accepts (the API server's default cap)
MAX_BODY = 512 << 20

#: The port a knurlogic on ANOTHER node is expected to answer on, which is
#: the one this page launches models on.
_SERVE_PORT: dict = {"n": 8080, "ui": 8899}


_MM: dict = {"doc": None, "at": 0.0}

#: The other machines this page knows (cluster/peers.py); None until the
#: page starts, so importing this module starts nothing.
PEERS: Peers | None = None


#: one per process start
_BOOT_ID = uuid.uuid4().hex


def _status_fn(_n=0):
    """A status for a box that is serving nothing.

    The node, its machine and its memory map are all still real -- that is
    the whole content of this mode. `artifact` is simply absent, and the
    page already handles that: it is the same shape `serve` emits.

    A peer's own memory map arrives only if a knurlogic there answers on the
    network.
    """
    # Reused for a few seconds, as `serve` does: the map runs `top`, which
    # takes over a second on a loaded box, and the page and every peer
    # asking for it would otherwise each pay that.
    now = time.time()
    if _MM["doc"] is None or now - _MM["at"] > 4.0:
        try:
            _MM["doc"] = loaded.memory_map()
        except (OSError, subprocess.SubprocessError, ValueError, KeyError,
                AttributeError):
            _MM["doc"] = None
        _MM["at"] = now
    mm = _MM["doc"]

    me = identity.identity()
    # This machine, always, under its own name.
    snaps = [status.snapshot(node=me["name"], role="local", memory_map=mm)]

    # Then every peer: a node that answers for itself is the best witness
    # of itself.
    peers = PEERS.all() if PEERS else []
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
    # what a coordinator page needs of this machine to place a rank on it
    try:
        from knurlogic.cluster import launch
        snaps[0]["cluster"] = launch.node_info(
            (snaps[0].get("memory") or {}).get("working_set_bytes") or 0)
    except (OSError, ValueError, AttributeError, KeyError, TypeError) as e:
        snaps[0]["cluster"] = {"error": f"{type(e).__name__}: {e}"}
    snap = status.aggregate(snaps)
    snap["wired"] = wired.advise(0)
    # liveness rides this GET: which process answered, and the control-plane
    # protocol it speaks (a peer that changed boot_id lost its ranks)
    from knurlogic.cluster import protocol
    snap["boot_id"] = _BOOT_ID
    snap["v"] = list(protocol.VERSION)
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


def _status_light(_n=0):
    """The liveness document: what every peer asks of this page every
    couple of seconds (/status.json?light=1). This machine's own node
    entry -- its cluster block and the memory map as last measured, never a
    new one unless that is over LIGHT_MAP_S old -- the peers as this page
    sees them, `boot_id` and protocol `v`. No aggregate, no wired advice,
    no Bonjour state: a browser's full /status.json has those. The node
    entry is built at most once per LIGHT_TTL_S however many peers ask (it
    runs a few `sysctl`s), by one request at a time: the others are
    answered with the last one, so N peers cost one build, never N at
    once."""
    now = time.time()
    if _LIGHT["doc"] is None or now - _LIGHT["at"] > LIGHT_TTL_S:
        # the first build is waited for; a refresh is done by whoever
        # gets there first, the rest are served the last entry
        if _LIGHT["doc"] is None:
            with _LIGHT_LOCK:
                _build_light(now)
        elif _LIGHT_LOCK.acquire(blocking=False):
            # a refresh runs off the request: the memory map walks every
            # process (seconds while a rank maps tens of GiB), and the
            # peers ask with a 1.5 s timeout -- the liveness answer must
            # never wait on it
            def refresh():
                try:
                    _build_light(time.time())
                except Exception:
                    logger.debug("light status refresh", exc_info=True)
                finally:
                    _LIGHT_LOCK.release()
            __import__("threading").Thread(
                target=refresh, daemon=True,
                name="knurlogic-light-status").start()
    from knurlogic.cluster import protocol
    return {"schema": status.SCHEMA, "nodes": [_LIGHT["doc"]],
            "boot_id": _BOOT_ID, "v": list(protocol.VERSION),
            "peers": [p.public() for p in (PEERS.all() if PEERS else [])]}


def _build_light(now: float) -> None:
    if _LIGHT["doc"] is not None and now - _LIGHT["at"] <= LIGHT_TTL_S:
        return                      # built while this request waited
    if _MM["doc"] is None or now - _MM["at"] > _map_age_limit(now):
        try:
            _MM["doc"] = loaded.memory_map()
        except (OSError, subprocess.SubprocessError, ValueError, KeyError,
                AttributeError):
            _MM["doc"] = None
        _MM["at"] = time.time()
    me = identity.identity()
    own = status.snapshot(node=me["name"], role="local",
                          memory_map=_MM["doc"])
    try:
        from knurlogic.cluster import launch
        own["cluster"] = launch.node_info(
            (own.get("memory") or {}).get("working_set_bytes") or 0)
    except (OSError, ValueError, AttributeError, KeyError, TypeError) as e:
        own["cluster"] = {"error": f"{type(e).__name__}: {e}"}
    _LIGHT.update(doc=own, at=time.time())


_LIGHT_LOCK = __import__("threading").Lock()
#: the light node entry, and how long one build of it serves
_LIGHT: dict = {"doc": None, "at": 0.0}
LIGHT_TTL_S = 1.5
#: the liveness document reuses the memory map up to this old while the
#: machine is quiet, and only HOT_MAP_S old for HOT_S after anything that
#: moves memory (a load, an unload, a request, a message from a peer) or
#: while a model here is answering one -- a card that trails a load by half
#: a minute reads as the machine not reporting
LIGHT_MAP_S = 30.0
HOT_MAP_S = 3.0
HOT_S = 30.0
_HOT: dict = {"until": 0.0}


def hot() -> None:
    """Memory is about to move here: read it often for a while."""
    _HOT["until"] = time.time() + HOT_S


def _answering() -> bool:
    """A model on this machine has a request in flight or queued, by the
    residency document the page already keeps (never a fresh survey)."""
    doc = documents._LOADED.get("doc") or {}
    for r in doc.get("resident") or []:
        q = r.get("requests") or {}
        if q.get("in_flight") or q.get("pending"):
            return True
    return False


def _map_age_limit(now: float) -> float:
    return HOT_MAP_S if now < _HOT["until"] or _answering() else LIGHT_MAP_S

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
    except (OSError, ValueError, AttributeError):
        return 0


def _answers(port: int) -> bool:
    import urllib.request
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1.5)
        return True
    except (OSError, http.client.HTTPException):
        return False


def children() -> list:
    """Every server knurlogic started, from any session, with its PHASE.

    `alive` alone was the trap: loading, serving and hung all read `alive:
    true`, and an agent that cannot tell them apart either guesses or waits
    forever. So each server says:

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
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
        pids = {}
    out = []
    for port, rec in sorted(registry().items()):
        pid = int(rec["pid"])
        mine = _CHILDREN.get(port)
        code = mine[0].poll() if mine and mine[0].pid == pid else None
        alive = code is None and is_our_server(pid)
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
        row = {"port": port, "artifact": rec.get("artifact"), "pid": pid,
               "log": str(log), "started": rec.get("started"),
               "seconds_since_start": (round(now - rec["t"]) if rec.get("t")
                                       else None)}
        if alive and _answers(port):
            held = pids.get(pid, 0)
            # the size the launch measured: a poll reads no model folder
            size = int(rec.get("bytes") or 0)
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
                row["advice"] = (f"no log output for {round(quiet or 0)}s while "
                                 f"loading. Stop waiting; read {log}. "
                                 f"unload(port={port}) if it is hung.")
        else:
            row["phase"] = "exited"
            if code is not None:
                row["exit_code"] = code
            row["log_tail"] = lines[-15:]
            # a server that refused to start says why in its log: carried
            # up as `refused`, the page's and state()'s reason
            from knurlogic.cluster.recovery import refusal_line
            why = refusal_line("\n".join(lines[-60:]))
            if why:
                row["refused"] = why
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


def _spawn_unlocked(path: str, port: int, tune: str = "default",
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
    from knurlogic.machine.servers import new_instance
    instance = new_instance()
    try:
        with open(log, "w") as fh:
            # Its own session, so it is not taken down with the terminal or
            # the MCP client that asked for it.
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                                    start_new_session=True)
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        return {"error": f"{type(e).__name__}: {e}"}
    _CHILDREN[port] = (proc, path)
    reg = registry()
    reg[port] = {"pid": proc.pid, "artifact": path, "log": str(log),
                 # measured once here (a launch is the user acting); the
                 # polls read it from this record, never the model folder
                 "bytes": _artifact_bytes(path),
                 "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "t": time.time(), "instance": instance}
    save_registry(reg)
    return {"starting": path, "port": port, "pid": proc.pid,
            "log": str(log), "instance": instance,
            "note": "the model is loading in its own process; poll `state` "
                    "(started_here) or GET /v1/models on the port"}


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


def _stop(port: int) -> dict:
    import os
    import signal

    from knurlogic.cluster import recovery
    recovery.cancel_port(port)          # asked for: never recovered
    reg = registry()
    rec = reg.get(port)
    if not rec:
        return {"error": f"knurlogic has no record of a server on port "
                         f"{port}; it stops only what it started"}
    if rec.get("job"):
        # rank 0 of a cluster job: the job stops, on every machine
        from knurlogic.cluster import launch
        out = launch.stop(rec["job"], reason="unloaded")
        return {**out, "stopped": rec.get("artifact"), "port": port}
    pid = int(rec["pid"])
    if not is_our_server(pid):
        reg.pop(port, None)
        save_registry(reg)
        _CHILDREN.pop(port, None)
        return {"error": f"the server on port {port} (pid {pid}) is already "
                         f"gone; record cleared", "log": rec.get("log")}
    os.kill(pid, signal.SIGTERM)
    # it saves its prompt cache to disk on the way out (seconds for tens of
    # GiB): killed sooner, what it had not written is lost
    from knurlogic.interfaces.http import STOP_SAVE_S
    end = time.monotonic() + STOP_SAVE_S + 5.0
    while time.monotonic() < end:
        time.sleep(0.25)
        if not is_our_server(pid):
            break
    else:
        os.kill(pid, signal.SIGKILL)
    mine = _CHILDREN.get(port)
    # answer when the process is gone and its memory is back, not when the
    # signal was sent: a caller that loads next must see the memory free
    gone = _wait_exit(pid, mine[0] if mine else None)
    if not gone:
        return {"stopped": rec.get("artifact"), "port": port, "pid": pid,
                "exiting": [pid],
                "note": f"pid {pid} is still exiting after "
                        f"{EXIT_WAIT_S:.0f} s; its memory is not free yet"}
    _CHILDREN.pop(port, None)
    reg.pop(port, None)
    save_registry(reg)
    return {"stopped": rec.get("artifact"), "port": port, "pid": pid}


#: how long an unload waits for the server process to be gone
EXIT_WAIT_S = 60.0


def _wait_exit(pid: int, proc=None, wait_s: float | None = None) -> bool:
    """True once `pid` has exited (reaped when `proc` is ours), polling up
    to `wait_s` (EXIT_WAIT_S)."""
    end = time.time() + (EXIT_WAIT_S if wait_s is None else wait_s)
    while True:
        if proc is not None:
            if proc.poll() is not None:
                return True
        elif not is_our_server(pid):
            return True
        if time.time() >= end:
            return False
        time.sleep(0.1)


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
                        return _stop(port)
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
    strategy (machine/strategy.py), default unless one was chosen."""
    from knurlogic.machine import strategy
    return strategy.get()


def _peer_by_id(node: str):
    for p in (PEERS.all() if PEERS else []):
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
    snap, _ = _status_fn()
    own: dict = next((n for n in snap.get("nodes") or []
                if n.get("role") in ("local", "server")), {})
    return launch.launch(
        dict(req, link=link), me=identity.identity(),
        peers=PEERS.all() if PEERS else [],
        local_info=own.get("cluster") or launch.node_info(),
        ui_port=_SERVE_PORT["ui"], serve_port=serve_port)


def clean(doc):
    from knurlogic.cluster.peers import clean as _clean
    return _clean(doc)


_ADDRS: dict = {}


def _addresses_of(host: str, ttl: float = 60.0) -> set:
    """The addresses a --peer NAME resolves to (cached a minute): the
    operator named that machine, so its address is trusted as the name
    is. An IP literal resolves to itself."""
    now = time.time()
    hit = _ADDRS.get(host)
    if hit and now - hit[0] < ttl:
        return hit[1]
    import socket
    try:
        got = {a[4][0].split("%")[0] for a in socket.getaddrinfo(
            host, None, proto=socket.IPPROTO_TCP)}
    except (OSError, UnicodeError):
        got = set()
    _ADDRS[host] = (now, got)
    return got


def _manual_hosts() -> list:
    """Every address of every --peer machine: the probe moves p.host to the
    fastest answering address, the machine may call from any other."""
    out: list = []
    for p in (PEERS.all() if PEERS else []):
        if "manual" in p.found_by:
            ks = getattr(p, "addresses", ())
            for h in [p.host, *(k.rpartition(":")[0] for k in ks)]:
                if h and h not in out:
                    out.append(h)
    return out


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
    if not (g.allows(local_ip) or ip in set(manual_hosts)
            or any(ip in _addresses_of(h) for h in manual_hosts)):
        return 403, {"error": f"{what} are taken over Thunderbolt or "
                              f"loopback, or from a peer named with --peer; "
                              f"this came from {ip}"}
    return None


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
        return 200, (stop or _stop)(port)
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
#: each peer's last good survey, {address: (time, entry)}
_PEER_LAST: dict = {}
#: the cluster jobs peers last reported, {job: doc} (running, and the ones
#: that ended lately with why): what a dropped connection is explained by
_PEER_JOBS: dict = {}
#: the peer page's relay prefix: /peer/v1/... reaches the model servers
#: that page itself started, by model name (peer_relay)
PEER_RELAY = "/peer"
#: the ONE route between pages: a POST of a protocol envelope
#: (cluster/transport.py); /peer/v1/ is otherwise the model relay above
MSG_PATH = "/peer/v1/msg"


def upstream(base: str, path: str) -> str:
    """The URL a request for `path` on the model at `base` goes to: the
    server itself when it is this machine's, the peer page's relay when it
    is a peer's."""
    t = _PEER_TARGETS.get(base)
    if t:
        return t["relay"] + PEER_RELAY + path
    return base + path


#: a model server's settings document, readable (/peek: a Read message) and
#: changeable (/apply: a Settings message) on a peer through that peer page
PEER_SETTINGS = "/settings.json"


def peer_settings(method: str, port, data, call=None) -> tuple:
    """A Settings (POST, `data` the knobs) or a Read with a port (GET,
    `data` the query), on the machine running the model, for a peer page's
    Settings (the caller has passed the peer gate): (status, doc). Only a
    server this machine started (the registry, still ours) -- never
    another port -- and only its /settings.json: GET reads it (tune and
    working_set_gib passed on), POST is its live apply, which refuses and
    reports per knob exactly as it does for this machine's own page. It is
    how a peer's live knob changes the peer, not this page."""
    import urllib.error
    import urllib.parse
    import urllib.request

    from knurlogic.machine.servers import is_our_server
    if not isinstance(port, int) or isinstance(port, bool):
        return 400, {"error": "name the model server by its port"}
    rec = registry().get(port)
    if not rec or not is_our_server(int(rec["pid"])):
        return 404, {"error": f"no model server this machine started on "
                              f"port {port}"}
    url = f"http://127.0.0.1:{port}{PEER_SETTINGS}"
    if method == "GET":
        q = data if isinstance(data, dict) else {}
        fwd = {k: str(q[k]) for k in PEEK_KEYS if q.get(k) is not None}
        if fwd:
            url += "?" + urllib.parse.urlencode(fwd)
        data = None
    else:
        if not isinstance(data, dict):
            return 400, {"error": "the body must be a JSON object of knobs"}
        data = json.dumps(data).encode()
    if call is None:
        def call(u, d, t):
            req = urllib.request.Request(
                u, data=d, method="POST" if d is not None else "GET",
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=t) as r:
                    return r.status, r.read()
            except urllib.error.HTTPError as e:
                return e.code, e.read()
    try:
        code, raw = call(url, data, APPLY_S)
        return code, json.loads(raw)
    except (OSError, ValueError, http.client.HTTPException) as e:
        return 502, {"error": f"{type(e).__name__}: {e}"}


# --- a MACHINE's own settings, set from any page ---------------------------
# The allowance and the strategy are each machine's own, kept in its own
# ~/.config. A page changes a peer's by asking that peer's page (the peer
# gate, a MachineSet message), which applies it to itself -- never by writing
# anything of the peer's here. The wired limit is not one of them: it is a
# `sudo sysctl` knurlogic never runs; a peer's command is read through /peek
# (wired_gib) and shown to run on that machine.

MACHINE_MAX = 1 << 10


def peer_machine(want) -> tuple:
    """A MachineSet message on the machine being changed (the caller has
    passed the peer gate): {"allowance_gib": N}, {"strategy": name} and/or
    {"settings": {...}} (the knurlogic-wide ones), applied here exactly as
    this machine's own page applies them. -> (status, doc): the machine's
    allowance and strategy after, with what applied, or the first
    refusal."""
    from knurlogic.interfaces.page import documents
    if isinstance(want, dict):
        want = {k: v for k, v in want.items() if v is not None}
    if not isinstance(want, dict) or not want or set(want) - {
            "allowance_gib", "strategy", "settings"}:
        return 400, {"error": "send {\"allowance_gib\": N}, "
                              "{\"strategy\": name} and/or "
                              "{\"settings\": {name: value}}"}
    applied = {}
    if "allowance_gib" in want:
        out = documents.set_allowance(json.dumps({"gib": want["allowance_gib"]}))
        if "error" in out:
            return 400, out
        applied.update(out["applied"])
    if "strategy" in want:
        out = documents.set_strategy(json.dumps({"preset": want["strategy"]}))
        if "error" in out:
            return 400, out
        applied.update(out["applied"])
    if "settings" in want:
        out = documents.set_knurlogic(json.dumps(want["settings"]))
        if "error" in out:
            return 400, out
        applied.update(out["applied"])
    return 200, {"allowance": documents.allowance_doc(),
                 "strategy": documents.strategy_doc(),
                 "knurlogic": documents.knurlogic_doc(), "applied": applied,
                 "machine": identity.identity().get("name") or ""}


def peer_pages() -> set:
    """The pages of peers answering this one, as http://host:port -- the
    only places a machine setting is sent, never an address from the
    request."""
    return {f"http://{p.key}" for p in (PEERS.all() if PEERS else [])
            if p.state == "answering"}


def machine_apply(where: str, body: bytes, post=None) -> tuple:
    """POST /machine.json?where=<peer page>: a machine setting for THAT
    machine, sent as a MachineSet message; where '' is this machine,
    applied here. -> (status, doc) as the machine answered."""
    base = (where or "").rstrip("/")
    if len(body or b"") > MACHINE_MAX:
        return 413, {"error": "a machine setting is small"}
    if not base:
        try:
            return peer_machine(json.loads(body or b""))
        except ValueError:
            return 400, {"error": "the body must be a JSON object"}
    if base not in peer_pages():
        return 403, {"error": f"not a machine answering this page: {base}"}
    try:
        want = json.loads(body or b"")
    except ValueError:
        want = None
    if not isinstance(want, dict):
        return 400, {"error": "the body must be a JSON object"}
    from knurlogic.cluster import transport
    post = post or transport.send
    try:
        out = post(base.removeprefix("http://"), "MachineSet", want)
    except (OSError, ValueError, http.client.HTTPException) as e:
        return 502, {"error": f"{base} did not take it: "
                              f"{type(e).__name__}: {e}"}
    if not isinstance(out, dict):
        return 502, {"error": f"{base} answered without a JSON object"}
    return (400 if out.get("error") else 200), out


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
    a Survey message (what its own /loaded.json says, never ?peers=1), so
    two pages asking each other cannot recurse. Each row is labelled with
    its machine and its address rewritten from the peer's loopback to the
    peer's address."""
    import threading

    from knurlogic.cluster import transport
    if fetch is None:
        def fetch(page, t):
            return transport.send(page, "Survey", {}, timeout=t)
    todo = [p for p in (peers.all() if peers else [])
            if p.state == "answering"]
    out: dict = {}

    def one(p):
        try:
            doc = fetch(p.key, timeout)
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
                          "jobs": js, "loads": doc.get("loads") or []}
        # peer survey: one peer's failure is its row's error; the others are still
        # listed
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
        got = out.get(p.key)
        if got and "error" not in got:
            _PEER_LAST[p.key] = (time.time(), got)
        was = _PEER_LAST.get(p.key)
        if got is None or ("error" in got and was):
            # Nothing came back in OUR deadline, or the Survey itself timed
            # out on a peer busy loading (a 101 GiB pipeline load did): its
            # status probe still answers, so it is a slow peer, not a gone
            # one. Show what it last said, labelled with its age -- dropping
            # it made a cluster load's % fall to this machine's share alone
            # and jump back (54 -> 38 -> 54). "no reply yet" when it never did.
            got = (dict(was[1], heard_ago=round(time.time() - was[0]))
                   if was else {
                       "machine": p.name or p.host, "address": p.key,
                       "resident": [], "late": True})
        res.append(got)
    targets, pjobs = {}, {}
    for m in res:
        for r in m["resident"]:
            if r.get("runtime") == "knurlogic" and r.get("where"):
                c = r.get("cluster") if isinstance(r.get("cluster"),
                                                   dict) else {}
                targets[r["where"].rstrip("/")] = {
                    "machine": m["machine"],
                    "relay": f"http://{m['address']}",
                    "job": str(c.get("job") or "")}
        for j in m.get("jobs") or []:
            if j.get("job"):
                pjobs[str(j["job"])] = j
    _PEER_TARGETS.clear()
    _PEER_TARGETS.update(targets)
    # a job that ended stays explainable after its page stops listing it
    _PEER_JOBS.update(pjobs)
    _PEER_AT[0] = time.time()
    return res


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
             # the rank writes it to its marker (cluster/jobs.progress)
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
                          and quiet > STALL_QUIET_S else "loading")
        elif e["total_bytes"] and e["bytes"] < WARM_FRACTION * e["total_bytes"]:
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
        except Exception:  # peer survey (logged)
            logger.debug("peer survey failed", exc_info=True)


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
    from knurlogic.interfaces.http import request_id as RID
    out = json.dumps(doc).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(out)))
    rid = RID.of(getattr(handler, "headers", None))
    if rid:
        handler.send_header(RID.HEADER, rid)
    try:
        handler.end_headers()
        handler.wfile.write(out)
    except (BrokenPipeError, ConnectionResetError):
        # the asker gave up (its timeout) before the answer: nothing to do
        handler.close_connection = True


def cluster_failure(base: str) -> str:
    """Why the model at `base` is a cluster job that cannot answer, or ""
    when it is not one. A peer's job is looked up in a fresh survey (its
    page reports the jobs that ended, with why); this machine's by port."""
    base = (base or "").rstrip("/")
    t = _PEER_TARGETS.get(base)
    if t is not None:
        job = t.get("job")
        if not job:
            return ""
        refresh_targets()
        e = _PEER_JOBS.get(job) or {}
        if e.get("phase") == "stopped" and e.get("reason"):
            return e["reason"]
        return (f"rank 0 of cluster job {job} on {t.get('machine')} "
                f"dropped the connection; the job is failing")
    try:
        from knurlogic.cluster import launch
        return launch.failure_of_port(urlparse(base).port)
    except (OSError, ValueError, TypeError, AttributeError):
        return ""


def cluster_failed(reason: str) -> dict:
    """The OpenAI-style error a rank 0 answers with when its ring fails
    (http/openai.py _status_of), for a job that could not answer at all."""
    return {"error": {"message": f"this model is split across machines and "
                                 f"its cluster job failed: {reason}",
                      "type": "server_error", "param": None,
                      "code": "cluster_failed"}}


def _stream(handler, url: str, body: bytes, timeout: float = 3600,
            base: str = "") -> None:
    """POST `body` to `url` and pass the answer back as it arrives, byte for
    byte: SSE, prefill keepalives and all. The upstream's status and
    Content-Type go with it; an upstream that cannot be reached is a 502 --
    or a 503 `cluster_failed` with the job's stop reason when `base` is a
    cluster job's rank 0 (dead, or its job stopped). When such a rank 0
    dies mid-way through an event stream, one `data: {"error": ...
    cluster_failed}` event goes out before the stream closes."""
    import urllib.error
    import urllib.request

    from knurlogic.interfaces.http import request_id as RID
    rid = RID.of(getattr(handler, "headers", None))
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json",
                                          **({RID.HEADER: rid} if rid else {}),
                                          **_client_headers(handler)},
                                 method="POST")
    try:
        up = urllib.request.urlopen(req, timeout=timeout)
        code, ctype = up.status, up.headers.get("Content-Type",
                                                "application/json")
    except urllib.error.HTTPError as e:
        up, code = e, e.code
        ctype = e.headers.get("Content-Type", "application/json")
    except (OSError, ValueError, http.client.HTTPException) as e:
        why = cluster_failure(base) if base else ""
        if why:
            _send_json(handler, 503, cluster_failed(why))
        else:
            _send_json(handler, 502, {"error": f"{type(e).__name__}: {e}"})
        return
    handler.send_response(code)
    handler.send_header("Content-Type", ctype)
    # the model server's echo, else the client's own id (an upstream that
    # predates the echo); either way the client gets its id back
    echo = RID.valid(up.headers.get(RID.HEADER)) or rid
    if echo:
        handler.send_header(RID.HEADER, echo)
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.close_connection = True
    sse = "text/event-stream" in (ctype or "")
    tail, cut, told = b"", False, False
    try:
        while True:
            try:
                chunk = (up.read1(8192) if hasattr(up, "read1")
                         else up.read(8192))
            except (OSError, http.client.HTTPException):
                cut = True          # the upstream died mid-answer
                break
            if not chunk:
                break
            told = told or b'"cluster_failed"' in tail + chunk
            tail = (tail + chunk)[-64:]
            handler.wfile.write(chunk)
            handler.wfile.flush()
        # An event stream that ends without its [DONE] was cut. From a
        # cluster job's rank 0 the client is told why, as one last event,
        # instead of a stream that simply stops -- unless the upstream (a
        # peer page's relay) already said so.
        if sse and base and code == 200 and not told and (
                cut or b"[DONE]" not in tail):
            why = cluster_failure(base)
            if why:
                handler.wfile.write(b"data: " + json.dumps(
                    cluster_failed(why)).encode() + b"\n\n")
                handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
        pass            # the client stopped listening; nothing to answer
    finally:
        up.close()


#: the client's labels (telemetry.md) and its cache retention: passed up
#: to the model server, which owns the prompt-cache entries by them
CLIENT_HEADERS = ("X-Client", "X-Client-Session", "X-Client-Run",
                  "X-Client-Role", "X-Cache-Retain", "X-Cache-Keep")


def _client_headers(handler) -> dict:
    h = getattr(handler, "headers", None)
    if h is None:
        return {}
    return {k: h.get(k) for k in CLIENT_HEADERS
            if isinstance(h.get(k), str) and h.get(k)}


#: the model server's prompt-cache endpoints the page forwards
PROMPT_CACHE_PATH = "/v1/prompt-cache"
PROMPT_CACHE_POSTS = tuple(PROMPT_CACHE_PATH + p
                           for p in ("/save", "/drop", "/pin", "/park"))


def prompt_cache_forward(handler, method: str, path: str, query: dict,
                         body: bytes, fetch=None, send=None) -> None:
    """GET /v1/prompt-cache, POST /v1/prompt-cache/{save,drop,pin} on the
    page: forwarded to the model server on this machine that serves the
    model named by a "model" query or body field; with none named, the only
    one; several and none named is a 400 listing them. The loopback
    operator only (the model server refuses anyone else; the page forwards
    from loopback, so it must refuse first). `send`: (url, method, body)
    -> (status, doc), for tests."""
    import ipaddress
    try:
        loop = ipaddress.ip_address(
            handler.client_address[0].split("%")[0]).is_loopback
    except (ValueError, AttributeError, IndexError):
        loop = False
    if not loop:
        _send_json(handler, 403, {"error": {
            "message": "the prompt cache is managed from this machine "
                       "(loopback) only", "type": "permission_error"}})
        return
    model = (query.get("model") or [None])[0]
    if model is None and body:
        try:
            got = json.loads(body)
            model = got.get("model") if isinstance(got, dict) else None
        except ValueError:
            model = None
    table = local_models(fetch)
    if model is not None:
        base = _resolve(table, model)
        if base is None:
            # a peer's model: its page's relay, like a chat (the peer
            # resolves the name again against the servers it started)
            far = _resolve(routable(fetch), model)
            if far is not None and _PEER_TARGETS.get(far) is not None:
                q = f"?model={quote(str(model))}" \
                    if method == "GET" else ""
                code, doc = (send or _send_up)(
                    upstream(far, path) + q, method,
                    body if method == "POST" else None)
                _send_json(handler, code, doc)
                return
            here = sorted(set(table) | set(routable(fetch)))
            _send_json(handler, 404, {"error": {
                "message": f"no running model {model!r} here or on a peer; "
                           f"running: {', '.join(here) or 'none'}",
                "type": "not_found"}, "models": here})
            return
    else:
        bases = set(table.values())
        if len(bases) != 1:
            _send_json(handler, 400, {"error": {
                "message": "name the model (a \"model\" query or body "
                           "field): " + (", ".join(sorted(table))
                                         or "none is running"),
                "type": "invalid_request_error"}, "models": sorted(table)})
            return
        base = bases.pop()
    code, doc = (send or _send_up)(base + path, method,
                                   body if method == "POST" else None)
    _send_json(handler, code, doc)


def _send_up(url: str, method: str, body: bytes | None):
    """(status, JSON doc) of one request to a model server."""
    import urllib.error
    import urllib.request
    req = urllib.request.Request(
        url, data=body if body is not None else None, method=method,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except ValueError:
            return e.code, {"error": {"message": str(e)}}
    except (OSError, ValueError) as e:
        return 502, {"error": {"message": f"{type(e).__name__}: {e}"}}


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
    _stream(handler, upstream(base, "/v1/chat/completions"), body,
            base=base)


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
    target = _PEER_TARGETS.get(base)
    if target:
        # a peer's model through that peer's page (a Settings message): its
        # server listens on the peer's loopback, not at `base`
        from knurlogic.cluster import transport
        try:
            out = (post or transport.send)(
                target["relay"].removeprefix("http://"), "Settings",
                {"port": urlparse(base).port, "values": want})
        except (OSError, ValueError, http.client.HTTPException) as e:
            return 502, {"error": f"{type(e).__name__}: {e}"}
        return (502 if out.get("error") and "applied" not in out
                else 200), out
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
        code, raw = post(base + PEER_SETTINGS,
                         json.dumps(want).encode(), APPLY_S)
        return code, json.loads(raw)      # JSON only, never an HTML page
    except (OSError, ValueError, http.client.HTTPException) as e:
        return 502, {"error": f"{type(e).__name__}: {e}"}


#: The paths the page's router forwards by `model`: the two chat surfaces
#: a client pointed at this page (Claude Code, an OpenAI SDK) uses, and
#: Claude Code's token count (it names the model too).
ROUTE_PATHS = ("/v1/messages", "/v1/chat/completions",
               "/v1/messages/count_tokens")
ROUTE_S = 2.0
_ROUTES: dict = {"at": 0.0, "map": {}, "docs": {}}
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
            except Exception:  # peer survey (logged)
                logger.debug("peer survey failed", exc_info=True)
    found: dict = {}
    docs: dict = {}

    def one(base):
        try:
            for m in fetch(upstream(base, "/v1/models"),
                           ROUTE_S).get("data") or []:
                if isinstance(m, dict) and m.get("id"):
                    found.setdefault(str(m["id"]), base)
                    docs.setdefault(str(m["id"]), m)
        # peer survey thread: one peer's silence is logged, the others are still asked
        except Exception:
            logger.debug("no model list from %s", base, exc_info=True)
    ts = [threading.Thread(target=one, args=(b,), daemon=True)
          for b in sorted(chat_targets())]
    for t in ts:
        t.start()
    end = time.time() + ROUTE_S
    for t in ts:
        t.join(max(end - time.time(), 0))
    out = dict(found)
    _ROUTES.update(at=now, map=out, docs=docs)
    return out


def route_models_document(fetch=None) -> dict:
    """GET /v1/models on the page: every model its router can reach, each
    with its server's own entry (sampling_defaults, thinking,
    context_length: what the page's chat reads), and the machine that
    answers for it (a split model's rank 0)."""
    from knurlogic.machine.identity import identity
    table = routable(fetch)
    docs = _ROUTES.get("docs") or {}
    here = identity().get("name")

    def machine(b):
        t = _PEER_TARGETS.get(b.rstrip("/"))
        return (t or {}).get("machine") if t is not None else here
    return {"object": "list", "data": [
        dict(docs.get(m) or {}, id=m, object="model", owned_by="knurlogic",
             server=b, machine=machine(b))
        for m, b in sorted(table.items())]}


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
    except (ValueError, AttributeError):
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
    _stream(handler, upstream(base, path), body, base=base)


def local_models(fetch=None, docs=None) -> dict:
    """{model id: base} over the servers THIS machine started (never the
    ones peers reported, so two pages relaying for each other cannot
    loop). What a peer page's relay resolves a model name against.
    `docs`, when given, is filled with each model's own /v1/models entry
    (sampling_defaults, context_length) for the relay's GET."""
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
                    if docs is not None:
                        docs.setdefault(str(m["id"]), m)
        # peer survey thread: one server's silence is logged, the others are still asked
        except Exception:
            logger.debug("no model list from %s", base, exc_info=True)
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
    if path == PROMPT_CACHE_PATH or path.startswith(PROMPT_CACHE_PATH + "/"):
        # a peer page managing the prompt cache of a model this machine
        # serves: resolved here by name, sent on to it from loopback
        if not (method == "GET" and path == PROMPT_CACHE_PATH or
                method == "POST" and path in PROMPT_CACHE_POSTS):
            _send_json(handler, 404, {"error": "not a relayed path"})
            return
        q = parse_qs(urlparse(getattr(handler, "path", "")).query)
        model = (q.get("model") or [None])[0]
        if model is None and body:
            try:
                got = json.loads(body)
                model = got.get("model") if isinstance(got, dict) else None
            except ValueError:
                model = None
        table = local_models(fetch)
        base = _resolve(table, model)
        if base is None:
            _send_json(handler, 404, {"error": {
                "message": f"no running model {model!r} on "
                           f"{identity.identity().get('name') or 'this machine'}",
                "type": "not_found"}, "models": sorted(table)})
            return
        code, doc = _send_up(base + path, method,
                             body if method == "POST" else None)
        _send_json(handler, code, doc)
        return
    docs: dict = {}
    table = local_models(fetch, docs)
    if method == "GET":
        if path != "/v1/models":
            _send_json(handler, 404, {"error": "not a relayed path"})
            return
        # each server's own entry, so a peer page's chat sees the model's
        # sampling_defaults and context_length, not just its name
        _send_json(handler, 200, {"object": "list", "data": [
            dict(docs.get(m) or {}, id=m, object="model",
                 owned_by="knurlogic")
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
    _stream(handler, base + path, body, base=base)


#: What `/peek` may read, and the query keys it passes along. Reads only:
#: a running model's settings and sampling defaults, a peer page's machine
#: settings. Nothing that changes anything is reachable through it.
PEEK_PATHS = ("/settings.json", "/v1/models", "/models.json")
PEEK_KEYS = ("tune", "working_set_gib", "wired_gib")
#: keys passed along for one path only: `rescan=1` reads a peer's model
#: folders again -- its picker was opened, never a poll
PEEK_PATH_KEYS = {"/models.json": ("rescan",),
                  # a peer's preview of a model it holds, named by identity
                  # (resolved there, from its own stores): its room is
                  # counted against ITS memory free now
                  "/settings.json": ("identity", "name", "kv_bits",
                                     "long_context")}


def _peek_keys(path: str) -> tuple:
    return PEEK_KEYS + PEEK_PATH_KEYS.get(path, ())
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


def _peek_peer(where: str, path: str, fwd: dict):
    """A read of a peer's document, sent as a Read message: a peer PAGE's
    own documents, or -- with its port -- one of its model servers'
    /settings.json. None when `where` is not a peer's (a model's /v1/models
    goes through the peer's relay, a local one direct)."""
    from knurlogic.cluster import transport
    t = _PEER_TARGETS.get(where)
    if t and path == PEER_SETTINGS:
        page, port = t["relay"].removeprefix("http://"), urlparse(where).port
    elif where in {f"http://{p.key}" for p in (PEERS.all() if PEERS else [])}:
        page, port = where.removeprefix("http://"), None
    else:
        return None
    try:
        out = transport.send(page, "Read", {"path": path, "query": fwd,
                                            **({"port": port} if port
                                               else {})})
    except (OSError, ValueError, http.client.HTTPException) as e:
        return 502, json.dumps({"error": f"{type(e).__name__}: {e}"})
    return 200, json.dumps(out)


def peek(q: dict, fetch=None) -> tuple:
    """GET /peek?where=<base>&path=<path>: another server's read-only
    document, for a page that cannot call another port or machine itself
    (a peer's, as a Read message).

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
    fwd = {k: q[k][0] for k in _peek_keys(path) if q.get(k)}
    if fetch is None:
        peer = _peek_peer(where, path, fwd)
        if peer is not None:
            return peer
    url = upstream(where, path)
    if fwd:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(fwd)
    if fetch is None:
        def fetch(u, t):
            with urllib.request.urlopen(u, timeout=t) as r:
                return r.read()
    try:
        body = fetch(url, PEEK_S)
        json.loads(body)          # pass on JSON only, never an HTML page
        return 200, body.decode() if isinstance(body, bytes) else body
    except (OSError, ValueError, http.client.HTTPException) as e:
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
                pid = txt.get("id", "") or PEERS.id_of_instance(
                    svc.get("name", ""))
                p = PEERS.add(svc["host"], svc["port"], "bonjour", id=pid)
                p.id = p.id or pid
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
    # Bonjour is optional; the failure is printed and peers can still be named
    except Exception as e:
        print(f"bonjour unavailable ({type(e).__name__}: {e}); peers can "
              f"still be named with --peer", file=sys.stderr)


def survey_here(routes: dict) -> tuple:
    """A Survey: what this page's own /loaded.json says (never with
    ?peers=1: pages asking each other must not recurse)."""
    h = routes.get("/loaded.json")
    if h is None:
        return 404, {"error": "this page lists nothing resident"}
    body, _ctype = h({}, 0)
    doc = json.loads(body)
    if isinstance(doc, dict):
        # fresh memory and ranks still exiting: what a coordinator placing
        # a launch right after an unload needs, not the last status
        from knurlogic.cluster import launch
        doc["available_bytes"] = launch.available_now()
        doc["exiting"] = len(launch._exiting(launch.J.registry()))
    return 200, doc


#: the documents a Read may fetch from a page, by path (the allow-list;
#: anything else is refused): /peek's page-to-peer reads
READ_PAGE_PATHS = ("/settings.json", "/models.json", "/v1/models")


def read_here(routes: dict, req: dict) -> tuple:
    """A Read on this machine: a page document (READ_PAGE_PATHS), or --
    with `port` -- a model server this machine started, through
    peer_settings. Reads only."""
    path = req.get("path")
    q: dict = req["query"] if isinstance(req.get("query"), dict) else {}
    if req.get("port") is not None:
        if path != PEER_SETTINGS:
            return 403, {"error": f"not a readable path: {str(path)[:60]!r}"}
        return peer_settings("GET", req.get("port"), q)
    if path not in READ_PAGE_PATHS or path not in PEEK_PATHS:
        return 403, {"error": f"not a readable path: {str(path)[:60]!r}"}
    if path == "/v1/models":
        return 200, route_models_document()
    h = routes.get(path)
    if h is None:
        return 404, {"error": "no such document here"}
    body, _ctype = h({k: [str(v)] for k, v in q.items()
                      if k in _peek_keys(path)}, 0)
    try:
        return 200, json.loads(body)
    except ValueError:
        return 502, {"error": "that document is not JSON"}


def peer_table(routes: dict) -> dict:
    """Message kind -> handler(body) -> (status, doc): every kind a page
    answers on MSG_PATH. Anything not here (Hello, Heartbeat, ...) is
    refused by the dispatcher as a typed Failure."""
    from knurlogic.cluster import launch

    def changes(fn):
        def run(body):
            out = fn(body)
            documents._LOADED["doc"] = None     # residency may have changed
            return out
        return run
    t: dict = {k: changes(lambda b, k=k: launch.peer_step(k, b))
               for k in launch.CLUSTER_KINDS}
    t["Load"] = changes(lambda b: peer_launch(dict(b, action="load")))
    t["Unload"] = changes(lambda b: peer_launch(dict(b, action="unload")))
    t["MachineSet"] = changes(peer_machine)
    t["Settings"] = lambda b: peer_settings(
        "POST", b.get("port"), b.get("values"))
    t["Survey"] = lambda b: survey_here(routes)
    t["Read"] = lambda b: read_here(routes, b)
    return t


def make_handler(routes: dict, gate=None, allow_origins=(),
                 allow_hosts=(), gate_for_peers=None):
    """The page's request handler: its routes, the router, the proxies, and
    the guards in front of every one of them. `gate_for_peers`: the
    /peer/ gate's link check (default cluster/links.Gate; tests pass
    their own)."""

    table = peer_table(routes)

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            # the page's files and documents are always asked for again:
            # a restart shows new code, never a stale module
            self.send_header("Cache-Control", "no-cache")
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
            if u.path.rstrip("/") == MSG_PATH:
                self._peer_msg("GET")
                return
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
            if u.path.rstrip("/") == PROMPT_CACHE_PATH:
                prompt_cache_forward(self, "GET", PROMPT_CACHE_PATH,
                                     parse_qs(u.query), b"")
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
            hot()               # a load, a chat, a peer's message: memory moves
            u = urlparse(self.path)
            if u.path.rstrip("/") == MSG_PATH:
                self._peer_msg("POST")
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
            if u.path.startswith(PROMPT_CACHE_PATH + "/"):
                prompt_cache_forward(self, "POST", u.path.rstrip("/"),
                                     parse_qs(u.query),
                                     self.rfile.read(n) if n else b"")
                return
            if u.path.rstrip("/") == "/machine.json":
                where = (parse_qs(u.query).get("where") or [""])[0]
                code, doc = machine_apply(where, self.rfile.read(n)
                                          if n else b"")
                self._send(json.dumps(doc).encode(), "application/json",
                           code)
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
            manual = _manual_hosts()
            refused = peer_refusal(
                self.headers, self.client_address[0],
                self.connection.getsockname()[0], manual_hosts=manual,
                what="relayed requests")
            if refused:
                self.close_connection = True
                _send_json(self, *refused)
                return
            if self._refuse_chunked():
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

        def _refuse_chunked(self) -> bool:
            """A peer route's body is a plain Content-Length body -- never
            Transfer-Encoding, the framing a smuggled request hides behind:
            411 and the connection closed. -> True when refused."""
            if not self.headers.get("Transfer-Encoding"):
                return False
            self.close_connection = True
            _send_json(self, 411, {"error": "send the body with a "
                       "Content-Length and no Transfer-Encoding"})
            return True

        def _peer_msg(self, method: str):
            """MSG_PATH, the ONE route between pages: POST only (a GET is a
            405), the peer gate (no Origin; loopback, Thunderbolt or a
            --peer address), a plain Content-Length body of at most
            PEER_MAX -- never Transfer-Encoding -- then the envelope
            (cluster/transport.handle) and the kind's handler."""
            from knurlogic.cluster import launch, transport
            if method != "POST":
                self.close_connection = True
                out = json.dumps({"error": "POST a protocol envelope"}
                                 ).encode()
                self.send_response(405)
                self.send_header("Allow", "POST")
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
                return
            if self._refuse_chunked():
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = -1
            if not 0 <= n <= launch.PEER_MAX:
                self.close_connection = True
                _send_json(self, 413, {"error": "a peer message is small"})
                return
            manual = _manual_hosts()
            refused = peer_refusal(
                self.headers, self.client_address[0],
                self.connection.getsockname()[0], gate=gate_for_peers,
                manual_hosts=manual, what="peer messages")
            body = self.rfile.read(n) if n else b""
            if refused:
                self.close_connection = True
                _send_json(self, *refused)
                return
            code, doc = transport.handle(body, table)
            _send_json(self, code, doc)

    return H


def serve_ui(host: str, port: int, serve_port: int, peers=(),
             allow_origins=(), allow_hosts=(), offline: bool = False,
             menubar: bool = False, open_page: bool = False) -> int:
    global PEERS
    from knurlogic.interfaces.page import updates
    updates.start_for_page(offline)
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
    routes = documents.routes(
        status_fn=_status_fn,
        settings_fn=documents.machine_settings(),
        models_fn=documents.models_document(serving=""),
        loaded_fn=_loaded_fn(),
        load_fn=_load_fn(serve_port))
    # /status.json?light=1: the liveness document every peer polls
    full_status = routes["/status.json"]
    routes["/status.json"] = lambda q, _n=0: (
        documents._json(_status_light()) if (q.get("light") or [""])[0]
        else full_status(q, _n))
    # the knurlogic allowance: THIS machine's only, and set only by a POST
    # the page sends when its user applies it; a peer's is read from that
    # peer's /settings.json through /peek
    routes["/allowance.json"] = lambda _q, _n=0: documents._json(
        documents.allowance_doc())
    routes["POST /allowance.json"] = lambda _q, _n=0, body=None: documents._json(
        documents.set_allowance(body))
    # the knurlogic strategy: this machine's default launch preset
    routes["/strategy.json"] = lambda _q, _n=0: documents._json(
        documents.strategy_doc())
    # a newer knurlogic on PyPI (asked once at page start, page/updates.py)
    routes["/release.json"] = lambda _q, _n=0: documents._json(
        updates.release_doc())
    # the knurlogic-wide settings: compaction, identical results across chips
    routes["/knurlogic.json"] = lambda _q, _n=0: documents._json(
        documents.knurlogic_doc())
    routes["POST /knurlogic.json"] = lambda _q, _n=0, body=None: documents._json(
        documents.set_knurlogic(body))
    routes["POST /strategy.json"] = lambda _q, _n=0, body=None: documents._json(
        documents.set_strategy(body))

    from knurlogic.interfaces.page import hub
    routes["/hub/search.json"] = lambda q, _n=0: documents._json(
        hub.search((q.get("q") or [""])[0]))
    routes["/hub/repo.json"] = lambda q, _n=0: documents._json(
        hub.repo((q.get("id") or [""])[0]))
    routes["/hub/downloads.json"] = lambda _q, _n=0: documents._json(
        hub.downloads())
    routes["POST /hub/download.json"] = lambda _q, _n=0, body=None: (
        documents._json(hub.act(body)))

    H = make_handler(routes, gate, allow_origins, allow_hosts)
    from knurlogic.cluster import launch
    launch.start_watching_existing()

    srv = ThreadingHTTPServer((bind, port), H)
    if gate:
        tb = [i["ip"] for i in links.thunderbolt()]
        urls = ", ".join(f"http://{ip}:{port}" for ip in tb)
        print(f"knurlogic  cluster mode: answering on "
              f"{urls or 'no Thunderbolt link yet'}"
              f" and http://127.0.0.1:{port}; Wi-Fi and Ethernet refused")
    else:
        print(f"knurlogic  http://{host}:{port}")
    print(f"  open http://127.0.0.1:{port}/ in your browser "
          "(--open does it for you).")
    print("  nothing loaded, no model required.")
    print(f"  loading from the page starts `knurlogic serve` on port "
          f"{serve_port}.")
    if open_page:
        _open_when_up(host, port)
    if menubar:
        import signal

        from knurlogic.interfaces import menubar as _mb
        # the menu's Quit sends SIGINT; a page started in the background
        # inherits SIGINT ignored, which would make that Quit do nothing
        signal.signal(signal.SIGINT, signal.default_int_handler)
        _mb.spawn(port)
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
        from knurlogic.cluster import launch
        for job in list(launch.SPECS):
            launch.stop(job, reason="the page that started it closed")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="knurlogic ui",
        description="the page, without loading anything: every model on the "
                    "disk, everything resident in every runtime, and where "
                    "the memory went")
    p.add_argument("--host", default="cluster",
                   help="`cluster` (default): answer on the Thunderbolt "
                        "link(s) and loopback only, advertise there, refuse "
                        "Wi-Fi and Ethernet; a Mac with no Thunderbolt link "
                        "answers on loopback. 127.0.0.1: this machine only. "
                        "Or an address.")
    p.add_argument("--port", type=int, default=8899)
    p.add_argument("--serve-port", type=int, default=8080,
                   help="the port a model loaded from this page is served on")
    p.add_argument("--peer", action="append", default=[],
                   metavar="HOST[:PORT]",
                   help="another machine's knurlogic page (repeatable). "
                        "Naming it on one side lists it (it learns this "
                        "machine from the request), but over Ethernet or "
                        "Wi-Fi each side's gate needs the other named: name "
                        "each machine on the other, or link them with "
                        "Thunderbolt. Remembered once it answers.")
    p.add_argument("--allow-origin", action="append", default=[],
                   metavar="URL", help="a web page origin allowed to call "
                   "this page's API from a browser (repeatable)")
    p.add_argument("--allow-host", action="append", default=[],
                   metavar="NAME", help="a DNS name this machine is reached "
                   "by, beyond localhost, IPs, .local and its hostname")
    p.add_argument("--offline", action="store_true",
                   help="skip the once-per-start check that asks Hugging "
                        "Face whether a downloaded model has an update "
                        "(HF_HUB_OFFLINE=1 does the same)")
    p.add_argument("--no-menubar", action="store_true",
                   help="do not show the macOS menu-bar icon")
    p.add_argument("--open", action="store_true",
                   help="open the page in the default browser once it is "
                        "up (not over SSH)")
    a = p.parse_args(argv)
    from knurlogic.machine import folders
    # once: a folder the old variables name is remembered, so the next
    # start needs no variable
    for f in folders.adopt_from_env():
        print(f"knurlogic  remembered model folder {f} "
              f"(knurlogic models folders)")
    peers = []
    for spec in a.peer:
        host, _, port = spec.rpartition(":") if ":" in spec else (spec, "", "")
        peers.append((host, int(port) if port.isdigit() else a.port))
    return serve_ui(a.host, a.port, a.serve_port, peers,
                    allow_origins=a.allow_origin, allow_hosts=a.allow_host,
                    offline=a.offline, menubar=not a.no_menubar,
                    open_page=a.open)


def _open_when_up(host: str, port: int) -> None:
    """Open the page in the default browser once it answers (`--open`).
    Not over SSH: the browser would open on the remote Mac's screen."""
    import os
    import threading
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
        return
    shown = "127.0.0.1" if host in ("cluster", "0.0.0.0", "::", "") else host
    url = f"http://{shown}:{port}/"

    def go():
        import webbrowser
        try:
            webbrowser.open(url)
        except Exception:  # no browser here: the printed URL still works
            logger.debug("could not open %s", url, exc_info=True)

    # the socket is bound and listening already; serve_forever answers
    # within a moment of this timer
    t = threading.Timer(0.5, go)
    t.daemon = True
    t.start()


def _wire() -> None:
    """Give launch and recovery what they need of this page. Each is
    a late-bound lambda, so a swapped PEERS or mcp.load is what they see."""
    from knurlogic.cluster import launch, recovery
    from knurlogic.interfaces import mcp
    launch.status_fn = lambda: _status_fn()
    launch.peers_fn = lambda: PEERS.all() if PEERS else []
    recovery.peers_fn = lambda: PEERS.all() if PEERS else []
    recovery.child_fn = lambda port: _CHILDREN.get(port)
    recovery.answers_fn = lambda port: _answers(port)
    recovery.load_fn = lambda **kw: mcp.load(**kw)


_wire()
