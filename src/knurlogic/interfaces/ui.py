"""`knurlogic ui` -- the page, with nothing loaded and nothing else running.

THE POINT: knurlogic must not need exo, and it must not need a model already
loaded. It sits ON TOP of whatever is there. Open this and you see every
model on the disk, everything resident in every runtime, and where the memory
went -- on a machine with no exo, no ollama and no weights in RAM, all of
which are ordinary states rather than errors.

`serve` is already exo-free; only `--cluster` reaches for exo. What was
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
from knurlogic.interfaces import cluster  # noqa: E402

#: Children started from the page: {port: (Popen, artifact path)}.
_CHILDREN: dict = {}

#: The port a knurlogic on ANOTHER node is expected to answer on, which is
#: the one this page launches models on.
_SERVE_PORT: dict = {"n": 8080, "ui": 8899}


#: Where to look for exo, only to ask WHO IS THERE. knurlogic does not need
#: exo to run; it needs it to know about the other machines, because exo is
#: the thing that already tracks them.
from knurlogic.machine.exo import EXO_URL  # noqa: E402  one home


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
        exo_nodes = cluster.inventory(EXO_URL)
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
            snaps.append({**p.node, "role": "remote",
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
        snaps.append({**cluster._snapshot_for(
            n, local, {}, None, peer_port=_SERVE_PORT["ui"]),
            "found_by": ["exo"]})
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


def _spawn(path: str, port: int, tune: str = "balanced",
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
        act = req.get("action")
        target, where = req.get("target") or "", req.get("where") or ""
        try:
            # Through the MCP's own functions: the page refuses what an agent
            # is refused -- will not fit, memory still moving -- in the same
            # words. The page loading past a check the MCP enforces would be
            # a capability on one side only.
            if act == "load":
                from knurlogic.interfaces import mcp
                return mcp.load(artifact=target,
                                port=int(req.get("port") or serve_port),
                                tune=req.get("tune") or "balanced",
                                sets=req.get("sets") or {},
                                force=bool(req.get("force")))
            if act == "exo-load":
                from knurlogic.interfaces import mcp
                return mcp.place(model=target, force=bool(req.get("force")))
            if act == "unload":
                # Ours to stop only if we started it. Anything else is
                # somebody's server and not this page's to kill.
                for port, rec in registry().items():
                    if rec.get("artifact") == target or str(port) == str(target):
                        return _stop(port)
                return {"error": "this page did not start that; stop it "
                                 "where it was started"}
            if act == "exo-unload":
                from knurlogic.interfaces import mcp
                return mcp.unplace(instance_id=target)
            if act == "ollama-unload":
                return loaded.ollama_unload(where, target)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
        return {"error": f"unknown action {act!r}"}
    return handler


def chat_targets() -> set:
    """Endpoints the page may send a chat to: exo, and servers knurlogic
    started that are still ours. A fixed allow-list, so the proxy cannot be
    pointed at an arbitrary address by whatever is in the request."""
    from knurlogic.machine.exo import EXO_URL
    from knurlogic.machine.servers import is_our_server
    out = {EXO_URL.rstrip("/")}
    for port, rec in registry().items():
        if is_our_server(int(rec["pid"])):
            out.add(f"http://127.0.0.1:{port}")
    return out


def proxy_chat(handler, where: str, body: bytes) -> None:
    """POST /chat?where=<base>: forward a chat request to a running model
    and stream its answer back as it arrives.

    The control page serves no model, so its chat has to reach the one the
    person clicked -- on exo, or on a server `load` started -- and a browser
    will not let a page on this port call another port directly. Streaming
    is passed through byte for byte: SSE, prefill keepalives and all."""
    import urllib.error
    import urllib.request
    base = (where or "").rstrip("/")
    if base not in chat_targets():
        body_out = json.dumps({"error": f"not a running model this page "
                                        f"knows: {base or '(none)'}"}).encode()
        handler.send_response(403)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body_out)))
        handler.end_headers()
        handler.wfile.write(body_out)
        return
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    try:
        up = urllib.request.urlopen(req, timeout=3600)
        code, ctype = up.status, up.headers.get("Content-Type",
                                                "application/json")
    except urllib.error.HTTPError as e:
        up, code = e, e.code
        ctype = e.headers.get("Content-Type", "application/json")
    except Exception as e:
        msg = json.dumps({"error": f"{type(e).__name__}: {e}"}).encode()
        handler.send_response(502)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(msg)))
        handler.end_headers()
        handler.wfile.write(msg)
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
        pass            # the page stopped listening; nothing to answer
    finally:
        up.close()


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
        if reachable:
            d.if_index = dsd.interface_of(host)
            d.register(f"{me['name']} {me['id'][:6]}", port,
                       {"id": me["id"], "name": me["name"],
                        "ver": __version__, "schema": status.SCHEMA,
                        "role": "ui"})
            d.if_index = 0
        d.browse()
        DISCOVERY = d.start()
    except Exception as e:
        print(f"bonjour unavailable ({type(e).__name__}: {e}); peers can "
              f"still be named with --peer", file=sys.stderr)


def serve_ui(host: str, port: int, serve_port: int, peers=()) -> int:
    global PEERS
    _SERVE_PORT["n"] = serve_port
    _SERVE_PORT["ui"] = port
    from knurlogic.cluster.peers import Peers
    me = identity.identity()
    reachable = host not in ("127.0.0.1", "localhost", "::1")
    PEERS = Peers(me, port, manual=peers, reachable=reachable).start()
    _start_discovery(me, host, port, reachable)
    routes = web.routes(
        status_fn=_status_fn,
        settings_fn=web.machine_settings(),
        models_fn=web.models_document(serving=""),
        loaded_fn=web.loaded_document(),
        load_fn=_load_fn(serve_port))

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

        def do_GET(self):
            u = urlparse(self.path)
            intro = self.headers.get("X-Knurlogic-Peer")
            if intro and PEERS:
                PEERS.introduce(self.client_address[0], intro)
            h = routes.get(u.path.rstrip("/") or "/")
            if h is None:
                self._send(b"not found", "text/plain", 404)
                return
            body, ctype = h(parse_qs(u.query), 0)
            self._send(body, ctype)

        def do_POST(self):
            u = urlparse(self.path)
            if u.path.rstrip("/") == "/chat":
                n = int(self.headers.get("Content-Length") or 0)
                where = (parse_qs(u.query).get("where") or [""])[0]
                proxy_chat(self, where, self.rfile.read(n) if n else b"")
                return
            h = routes.get("POST " + (u.path.rstrip("/") or "/"))
            if h is None:
                self._send(b"not found", "text/plain", 404)
                return
            n = int(self.headers.get("Content-Length") or 0)
            body, ctype = h(parse_qs(u.query), 0, self.rfile.read(n) if n
                            else b"")
            self._send(body, ctype)

    srv = ThreadingHTTPServer((host, port), H)
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
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="knurlogic ui",
        description="the page, without loading anything: every model on the "
                    "disk, everything resident in every runtime, and where "
                    "the memory went")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8899)
    p.add_argument("--serve-port", type=int, default=8080,
                   help="the port a model loaded from this page is served on")
    p.add_argument("--peer", action="append", default=[],
                   metavar="HOST[:PORT]",
                   help="another machine's knurlogic page (repeatable). "
                        "Naming it on one side is enough: it learns this "
                        "machine from the request. Remembered once it "
                        "answers.")
    a = p.parse_args(argv)
    peers = []
    for spec in a.peer:
        host, _, port = spec.rpartition(":") if ":" in spec else (spec, "", "")
        peers.append((host, int(port) if port.isdigit() else a.port))
    return serve_ui(a.host, a.port, a.serve_port, peers)
