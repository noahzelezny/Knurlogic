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

from . import loaded, status, web, wired

#: Children started from the page: {port: (Popen, artifact path)}.
_CHILDREN: dict = {}

#: The port a knurlogic on ANOTHER node is expected to answer on, which is
#: the one this page launches models on.
_SERVE_PORT: dict = {"n": 8080}


#: Where to look for exo, only to ask WHO IS THERE. knurlogic does not need
#: exo to run; it needs it to know about the other machines, because exo is
#: the thing that already tracks them.
EXO_URL = "http://127.0.0.1:52415"


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
    from . import cluster

    mm = None
    try:
        mm = loaded.memory_map()
    except Exception:
        pass

    nodes = []
    try:
        nodes = cluster.inventory(EXO_URL)
    except Exception:
        nodes = []

    if not nodes:
        snaps = [status.snapshot(node="local", role="local", memory_map=mm)]
    else:
        local = _local_name(nodes)
        snaps = []
        for n in nodes:
            is_local = n.name == local
            snaps.append(cluster._snapshot_for(
                n, local, {}, None, peer_port=_SERVE_PORT["n"])
                if not is_local else
                status.snapshot(node=n.name, role="local", memory_map=mm,
                                memory_fn=lambda n=n: _local_memory(n, mm)))
    snap = status.aggregate(snaps)
    snap["wired"] = wired.advise(0)
    return snap, status.render_cluster(snap)


def _spawn(path: str, port: int, tune: str = "balanced",
           sets: dict | None = None) -> dict:
    """Start `knurlogic serve` for one artifact, on its own port.

    Deliberately a child process rather than an in-process load: the
    settings that matter are resolved and put in the ENVIRONMENT before the
    engine imports anything, and that cannot be done to a process that is
    already running. A fresh process is the only way the knobs are honoured
    in full -- which is the difference between this and switching a model
    inside a running server.
    """
    if not Path(path).exists():
        return {"error": f"no such artifact: {path}"}
    live = _CHILDREN.get(port)
    if live and live[0].poll() is None:
        return {"error": f"port {port} is already serving {live[1]}"}
    cmd = [sys.executable, "-m", "knurlogic.cli", "serve", path,
           "--port", str(port), "--tune", tune]
    # Settings chosen at LAUNCH, which for most of these is the only moment
    # they can be chosen: they are read at import and compiled into kernel
    # source, so a running server cannot be told about them.
    for k, v in sorted((sets or {}).items()):
        cmd += ["--set", f"{k}={v}"]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.STDOUT)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    _CHILDREN[port] = (proc, path)
    return {"starting": path, "port": port, "pid": proc.pid,
            "note": "the model is loading in its own process; it appears "
                    "under LOADED when it answers"}


def _stop(port: int) -> dict:
    live = _CHILDREN.get(port)
    if not live or live[0].poll() is not None:
        return {"error": f"nothing started from here on port {port}"}
    live[0].terminate()
    try:
        live[0].wait(timeout=10)
    except Exception:
        live[0].kill()
    _CHILDREN.pop(port, None)
    return {"stopped": live[1], "port": port}


def _load_fn(serve_port: int):
    def handler(_q: dict, body=None) -> dict:
        try:
            req = json.loads(body or b"{}")
        except Exception:
            req = {}
        act = req.get("action")
        target, where = req.get("target") or "", req.get("where") or ""
        try:
            if act == "load":
                return _spawn(target, int(req.get("port") or serve_port),
                              req.get("tune") or "balanced",
                              req.get("sets") or {})
            if act == "unload":
                # Ours to stop only if we started it. Anything else is
                # somebody's server and not this page's to kill.
                for port, (proc, path) in list(_CHILDREN.items()):
                    if path == target or str(port) == str(target):
                        return _stop(port)
                return {"error": "this page did not start that; stop it "
                                 "where it was started"}
            if act == "exo-unload":
                return loaded.exo_unload(where, target)
            if act == "ollama-unload":
                return loaded.ollama_unload(where, target)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
        return {"error": f"unknown action {act!r}"}
    return handler


def serve_ui(host: str, port: int, serve_port: int) -> int:
    _SERVE_PORT["n"] = serve_port
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
            h = routes.get(u.path.rstrip("/") or "/")
            if h is None:
                self._send(b"not found", "text/plain", 404)
                return
            body, ctype = h(parse_qs(u.query), 0)
            self._send(body, ctype)

        def do_POST(self):
            u = urlparse(self.path)
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
        for proc, _p in _CHILDREN.values():
            if proc.poll() is None:
                proc.terminate()
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
    a = p.parse_args(argv)
    return serve_ui(a.host, a.port, a.serve_port)
