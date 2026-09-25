"""`knurlogic serve --cluster` -- wrap exo, do not rebuild it.

An artifact that does not fit one box is the case Knurlogic exists for, and
distributed inference is not something to write again: exo already places
shards, already shards the model, and already speaks OpenAI. So this WRAPS
it, the same way `serve` wraps mlx-lm's server. What Knurlogic adds is what
it adds anywhere -- the settings resolved per node BEFORE anything loads, and
one status that says what every node is actually holding.

WHAT IS AND IS NOT APPLIED, because the distinction is the whole honesty of
this command:

  * The resolution for the node we LAUNCH is applied, as env, before exo's
    process starts -- which is the same ordering rule as `serve`: a VQ
    artifact's bundled runtime reads its knobs at import, and exo's workers
    are spawned processes that inherit this environment.
  * The resolution for a node we did not launch is REPORTED, not applied.
    Nothing here can reach into another machine's process, and printing
    settings that did not take effect is how a benchmark ends up measuring
    the same value twice (vqlab F33).

Attaching to an exo that is already running is the default, because that is
what is actually running day to day. `--launch` starts one.

    knurlogic serve <artifact> --cluster [--exo URL] [--launch]
                               [--node NAME:GIB]...
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from knurlogic.engine import override
from knurlogic.machine import status
from knurlogic.interfaces import web
from knurlogic.machine.artifact import Artifact
from knurlogic.tuning.resolve import Node, resolve_cluster

GIB = 1 << 30
from knurlogic.machine.exo import EXO_URL as DEFAULT_EXO  # noqa: E402


# --- talking to exo: cluster/exo.py (re-exported for run() and tests) ---

from knurlogic.cluster.exo import (  # noqa: E402,F401
    ExoNode, PEER_PORTS, _PEER, _PEER_METRICS, _bytes, _get, _node_ip, _pick,
    _snapshot_for, _swap_used, inventory, peer_memory_map)


# --- launching one ----------------------------------------------------------

def _wait_for(exo_url: str, seconds: float) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            _get(f"{exo_url}/state", timeout=2.0)
            return True
        except (urllib.error.URLError, OSError, ValueError):
            time.sleep(1.0)
    return False


def override_paths():
    """Where the installer and its evidence live. Stable, so a crashed run
    can still be asked what it applied."""
    root = Path(os.environ.get("XDG_CACHE_HOME",
                               Path.home() / ".cache")) / "knurlogic"
    return root / "override", root / "override.log"


def launch(cmd: list, env: dict, exo_url: str, wait: float = 120.0):
    """Start exo with this environment already set.

    Set BEFORE the process exists, which is the same reason `serve` sets it
    before the model loads -- and it is what makes overrides possible at all:
    exo's runner is a spawned process, so the only thing that reaches it is
    the environment this call hands over.
    """
    child = os.environ.copy()
    child.update(env)
    proc = subprocess.Popen(cmd, env=child)
    if not _wait_for(exo_url, wait):
        proc.terminate()
        raise RuntimeError(
            f"launched {' '.join(cmd)} but {exo_url}/state never answered in "
            f"{wait:.0f}s -- exo did not come up, and serving a front end for "
            f"a cluster that is not there would only move the error later")
    return proc


def _default_cmd() -> list:
    exe = shutil.which("exo")
    if exe:
        return [exe]
    raise RuntimeError(
        "no `exo` on PATH. exo runs in its own environment (python 3.13, its "
        "own rust bindings) and guessing an interpreter for it is how you end "
        "up measuring the wrong env -- pass --exo-cmd with the command that "
        "starts exo on this box.")


# --- the front end ----------------------------------------------------------

def _front(host: str, port: int, exo_url: str, status_fn,
           settings_fn=None, serving: str = ""):
    """Knurlogic's own port: /status, /status.json, /, everything else to exo.

    The OpenAI surface is exo's and is proxied untouched. There is no second
    implementation of chat completions here and there should never be one.
    """
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, urlparse

    own = web.routes(status_fn=lambda _n=0: status_fn(),
                     settings_fn=settings_fn,
                     models_fn=web.models_document(serving=serving),
                     loaded_fn=web.loaded_document())

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, body: bytes, ctype: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _proxy(self, body=None):
            req = urllib.request.Request(
                exo_url + self.path, data=body, method=self.command,
                headers={k: v for k, v in self.headers.items()
                         if k.lower() not in ("host", "content-length")})
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    data = r.read()
                    self._send(data, r.headers.get("Content-Type",
                                                   "application/json"),
                               r.status)
            except urllib.error.HTTPError as e:
                self._send(e.read(), e.headers.get("Content-Type",
                                                   "application/json"),
                           e.code)
            except Exception as e:
                self._send(json.dumps({"error": f"exo at {exo_url}: {e}"})
                           .encode(), "application/json", 502)

        def do_GET(self):
            u = urlparse(self.path)
            handler = own.get(u.path.rstrip("/") or "/")
            if handler is None:
                return self._proxy()
            body, ctype = handler(parse_qs(u.query))
            return self._send(body, ctype)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            return self._proxy(self.rfile.read(n) if n else None)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer((host, port), H)
    srv.serve_forever()
    return 0


# --- the command ------------------------------------------------------------

def _parse_node(spec: str) -> Node:
    name, _, gib = spec.partition(":")
    if not gib:
        raise ValueError(f"--node wants NAME:GIB, got {spec!r}")
    return Node(name, int(float(gib) * GIB))


def run(path: str, host: str, port: int, profile: str, exo_url: str,
        nodes: list, do_launch: bool, exo_cmd: list, local: str | None,
        tune: str = "balanced", draft: bool = True) -> int:
    a = Artifact.load(path)
    print(f"artifact  {a.path.name}  ({a.model_type}, {a.gib:.1f} GiB)")
    print(f"cluster   exo at {exo_url}")

    declared = [_parse_node(s) for s in nodes]
    seen = []
    if not do_launch or _wait_for(exo_url, 0.1):
        try:
            seen = inventory(exo_url)
        except Exception as e:
            if not declared:
                print(f"cannot read {exo_url}/state ({e}) and no --node given,"
                      f" so there is nothing to resolve against.",
                      file=sys.stderr)
                return 2

    if declared:
        budget = declared
        print(f"nodes     {len(declared)} declared with --node")
    else:
        budget = [Node(n.name, n.ram_total) for n in seen]
        print(f"nodes     {len(seen)} from exo's own state")
        print("          memory here is exo's SYSTEM RAM, not the Metal "
              "working set; --node NAME:GIB overrides it")
    if not budget:
        print("no nodes: exo reports none and none were declared.",
              file=sys.stderr)
        return 2

    declared_ws = {n.name: n.working_set_bytes for n in declared}
    c = resolve_cluster(a, budget, profile=profile, tune=tune)
    # Drafting in exo is EXO_MTP, and the fork requires it identical on
    # every node. A packed head is used because it is there -- the same rule
    # `serve` follows -- and `--no-draft` is the one way to say otherwise.
    from knurlogic.engine import mtp
    head = mtp.find_head(a.path)
    if head is not None:
        for r in c.nodes.values():
            r.env["EXO_MTP"] = "1" if draft else "0"
        c.notes.append(f"drafting {'on' if draft else 'off (--no-draft)'} "
                       f"on every node: {head.describe()}")
    for n in c.notes:
        print(f"  note: {n}")
    for w in c.warnings:
        print(f"  WARNING: {w}", file=sys.stderr)

    local = local or (budget[0].name if do_launch else None)
    print()
    for name, r in c:
        applied = (do_launch and name == local)
        how = ("APPLIED to the exo we launch" if applied else
               "reported only -- set these on that node yourself")
        print(f"{name}  ({how})")
        for k, v in sorted(r.env.items()):
            print(f"  {k}={v}")

    # Overrides reach only a process we start, because the mechanism rides on
    # the environment -- so they are installed here and nowhere else.
    ov_root, ov_log = override_paths()
    overrides = override.load()
    ov_env = {}
    if overrides and do_launch:
        if ov_log.exists():
            ov_log.unlink()      # this launch's evidence, not the last one's
        ov_env = override.install(overrides, root=ov_root, log=ov_log)
        print(f"\noverrides  {len(overrides)} installed for the process we "
              f"launch (and every runner it spawns)")
        for o in overrides:
            print(f"  {o.state:<9s} {o.module}"
                  + (f"  against {o.against}" if o.against else ""))
    elif overrides:
        print(f"\noverrides  {len(overrides)} declared but NOT applied: they "
              f"reach only a process Knurlogic starts, and this attached to "
              f"one that was already running. Use --launch.")

    proc = None
    if do_launch:
        cmd = exo_cmd or _default_cmd()
        print(f"\nlaunching {' '.join(cmd)}")
        node_env = c.nodes[local].env if local in c.nodes else {}
        proc = launch(cmd, {**node_env, **ov_env}, exo_url)
    elif not _wait_for(exo_url, 5):
        print(f"nothing answering at {exo_url}/state. Start exo, or pass "
              f"--launch.", file=sys.stderr)
        return 2

    def _status_fn():
        try:
            live = inventory(exo_url)
        except Exception:
            live = []
        by_name = {n.name: n for n in live}
        snaps = []
        for name in c.nodes:
            n = by_name.get(name)
            if n is None:
                snaps.append(status.snapshot(node=name, role="declared",
                                             reachable=False,
                                             memory_fn=lambda: {
                                                 "available": False}))
            else:
                snaps.append(_snapshot_for(n, local, c.nodes[name].env, a,
                                           declared_ws.get(name, 0),
                                           peer_port=port))
        for n in live:                      # nodes exo sees that we did not
            if n.name not in c.nodes:
                snaps.append(_snapshot_for(n, local, {}, a,
                                           peer_port=port))
        snap = status.aggregate(snaps, artifact=a)
        if overrides:
            snap["overrides"] = override.status(
                overrides, ov_log if do_launch else None)
        text = status.render_cluster(snap)
        if snap.get("overrides"):
            text += "\n\n" + override.render(snap["overrides"])
        return snap, text

    # The settings panel answers for ONE node: the knobs are per process and
    # a single dict for a cluster would put the wrong ones on the wrong box.
    # Which node it speaks for is named in the document.
    shown = local if local in c.nodes else next(iter(c.nodes))
    shown_ws = next((n.working_set_bytes for n in c.inventory
                     if n.name == shown), 0)

    def _resolve_for(ws_bytes, tune_name):
        return resolve_cluster(a, [Node(shown, ws_bytes)],
                               profile=profile, tune=tune_name).nodes[shown]

    settings_fn = web.settings_document(
        a, live_env=dict(c.nodes[shown].env), live_tune=tune,
        live_working_set=shown_ws, resolve_fn=_resolve_for,
        # Nothing here can reach into another machine's process, so no knob
        # is live from this side however it is read on that side.
        restart_why=("this is a node's own process; knurlogic reports these "
                     "and does not reach into it"),
        wired_advice={"known": False,
                      "note": f"these knobs are for node {shown!r}; a wired "
                              f"limit is per machine and knurlogic only reads "
                              f"the one on the box it runs on"})

    print(f"\nserving on http://{host}:{port}/v1  (proxied to exo; "
          f"ctrl-c to stop)")
    print(f"  /status and /status.json aggregate every node", flush=True)
    try:
        return _front(host, port, exo_url, _status_fn, settings_fn,
                      serving=a.path.name if a else "")
    except KeyboardInterrupt:
        return 0
    finally:
        if proc is not None:
            proc.terminate()
