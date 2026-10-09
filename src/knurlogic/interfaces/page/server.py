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
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from knurlogic.interfaces import spawn
from knurlogic.interfaces.page import (
    documents,
    loads,
    messages,
    nodes,
    peek,
    peers,
    prompt_cache,
    relay,
    router,
)
from knurlogic.machine import identity

logger = logging.getLogger(__name__)


#: the largest body the page accepts (the API server's default cap)
MAX_BODY = 512 << 20


def make_handler(routes: dict, gate=None, allow_origins=(),
                 allow_hosts=(), gate_for_peers=None):
    """The page's request handler: its routes, the router, the proxies, and
    the guards in front of every one of them. `gate_for_peers`: the
    /peer/ gate's link check (default cluster/links.Gate; tests pass
    their own)."""
    wire()

    table = messages.peer_table(routes)

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
            if u.path.rstrip("/") == peers.MSG_PATH:
                self._peer_msg("GET")
                return
            if u.path.startswith(peers.PEER_RELAY + "/v1/"):
                self._peer_relay("GET", u.path)
                return
            if self._gated():
                return
            intro = self.headers.get("X-Knurlogic-Peer")
            if intro and nodes.PEERS:
                nodes.PEERS.introduce(self.client_address[0], intro)
            if u.path.rstrip("/") == "/v1/models":
                self._send(json.dumps(router.route_models_document()).encode(),
                           "application/json")
                return
            if u.path.rstrip("/") == prompt_cache.PROMPT_CACHE_PATH:
                prompt_cache.prompt_cache_forward(
                    self, "GET", prompt_cache.PROMPT_CACHE_PATH,
                    parse_qs(u.query), b"")
                return
            if u.path.rstrip("/") == "/peek":
                code, doc = peek.peek(parse_qs(u.query))
                self._send(doc.encode(), "application/json", code)
                return
            h = routes.get(u.path.rstrip("/") or "/")
            if h is None:
                self._send(b"not found", "text/plain", 404)
                return
            body, ctype = h(parse_qs(u.query), 0)
            self._send(body, ctype)

        def do_POST(self):
            nodes.hot()               # a load, a chat, a peer's message: memory moves
            u = urlparse(self.path)
            if u.path.rstrip("/") == peers.MSG_PATH:
                self._peer_msg("POST")
                return
            if u.path.startswith(peers.PEER_RELAY + "/v1/"):
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
                router.proxy_chat(self, where, self.rfile.read(n) if n else b"")
                return
            if u.path.rstrip("/") in router.ROUTE_PATHS:
                router.route(self, u.path.rstrip("/"),
                      self.rfile.read(n) if n else b"")
                return
            if u.path.startswith(prompt_cache.PROMPT_CACHE_PATH + "/"):
                prompt_cache.prompt_cache_forward(self, "POST", u.path.rstrip("/"),
                                     parse_qs(u.query),
                                     self.rfile.read(n) if n else b"")
                return
            if u.path.rstrip("/") == "/machine.json":
                where = (parse_qs(u.query).get("where") or [""])[0]
                code, doc = peers.machine_apply(where, self.rfile.read(n)
                                          if n else b"")
                self._send(json.dumps(doc).encode(), "application/json",
                           code)
                return
            if u.path.rstrip("/") == "/apply":
                where = (parse_qs(u.query).get("where") or [""])[0]
                if n > peek.APPLY_MAX:
                    self._send(b"a settings change is small", "text/plain",
                               413)
                    return
                code, doc = peek.apply_settings(where, self.rfile.read(n)
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
            manual = peers.manual_hosts()
            refused = peers.peer_refusal(
                self.headers, self.client_address[0],
                self.connection.getsockname()[0], manual_hosts=manual,
                what="relayed requests")
            if refused:
                self.close_connection = True
                router.send_json(self, *refused)
                return
            if self._refuse_chunked():
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = -1
            if not 0 <= n <= MAX_BODY:
                self.close_connection = True
                router.send_json(self, 400, {"error": f"Content-Length must be a "
                                       f"number of bytes up to {MAX_BODY}"})
                return
            body = self.rfile.read(n) if n else b""
            relay.peer_relay(self, method,
                       path[len(peers.PEER_RELAY):].rstrip("/"), body)

        def _refuse_chunked(self) -> bool:
            """A peer route's body is a plain Content-Length body -- never
            Transfer-Encoding, the framing a smuggled request hides behind:
            411 and the connection closed. -> True when refused."""
            if not self.headers.get("Transfer-Encoding"):
                return False
            self.close_connection = True
            router.send_json(self, 411, {"error": "send the body with a "
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
                router.send_json(self, 413, {"error": "a peer message is small"})
                return
            manual = peers.manual_hosts()
            refused = peers.peer_refusal(
                self.headers, self.client_address[0],
                self.connection.getsockname()[0], gate=gate_for_peers,
                manual_hosts=manual, what="peer messages")
            body = self.rfile.read(n) if n else b""
            if refused:
                self.close_connection = True
                router.send_json(self, *refused)
                return
            code, doc = transport.handle(body, table)
            router.send_json(self, code, doc)

    return H


def serve_ui(host: str, port: int, serve_port: int, peers=(),
             allow_origins=(), allow_hosts=(), offline: bool = False,
             menubar: bool = False, open_page: bool = False) -> int:
    from knurlogic.interfaces.page import updates
    updates.start_for_page(offline)
    spawn.SERVE_PORT["n"] = serve_port
    spawn.SERVE_PORT["ui"] = port
    from knurlogic.cluster import links
    from knurlogic.cluster.peers import Peers
    me = identity.identity()
    # `cluster`: every address bound, only loopback and Thunderbolt
    # answered (cluster/links.Gate), advertised on Thunderbolt only.
    gate = links.Gate() if host == "cluster" else None
    bind = "0.0.0.0" if gate else host
    reachable = host not in ("127.0.0.1", "localhost", "::1")
    nodes.PEERS = Peers(me, port, manual=peers, reachable=reachable).start()
    nodes.start_discovery(me, host, port, reachable)
    routes = documents.routes(
        status_fn=nodes.status_fn,
        settings_fn=documents.machine_settings(),
        models_fn=documents.models_document(serving=""),
        loaded_fn=loads.loaded_fn(),
        load_fn=loads.load_fn(serve_port))
    # /status.json?light=1: the liveness document every peer polls
    full_status = routes["/status.json"]
    routes["/status.json"] = lambda q, _n=0: (
        documents.json_reply(nodes.status_light()) if (q.get("light") or [""])[0]
        else full_status(q, _n))
    # the knurlogic allowance: THIS machine's only, and set only by a POST
    # the page sends when its user applies it; a peer's is read from that
    # peer's /settings.json through /peek
    routes["/allowance.json"] = lambda _q, _n=0: documents.json_reply(
        documents.allowance_doc())
    routes["POST /allowance.json"] = lambda _q, _n=0, body=None: documents.json_reply(
        documents.set_allowance(body))
    # the knurlogic strategy: this machine's default launch preset
    routes["/strategy.json"] = lambda _q, _n=0: documents.json_reply(
        documents.strategy_doc())
    # a newer knurlogic on PyPI (asked once at page start, page/updates.py)
    routes["/release.json"] = lambda _q, _n=0: documents.json_reply(
        updates.release_doc())
    # the knurlogic-wide settings: compaction, identical results across chips
    routes["/knurlogic.json"] = lambda _q, _n=0: documents.json_reply(
        documents.knurlogic_doc())
    routes["POST /knurlogic.json"] = lambda _q, _n=0, body=None: documents.json_reply(
        documents.set_knurlogic(body))
    routes["POST /strategy.json"] = lambda _q, _n=0, body=None: documents.json_reply(
        documents.set_strategy(body))

    from knurlogic.interfaces.page import hub
    routes["/hub/search.json"] = lambda q, _n=0: documents.json_reply(
        hub.search((q.get("q") or [""])[0]))
    routes["/hub/repo.json"] = lambda q, _n=0: documents.json_reply(
        hub.repo((q.get("id") or [""])[0]))
    routes["/hub/downloads.json"] = lambda _q, _n=0: documents.json_reply(
        hub.downloads())
    routes["POST /hub/download.json"] = lambda _q, _n=0, body=None: (
        documents.json_reply(hub.act(body)))

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
        for port in list(spawn.CHILDREN):
            spawn.stop(port)
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


def wire() -> None:
    """Give launch and recovery what they need of this page; every page
    handler (make_handler) calls it, never this module's import. Each is
    a late-bound lambda, so a swapped PEERS or mcp lifecycle.load is what they see."""
    from knurlogic.cluster import launch, recovery
    from knurlogic.interfaces.mcp import lifecycle
    launch.status_fn = lambda: nodes.status_fn()
    launch.peers_fn = lambda: nodes.PEERS.all() if nodes.PEERS else []
    recovery.peers_fn = lambda: nodes.PEERS.all() if nodes.PEERS else []
    recovery.child_fn = lambda port: spawn.CHILDREN.get(port)
    recovery.answers_fn = lambda port: spawn.answers(port)
    recovery.load_fn = lambda **kw: lifecycle.load(**kw)

