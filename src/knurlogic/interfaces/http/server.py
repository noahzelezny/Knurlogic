"""knurlogic's own HTTP server (docs/SERVER.md): stdlib ThreadingHTTPServer,
one thread per connection, every model operation handed to the scheduler.

  POST /v1/chat/completions  /chat/completions  /v1/completions   openai.py
  POST /v1/messages          Anthropic, in-process over openai.py
  GET  /v1/models            the served model, capabilities and size
  GET  /v1/residency         what is loaded, its state and memory  (the harness)
  POST /v1/ensure            load a model if it is not; optionally wait
  GET  /health
  and the page's routes (web.routes): /, /status.json, /settings.json, ...

A write that fails (the client went away) cancels the Job, so the
scheduler frees its row on the next step.

BROWSERS. A web page open in the user's browser can send requests to
localhost; without care, any site could make this server load a model.
Browsers mark their requests with `Origin`, and ordinary clients (SDKs,
curl, harnesses) send none. So a request carrying an Origin is answered
only when that origin is this server itself (the page it serves) or one
the operator allowed (`--allow-origin`), and only those get CORS headers.
The Host header must name this machine (a hostname, `.local` name or IP
literal), which stops DNS rebinding -- a foreign domain re-pointed at
127.0.0.1 to look same-origin.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional
from urllib.parse import parse_qs, urlparse

from . import openai as O

logger = logging.getLogger(__name__)

CHAT_PATHS = ("/v1/chat/completions", "/chat/completions")

#: Largest request body read, bytes (--max-request-mib). The body is read
#: into memory before anything else can judge it, so it is bounded first.
#: Generous: images are bounded separately by the image store's memory.
DEFAULT_MAX_BODY = 512 * 1024 * 1024


class BodyError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class App:
    """What the handler needs: the scheduler, what is served, the page's
    routes and a few hooks the caller fills in."""

    def __init__(self, scheduler, *, served: Callable[[], dict],
                 routes: Optional[dict] = None,
                 gate=None,
                 concurrency: Callable[[], str] = None,
                 residency: Callable[[], dict] = None,
                 ensure: Callable[[dict], dict] = None,
                 max_body: int = DEFAULT_MAX_BODY,
                 allow_origins: tuple = (), allow_hosts: tuple = ()):
        from knurlogic.engine.serve import thinking
        from knurlogic.interfaces import messages
        self.scheduler = scheduler
        self.served = served
        self.routes = routes or {}
        self.gate = gate
        self.concurrency = concurrency
        self.residency = residency
        self.ensure = ensure
        self.max_body = int(max_body)
        self.allow_origins = {o.rstrip("/") for o in allow_origins}
        self.allow_hosts = {h.lower() for h in allow_hosts}
        self.translate = thinking.translate
        self.requests = 0
        self.messages = messages.handler_over(self._transport,
                                              served().get("id", ""))

    def has_vision(self) -> bool:
        from knurlogic.engine.serve import state
        return state.VISION.get("serve") is not None

    def submit(self, body: dict, chat: bool):
        from knurlogic.machine.artifact import sampling_defaults
        path = self.scheduler.host.path
        job, ctx = O.build_job(body, chat=chat,
                               translate=self.translate if chat else None,
                               has_vision=self.has_vision,
                               sampling_defaults=(sampling_defaults(path)
                                                  if path else None))
        self.requests += 1
        self.scheduler.submit(job)
        host = self.scheduler.host

        def decode(t):
            # resolved when used: a request that arrived during a load has
            # the tokenizer by the time it has tokens
            tok = host.tokenizer
            return tok.decode([t]) if tok is not None else ""
        reply = O.Reply(job, ctx, served=self.served().get("id", ""),
                        decode=decode)
        return job, reply

    def _transport(self, oai: dict):
        """/v1/messages -> the OpenAI surface, in-process."""
        from knurlogic.interfaces.messages import TransportError
        try:
            job, reply = self.submit(oai, chat=True)
            first = reply.first()
            if first[0] == "error":
                raise O._status_of(first[1])
            if oai.get("stream"):
                def events():
                    try:
                        yield from reply.events(first)
                    finally:
                        job.cancel()     # abandoned mid-stream: free the row
                return events()
            return reply.complete(first)
        except O.ApiError as e:
            raise TransportError(e.status, str(e)) from e


class Handler(BaseHTTPRequestHandler):
    app: App = None                    # set on the subclass by serve()
    server_version = "knurlogic"

    def log_message(self, fmt, *args):  # quiet, as mlx-lm's is not
        logger.debug("%s " + fmt, self.address_string(), *args)

    # ------------------------------------------------------------ helpers

    def _send(self, code: int, body: bytes, ctype="application/json",
              headers: Optional[dict] = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj, headers=None) -> None:
        self._send(code, json.dumps(obj).encode(), headers=headers)

    def _error(self, e: O.ApiError) -> None:
        self._json(e.status, e.body())

    def _cors(self) -> None:
        o = self.headers.get("Origin")
        if o and o.rstrip("/") in self.app.allow_origins:
            self.send_header("Access-Control-Allow-Origin", o)
            self.send_header("Vary", "Origin")

    def _refused_browser(self) -> bool:
        why = browser_refusal(self.headers, self.app.allow_origins,
                              self.app.allow_hosts)
        if why is None:
            return False
        self._send(403, why.encode(), "text/plain; charset=utf-8")
        return True

    def _gated(self) -> bool:
        g = self.app.gate
        if g is None:
            return False
        local = self.connection.getsockname()[0]
        if g.allows(local):
            return False
        self._send(403, g.refusal(local), "text/plain; charset=utf-8")
        return True

    def _body(self) -> bytes:
        """The request body, bounded before a byte is read."""
        cl = self.headers.get("Content-Length")
        if self.headers.get("Transfer-Encoding"):
            # with a Content-Length too, the two framings disagree about
            # where this request ends and the next begins (smuggling);
            # alone, it is chunked, which this server does not read
            raise BodyError(411, "send the body with a Content-Length "
                                 "and no Transfer-Encoding (chunked "
                                 "uploads are not accepted)")
        if cl is None:
            return b""
        try:
            n = int(cl)
        except ValueError:
            raise BodyError(400, f"Content-Length {cl!r} is not a number")
        if n < 0:
            raise BodyError(400, "Content-Length is negative")
        if n > self.app.max_body:
            raise BodyError(413, f"the request body is {n} bytes; the "
                                 f"maximum is {self.app.max_body} bytes "
                                 f"(--max-request-mib)")
        return self.rfile.read(n) if n else b""

    # -------------------------------------------------------------- verbs

    def do_OPTIONS(self):
        o = (self.headers.get("Origin") or "").rstrip("/")
        self.send_response(204)
        if o and o in self.app.allow_origins:
            self._cors()
            self.send_header("Access-Control-Allow-Methods",
                             "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers",
                             "Content-Type, Authorization, x-api-key, "
                             "anthropic-version")
        self.end_headers()

    def do_GET(self):
        self._guarded(self._get)

    def do_POST(self):
        self._guarded(self._post)

    def _guarded(self, fn) -> None:
        """Any error a route did not answer itself is a 500 with a body --
        never a dropped connection and a traceback on stderr only."""
        try:
            if self._gated() or self._refused_browser():
                return
            fn()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            logger.exception("%s %s failed", self.command, self.path)
            try:
                self._error(O.ApiError(500, f"{type(e).__name__}: {e}"))
            except Exception:
                pass

    def _get(self):
        u = urlparse(self.path)
        path = u.path.rstrip("/") or "/"
        if path == "/v1/models":
            from knurlogic.machine.artifact import (context_length,
                                                    sampling_defaults)
            path = self.app.scheduler.host.path
            return self._json(200, O.models_document(
                self.app.served(),
                sampling_defaults(path) if path else {},
                context_length(path) if path else 0))
        if path == "/health":
            return self._json(200, {"status": "ok", "server": "knurlogic",
                                    "model": self.app.scheduler.host.state})
        if path == "/v1/residency" and self.app.residency:
            return self._json(200, self.app.residency())
        h = self.app.routes.get(path)
        if h is None:
            return self._json(404, {"error": {"message": f"no route {path}",
                                              "type": "not_found"}})
        body, ctype = h(parse_qs(u.query), self.app.requests)
        self._send(200, body, ctype)

    def _post(self):
        u = urlparse(self.path)
        path = u.path.rstrip("/") or "/"
        try:
            raw = self._body()
        except BodyError as e:
            self.close_connection = True       # the unread body is left
            return self._error(O.ApiError(e.status, str(e)))
        if path in CHAT_PATHS or path == "/v1/completions":
            return self._inference(raw, chat=path != "/v1/completions")
        if path == "/v1/messages":
            return self._raw(self.app.messages, raw)
        if path == "/v1/messages/count_tokens":
            return self._count_tokens(raw)
        if path == "/v1/ensure" and self.app.ensure:
            try:
                body = json.loads(raw or b"{}")
            except ValueError:
                return self._error(O.ApiError(400, "body must be JSON"))
            try:
                return self._json(200, self.app.ensure(body))
            except O.ApiError as e:
                return self._error(e)
        h = self.app.routes.get("POST " + path)
        if h is None:
            return self._json(404, {"error": {"message": f"no route {path}",
                                              "type": "not_found"}})
        if getattr(h, "raw", False):
            return self._raw(h, raw)
        out, ctype = h(parse_qs(u.query), 0, raw)
        self._send(200, out, ctype)

    # ---------------------------------------------------------- inference

    def _headers(self) -> dict:
        c = self.app.concurrency
        return {"X-Knurlogic-Concurrency": c()} if c else {}

    def _inference(self, raw: bytes, chat: bool) -> None:
        try:
            body = json.loads(raw or b"{}")
        except ValueError as e:
            return self._error(O.ApiError(400, f"invalid JSON: {e}"))
        try:
            job, reply = self.app.submit(body, chat)
        except O.ApiError as e:
            return self._error(e)
        first = reply.first()
        if first[0] == "error":
            return self._error(O._status_of(first[1]))
        if not reply.ctx["stream"]:
            try:
                out = reply.complete(first)
            except O.ApiError as e:
                return self._error(e)
            return self._json(200, out, self._headers())
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self._cors()
        for k, v in self._headers().items():
            self.send_header(k, v)
        self.end_headers()
        self.close_connection = True
        try:
            for chunk in reply.events(first):
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            job.cancel()           # the scheduler frees the row

    def _count_tokens(self, raw: bytes) -> None:
        """POST /v1/messages/count_tokens: what the prompt would be, in the
        served model's tokens, rendered by its own template -- Claude Code
        asks this to manage its context. Image blocks are left out (their
        cost is known only once encoded), so with images it is a floor."""
        from knurlogic.engine.runtime import prompt as P
        from knurlogic.interfaces import messages as M
        try:
            req = json.loads(raw or b"{}")
            oai = M.to_openai(req if isinstance(req, dict) else {})
        except ValueError:
            return self._error(O.ApiError(400, "body must be JSON"))
        tok = getattr(self.app.scheduler.host, "tokenizer", None)
        if tok is None:
            return self._error(O.ApiError(503, "no model is loaded",
                                          type_="server_error"))
        msgs = []
        for m in oai.get("messages") or []:
            c = m.get("content")
            if isinstance(c, list):
                c = [p for p in c
                     if isinstance(p, dict) and p.get("type") == "text"]
            msgs.append(dict(m, content=c))
        try:
            prompt, *_ = P.tokenize(None, tok, P.ChatRequest(
                "chat", "", msgs, oai.get("tools") or None), P.PromptArgs())
        except P.PromptError as e:
            return self._error(O.ApiError(400, str(e)))
        return self._json(200, {"input_tokens": len(prompt)})

    def _raw(self, handler, raw: bytes) -> None:
        """A handler that writes its own response (an event stream has no
        length to declare, so it ends when the socket does)."""
        def start(code, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self._cors()
            self.end_headers()

        def write(b):
            self.wfile.write(b)
            self.wfile.flush()
        self.close_connection = True
        try:
            handler(raw, write, start)
        except (BrokenPipeError, ConnectionResetError):
            pass


def browser_refusal(headers, allow_origins=(), allow_hosts=()):
    """Why a request is refused, or None -- the browser guard both of
    knurlogic's servers (this one and `knurlogic ui`) apply:

      Host    must name this machine, or be allowed (--allow-host): a
              foreign domain re-pointed at this address (DNS rebinding)
              would otherwise look same-origin to the browser.
      Origin  a browser marks its requests with one; answered only when it
              is this server itself or allowed (--allow-origin). Ordinary
              clients (SDKs, curl, harnesses) send none."""
    host = (headers.get("Host") or "").strip()
    if host and not host_is_local(host, allow_hosts):
        name = host.rsplit(":", 1)[0] if not host.endswith("]") else host
        return (f"Host {host!r} is not this machine (DNS-rebinding guard). "
                f"If you reach it by that name, start it with "
                f"--allow-host {name}.")
    o = (headers.get("Origin") or "").rstrip("/")
    if o and o not in (f"http://{host}", f"https://{host}") \
            and o not in {x.rstrip("/") for x in allow_origins}:
        return (f"requests from the web page at {o} are not answered: a "
                f"page in a browser must not be able to drive this server. "
                f"Start it with --allow-origin {o} to allow that page.")
    return None


def host_is_local(host_header: str, allow_hosts=()) -> bool:
    """Does a Host header name this machine: localhost, an IP literal, a
    `.local` / `.localhost` name, this machine's hostname, or a name the
    operator allowed?"""
    import ipaddress
    h = host_header.strip()
    if h.startswith("["):                       # [::1]:8080
        h = h[1:h.find("]")] if "]" in h else h
    elif h.count(":") == 1:
        h = h.split(":", 1)[0]
    h = h.lower().rstrip(".")
    if h in {x.lower() for x in allow_hosts}:
        return True
    if h in ("localhost",) or h.endswith((".localhost", ".local")):
        return True
    try:
        ipaddress.ip_address(h)
        return True
    except ValueError:
        pass
    # This machine's own name, exactly. Never "the first label matches":
    # noahs-mac.attacker.example would pass that, and a name an attacker
    # controls can be pointed at 127.0.0.1 (DNS rebinding) -- the Origin
    # check compares against this same Host, so it would pass too. A name
    # this machine is reached by (a tailnet, a LAN domain) is --allow-host.
    return h in _machine_names()


_NAMES: set = set()


def _machine_names() -> set:
    """This machine's hostname, once. Not getfqdn(): that is a DNS lookup
    (seconds on a network without reverse DNS), and `.local` names are
    accepted above anyway."""
    import socket
    if not _NAMES:
        n = socket.gethostname().lower()
        _NAMES.update({n, n.split(".")[0]})
    return _NAMES


def make_server(app: App, host: str, port: int) -> ThreadingHTTPServer:
    H = type("KnurlogicHandler", (Handler,), {"app": app})
    srv = ThreadingHTTPServer((host, port), H)
    srv.daemon_threads = True
    return srv
