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


class App:
    """What the handler needs: the scheduler, what is served, the page's
    routes and a few hooks the caller fills in."""

    def __init__(self, scheduler, *, served: Callable[[], dict],
                 routes: Optional[dict] = None,
                 gate=None, image_limit: Callable = None,
                 concurrency: Callable[[], str] = None,
                 residency: Callable[[], dict] = None,
                 ensure: Callable[[dict], dict] = None):
        from knurlogic.engine.serve import thinking
        from knurlogic.interfaces import messages
        self.scheduler = scheduler
        self.served = served
        self.routes = routes or {}
        self.gate = gate
        self.image_limit = image_limit
        self.concurrency = concurrency
        self.residency = residency
        self.ensure = ensure
        self.translate = thinking.translate
        self.requests = 0
        self.messages = messages.handler_over(self._transport,
                                              served().get("id", ""))

    def has_vision(self) -> bool:
        from knurlogic.engine.serve import state
        return state.VISION.get("serve") is not None

    def submit(self, body: dict, chat: bool):
        job, ctx = O.build_job(body, chat=chat,
                               translate=self.translate if chat else None,
                               has_vision=self.has_vision,
                               image_limit=self.image_limit)
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
        self.send_header("Access-Control-Allow-Origin", "*")

    def _gated(self) -> bool:
        g = self.app.gate
        if g is None:
            return False
        local = self.connection.getsockname()[0]
        if g.allows(local):
            return False
        self._send(403, g.refusal(local), "text/plain; charset=utf-8")
        return True

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        return raw

    # -------------------------------------------------------------- verbs

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):
        if self._gated():
            return
        u = urlparse(self.path)
        path = u.path.rstrip("/") or "/"
        if path == "/v1/models":
            return self._json(200, O.models_document(self.app.served()))
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

    def do_POST(self):
        if self._gated():
            return
        u = urlparse(self.path)
        path = u.path.rstrip("/") or "/"
        raw = self._body()
        if path in CHAT_PATHS or path == "/v1/completions":
            return self._inference(raw, chat=path != "/v1/completions")
        if path == "/v1/messages":
            return self._raw(self.app.messages, raw)
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


def make_server(app: App, host: str, port: int) -> ThreadingHTTPServer:
    H = type("KnurlogicHandler", (Handler,), {"app": app})
    srv = ThreadingHTTPServer((host, port), H)
    srv.daemon_threads = True
    return srv
