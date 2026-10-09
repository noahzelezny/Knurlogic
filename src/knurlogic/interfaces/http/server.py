"""knurlogic's own HTTP server: stdlib ThreadingHTTPServer, one thread per
connection, every model operation handed to the scheduler.

Serves /v1/chat/completions, /v1/completions, /v1/messages, /v1/responses,
/api/* (Ollama), /v1/models,
/v1/residency, /v1/ensure, /health and the page's routes. A failed write
cancels the Job. A request carrying an Origin is answered only for this
server's own origin or one allowed with --allow-origin, and the Host header
must name this machine (stops DNS rebinding).

Design: docs/design/server.md.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import select
import socket
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from knurlogic.engine.templates import TEMPLATE_ERRORS

from . import openai as O
from . import telemetry as T
from .compaction import CompactingChat
from .prompt_cache import PromptCacheHandlers

logger = logging.getLogger(__name__)

CHAT_PATHS = ("/v1/chat/completions", "/chat/completions")

#: the inference routes: each opens a ledger Request (telemetry.py) named
#: by its api; the second value says whether its stream carries
#: knurlogic.progress events without the client asking. The Anthropic SDK
#: skips an event name it does not know; the OpenAI SDK does not (it hands
#: the event's data on as a chunk, which has no `choices`), so the OpenAI
#: shapes carry it only for a client that names itself with X-Client.
#: Ollama's stream is NDJSON, not SSE: it has no events.
INFERENCE = {"/v1/chat/completions": ("chat", False),
             "/chat/completions": ("chat", False),
             "/v1/completions": ("completions", False),
             "/v1/messages": ("messages", True),
             "/v1/responses": ("responses", False),
             "/api/chat": ("ollama", None),
             "/api/generate": ("ollama", None)}

#: How often a streamed request waiting for admission is told where it
#: stands (knurlogic.progress, phase "queue"), seconds.
PROGRESS_S = 1.0

#: Largest request body read, bytes (--max-request-mib). The body is read
#: into memory before anything else can judge it, so it is bounded first.
#: Generous: images are bounded separately by the image store's memory.
DEFAULT_MAX_BODY = 512 * 1024 * 1024


#: How often a non-streamed request's connection is checked for a hang-up
#: while it waits, seconds.
HANGUP_POLL_S = 1.0

#: The connection of the request this handler thread is serving, for
#: App.submit's hang-up watch (_watch_hangup).
_CONN = threading.local()


def _hung_up(sock) -> bool:
    """True when the peer has closed: readable with nothing to read (EOF)
    or an error. Peeks, so a request's bytes are never consumed; data
    waiting (a pipelined request) is not a hang-up."""
    try:
        ready, _, _ = select.select([sock], [], [], 0)
        if not ready:
            return False
        return sock.recv(1, socket.MSG_PEEK) == b""
    except BlockingIOError:
        return False
    except (OSError, ValueError):
        return True


def _watch_hangup(job, sock, done: threading.Event) -> None:
    """Cancel `job` when its client hangs up while a non-streamed reply is
    being made. Nothing is written until such a reply is complete, so the
    failed write that stops a stream never happens: a client that timed
    out and resent its request (the harness's 120 s read timeout, 2026-10-06)
    left every copy prefilling 170k tokens to the end. Off with
    KNURLOGIC_DISCONNECT_POLL=off, for a client that half-closes its side
    after sending (none known)."""
    while not done.wait(HANGUP_POLL_S):
        if job.cancelled:
            return
        if _hung_up(sock):
            logger.info("a non-streamed request's client hung up: "
                        "cancelling it")
            job.cancel()
            return


class BodyError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class App(CompactingChat):
    """What the handler needs: the scheduler, what is served, the page's
    routes and a few hooks the caller fills in."""

    def __init__(self, scheduler, *, served: Callable[[], dict],
                 routes: dict | None = None,
                 gate=None,
                 concurrency: Callable[[], str] = None,
                 residency: Callable[[], dict] = None,
                 ensure: Callable[[dict], dict] = None,
                 max_body: int = DEFAULT_MAX_BODY,
                 allow_origins: tuple = (), allow_hosts: tuple = ()):
        from knurlogic.engine.model import thinking
        from knurlogic.interfaces.http import messages, ollama, responses
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
        self._count_lock = threading.Lock()
        self.messages = messages.handler_over(self._transport,
                                              served().get("id", ""))
        self.responses = responses.handler_over(self._transport,
                                                served().get("id", ""))
        self.ollama_chat = ollama.handler_over(
            self._transport, served().get("id", ""), generate=False)
        self.ollama_generate = ollama.handler_over(
            self._transport, served().get("id", ""), generate=True)

    def has_vision(self) -> bool:
        """Whether an image request is refused up front. Until the model
        has loaded, vision is not known yet: the request queues like a text
        one, and the scheduler refuses it at admission if the loaded model
        has none (otherwise an image sent during a load gets a 400 "no
        vision" while a text request waits and is served)."""
        from knurlogic.engine.model import state
        if getattr(self.scheduler.host, "state", "ready") != "ready":
            return True
        return state.VISION.get("serve") is not None

    def _count(self) -> None:
        """One more request served (the handler threads race otherwise)."""
        with self._count_lock:
            self.requests += 1

    def submit(self, body: dict, chat: bool, extra: dict = None):
        from knurlogic.machine.artifact import sampling_defaults
        received = time.perf_counter()
        path = self.scheduler.host.path
        job, ctx = O.build_job(body, chat=chat,
                               translate=self.translate if chat else None,
                               has_vision=self.has_vision,
                               sampling_defaults=(sampling_defaults(path)
                                                  if path else None))
        if chat:
            ctx["window"] = self.window()
        ctx.update(extra or {})
        self._count()
        job.received = received     # usage.knurlogic.timing: http_build
        rec = getattr(_CONN, "record", None)    # the ledger's Request
        # the owner of the prompt-cache entries it makes: the ledger's
        # labels (the same parsing and byte limits), and X-Cache-Retain
        hdrs = getattr(_CONN, "headers", None)
        from knurlogic.machine import ledger as L
        labs = rec.labels if rec is not None else L.labels(hdrs)
        job.session, job.role, job.run = (labs.get("session"),
                                           labs.get("role"), labs.get("run"))
        job.pin = bool(hdrs) and (hdrs.get("X-Cache-Retain") or ""
                                  ).strip().lower() == "pin"
        job.keep_latest = bool(hdrs) and (hdrs.get("X-Cache-Keep") or ""
                                          ).strip().lower() == "latest"
        if rec is not None:
            job.request_id = rec.id    # X-Request-Id: the client's, or a ULID
            rec.model = self.served().get("id") or None
            rec.jobs.append(job)
            ctx["record"], ctx["progress"] = rec, rec.progress
        self.scheduler.submit(job)
        conn = getattr(_CONN, "sock", None)
        if conn is not None and not ctx.get("stream") and \
                os.environ.get("KNURLOGIC_DISCONNECT_POLL", "on") != "off":
            threading.Thread(target=_watch_hangup,
                             args=(job, conn, _CONN.done),
                             daemon=True).start()
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
        from knurlogic.interfaces.http.messages import TransportError
        try:
            kind, val = self.chat(oai)
            return val
        except O.ApiError as e:
            raise TransportError(e.status, str(e),
                                 getattr(e, "retry_after", None)) from e

    # ------------------------------------------------------- context

    def window(self) -> int:
        """The context a request may use: the server's cap
        (KNURLOGIC_CONTEXT_LENGTH) else the model's own window; 0 when
        neither is known."""
        from knurlogic.engine.runtime.scheduler import _context_cap
        cap = _context_cap()
        if cap:
            return cap
        path = getattr(self.scheduler.host, "path", None)
        if not path:
            return 0
        from knurlogic.machine.artifact import context_length
        return context_length(path)

    def count(self, messages: list, tools=None) -> int:
        """A chat prompt's length in the served model's tokens, rendered
        by its own template; images left out (a floor with them). 0 with
        no model loaded."""
        from knurlogic.engine.runtime import prompt as P
        tok = getattr(self.scheduler.host, "tokenizer", None)
        if tok is None:
            return 0
        msgs = []
        for m in messages or []:
            c = m.get("content")
            if isinstance(c, list):
                c = [p for p in c
                     if isinstance(p, dict) and p.get("type") == "text"]
            msgs.append(dict(m, content=c))
        prompt, *_ = P.tokenize(None, tok, P.ChatRequest(
            "chat", "", msgs, tools or None), P.PromptArgs())
        return len(prompt)


#: How long a streamed request may wait for its first event (it is queued
#: behind other rows, or waiting for memory) before the reply goes out as a
#: 200 kept alive by comments. Until something is written, a client that hung
#: up or timed out cannot be noticed, and its job would hold its place in the
#: queue and then prefill for nobody (2026-10-05: a dead caller's 55k-token
#: prompt prefilled for ~7 min ahead of a live one, which got no bytes for
#: 600 s, timed out, and retried behind its own ghost).
QUEUED_KEEPALIVE_S = 5.0


def _queued(job, reply, sched=None):
    """SSE for a streamed request still waiting for its first event: a
    keepalive comment every QUEUED_KEEPALIVE_S until it starts (and, when
    its stream carries them, a knurlogic.progress "queue" event with the
    requests ahead of it every PROGRESS_S), then the reply as usual. A
    refusal that arrives after the 200 is an error event. A failed write
    (the client is gone) closes this generator, and the job is cancelled:
    the scheduler drops a cancelled job from the queue."""
    progress = bool(getattr(reply, "ctx", {}).get("progress"))
    tick = PROGRESS_S if progress else QUEUED_KEEPALIVE_S
    waited = QUEUED_KEEPALIVE_S      # a comment at once, then every 5 s
    try:
        while True:
            if progress:
                f = getattr(sched, "ahead", None)
                try:
                    ahead = f(job) if callable(f) else 0
                except (TypeError, ValueError):
                    ahead = 0
                yield reply.progress("queue", ahead=ahead)
            if waited >= QUEUED_KEEPALIVE_S:
                waited = 0.0
                yield b": keepalive queued\n\n"
            try:
                first = reply.first(timeout=tick)
                break
            except queue.Empty:
                waited += tick
        if first[0] == "error":
            yield O._data(O._status_of(first[1]).body())
            yield b"data: [DONE]\n\n"
            return
        yield from reply.events(first)
    finally:
        job.cancel()


def _load_state(host) -> str:
    """The host's load state for /v1/models (loading, warming, ready,
    failed, empty); "ready" for a host that does not say (a test's fake)."""
    status = getattr(host, "status", None)
    if not callable(status):
        return "ready"
    try:
        return str(status().get("state") or "ready")
    except Exception:       # a status that fails must not fail /v1/models
        return "ready"


def _memory_short(sched) -> str | None:
    """Why the scheduler could not admit a minimal prompt now, or None."""
    f = getattr(sched, "memory_short", None)
    return f() if callable(f) else None


class Handler(PromptCacheHandlers, T.TelemetryHandlers,
              BaseHTTPRequestHandler):
    app: App = None  # type: ignore[assignment]  # set on the subclass by serve()
    server_version = "knurlogic"

    def log_message(self, fmt, *args):  # quiet, as mlx-lm's is not
        logger.debug("%s " + fmt, self.address_string(), *args)

    # ------------------------------------------------------------ helpers

    def _send(self, code: int, body: bytes, ctype="application/json",
              headers: dict | None = None) -> None:
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
        ra = getattr(e, "retry_after", None)
        self._json(e.status, e.body(),
                   headers={"Retry-After": str(int(ra))} if ra else None)

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
            raise BodyError(400, f"Content-Length {cl!r} is not a number") from None
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
                             "anthropic-version, X-Request-Id, X-Client, "
                             "X-Client-Session, X-Client-Run, "
                             "X-Client-Role, X-Cache-Retain, X-Cache-Keep")
        self.end_headers()

    def do_GET(self):
        self._guarded(self._get)

    def send_response(self, code, message=None):
        rec = getattr(self, "_record", None)
        if rec is not None and rec.status is None:
            rec.status = int(code)
        super().send_response(code, message)

    def end_headers(self):
        # X-Request-Id on every answer to an inference request: success,
        # stream or error alike -- its ledger row's id: the client's own
        # X-Request-Id echoed exactly, else a ULID (telemetry.py)
        rec = getattr(self, "_record", None)
        if rec is not None:
            self.send_header(T.HEADER, rec.id)
        super().end_headers()

    def do_POST(self):
        _CONN.sock, _CONN.done = self.connection, threading.Event()
        _CONN.headers = self.headers
        route = INFERENCE.get(urlparse(self.path).path.rstrip("/"))
        rec = None
        if route is not None:
            api, events = route
            # the Messages stream always carries progress; an OpenAI shape
            # only for a client that names itself; Ollama never (INFERENCE)
            rec = T.Request(api, self.headers, progress=bool(
                events or (events is not None
                           and self.headers.get("X-Client"))))
        self._record = _CONN.record = rec
        try:
            self._guarded(self._post)
        finally:
            _CONN.done.set()
            _CONN.sock = None
            _CONN.record = None
            _CONN.headers = None
            self._record = None
            if rec is not None:
                rec.close()        # one ledger row, at the last byte

    def _guarded(self, fn) -> None:
        """Any error a route did not answer itself is a 500 with a body --
        never a dropped connection and a traceback on stderr only."""
        try:
            if self._gated() or self._refused_browser():
                return
            fn()
        except (BrokenPipeError, ConnectionResetError):
            pass
        # HTTP handler top level: any failure is a 500 with a body (logged)
        except Exception as e:
            logger.exception("%s %s failed", self.command, self.path)
            try:
                self._error(O.ApiError(500, f"{type(e).__name__}: {e}"))
            except OSError:
                pass    # the client is gone; the failure is logged above

    def _get(self):
        u = urlparse(self.path)
        path = u.path.rstrip("/") or "/"
        if path == "/v1/models":
            from knurlogic.machine.artifact import context_length, sampling_defaults
            path = self.app.scheduler.host.path
            from knurlogic.engine.model import thinking as TH
            try:
                think = TH.levels(TH.template_of(path)) if path else None
            except (ImportError, OSError, *TEMPLATE_ERRORS):
                think = None
            return self._json(200, O.models_document(
                self.app.served(),
                sampling_defaults(path) if path else {},
                context_length(path) if path else 0, think,
                load_state=_load_state(self.app.scheduler.host),
                memory_short=_memory_short(self.app.scheduler)))
        if path in ("/api/tags", "/api/version"):
            return self._ollama_get(path)
        if path == "/health":
            return self._json(200, {"status": "ok", "server": "knurlogic",
                                    "model": self.app.scheduler.host.state})
        if path == "/v1/usage":
            return self._usage(parse_qs(u.query))
        if path == "/v1/residency" and self.app.residency:
            return self._json(200, self.app.residency())
        if path == "/v1/prompt-cache":
            return self._list_prompt_cache()
        h = self.app.routes.get(path)
        if h is None:
            return self._json(404, {"error": {"message": f"no route {path}",
                                              "type": "not_found"}})
        body, ctype = h(parse_qs(u.query), self.app.requests)
        self._send(200, body, ctype)

    def _loopback(self) -> bool:
        import ipaddress
        try:
            return ipaddress.ip_address(
                self.client_address[0].split("%")[0]).is_loopback
        except ValueError:
            return False

    def _ollama_get(self, path: str) -> None:
        from knurlogic import __version__
        from knurlogic.interfaces.http import ollama
        if path == "/api/version":
            return self._json(200, {"version": __version__})
        return self._json(200, ollama.tags_document(
            self.app.served(), self.app.scheduler.host.path))

    def _ollama_show(self) -> None:
        from knurlogic.engine.model import thinking as TH
        from knurlogic.interfaces.http import ollama
        from knurlogic.machine.artifact import context_length
        path = self.app.scheduler.host.path
        try:
            think = TH.levels(TH.template_of(path)) if path else None
        except (ImportError, OSError, *TEMPLATE_ERRORS):
            think = None
        return self._json(200, ollama.show_document(
            self.app.served(), path, context_length(path) if path else 0,
            think))

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
        if path == "/v1/responses":
            return self._raw(self.app.responses, raw)
        if path == "/api/chat":
            return self._raw(self.app.ollama_chat, raw)
        if path == "/api/generate":
            return self._raw(self.app.ollama_generate, raw)
        if path == "/api/show":
            return self._ollama_show()
        if path == "/v1/messages/count_tokens":
            return self._count_tokens(raw)
        if path == "/v1/prompt-cache/save":
            return self._save_prompt_cache(raw)
        if path == "/v1/prompt-cache/drop":
            return self._drop_prompt_cache(raw)
        if path == "/v1/prompt-cache/pin":
            return self._pin_prompt_cache(raw)
        if path == "/v1/prompt-cache/park":
            return self._park_prompt_cache(raw)
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
        if chat:
            return self._chat(body)
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

    def _chat(self, body) -> None:
        try:
            kind, val = self.app.chat(body)
        except O.ApiError as e:
            return self._error(e)
        if kind == "json":
            return self._json(200, val, self._headers())
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self._cors()
        for k, v in self._headers().items():
            self.send_header(k, v)
        self.end_headers()
        self.close_connection = True
        try:
            for chunk in val:
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            val.close()            # cancels the Job: the row is freed

    def _count_tokens(self, raw: bytes) -> None:
        """POST /v1/messages/count_tokens: what the prompt would be, in the
        served model's tokens, rendered by its own template -- Claude Code
        asks this to manage its context. Image blocks are left out (their
        cost is known only once encoded), so with images it is a floor."""
        from knurlogic.engine.runtime import prompt as P
        from knurlogic.interfaces.http import messages as M
        try:
            req = json.loads(raw or b"{}")
            oai = M.to_openai(req if isinstance(req, dict) else {})
        except ValueError:
            return self._error(O.ApiError(400, "body must be JSON"))
        if getattr(self.app.scheduler.host, "tokenizer", None) is None:
            return self._error(O.ApiError(503, "no model is loaded",
                                          type_="server_error",
                                          retry_after=5))
        try:
            # as the model would see it: resent compactions folded in
            from knurlogic.context_management import compaction as C
            from knurlogic.context_management import context_edits as E
            msgs, _ = E.view(oai.get("messages") or [],
                             C.settings()["keep"])
            n = self.app.count(msgs, oai.get("tools"))
        except P.PromptError as e:
            return self._error(O.ApiError(400, str(e)))
        return self._json(200, {"input_tokens": n})

    def _raw(self, handler, raw: bytes) -> None:
        """A handler that writes its own response (an event stream has no
        length to declare, so it ends when the socket does)."""
        def start(code, ctype, headers=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            for k, v in (headers or {}).items():
                self.send_header(k, v)
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
    # my-mac.attacker.example would pass that, and a name an attacker
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
