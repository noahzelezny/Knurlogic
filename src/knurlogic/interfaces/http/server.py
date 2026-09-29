"""knurlogic's own HTTP server (docs/SERVER.md): stdlib ThreadingHTTPServer,
one thread per connection, every model operation handed to the scheduler.

  POST /v1/chat/completions  /chat/completions  /v1/completions   openai.py
  POST /v1/messages          Anthropic, in-process over openai.py
  GET  /v1/models            the served model, capabilities and size
  GET  /v1/residency         what is loaded, its state and memory
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
        from knurlogic.interfaces.http import messages
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

    def has_vision(self) -> bool:
        """Whether an image request is refused up front. Until the model
        has loaded, vision is not known yet: the request queues like a text
        one, and the scheduler refuses it at admission if the loaded model
        has none (M4 2026-09-28: an image sent during the 27B's load got a
        400 "no vision" while a text request waited and was served)."""
        from knurlogic.engine.serve import state
        if getattr(self.scheduler.host, "state", "ready") != "ready":
            return True
        return state.VISION.get("serve") is not None

    def _count(self) -> None:
        """One more request served (the handler threads race otherwise)."""
        with self._count_lock:
            self.requests += 1

    def submit(self, body: dict, chat: bool, extra: dict = None):
        from knurlogic.machine.artifact import sampling_defaults
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

    def _generate(self, body: dict) -> dict:
        """One non-streaming completion, in-process (the summary pass).
        A template with no way to turn thinking off is asked again
        without the ask."""
        try:
            job, reply = self.submit(body, chat=True)
        except O.ApiError:
            if "reasoning_effort" not in body:
                raise
            body = {k: v for k, v in body.items() if k != "reasoning_effort"}
            job, reply = self.submit(body, chat=True)
        first = reply.first()
        if first[0] == "error":
            raise O._status_of(first[1])
        return reply.complete(first)

    def _warm(self, body: dict) -> None:
        """Prefill a compacted prompt so it is in the prompt cache before
        the client's next turn asks for it (pause_after_compaction: no
        continuation prefilled it). One token; nobody waits on it."""
        def run():
            try:
                job, reply = self.submit(dict(body, max_tokens=1,
                                              stream=False), chat=True)
                reply.complete(reply.first())
            except Exception as e:
                logger.debug("warming the compacted prompt: %s", e)
        threading.Thread(target=run, daemon=True).start()

    def chat(self, body: dict):
        """A chat request through context management, then the engine:
        ("json", completion) or ("stream", SSE byte iterator); ApiError to
        refuse. A request that needs a summary pass and streams is
        answered 200 at once and kept alive while the summary is written;
        a later failure is an error event."""
        from knurlogic.context_management import compaction as C
        from knurlogic.context_management import context_edits as E
        if not isinstance(body, dict):
            raise O.ApiError(400, "the request body must be a JSON object")
        try:
            run, out, pending = C.prepare(
                body, count=self.count,
                window=self.window() if (body.get("context_management")
                                         or C.settings()["auto"]) else 0)
        except E.EditError as e:
            raise O.ApiError(400, str(e), param="context_management")
        except Exception as e:
            # a template that cannot render this history is the engine's
            # to refuse, as it would without context management
            if not isinstance(e, ValueError):
                raise
            run, out, pending = dict(body), C.Outcome(), None
            run.pop("context_management", None)

        def extra():
            return {"compaction": out.compaction, "applied": out.applied,
                    "iteration": out.iteration}

        def pause_doc():
            import time
            import uuid
            self._warm(run2[0])
            return C.paused(out, id_=f"chatcmpl-{uuid.uuid4().hex}",
                            created=int(time.time()),
                            model=self.served().get("id", "")
                            or body.get("model") or "default",
                            context={"tokens": 0, "window": self.window()})

        run2 = [run]
        if pending is not None and not body.get("stream"):
            run2[0] = C.summarize(run, pending, out, self._generate)
            if out.pause:
                return "json", pause_doc()
            pending = None
        if pending is None:
            job, reply = self.submit(run2[0], chat=True, extra=extra())
            first = reply.first()
            if first[0] == "error":
                raise O._status_of(first[1])
            if not reply.ctx["stream"]:
                return "json", reply.complete(first)

            def events():
                try:
                    yield from reply.events(first)
                finally:
                    job.cancel()         # abandoned mid-stream: free the row
            return "stream", events()

        def summarized():
            box = {}

            def work():
                try:
                    box["run"] = C.summarize(run, pending, out,
                                             self._generate)
                except BaseException as e:  # any end of the worker becomes the reply
                    box["error"] = e
            t = threading.Thread(target=work, daemon=True)
            t.start()
            while t.is_alive():
                t.join(5)
                if t.is_alive():
                    yield b": keepalive compaction\n\n"
            if "error" in box:
                yield O._data(O._status_of(box["error"]).body())
                yield b"data: [DONE]\n\n"
                return
            run2[0] = box["run"]
            if out.pause:
                doc = pause_doc()
                msg = doc["choices"][0]["message"]
                base = {"id": doc["id"], "object": "chat.completion.chunk",
                        "created": doc["created"], "model": doc["model"]}
                yield O._data(dict(base, choices=[{
                    "index": 0, "finish_reason": None,
                    "delta": {"role": "assistant",
                              "compaction": msg["compaction"]}}]))
                yield O._data(dict(base, choices=[],
                                   context_management=doc[
                                       "context_management"]))
                yield O._data(dict(base, choices=[{
                    "index": 0, "finish_reason": "compaction",
                    "delta": {}}]))
                if (body.get("stream_options") or {}).get("include_usage"):
                    yield O._data(dict(base, choices=[],
                                       usage=doc["usage"]))
                yield b"data: [DONE]\n\n"
                return
            try:
                job, reply = self.submit(run2[0], chat=True, extra=extra())
            except O.ApiError as e:
                yield O._data(e.body())
                yield b"data: [DONE]\n\n"
                return
            try:
                first = reply.first()
                if first[0] == "error":
                    yield O._data(O._status_of(first[1]).body())
                    yield b"data: [DONE]\n\n"
                    return
                yield from reply.events(first)
            finally:
                job.cancel()
        return "stream", summarized()


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
                pass    # the client is gone; the failure is logged above

    def _get(self):
        u = urlparse(self.path)
        path = u.path.rstrip("/") or "/"
        if path == "/v1/models":
            from knurlogic.machine.artifact import (context_length,
                                                    sampling_defaults)
            path = self.app.scheduler.host.path
            from knurlogic.engine.serve import thinking as TH
            try:
                think = TH.levels(TH.template_of(path)) if path else None
            except Exception:
                think = None
            return self._json(200, O.models_document(
                self.app.served(),
                sampling_defaults(path) if path else {},
                context_length(path) if path else 0, think))
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
