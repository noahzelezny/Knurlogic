"""The page's router to model servers: one address for every model here and
on peers. POST /v1/messages, /v1/chat/completions and count_tokens by
`model` (`route`), the page's chat (`proxy_chat`), GET /v1/models
(`route_models_document`), and the streaming that carries the answer back
(`_stream`, with a cluster job's failure said in its words)."""

from __future__ import annotations

import http.client
import json
import logging
import time
from urllib.parse import urlparse

from knurlogic.interfaces.page import nodes, peers
from knurlogic.machine.servers import registry

logger = logging.getLogger(__name__)


def refresh_targets() -> None:
    """Ask the peers what they serve NOW and forget the router's cached
    table: after a launch, and once before refusing a model this page does
    not know, so a chat to a job launched a moment ago is not refused until
    the page's next survey."""
    _ROUTES["at"] = 0.0
    if nodes.PEERS is not None:
        try:
            peers.peer_residency(nodes.PEERS)
        except Exception:  # peer survey (logged)
            logger.debug("peer survey failed", exc_info=True)


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
    out.update(peers._PEER_TARGETS)
    return out


def _send_json(handler, code: int, doc) -> None:
    from knurlogic.interfaces.http import telemetry as T
    out = json.dumps(doc).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(out)))
    rid = T.id_of(getattr(handler, "headers", None))
    if rid:
        handler.send_header(T.HEADER, rid)
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
    t = peers._PEER_TARGETS.get(base)
    if t is not None:
        job = t.get("job")
        if not job:
            return ""
        refresh_targets()
        e = peers._PEER_JOBS.get(job) or {}
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

    from knurlogic.interfaces.http import telemetry as T
    rid = T.id_of(getattr(handler, "headers", None))
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json",
                                          **({T.HEADER: rid} if rid else {}),
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
    echo = T.valid_id(up.headers.get(T.HEADER)) or rid
    if echo:
        handler.send_header(T.HEADER, echo)
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
    _stream(handler, peers.upstream(base, "/v1/chat/completions"), body,
            base=base)


#: The paths the page's router forwards by `model`: the two chat surfaces
#: a client pointed at this page (Claude Code, an OpenAI SDK) uses, and
#: Claude Code's token count (it names the model too).
ROUTE_PATHS = ("/v1/messages", "/v1/chat/completions",
               "/v1/messages/count_tokens")


ROUTE_S = 2.0


_ROUTES: dict = {"at": 0.0, "map": {}, "docs": {}}


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
        if nodes.PEERS is not None and now - peers._PEER_AT[0] > PEER_SURVEY_MAX_AGE_S:
            try:
                peers.peer_residency(nodes.PEERS)
            except Exception:  # peer survey (logged)
                logger.debug("peer survey failed", exc_info=True)
    found: dict = {}
    docs: dict = {}

    def one(base):
        try:
            for m in fetch(peers.upstream(base, "/v1/models"),
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
        t = peers._PEER_TARGETS.get(b.rstrip("/"))
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
    _stream(handler, peers.upstream(base, path), body, base=base)


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
