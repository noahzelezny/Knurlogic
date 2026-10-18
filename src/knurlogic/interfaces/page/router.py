"""The page's router to model servers: one address for every model here and
on peers. POST /v1/messages, /v1/chat/completions and count_tokens by
`model` (`route`; a model on two Macs is two instances, picked by
`<id>@<machine>`, X-Knurlogic-Machine or the session: `pick`), the page's
chat (`proxy_chat`), GET /v1/models (`route_models_document`), and the
streaming that carries the answer back (`stream`, with a cluster job's
failure said in its words).

Errors: the page's own endpoints answer {"error": "<why>"}; a route that
forwards an API (/v1/messages, /v1/chat/completions, the prompt cache)
answers in that API's shape -- api_error, which both the Anthropic and the
OpenAI clients read, or the model server's own document passed on."""

from __future__ import annotations

import json
import logging
import time
from urllib.parse import urlparse

from knurlogic.cluster import transport
from knurlogic.interfaces.http.guard import hung_up
from knurlogic.interfaces.http.respond import send_json
from knurlogic.interfaces.page import nodes, peers
from knurlogic.machine.http_client import NET_ERRORS, open_url, request
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
    out.update(peers.PEER_TARGETS)
    return out


def cluster_failure(base: str) -> str:
    """Why the model at `base` is a cluster job that cannot answer, or ""
    when it is not one. A peer's job is looked up in a fresh survey (its
    page reports the jobs that ended, with why); this machine's by port."""
    base = (base or "").rstrip("/")
    t = peers.PEER_TARGETS.get(base)
    if t is not None:
        job = t.get("job")
        if not job:
            return ""
        refresh_targets()
        e = peers.PEER_JOBS.get(job) or {}
        if e.get("phase") == "stopped" and e.get("reason"):
            return e["reason"]
        return (f"rank 0 of cluster job {job} on {t.get('machine')} "
                f"dropped the connection; the job is failing")
    try:
        from knurlogic.cluster import watching
        return watching.failure_of_port(urlparse(base).port)
    except (OSError, ValueError, TypeError, AttributeError):
        return ""


def api_error(status_type: str, message: str, **extra) -> dict:
    """An error on a forwarded API route, in the shape both the Anthropic
    and the OpenAI clients read as one (error.type, error.message)."""
    return {"type": "error", "error": {"type": status_type,
                                       "message": message}, **extra}


def cluster_failed(reason: str) -> dict:
    """The OpenAI-style error a rank 0 answers with when its ring fails
    (http/openai.py status_of), for a job that could not answer at all."""
    return {"error": {"message": f"this model is split across machines and "
                                 f"its cluster job failed: {reason}",
                      "type": "server_error", "param": None,
                      "code": "cluster_failed"}}


def stream(handler, url: str, body: bytes, timeout: float = 3600,
            base: str = "") -> None:
    """POST `body` to `url` and pass the answer back as it arrives, byte for
    byte: SSE, prefill keepalives and all. The upstream's status and
    Content-Type go with it; an upstream that cannot be reached is a 502 --
    or a 503 `cluster_failed` with the job's stop reason when `base` is a
    cluster job's rank 0 (dead, or its job stopped). When such a rank 0
    dies mid-way through an event stream, one `data: {"error": ...
    cluster_failed}` event goes out before the stream closes."""
    from knurlogic.interfaces.http import telemetry as T
    rid = T.id_of(getattr(handler, "headers", None))
    try:
        up = open_url(url, body, "POST", timeout,
                      {**({T.HEADER: rid} if rid else {}),
                       **_client_headers(handler)})
        code = up.status
        ctype = up.headers.get("Content-Type", "application/json")
    except NET_ERRORS as e:
        why = cluster_failure(base) if base else ""
        if why:
            send_json(handler, 503, cluster_failed(why))
        else:
            send_json(handler, 502, api_error("api_error",
                                              f"{type(e).__name__}: {e}"))
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
            except NET_ERRORS:
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


def send_up(url: str, method: str, body: bytes | None):
    """(status, JSON doc) of one request to a model server; an error of
    its own is in the API's shape (`api_error`)."""
    try:
        code, raw = request(url, body, method, timeout=600)
    except NET_ERRORS as e:
        return 502, api_error("api_error", f"{type(e).__name__}: {e}")
    try:
        return code, json.loads(raw or b"{}")
    except ValueError as e:
        if code >= 400:
            return code, api_error("api_error", f"HTTP Error {code}")
        return 502, api_error("api_error", f"{type(e).__name__}: {e}")


def proxy_chat(handler, where: str, body: bytes) -> None:
    """POST /chat?where=<base>: forward a chat request to a running model
    and stream its answer back as it arrives.

    The control page serves no model, so its chat has to reach the one the
    person clicked -- a server `load` started -- and a browser
    will not let a page on this port call another port directly."""
    base = (where or "").rstrip("/")
    if not known_target(base):
        send_json(handler, 403, {"error": f"not a running model this page "
                                           f"knows: {base or '(none)'}"})
        return
    stream(handler, peers.upstream(base, "/v1/chat/completions"), body,
            base=base)


#: The paths the page's router forwards by `model`: the two chat surfaces
#: a client pointed at this page (Claude Code, an OpenAI SDK) uses, and
#: Claude Code's token count (it names the model too).
ROUTE_PATHS = ("/v1/messages", "/v1/chat/completions",
               "/v1/messages/count_tokens")


ROUTE_S = 2.0


_ROUTES: dict = {"at": 0.0, "all": {}, "docs": {}}


#: how old that may be before the router asks again itself
PEER_SURVEY_MAX_AGE_S = 30.0


def fetch_json(url: str, timeout: float):
    """A server's JSON document; raises when it does not answer one."""
    return json.loads(request(url, timeout=timeout)[1])


def instances_at(bases, fetch=None, docs: dict | None = None) -> dict:
    """{model id: [base, ...]} over `bases`, each asked its /v1/models --
    the id a client names is the one the server answers to -- in parallel
    with one deadline (ROUTE_S); one that does not answer is left out.
    Every server of an id, in a fixed order (_ordered): the same model on
    two Macs is two instances, never whichever answered first. `docs`,
    when given, is filled with each model's own /v1/models entry
    (sampling_defaults, thinking, context_length)."""
    fetch = fetch or fetch_json
    found: dict = {}

    def one(base):
        try:
            for m in fetch(peers.upstream(base, "/v1/models"),
                           ROUTE_S).get("data") or []:
                if isinstance(m, dict) and m.get("id"):
                    found.setdefault(str(m["id"]), set()).add(base)
                    if docs is not None:
                        docs.setdefault(str(m["id"]), m)
        # one server's silence is logged, the others are still asked
        except Exception:
            logger.debug("no model list from %s", base, exc_info=True)
    transport.parallel(one, sorted(bases), deadline=ROUTE_S)
    return {m: _ordered(bs) for m, bs in found.items()}


def models_at(bases, fetch=None, docs: dict | None = None) -> dict:
    """{model id: base}: each id's first instance (instances_at)."""
    return {m: bs[0] for m, bs in instances_at(bases, fetch, docs).items()}


def _ordered(bases) -> list:
    """This machine's servers first (no relay hop), then each peer's, by
    machine and address: a stable choice, the same every time."""
    def key(b):
        t = peers.PEER_TARGETS.get(b.rstrip("/"))
        return (t is not None, (t or {}).get("machine") or "", b)
    return sorted(bases, key=key)


def machine_of(base: str) -> tuple:
    """(name, node id) of the machine serving at `base`: this one, or the
    peer that reported it (its id from the page's peer list; "" unknown)."""
    from knurlogic.machine.identity import identity
    t = peers.PEER_TARGETS.get(base.rstrip("/"))
    if t is None:
        me = identity()
        return me.get("name") or "", me.get("id") or ""
    name = t.get("machine") or ""
    known = nodes.PEERS.all() if nodes.PEERS is not None else []
    pid = next((p.id for p in known if p.name == name and p.id), "")
    return name, pid


def instances(fetch=None, ttl: float = 5.0) -> dict:
    """{model id: [base, ...]} for every running knurlogic server this
    page knows (chat_targets: its own and the ones peers reported), by
    instances_at. Cached for a few seconds: Claude Code sends several
    requests a turn."""
    now = time.time()
    if fetch is None:
        if now - _ROUTES["at"] < ttl:
            return {m: list(bs) for m, bs in _ROUTES["all"].items()}
        # the page's own polling refreshes what peers serve, but a client
        # (Claude Code) may call before anyone has opened the page: the
        # router then saw only this machine
        if nodes.PEERS is not None and now - peers.PEER_AT[0] > PEER_SURVEY_MAX_AGE_S:
            try:
                peers.peer_residency(nodes.PEERS)
            except Exception:  # peer survey (logged)
                logger.debug("peer survey failed", exc_info=True)
    docs: dict = {}
    out = instances_at(chat_targets(), fetch, docs)
    _ROUTES.update(at=now, all=out, docs=docs)
    return out


def routable(fetch=None, ttl: float = 5.0) -> dict:
    """{model id: base}: each running model's default instance (the first
    of instances())."""
    return {m: bs[0] for m, bs in instances(fetch, ttl).items()}


def route_models_document(fetch=None) -> dict:
    """GET /v1/models on the page: every model its router can reach, each
    with its server's own entry (sampling_defaults, thinking,
    context_length: what the page's chat reads), the machine that answers
    for it by default (a split model's rank 0), and every instance: a
    model on two Macs is named `<id>@<node id>` to reach one of them."""
    table = instances(fetch)
    docs = _ROUTES.get("docs") or {}

    def inst(m, b):
        name, mid = machine_of(b)
        return {"model": f"{m}@{mid}" if mid else m, "machine": name,
                "machine_id": mid, "server": b}
    out = []
    for m, bs in sorted(table.items()):
        name, mid = machine_of(bs[0])
        out.append(dict(docs.get(m) or {}, id=m, object="model",
                        owned_by="knurlogic", server=bs[0], machine=name,
                        machine_id=mid, instances=[inst(m, b) for b in bs]))
    return {"object": "list", "data": out}


#: the instance each session was last served by, {session: base}: its
#: prompt cache is there, so its next request goes there too. Bounded
#: (oldest forgotten first).
_AFFINITY: dict = {}
AFFINITY_MAX = 4096
#: the header that names an instance's machine, beside `<id>@<machine>`
MACHINE_HEADER = "X-Knurlogic-Machine"


def split_instance(model) -> tuple:
    """(model id, machine) of a `model` field: `<id>@<machine>` names an
    instance (node id or machine name); a plain id names none."""
    if isinstance(model, str) and "@" in model:
        mid, _, machine = model.rpartition("@")
        if mid and machine:
            return mid, machine
    return model, ""


def pick(bases: list, machine: str = "", session: str = "") -> str | None:
    """The instance a request goes to: the one on `machine` (a node id or
    a machine name, any case) when it names one, None if no instance is
    there; else the one its session was served by last; else the first."""
    if machine:
        want = machine.lower()
        for b in bases:
            name, mid = machine_of(b)
            if want in (mid.lower(), name.lower()):
                return b
        return None
    last = _AFFINITY.get(session) if session else None
    return last if last in bases else bases[0] if bases else None


def _remember(session: str, base: str) -> None:
    if not session:
        return
    _AFFINITY.pop(session, None)
    _AFFINITY[session] = base
    while len(_AFFINITY) > AFFINITY_MAX:
        _AFFINITY.pop(next(iter(_AFFINITY)))


def route(handler, path: str, body: bytes, fetch=None) -> None:
    """POST /v1/messages or /v1/chat/completions on the page: forward to
    the running knurlogic server whose model id is the request's `model`.

    Claude Code takes ONE base URL and names a model per tier; each
    knurlogic server holds one model. So the page, which already knows
    every running server here and on its peers, is the one address that
    can hand each request to the server holding the model it names. Only
    to those servers -- the same allow-list the page's chat uses. A model
    running on two Macs: `<id>@<machine>` or the X-Knurlogic-Machine
    header picks one; a session stays on the one that served it."""
    try:
        doc = json.loads(body or b"{}")
    except ValueError:
        doc = None
    asked = doc.get("model") if isinstance(doc, dict) else None
    model, machine = split_instance(asked)
    machine = machine or (handler.headers.get(MACHINE_HEADER) or "").strip()
    session = (handler.headers.get("X-Client-Session") or "").strip()
    table = instances(fetch)
    base = pick(table.get(model) or [], machine, session) \
        if isinstance(model, str) else None
    if base is None:
        # launched a moment ago? ask the peers once more before refusing
        refresh_targets()
        table = instances(fetch, ttl=0)
        base = pick(table.get(model) or [], machine, session) \
            if isinstance(model, str) else None
    if base is None and isinstance(model, str) and fetch is None \
            and not machine:
        # a split job still starting: its rank 0 serves (and queues) only
        # once every rank has loaded, so until then nothing here answers
        # for the model. Wait for it, as a one-Mac server would queue.
        base, _ = _await_launch(handler, model, routable(fetch))
    if base is None:
        running = sorted(table)
        where = f" on {machine!r}" if machine else ""
        # say it in the shape either client reads as an error
        send_json(handler, 404, api_error(
            "not_found_error", f"no running model {model!r}{where}; "
                               f"running: {', '.join(running) or 'none'}",
            models=running))
        return
    _remember(session, base)
    if asked != model:            # `<id>@<machine>`: the server knows <id>
        body = json.dumps(dict(doc, model=model)).encode()
    stream(handler, peers.upstream(base, path), body, base=base)


#: How long a request waits for a split job still joining or loading; and,
#: once its ranks are loaded, for its rank 0 to answer (seconds)
LAUNCH_WAIT_S = 1800.0
READY_GRACE_S = 60.0


def _starting_jobs() -> tuple:
    """(any job joining or loading, any job loaded) among the split jobs
    with a rank here."""
    try:
        from knurlogic.cluster import watching
        js = watching.jobs_document()
    except (OSError, ValueError, KeyError, AttributeError, TypeError):
        return False, False
    ph = {j.get("phase") for j in js if isinstance(j, dict)}
    return bool(ph & {"joining", "loading"}), "ready" in ph


def _await_launch(handler, model: str, table: dict | None = None,
                  poll: float = 1.0) -> tuple:
    """(base, table) once `model` is served, or (None, table) when no
    split job here is starting, the wait runs out, or the client hangs
    up. A job counted ready gets READY_GRACE_S for its server to answer."""
    start = time.time()
    ready_since = None
    table = dict(table or {})
    while time.time() - start < LAUNCH_WAIT_S:
        starting, loaded = _starting_jobs()
        if not starting:
            if not loaded:
                break
            ready_since = ready_since or time.time()
            if time.time() - ready_since > READY_GRACE_S:
                break
        if hung_up(handler.connection):
            break
        time.sleep(poll)
        refresh_targets()
        table = routable(ttl=0)
        base = table.get(model)
        if base:
            return base, table
    return None, table


def local_models(fetch=None, docs=None) -> dict:
    """models_at over the servers THIS machine started (never the ones
    peers reported, so two pages relaying for each other cannot loop):
    what a peer page's relay resolves a model name against."""
    from knurlogic.machine.servers import is_our_server
    return models_at([f"http://127.0.0.1:{port}" for port, rec in
                      registry().items() if is_our_server(int(rec["pid"]))],
                     fetch, docs)


def requested_model(query: dict, body: bytes):
    """The model a request names: its `?model=` query, else its JSON
    body's "model" field; None when it names none."""
    model = (query.get("model") or [None])[0]
    if model is None and body:
        try:
            got = json.loads(body)
        except ValueError:
            return None
        model = got.get("model") if isinstance(got, dict) else None
    return model


def resolve(table: dict, model):
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
