"""Reading and changing another server's settings through the page: GET
/peek (a running model's or a peer page's read-only documents), POST
/apply (a running model's live knobs), and the peer side of both for a
model server on this machine (`peer_settings`)."""

from __future__ import annotations

import http.client
import json
from urllib.parse import urlparse

from knurlogic.interfaces.page import nodes, peers, router
from knurlogic.machine.servers import registry

#: a model server's settings document, readable (/peek: a Read message) and
#: changeable (/apply: a Settings message) on a peer through that peer page
PEER_SETTINGS = "/settings.json"


def peer_settings(method: str, port, data, call=None) -> tuple:
    """A Settings (POST, `data` the knobs) or a Read with a port (GET,
    `data` the query), on the machine running the model, for a peer page's
    Settings (the caller has passed the peer gate): (status, doc). Only a
    server this machine started (the registry, still ours) -- never
    another port -- and only its /settings.json: GET reads it (tune and
    working_set_gib passed on), POST is its live apply, which refuses and
    reports per knob exactly as it does for this machine's own page. It is
    how a peer's live knob changes the peer, not this page."""
    import urllib.error
    import urllib.parse
    import urllib.request

    from knurlogic.machine.servers import is_our_server
    if not isinstance(port, int) or isinstance(port, bool):
        return 400, {"error": "name the model server by its port"}
    rec = registry().get(port)
    if not rec or not is_our_server(int(rec["pid"])):
        return 404, {"error": f"no model server this machine started on "
                              f"port {port}"}
    url = f"http://127.0.0.1:{port}{PEER_SETTINGS}"
    if method == "GET":
        q = data if isinstance(data, dict) else {}
        fwd = {k: str(q[k]) for k in PEEK_KEYS if q.get(k) is not None}
        if fwd:
            url += "?" + urllib.parse.urlencode(fwd)
        data = None
    else:
        if not isinstance(data, dict):
            return 400, {"error": "the body must be a JSON object of knobs"}
        data = json.dumps(data).encode()
    if call is None:
        def call(u, d, t):
            req = urllib.request.Request(
                u, data=d, method="POST" if d is not None else "GET",
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=t) as r:
                    return r.status, r.read()
            except urllib.error.HTTPError as e:
                return e.code, e.read()
    try:
        code, raw = call(url, data, APPLY_S)
        return code, json.loads(raw)
    except (OSError, ValueError, http.client.HTTPException) as e:
        return 502, {"error": f"{type(e).__name__}: {e}"}


#: the largest settings change `/apply` forwards; a knob set is a few bytes
APPLY_MAX = 16 << 10


APPLY_S = 10.0


def apply_settings(where: str, body: bytes, post=None) -> tuple:
    """POST /apply?where=<base>: forward a live-knob change to a running
    model server's own POST /settings.json and hand back its per-knob report.

    Only a knurlogic model server this page knows (chat_targets) -- never a
    peer's page, never another path -- a small JSON object, a short
    deadline. The page sends it only when the person clicks Apply; which
    knobs actually move is the model server's to decide and report."""
    import urllib.error
    import urllib.request
    base = (where or "").rstrip("/")
    if not router.known_target(base):
        return 403, {"error": f"not a running model this page knows: "
                              f"{base or '(none)'}"}
    if len(body or b"") > APPLY_MAX:
        return 413, {"error": f"a settings change is at most {APPLY_MAX} "
                              f"bytes"}
    try:
        want = json.loads(body or b"")
    except ValueError:
        want = None
    if not isinstance(want, dict):
        return 400, {"error": "the body must be a JSON object of knobs"}
    target = peers.PEER_TARGETS.get(base)
    if target:
        # a peer's model through that peer's page (a Settings message): its
        # server listens on the peer's loopback, not at `base`
        from knurlogic.cluster import transport
        try:
            out = (post or transport.send)(
                target["relay"].removeprefix("http://"), "Settings",
                {"port": urlparse(base).port, "values": want})
        except (OSError, ValueError, http.client.HTTPException) as e:
            return 502, {"error": f"{type(e).__name__}: {e}"}
        return (502 if out.get("error") and "applied" not in out
                else 200), out
    if post is None:
        def post(url, data, t):
            req = urllib.request.Request(
                url, data=data, method="POST",
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=t) as r:
                    return r.status, r.read()
            except urllib.error.HTTPError as e:
                return e.code, e.read()
    try:
        code, raw = post(base + PEER_SETTINGS,
                         json.dumps(want).encode(), APPLY_S)
        return code, json.loads(raw)      # JSON only, never an HTML page
    except (OSError, ValueError, http.client.HTTPException) as e:
        return 502, {"error": f"{type(e).__name__}: {e}"}


#: What `/peek` may read, and the query keys it passes along. Reads only:
#: a running model's settings and sampling defaults, a peer page's machine
#: settings. Nothing that changes anything is reachable through it.
PEEK_PATHS = ("/settings.json", "/v1/models", "/models.json")


PEEK_KEYS = ("tune", "working_set_gib", "wired_gib")


#: keys passed along for one path only: `rescan=1` reads a peer's model
#: folders again -- its picker was opened, never a poll
PEEK_PATH_KEYS = {"/models.json": ("rescan",),
                  # a peer's preview of a model it holds, named by identity
                  # (resolved there, from its own stores): its room is
                  # counted against ITS memory free now
                  "/settings.json": ("identity", "name", "kv_bits",
                                     "long_context")}


def peek_keys(path: str) -> tuple:
    return PEEK_KEYS + PEEK_PATH_KEYS.get(path, ())


PEEK_S = 3.0


def peek_targets() -> set:
    """Addresses `/peek` may read from: the running models the chat proxy
    already allows, and the pages of peers that are answering -- machines
    this page polls anyway, never an address taken from the request."""
    out = set(router.chat_targets())
    for p in (nodes.PEERS.all() if nodes.PEERS else []):
        if p.state == "answering":
            out.add(f"http://{p.key}")
    return out


def _peek_peer(where: str, path: str, fwd: dict):
    """A read of a peer's document, sent as a Read message: a peer PAGE's
    own documents, or -- with its port -- one of its model servers'
    /settings.json. None when `where` is not a peer's (a model's /v1/models
    goes through the peer's relay, a local one direct)."""
    from knurlogic.cluster import transport
    t = peers.PEER_TARGETS.get(where)
    if t and path == PEER_SETTINGS:
        page, port = t["relay"].removeprefix("http://"), urlparse(where).port
    elif where in {f"http://{p.key}"
                   for p in (nodes.PEERS.all() if nodes.PEERS else [])}:
        page, port = where.removeprefix("http://"), None
    else:
        return None
    try:
        out = transport.send(page, "Read", {"path": path, "query": fwd,
                                            **({"port": port} if port
                                               else {})})
    except (OSError, ValueError, http.client.HTTPException) as e:
        return 502, json.dumps({"error": f"{type(e).__name__}: {e}"})
    return 200, json.dumps(out)


def peek(q: dict, fetch=None) -> tuple:
    """GET /peek?where=<base>&path=<path>: another server's read-only
    document, for a page that cannot call another port or machine itself
    (a peer's, as a Read message).

    (status, body): the upstream's JSON as it came, or a JSON error. GET
    only, a fixed list of paths and targets, a short deadline: Settings on
    a peer's tab reads that peer and can never set anything on it."""
    import urllib.parse
    import urllib.request
    where = ((q.get("where") or [""])[0] or "").rstrip("/")
    path = (q.get("path") or [""])[0]
    if path not in PEEK_PATHS:
        return 403, json.dumps({"error": f"not a readable path: {path!r}"})
    if where not in peek_targets():
        return 403, json.dumps({"error": f"not a server this page knows: "
                                         f"{where or '(none)'}"})
    fwd = {k: q[k][0] for k in peek_keys(path) if q.get(k)}
    if fetch is None:
        peer = _peek_peer(where, path, fwd)
        if peer is not None:
            return peer
    url = peers.upstream(where, path)
    if fwd:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(fwd)
    if fetch is None:
        def fetch(u, t):
            with urllib.request.urlopen(u, timeout=t) as r:
                return r.read()
    try:
        body = fetch(url, PEEK_S)
        json.loads(body)          # pass on JSON only, never an HTML page
        return 200, body.decode() if isinstance(body, bytes) else body
    except (OSError, ValueError, http.client.HTTPException) as e:
        return 502, json.dumps({"error": f"{type(e).__name__}: {e}"})
