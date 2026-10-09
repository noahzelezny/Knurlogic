"""The page's /v1/prompt-cache: forwarded to the model server (here, or a
peer's through its relay) that serves the named model. The model
server's own endpoints are interfaces/http/prompt_cache.py; this is the
page's routing to them.

Also the pieces of saved entries between machines that do not share model
storage (the bottom of this file): each page serves its own cache roots
(ENTRIES_PATH, PIECE_PATH), and the model servers it starts fetch what
they lack from the peers' pages (PageFetcher, installed for
engine/prompt_cache/transfer.py)."""

from __future__ import annotations

import json
import os
from urllib.parse import quote

from knurlogic.interfaces.page import peers, router

#: the model server's prompt-cache endpoints the page forwards
PROMPT_CACHE_PATH = "/v1/prompt-cache"


PROMPT_CACHE_POSTS = tuple(PROMPT_CACHE_PATH + p
                           for p in ("/save", "/drop", "/pin", "/park"))


def prompt_cache_forward(handler, method: str, path: str, query: dict,
                         body: bytes, fetch=None, send=None) -> None:
    """GET /v1/prompt-cache, POST /v1/prompt-cache/{save,drop,pin} on the
    page: forwarded to the model server on this machine that serves the
    model named by a "model" query or body field; with none named, the only
    one; several and none named is a 400 listing them. The loopback
    operator only (the model server refuses anyone else; the page forwards
    from loopback, so it must refuse first). `send`: (url, method, body)
    -> (status, doc), for tests."""
    import ipaddress
    try:
        loop = ipaddress.ip_address(
            handler.client_address[0].split("%")[0]).is_loopback
    except (ValueError, AttributeError, IndexError):
        loop = False
    if not loop:
        router.send_json(handler, 403, {"error": {
            "message": "the prompt cache is managed from this machine "
                       "(loopback) only", "type": "permission_error"}})
        return
    model = (query.get("model") or [None])[0]
    if model is None and body:
        try:
            got = json.loads(body)
            model = got.get("model") if isinstance(got, dict) else None
        except ValueError:
            model = None
    table = router.local_models(fetch)
    if model is not None:
        base = router.resolve(table, model)
        if base is None:
            # a peer's model: its page's relay, like a chat (the peer
            # resolves the name again against the servers it started)
            far = router.resolve(router.routable(fetch), model)
            if far is not None and peers.PEER_TARGETS.get(far) is not None:
                q = f"?model={quote(str(model))}" \
                    if method == "GET" else ""
                code, doc = (send or router.send_up)(
                    peers.upstream(far, path) + q, method,
                    body if method == "POST" else None)
                router.send_json(handler, code, doc)
                return
            here = sorted(set(table) | set(router.routable(fetch)))
            router.send_json(handler, 404, {"error": {
                "message": f"no running model {model!r} here or on a peer; "
                           f"running: {', '.join(here) or 'none'}",
                "type": "not_found"}, "models": here})
            return
    else:
        bases = set(table.values())
        if len(bases) != 1:
            router.send_json(handler, 400, {"error": {
                "message": "name the model (a \"model\" query or body "
                           "field): " + (", ".join(sorted(table))
                                         or "none is running"),
                "type": "invalid_request_error"}, "models": sorted(table)})
            return
        base = bases.pop()
    code, doc = (send or router.send_up)(base + path, method,
                                   body if method == "POST" else None)
    router.send_json(handler, code, doc)


# --- pieces of saved entries, between machines -----------------------------
# Machines that do not share model storage keep their own cache roots; a
# rank that lacks some files of an entry fetches them from the peer that
# holds them (engine/prompt_cache/transfer.py). Each page serves its own
# roots: a listing per model key, and one piece at a time. Read-only, and
# no name in a request is ever a path (transfer.piece_path).

#: ?key=<key id> -> transfer.listing: the entries held here
ENTRIES_PATH = PROMPT_CACHE_PATH + "/entries"
#: ?key=&entry=&layer=<slot index | draft | entry> -> the file's bytes
PIECE_PATH = PROMPT_CACHE_PATH + "/piece"
#: the answering peer pages, for a model server on this machine (loopback)
PEERS_PATH = PROMPT_CACHE_PATH + "/peers"
PIECE_PATHS = (ENTRIES_PATH, PIECE_PATH, PEERS_PATH)
#: set by a page for the model servers it starts: its own port, where
#: they ask for PEERS_PATH
PAGE_PORT_ENV = "KNURLOGIC_PAGE_PORT"


def _loopback(handler) -> bool:
    import ipaddress
    try:
        return ipaddress.ip_address(
            handler.client_address[0].split("%")[0]).is_loopback
    except (ValueError, AttributeError, IndexError):
        return False


def piece_route(handler, path: str, query: dict, gate=None, roots=None,
                manual=None) -> None:
    """GET ENTRIES_PATH / PIECE_PATH: the peer gate (no Origin; loopback,
    Thunderbolt or a --peer address), as every page-to-page route. GET
    PEERS_PATH: loopback only, no Origin. `roots` / `manual`: for tests."""
    from knurlogic.engine.prompt_cache import transfer
    if path == PEERS_PATH:
        if handler.headers.get("Origin") is not None or \
                not _loopback(handler):
            handler.close_connection = True
            router.send_json(handler, 403, {"error": "the peer list is for "
                                            "this machine's model servers"})
            return
        router.send_json(handler, 200, {"pages": sorted(peers.peer_pages())})
        return
    refused = peers.peer_refusal(
        handler.headers, handler.client_address[0],
        handler.connection.getsockname()[0], gate=gate,
        manual_hosts=peers.manual_hosts() if manual is None else manual,
        what="prompt cache pieces")
    if refused:
        handler.close_connection = True
        router.send_json(handler, *refused)
        return

    def one(name):
        v = query.get(name) or []
        return v[0] if len(v) == 1 else None
    key = one("key")
    if not transfer.is_key_id(key):
        router.send_json(handler, 400, {"error": "key is a 16-hex key id"})
        return
    if path == ENTRIES_PATH:
        router.send_json(handler, 200, {"key": key, "entries":
                                        transfer.listing(key, roots)})
        return
    entry, piece = one("entry"), transfer.parse_piece(one("layer"))
    if piece is None or entry is None:
        router.send_json(handler, 400, {"error": "entry is a tokens hash, "
                                        "layer a slot index, draft or "
                                        "entry"})
        return
    f = transfer.piece_path(key, entry, piece, roots)
    if f is None:
        router.send_json(handler, 404, {"error": "no such piece here"})
        return
    import shutil
    try:
        fh = open(f, "rb")
    except OSError:
        router.send_json(handler, 404, {"error": "no such piece here"})
        return
    with fh:
        handler.send_response(200)
        handler.send_header("Content-Type", "application/octet-stream")
        handler.send_header("Content-Length",
                            str(os.fstat(fh.fileno()).st_size))
        handler.end_headers()
        try:
            shutil.copyfileobj(fh, handler.wfile, 1 << 20)
        except (BrokenPipeError, ConnectionResetError):
            pass


class PageFetcher:
    """engine/prompt_cache/transfer's fetcher over the pages: the peers
    are the pages answering this machine's page; each is asked for its
    listing and its pieces directly."""

    def __init__(self, page: str, timeout: float = 30.0):
        self.page = page.rstrip("/")
        self.timeout = timeout

    def _open(self, url: str):
        import urllib.request
        return urllib.request.urlopen(url, timeout=self.timeout)

    def peers(self) -> list:
        with self._open(self.page + PEERS_PATH) as r:
            got = json.loads(r.read() or b"{}")
        return [p for p in (got.get("pages") or [])
                if isinstance(p, str) and p.startswith("http://")]

    def listing(self, peer: str, key_id: str):
        with self._open(f"{peer}{ENTRIES_PATH}?key={quote(key_id)}") as r:
            got = json.loads(r.read() or b"{}")
        es = got.get("entries") if isinstance(got, dict) else None
        return es if isinstance(es, dict) else None

    def get(self, peer: str, key_id: str, entry: str, piece, dest,
            size: int) -> int:
        """The piece into `dest`: at most `size` + 1 bytes (one more than
        listed is a mismatch the caller refuses)."""
        url = (f"{peer}{PIECE_PATH}?key={quote(key_id)}"
               f"&entry={quote(entry)}&layer={quote(str(piece))}")
        n = 0
        with self._open(url) as r, open(dest, "wb") as out:
            while n <= size:
                buf = r.read(min(1 << 20, size + 1 - n))
                if not buf:
                    break
                out.write(buf)
                n += len(buf)
        return n


def install_fetcher(environ=None) -> bool:
    """A model server a page started: fetch missing cache pieces from that
    page's peers (PAGE_PORT_ENV names the page). -> installed."""
    from knurlogic.engine.prompt_cache import transfer
    port = (os.environ if environ is None else environ).get(PAGE_PORT_ENV,
                                                            "")
    if not port.isdigit():
        return False
    transfer.use(PageFetcher(f"http://127.0.0.1:{int(port)}"))
    return True
