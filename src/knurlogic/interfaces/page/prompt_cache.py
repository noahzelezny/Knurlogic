"""The page's /v1/prompt-cache: forwarded to the model server (here, or a
peer's through its relay) that serves the named model. The model
server's own endpoints are interfaces/http/prompt_cache.py; this is the
page's routing to them."""

from __future__ import annotations

import json
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
        router._send_json(handler, 403, {"error": {
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
        base = router._resolve(table, model)
        if base is None:
            # a peer's model: its page's relay, like a chat (the peer
            # resolves the name again against the servers it started)
            far = router._resolve(router.routable(fetch), model)
            if far is not None and peers._PEER_TARGETS.get(far) is not None:
                q = f"?model={quote(str(model))}" \
                    if method == "GET" else ""
                code, doc = (send or router._send_up)(
                    peers.upstream(far, path) + q, method,
                    body if method == "POST" else None)
                router._send_json(handler, code, doc)
                return
            here = sorted(set(table) | set(router.routable(fetch)))
            router._send_json(handler, 404, {"error": {
                "message": f"no running model {model!r} here or on a peer; "
                           f"running: {', '.join(here) or 'none'}",
                "type": "not_found"}, "models": here})
            return
    else:
        bases = set(table.values())
        if len(bases) != 1:
            router._send_json(handler, 400, {"error": {
                "message": "name the model (a \"model\" query or body "
                           "field): " + (", ".join(sorted(table))
                                         or "none is running"),
                "type": "invalid_request_error"}, "models": sorted(table)})
            return
        base = bases.pop()
    code, doc = (send or router._send_up)(base + path, method,
                                   body if method == "POST" else None)
    router._send_json(handler, code, doc)
