"""PEER_RELAY/v1/...: a peer page reaching a model this machine started --
chat, messages, count_tokens, /v1/models and the prompt cache, resolved
by model name against the servers started here (`peer_relay`)."""

from __future__ import annotations

import json
from urllib.parse import parse_qs, urlparse

from knurlogic.interfaces.page import prompt_cache, router
from knurlogic.machine import identity


def peer_relay(handler, method: str, path: str, body: bytes,
               fetch=None) -> None:
    """GET /peer/v1/models, POST /peer/v1/chat/completions, /v1/messages,
    /v1/messages/count_tokens -- on the machine holding the model, for a
    peer page whose router or chat names it. The caller has passed
    peer_refusal. Resolved by model name against the servers this machine
    started (local_models) and streamed back as it arrives."""
    if path == prompt_cache.PROMPT_CACHE_PATH or path.startswith(
            prompt_cache.PROMPT_CACHE_PATH + "/"):
        # a peer page managing the prompt cache of a model this machine
        # serves: resolved here by name, sent on to it from loopback
        if not (method == "GET" and path == prompt_cache.PROMPT_CACHE_PATH or
                method == "POST" and path in prompt_cache.PROMPT_CACHE_POSTS):
            router.send_json(handler, 404, {"error": "not a relayed path"})
            return
        q = parse_qs(urlparse(getattr(handler, "path", "")).query)
        model = (q.get("model") or [None])[0]
        if model is None and body:
            try:
                got = json.loads(body)
                model = got.get("model") if isinstance(got, dict) else None
            except ValueError:
                model = None
        table = router.local_models(fetch)
        base = router.resolve(table, model)
        if base is None:
            router.send_json(handler, 404, {"error": {
                "message": f"no running model {model!r} on "
                           f"{identity.identity().get('name') or 'this machine'}",
                "type": "not_found"}, "models": sorted(table)})
            return
        code, doc = router.send_up(base + path, method,
                             body if method == "POST" else None)
        router.send_json(handler, code, doc)
        return
    docs: dict = {}
    table = router.local_models(fetch, docs)
    if method == "GET":
        if path != "/v1/models":
            router.send_json(handler, 404, {"error": "not a relayed path"})
            return
        # each server's own entry, so a peer page's chat sees the model's
        # sampling_defaults and context_length, not just its name
        router.send_json(handler, 200, {"object": "list", "data": [
            dict(docs.get(m) or {}, id=m, object="model",
                 owned_by="knurlogic")
            for m in sorted(table)]})
        return
    if path not in router.ROUTE_PATHS:
        router.send_json(handler, 404, {"error": "not a relayed path"})
        return
    try:
        doc = json.loads(body or b"{}")
    except ValueError:
        doc = None
    if not isinstance(doc, dict):
        router.send_json(handler, 400, {"error": "the body must be a JSON object"})
        return
    model = doc.get("model")
    base = router.resolve(table, model)
    if base is None:
        router.send_json(handler, 404, {
            "type": "error",
            "error": {"type": "not_found_error",
                      "message": f"no running model {model!r} on "
                                 f"{identity.identity().get('name') or 'this machine'}"
                                 f"; running: "
                                 f"{', '.join(sorted(table)) or 'none'}"},
            "models": sorted(table)})
        return
    router.stream(handler, base + path, body, base=base)
