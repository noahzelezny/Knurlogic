"""The one JSON reply writer for handlers that are not the model server's
own (the page and its router, relay and proxies): send_json(handler, code,
doc, headers) -- the client's X-Request-Id echoed back, and a client that
hung up before its answer is not an error."""

from __future__ import annotations

import json

from . import telemetry as T


def send_json(handler, code: int, doc, headers: dict | None = None) -> None:
    out = json.dumps(doc).encode()
    handler.send_response(code)
    for k, v in (headers or {}).items():
        handler.send_header(k, v)
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
