"""The one JSON reply writer, for every handler (the page and its router,
relay and proxies, and the model server): send_json(handler, code, doc,
headers) -- the client's X-Request-Id echoed back (unless the handler's
ledger record stamps it, http/server.py end_headers), and a client that
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
    if rid and getattr(handler, "_record", None) is None:
        handler.send_header(T.HEADER, rid)
    try:
        handler.end_headers()
        handler.wfile.write(out)
    except (BrokenPipeError, ConnectionResetError):
        handler.close_connection = True
