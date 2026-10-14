"""JSON over HTTP, as a client: the one place knurlogic asks a local or peer
server for a JSON document.

  get_json(url)            the document, or None when the server is absent,
                           slow, or answers something that is not JSON --
                           for surveys, where absent is an answer
  send_json(url, payload)  a request with a JSON body (or none); the reply's
                           document, {} for an empty one; errors raise
"""
from __future__ import annotations

import http.client
import json
import urllib.request


def get_json(url: str, timeout: float = 1.5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode() or "null")
    except (OSError, ValueError, http.client.HTTPException):
        return None


def send_json(url: str, payload=None, method: str = "POST",
              timeout: float = 30.0, headers: dict | None = None):
    data = None if payload is None else json.dumps(payload).encode()
    hdr = dict(headers or {})
    if data:
        hdr["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=hdr)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode()
    return json.loads(body) if body.strip() else {}
