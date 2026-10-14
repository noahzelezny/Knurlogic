"""JSON over HTTP, as a client: the one place knurlogic asks a local or peer
server for a document.

  get_json(url)            the document, or None when the server is absent,
                           slow, or answers something that is not JSON --
                           for surveys, where absent is an answer
  send_json(url, payload)  a request with a JSON body (or none); the reply's
                           document, {} for an empty one; an HTTP error
                           status raises urllib.error.HTTPError
  request(url, data)       (status, raw body) of one request; an HTTP error
                           status is an answer, not an exception
  open_url(url, data)      the open response, for a caller that streams it;
                           an HTTP error status is returned as its response

Every call raises only one of NET_ERRORS: no answer came (refused, timed
out, a broken reply), or the answer did not parse (ValueError; HTTPError is
an OSError). `get_json` catches them all; `request` and `open_url` turn an
HTTP error status into an answer; `send_json` lets it raise. Their `data`
is sent as application/json.
"""
from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request

#: what a call to a local or peer server can raise: no answer, or one
#: that does not parse
NET_ERRORS = (OSError, ValueError, http.client.HTTPException)


def get_json(url: str, timeout: float = 1.5):
    """The JSON document at `url`, or None on any of NET_ERRORS."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode() or "null")
    except NET_ERRORS:
        return None


def send_json(url: str, payload=None, method: str = "POST",
              timeout: float = 30.0, headers: dict | None = None):
    """The reply document to `payload` sent as JSON ({} for an empty
    reply). Raises NET_ERRORS, including HTTPError for a 4xx/5xx status."""
    data = None if payload is None else json.dumps(payload).encode()
    hdr = dict(headers or {})
    if data:
        hdr["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=hdr)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode()
    return json.loads(body) if body.strip() else {}


def open_url(url: str, data: bytes | None = None, method: str | None = None,
             timeout: float = 30.0, headers: dict | None = None):
    """The response to one request (.status, .headers, .read); GET without
    `data`, POST with it unless `method` says otherwise."""
    hdr = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(
        url, data=data, headers=hdr,
        method=method or ("GET" if data is None else "POST"))
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        return e


def request(url: str, data: bytes | None = None, method: str | None = None,
            timeout: float = 30.0, headers: dict | None = None) -> tuple:
    """(status, raw body) of one request."""
    with open_url(url, data, method, timeout, headers) as r:
        return r.status, r.read()
