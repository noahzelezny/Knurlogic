"""The telemetry contract's server side (docs/design/telemetry.md,
`telemetry: 1`): the Request opened at the handler and closed on its last
byte into one ledger row (machine/ledger.py), and the SSE
`knurlogic.progress` event.

A Request's id is the client's X-Request-Id when it sent a valid one
(valid_id: 1..128 printable ASCII), echoed exactly; otherwise a ULID
minted here. Either way it is the X-Request-Id answered,
usage.knurlogic.request_id, and the row's id. The page's router passes a
client's header on and returns the model server's id, falling back to the
client's own when an upstream sends none (interfaces/page/router.py).

A value that is not a valid id is not one this server will repeat into a
header, and is ignored rather than refused.

TelemetryHandlers (a mixin of server.Handler) answers GET /v1/usage from
this machine's ledger."""
from __future__ import annotations

import json
import time

from knurlogic.machine import ledger as L

from . import openai as O

VERSION = 1
EVENT = "knurlogic.progress"

HEADER = "X-Request-Id"
MAX_LEN = 128


def valid_id(value) -> str | None:
    """`value` if it may be echoed, else None."""
    if not isinstance(value, str) or not 0 < len(value) <= MAX_LEN:
        return None
    if any(not (0x20 <= ord(ch) <= 0x7E) for ch in value):
        return None
    return value


def id_of(headers) -> str | None:
    """The request's id from its headers (any case), or None."""
    try:
        return valid_id(headers.get(HEADER)) if headers is not None else None
    except AttributeError:
        return None


_MACHINE: list = []


def _machine() -> str | None:
    if not _MACHINE:
        try:
            from knurlogic.machine.identity import identity
            _MACHINE.append(identity().get("id"))
        except Exception:      # an id is a label; its lack never fails a request
            _MACHINE.append(None)
    return _MACHINE[0]


class Request:
    """One HTTP inference request, as the ledger records it."""

    def __init__(self, api: str, headers, *, progress: bool = False):
        self.id = id_of(headers) or L.ulid()
        self.api = api
        self.ts_start = time.time()
        self.labels = L.labels(headers)
        #: whether its stream carries knurlogic.progress events
        self.progress = progress
        self.model: str | None = None
        self.tokens = {"prompt_tokens": 0, "cached_tokens": 0,
                       "output_tokens": 0, "disk_tokens": 0}
        self.timing: dict = {}
        self.finish: str | None = None
        self.status: int | None = None
        self.jobs: list = []
        self.closed = False

    def done(self, usage: dict, finish: str | None) -> None:
        """A job's final usage (a summary pass and the answer both count)."""
        u = usage or {}
        self.tokens["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
        self.tokens["cached_tokens"] += int(
            (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        self.tokens["output_tokens"] += int(u.get("completion_tokens") or 0)
        # the cached tokens that came from a prompt cache restored from
        # disk (usage.knurlogic.cache.disk; engine/prompt_cache/disk)
        disk = (((u.get("knurlogic") or {}).get("cache") or {})
                .get("disk") or {})
        self.tokens["disk_tokens"] += int(disk.get("tokens") or 0)
        self.timing = (u.get("knurlogic") or {}).get("timing") or {}
        self.finish = finish or "stop"

    def failed(self) -> None:
        self.finish = "error"

    def row(self) -> dict:
        t = self.timing
        finish = self.finish
        if finish is None:
            gone = any(getattr(j, "cancelled", False) for j in self.jobs)
            finish = ("error" if (self.status or 200) >= 400
                      else "cancelled" if gone or self.status else "error")
        return {"id": self.id, "ts_start": self.ts_start,
                "ts_end": time.time(), "machine": _machine(),
                "model": self.model, "api": self.api,
                "key_id": "anonymous", **self.labels, **self.tokens,
                "queue_ms": t.get("queue_ms"),
                "prefill_ms": t.get("prefill_ms"),
                "prefill_tps": t.get("prefill_tps"),
                "decode_ms": t.get("decode_ms"),
                "decode_tps": t.get("decode_tps"),
                "finish": finish, "status": self.status}

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            L.record(self.row())


def progress(request_id, phase: str, done: int = 0, total: int = 0,
             tps=None, ahead: int = 0) -> bytes:
    """One SSE `knurlogic.progress` event."""
    data = {"type": EVENT, "request_id": request_id, "phase": phase,
            "done": int(done), "total": int(total), "tps": tps,
            "queue": {"ahead": int(ahead)}}
    return f"event: {EVENT}\ndata: {json.dumps(data)}\n\n".encode()


class TelemetryHandlers:
    """Handler's GET /v1/usage (interfaces/http/server)."""

    def _usage(self, q: dict) -> None:
        """GET /v1/usage?since=&until=&group=&key=: this machine's ledger
        summed by one label (fleet.md, "Reading it back"). Until keys
        exist, the loopback operator only."""
        import ipaddress

        try:
            loop = ipaddress.ip_address(
                self.client_address[0].split("%")[0]).is_loopback
        except ValueError:
            loop = False
        if not loop:
            return self._json(403, {"error": {
                "message": "usage is read on this machine (loopback) only",
                "type": "permission_error"}})

        def one(name, default=None):
            return (q.get(name) or [default])[0]
        try:
            since = float(one("since", 0))
            until = float(one("until")) if one("until") else None
            group = one("group", "model")
            rows = L.ledger().summary(since, until, group, one("key"))
        except ValueError as e:
            return self._error(O.ApiError(400, str(e)))
        return self._json(200, {"object": "usage", "since": since,
                                "until": until, "group": group,
                                "summary": rows})
