"""The one client (and the one decoder) of the control plane.

Every page-to-page message is an envelope (cluster/protocol.py) POSTed to
ONE route, `MSG_PATH`. `send` is the client: it wraps a kind's body in an
envelope, posts it, and hands back the reply's body as a dict (today's wire
keys), or raises:

  PeerUnreachable   the address did not answer (an OSError)
  VersionMismatch   the peer speaks another major version, or no protocol
                    at all (a 404, or a reply that is not an envelope): the
                    text says which machine to update
  PeerRefused       a plain refusal from the peer's gate (403, 411, 413)

A typed refusal (a `Failure` reply, or an error a handler gave) is not an
exception: it comes back as `{"error": reason, "failure_kind": kind}`, the
shape every caller already reads.

`handle` is the server half: decode, check, dispatch to a table, wrap the
handler's (status, doc) in a reply envelope. It never reads an address out
of a message: replies go back on the connection, stops and heartbeats to the
addresses the receiver already knows.

ONE timeout table (`TIMEOUTS`) for every call between pages.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable

from knurlogic.cluster import NET_ERRORS
from knurlogic.cluster import jobs as J
from knurlogic.cluster import protocol as P

log = logging.getLogger(__name__)

MSG_PATH = "/peer/v1/msg"

#: seconds a call may take, by message kind; "status" is the liveness GET
#: every peer answers (a light document, see the page's /status.json?light=1)
TIMEOUTS: dict = {
    "status": 1.5,
    "Survey": 2.5,
    "Prepare": 30.0,
    "Start": 30.0,
    # the peer answers once its ranks are gone: its grace, then its reap
    "Stop": J.GRACE_S + J.REAP_S + 10,
    "JobState": 5.0,
    "Shape": 30.0,
    "Load": 60.0,
    "Unload": 90.0,
    "MachineSet": 10.0,
    "Settings": 10.0,
    "Read": 3.0,
}

#: the request kind -> the kind its reply is (others reply with their own)
REPLY_KIND = {"Prepare": "PrepareReply", "Start": "Started",
              "Stop": "Stopped", "Survey": "Residency"}


def timeout_for(kind: str) -> float:
    return TIMEOUTS.get(kind, TIMEOUTS["Prepare"])


class PeerUnreachable(OSError):
    """The address did not answer."""


class PeerRefused(P.ProtocolError):
    """A plain refusal from the peer's gate, not a typed reply."""


def no_protocol(who: str) -> str:
    return (f"{who} does not speak this protocol (no /peer/v1/msg): "
            f"update knurlogic on {who}")


_SENDER: list = [""]


def sender_id() -> str:
    if not _SENDER[0]:
        try:
            from knurlogic.machine import identity
            _SENDER[0] = str(identity.identity().get("id") or "")
        except (OSError, ValueError, KeyError):
            pass
    return _SENDER[0]


def encode(kind: str, doc: dict, job: str | None = None,
           sender: str | None = None) -> tuple:
    """(bytes, seq): the envelope for `kind` around `doc`."""
    if kind not in P.kinds():
        raise P.ProtocolError(f"unknown message kind {kind!r}")
    seq = P.next_seq()
    env = {"v": list(P.VERSION), "kind": kind,
           "from": sender_id() if sender is None else sender, "seq": seq,
           "ts": time.time(), "body": doc}
    job = job if job is not None else doc.get("job")
    if isinstance(job, str):
        env["job"] = job
    return json.dumps(env).encode(), seq


def decode_reply(code: int, raw: bytes, who: str, seq: int | None = None):
    """The reply body as a clean dict, from the peer's HTTP answer."""
    try:
        out = json.loads(raw)
    except (ValueError, TypeError):
        out = None
    if not isinstance(out, dict):
        raise P.VersionMismatch(no_protocol(who))
    if "kind" not in out or "v" not in out:
        # no envelope: a route this peer does not have (404), or its
        # gate's plain refusal
        if code == 404:
            raise P.VersionMismatch(no_protocol(who))
        if isinstance(out.get("error"), str):
            raise PeerRefused(out["error"][:500])
        raise P.VersionMismatch(no_protocol(who))
    P.check_version(out["v"], who)
    re_ = out.get("re")
    if seq is not None and re_ is not None and re_ != seq:
        raise P.ProtocolError(f"{who} answered another request")
    from knurlogic.cluster.peers import clean
    body = clean(out.get("body") if isinstance(out.get("body"), dict) else {})
    if out["kind"] == "Failure":
        return {"error": str(body.get("reason") or "refused")[:500],
                "failure_kind": body.get("kind") or "refusal"}
    return body


def send(page: str, kind: str, doc: dict | None = None, *,
         job: str | None = None, timeout: float | None = None,
         who: str | None = None) -> dict:
    """Post `kind` with body `doc` to the page at `page` (host:port) and
    return the reply's body."""
    doc = doc or {}
    data, seq = encode(kind, doc, job)
    who = who or page
    req = urllib.request.Request(
        f"http://{page}{MSG_PATH}", data=data, method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(
                req, timeout=timeout_for(kind) if timeout is None
                else timeout) as r:
            code, raw = r.status, r.read()
    except urllib.error.HTTPError as e:
        code, raw = e.code, e.read()
    except OSError as e:
        raise PeerUnreachable(f"{who}: {type(e).__name__}: {e}") from e
    return decode_reply(code, raw, who, seq)


def parallel(fn: Callable, items: list, limit: float | None = None) -> list:
    """fn(item) for every item at once; one item's failure is its result
    ({"ok": False, "error": ...}), the others go on."""
    out: list = [None] * len(items)

    def one(i, x):
        try:
            out[i] = fn(x)
        except (*NET_ERRORS, P.ProtocolError, AttributeError, KeyError,
                TypeError) as e:
            out[i] = {"ok": False,
                      "error": str(e) if isinstance(
                          e, (OSError, P.ProtocolError))
                      else f"{type(e).__name__}: {e}"}
    ts = [threading.Thread(target=one, args=(i, x), daemon=True)
          for i, x in enumerate(items)]
    for t in ts:
        t.start()
    end_in = (limit if limit is not None else max(TIMEOUTS.values())) + 5
    end = time.time() + end_in
    for t in ts:
        t.join(max(end - time.time(), 0))
    return out


def send_all(calls: list, *, local: Callable | None = None,
             send_fn: Callable | None = None) -> list:
    """`calls`: [(page | None, kind, doc)], each done at once; a page of
    None is this machine, answered by `local(kind, doc)`. -> the reply
    bodies in order, a failure being that entry's {"ok": False, "error"}."""
    send_fn = send_fn or send

    def one(c):
        page, kind, doc = c
        if page is None:
            return local(kind, doc)             # type: ignore[misc]
        return send_fn(page, kind, doc)
    return parallel(one, list(calls))


# ------------------------------------------------------------ server half

def failure_reply(reason: str, kind: str = "refusal", re_: int | None = None,
                  sender: str | None = None) -> dict:
    env = {"v": list(P.VERSION), "kind": "Failure",
           "from": sender_id() if sender is None else sender, "ts": 0.0,
           "body": P.Failure(kind=kind, reason=reason[:300]).to_wire()}
    if re_ is not None:
        env["re"] = re_
    return env


def reply(kind: str, code: int, doc: dict, re_: int | None) -> dict:
    """A handler's (code, doc) as a reply envelope: a bare {"error": ...}
    is a typed refusal, anything else the reply kind's own body."""
    if isinstance(doc, dict) and set(doc) <= {"error", "v"} and "error" in doc:
        return failure_reply(str(doc["error"]), re_=re_)
    env = {"v": list(P.VERSION), "kind": REPLY_KIND.get(kind, kind),
           "from": sender_id(), "ts": 0.0, "body": doc}
    if re_ is not None:
        env["re"] = re_
    return env


def handle(body: bytes, table: dict) -> tuple:
    """(status, document) for a POST to MSG_PATH whose gates have passed.

    `table`: kind -> fn(body: dict) -> (status, doc). Not an envelope: a
    plain 400 refusal. A different major version, an unknown kind or a bad
    body of a real envelope: a typed Failure(refusal). `from` is never used
    to pick an address."""
    try:
        doc = json.loads(body or b"")
    except ValueError:
        return 400, {"error": "the body must be a protocol envelope "
                              "(JSON): see docs/design/orchestration.md"}
    if not isinstance(doc, dict) or not all(k in doc for k in ("v", "kind")):
        return 400, {"error": "the body must be a protocol envelope "
                              "{v, kind, from, body}"}
    re_ = doc.get("seq") if isinstance(doc.get("seq"), int) else None
    try:
        P.check_version(doc.get("v"), str(doc.get("from") or "a peer")[:60])
    except P.VersionMismatch as e:
        return 400, failure_reply(str(e), re_=re_)
    except P.ProtocolError as e:
        return 400, {"error": str(e)}
    kind = doc.get("kind")
    if kind not in table:
        return 400, failure_reply(
            f"this machine does not handle message kind {str(kind)[:40]!r}",
            re_=re_)
    inner = doc.get("body", {})
    try:
        P.Message.from_wire(doc)                # required fields, shapes
    except P.ProtocolError as e:
        return 400, failure_reply(str(e), re_=re_)
    if not isinstance(inner, dict):
        return 400, failure_reply("body must be an object", re_=re_)
    try:
        code, out = table[kind](inner)
    except Exception:                   # a handler bug is not "no answer"
        log.exception("peer message handler %s failed", kind)
        return 500, failure_reply("this machine's handler failed",
                                  kind="failure", re_=re_)
    return code, reply(str(kind), code, out, re_)
