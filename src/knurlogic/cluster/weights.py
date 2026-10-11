"""Weight streaming for split loads, the cluster half
(docs/design/weight-streaming.md): who reads from whom and in what order
(`read_plan`), the page that serves a shard's byte range (`serve`), and the
rank's client for it (`reader`). The rank's own use of them is
engine/split/weights.py.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from knurlogic.cluster import transport

#: a resolved identity is kept this long (seconds) by the serving page: one
#: load asks for thousands of ranges
RESOLVE_TTL = 60.0
_RESOLVED: dict = {}


def read_plan(names: list, local: list, pages: list) -> list:
    """One entry per rank: {"from", "source", "turn", "ordered"}.

    `names`: each rank's machine name; `local`: whether its copy is on its
    own disk (Prepare's `local_copy`); `pages`: each machine's page as the
    job's link reaches it ("ip:port"). A rank on a mount fetches from the
    first rank's machine with a local copy and reads at once (turn 0);
    ranks with local copies read after (turn 1). Ordered only when both
    turns have ranks."""
    holders = [r for r, ok in enumerate(local) if ok]
    out = []
    for _r, ok in enumerate(local):
        src = "" if ok or not holders else pages[holders[0]]
        out.append({"from": src,
                    "source": names[holders[0]] if src else "",
                    "turn": 1 if ok else 0})
    ordered = bool(holders) and len(holders) < len(local)
    for e in out:
        e["ordered"] = ordered
    return out


def rank_env(entry) -> dict:
    """The rank's environment for its read-plan entry."""
    if not isinstance(entry, dict):
        return {}
    clean = {"from": str(entry.get("from") or "")[:80],
             "source": str(entry.get("source") or "")[:80],
             "turn": 1 if entry.get("turn") == 1 else 0,
             "ordered": bool(entry.get("ordered"))}
    return {"KNURLOGIC_WEIGHTS": json.dumps(clean)}


# ------------------------------------------------------------ serving page

def _local_dir(ident: str, name: str, resolve=None):
    now = time.time()
    hit = _RESOLVED.get((ident, name))
    if hit and now - hit[0] < RESOLVE_TTL:
        return hit[1]
    from knurlogic.machine import artifact as A
    p = (resolve or A.resolve_identity)(ident, name=name)
    if p and A.on_network(p):
        p = None                      # it would only relay the mount
    _RESOLVED[(ident, name)] = (now, p)
    return p


def serve(body: dict, resolve=None) -> tuple:
    """A Weights message on the machine holding the files: (status, a
    transport.Bytes of the range) or (status, {"error"})."""
    ident = body.get("identity")
    f, off, n = body.get("file"), body.get("offset"), body.get("length")
    if not isinstance(ident, str) or not isinstance(f, str) or \
            not all(isinstance(x, int) and not isinstance(x, bool) and x >= 0
                    for x in (off, n)):
        return 400, {"error": "Weights is {identity, file, offset, length}"}
    if "/" in f or "\\" in f or f.startswith(".") or \
            not f.endswith(".safetensors"):
        return 400, {"error": "file is a shard's file name"}
    name = Path(str(body.get("name") or "")).name
    d = _local_dir(ident, name, resolve)
    if not d:
        return 404, {"error": "no copy with that identity on this "
                              "machine's own disk"}
    p = Path(d) / f
    try:
        size = p.stat().st_size if p.is_file() else -1
    except OSError:
        size = -1
    if size < 0:
        return 404, {"error": f"no shard {f[:80]!r} in that artifact"}
    if off + n > size:
        return 416, {"error": f"{off}+{n} is past the shard's {size} bytes"}
    return 200, transport.Bytes(str(p), off, n)


# ------------------------------------------------------------- rank client

def reader(page: str, identity: str, name: str = "", fetch=None):
    """read(file, offset, length, into) for engine/split/weights.fill: each
    range a Weights message to `page`, one kept-alive connection per
    thread."""
    fetch = fetch or transport.fetch
    local = threading.local()

    def read(file, offset, length, into):
        import http.client
        c = getattr(local, "conn", None)
        if c is None:
            host, _, port = page.rpartition(":")
            c = local.conn = http.client.HTTPConnection(
                host, int(port), timeout=transport.timeout_for("Weights"))
        try:
            fetch(page, "Weights", {"identity": identity, "name": name,
                                    "file": file, "offset": offset,
                                    "length": length}, into, conn=c)
        except OSError:
            local.conn = None
            c.close()
            raise
    return read
