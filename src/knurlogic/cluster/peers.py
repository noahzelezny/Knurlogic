"""Peers: named, remembered, or introduced -- and why each is or is not
answering.

The night this was written, a second Mac sat silent for twenty minutes
because a firewall prompt was waiting on ITS screen, and nothing on either
machine said so. So every peer carries a state, and a state that is not
`answering` carries the fix, on the machine that can apply it:

  answering        its status came back
  not_answering    known (named, remembered or introduced) but the request
                   failed; `problem` says what the failure looked like
  version_mismatch answering, with a status schema this one does not read

A peer that stops answering is kept and shown with how long it has been
gone -- satellites sleep -- never dropped silently.

HOW A PEER LEARNS ABOUT US. Every status request carries an introduction
header (`X-Knurlogic-Peer: <id> <port>`); the receiving page records the
requester's address with that port as an `introduced` peer. So naming a
machine on ONE side is enough for both to know each other, and the side
that cannot be reached still finds out: it asks its peers what they see,
and a peer that lists it as `not_answering` is a measured fact -- "they
can see me and cannot connect", which on a Mac is almost always the
application firewall on THIS machine (docs/DISCOVERY.md, review item 1).

peers.json (~/.knurlogic/peers.json) is keyed by node id, versioned, and
written atomically. It holds addresses, not secrets.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from knurlogic.machine.status import SCHEMA

HEADER = "X-Knurlogic-Peer"
STORE_SCHEMA = 1
TIMEOUT_S = 3.0          # a busy peer was measured at 1.4 s
REFRESH_S = 4.0


def store_path() -> Path:
    return Path(os.environ.get("KNURLOGIC_HOME",
                               Path.home() / ".knurlogic")) / "peers.json"


@dataclass
class Peer:
    host: str
    port: int
    found_by: set = field(default_factory=set)
    id: str = ""
    name: str = ""
    state: str = "not_answering"
    problem: str = ""
    last_seen: float = 0.0          # last time it answered
    failing_since: float = 0.0
    node: dict | None = None        # its own entry from its status
    doc_peers: list = field(default_factory=list)   # what IT sees

    @property
    def key(self) -> str:
        return f"{self.host}:{self.port}"

    def public(self) -> dict:
        d = {"id": self.id, "name": self.name, "address": self.key,
             "found_by": sorted(self.found_by), "state": self.state}
        if self.problem:
            d["problem"] = self.problem
        if self.last_seen:
            d["last_seen"] = round(self.last_seen, 1)
        if self.failing_since:
            d["failing_seconds"] = round(time.time() - self.failing_since)
        return d


def _describe(exc: Exception, peer: Peer) -> str:
    """What a failed request LOOKED like, and what usually causes it."""
    s = f"{type(exc).__name__}: {exc}"
    where = peer.name or peer.host
    if "timed out" in s or isinstance(exc, TimeoutError):
        return (f"{where} did not answer in {TIMEOUT_S:.0f} s. If knurlogic "
                f"is running there, the macOS firewall on {where} is the "
                f"usual cause: allow incoming connections for its Python "
                f"(System Settings -> Network -> Firewall -> Options). "
                f"Asleep, or busy past {TIMEOUT_S:.0f} s, look the same.")
    if "refused" in s.lower():
        return (f"{where} refused the connection on port {peer.port}: "
                f"nothing is listening there. Start `knurlogic ui --host "
                f"<its address> --port {peer.port}` on {where}.")
    if "No route" in s or "unreachable" in s.lower():
        return (f"no route to {peer.host}: the cable, the network, or the "
                f"address changed.")
    return s


class Peers:
    """Every peer this page knows, refreshed in the background so a slow
    or absent peer never makes the page's own status slow."""

    def __init__(self, me: dict, my_port: int, manual=(), store=None,
                 fetch=None, reachable=True):
        self.me, self.my_port = me, my_port
        # A page bound to loopback does not introduce itself: the peer
        # would try the address, fail, and report a firewall problem this
        # machine does not have.
        self.reachable = reachable
        self.store = Path(store) if store else store_path()
        self._fetch = fetch or self._http
        self._lock = threading.Lock()
        self._peers: dict[str, Peer] = {}
        self._thread = None
        for host, port in manual:
            self.add(host, port, "manual")
        for rec in self._load().values():
            addr = rec.get("address", "")
            host, _, port = addr.rpartition(":")
            if host and port.isdigit():
                p = self.add(host, int(port), "remembered")
                p.id, p.name = rec.get("id", ""), rec.get("name", "")
                p.last_seen = rec.get("last_seen", 0.0)

    # -- sources --------------------------------------------------------
    def add(self, host: str, port: int, source: str) -> Peer:
        with self._lock:
            p = self._peers.get(f"{host}:{port}")
            if p is None:
                p = self._peers[f"{host}:{port}"] = Peer(host, int(port))
            p.found_by.add(source)
            return p

    def introduce(self, host: str, header: str) -> None:
        """A peer asked for our status and said who it is."""
        parts = (header or "").split()
        if len(parts) != 2 or not parts[1].isdigit():
            return
        pid, port = parts[0], int(parts[1])
        if pid == self.me.get("id"):
            return                                  # ourselves
        p = self.add(host, port, "introduced")
        p.id = p.id or pid

    # -- refresh --------------------------------------------------------
    def _http(self, url: str) -> dict:
        hdr = ({HEADER: f"{self.me.get('id', '')} {self.my_port}"}
               if self.reachable else {})
        req = urllib.request.Request(url, headers=hdr)
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            return json.loads(r.read())

    def _one(self, p: Peer) -> None:
        now = time.time()
        try:
            doc = self._fetch(f"http://{p.key}/status.json")
        except Exception as e:
            p.state, p.problem = "not_answering", _describe(e, p)
            p.failing_since = p.failing_since or now
            return
        own = next((n for n in doc.get("nodes") or []
                    if n.get("role") in ("local", "server")), None)
        if own is None or own.get("id") == self.me.get("id"):
            # Answered, but as someone else -- or as us, through a loop.
            p.state = "not_answering"
            p.problem = (f"{p.key} answered, but not as a knurlogic node"
                         if own is None else f"{p.key} is this machine")
            return
        p.id, p.name = own.get("id", p.id), own.get("node", p.name)
        p.node, p.doc_peers = own, doc.get("peers") or []
        p.last_seen, p.failing_since = now, 0.0
        if doc.get("schema") != SCHEMA:
            p.state = "version_mismatch"
            p.problem = (f"{p.name} speaks status schema {doc.get('schema')}"
                         f", this machine {SCHEMA}: update knurlogic on the "
                         f"older one")
        else:
            p.state, p.problem = "answering", ""

    def refresh(self) -> None:
        with self._lock:
            todo = list(self._peers.values())
        ts = [threading.Thread(target=self._one, args=(p,), daemon=True)
              for p in todo]
        for t in ts:
            t.start()
        for t in ts:
            t.join(TIMEOUT_S + 1)
        self._dedupe()
        self._save()

    def _dedupe(self) -> None:
        """One machine, one peer: the same id at two addresses keeps the one
        that answered most recently, and inherits the other's sources."""
        with self._lock:
            by_id: dict[str, Peer] = {}
            for k, p in list(self._peers.items()):
                if not p.id:
                    continue
                q = by_id.get(p.id)
                if q is None:
                    by_id[p.id] = p
                    continue
                keep, drop = (p, q) if p.last_seen >= q.last_seen else (q, p)
                keep.found_by |= drop.found_by
                self._peers.pop(drop.key, None)
                by_id[p.id] = keep

    def start(self) -> "Peers":
        if self._thread is None:
            def loop():
                while True:
                    try:
                        self.refresh()
                    except Exception:
                        pass
                    time.sleep(REFRESH_S)
            self._thread = threading.Thread(target=loop, daemon=True,
                                            name="knurlogic-peers")
            self._thread.start()
        return self

    # -- reading --------------------------------------------------------
    def all(self) -> list[Peer]:
        with self._lock:
            return sorted(self._peers.values(), key=lambda p: p.name or p.key)

    def seen_by_peers(self) -> list[dict]:
        """What the peers that answer say about THIS machine. The only way
        a machine behind its own firewall can find out: its outbound works,
        and the peer's view is a measurement, not a guess."""
        out = []
        for p in self.all():
            if p.state != "answering":
                continue
            for q in p.doc_peers:
                if q.get("id") == self.me.get("id"):
                    out.append({"peer": p.name or p.key,
                                "state": q.get("state"),
                                "address": q.get("address")})
        return out

    def self_problem(self) -> str:
        blocked = [s for s in self.seen_by_peers()
                   if s["state"] == "not_answering"]
        if not blocked:
            return ""
        who = ", ".join(s["peer"] for s in blocked)
        return (f"{who} can see this machine but cannot connect to it. The "
                f"macOS firewall on THIS machine is the usual cause: allow "
                f"incoming connections for this Python (System Settings -> "
                f"Network -> Firewall -> Options).")

    # -- memory ---------------------------------------------------------
    def _load(self) -> dict:
        try:
            d = json.loads(self.store.read_text())
            return d.get("peers", {}) if d.get("schema") == STORE_SCHEMA \
                else {}
        except Exception:
            return {}

    def _save(self) -> None:
        """Only peers that have answered at least once are remembered, so a
        typo in --peer does not haunt every later run."""
        old = self._load()
        changed = False
        for p in self.all():
            if not (p.id and p.last_seen):
                continue
            was = old.get(p.id) or {}
            # Rewritten when something a later run needs has changed, not
            # on every refresh: last_seen only counts once it is stale.
            if (was.get("address") != p.key or was.get("name") != p.name
                    or p.last_seen - was.get("last_seen", 0) > 300):
                old[p.id] = {"id": p.id, "name": p.name, "address": p.key,
                             "last_seen": round(p.last_seen, 1)}
                changed = True
        if not changed:
            return
        try:
            self.store.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.store.with_suffix(".tmp")
            tmp.write_text(json.dumps({"schema": STORE_SCHEMA, "peers": old},
                                      indent=1))
            os.replace(tmp, self.store)
        except OSError:
            pass
