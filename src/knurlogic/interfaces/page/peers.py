"""What this page knows of and does with its peers: the gate every /peer/
route passes (`peer_refusal`), the survey of what each peer serves
(`peer_residency`, which fills the chat targets `upstream` resolves
through a peer's relay), and a machine's own settings set from any page
(`peer_machine`, `machine_apply`)."""

from __future__ import annotations

import http.client
import json
import time
from urllib.parse import urlparse

from knurlogic.interfaces.page import nodes
from knurlogic.machine import identity

_ADDRS: dict = {}


def _addresses_of(host: str, ttl: float = 60.0) -> set:
    """The addresses a --peer NAME resolves to (cached a minute): the
    operator named that machine, so its address is trusted as the name
    is. An IP literal resolves to itself."""
    now = time.time()
    hit = _ADDRS.get(host)
    if hit and now - hit[0] < ttl:
        return hit[1]
    import socket
    try:
        got = {a[4][0].split("%")[0] for a in socket.getaddrinfo(
            host, None, proto=socket.IPPROTO_TCP)}
    except (OSError, UnicodeError):
        got = set()
    _ADDRS[host] = (now, got)
    return got


def manual_hosts() -> list:
    """Every address of every --peer machine: the probe moves p.host to the
    fastest answering address, the machine may call from any other."""
    out: list = []
    for p in (nodes.PEERS.all() if nodes.PEERS else []):
        if "manual" in p.found_by:
            ks = getattr(p, "addresses", ())
            for h in [p.host, *(k.rpartition(":")[0] for k in ks)]:
                if h and h not in out:
                    out.append(h)
    return out


def peer_refusal(headers, client_ip: str, local_ip: str, gate=None,
                 manual_hosts=(), what: str = "peer requests"):
    """The gate every /peer/ route shares: (status, doc) when refused, else
    None. No Origin header (a browser never reaches a peer route), and the
    connection arrived on loopback or Thunderbolt, or from a peer address
    named with --peer -- in every mode, whatever --host says."""
    from knurlogic.cluster.links import Gate
    if headers.get("Origin") is not None:
        return 403, {"error": "a web page cannot drive another machine"}
    ip = (client_ip or "").removeprefix("::ffff:")
    g = gate or Gate()
    if not (g.allows(local_ip) or ip in set(manual_hosts)
            or any(ip in _addresses_of(h) for h in manual_hosts)):
        return 403, {"error": f"{what} are taken over Thunderbolt or "
                              f"loopback, or from a peer named with --peer; "
                              f"this came from {ip}"}
    return None


#: How long the page waits for all peers together. A peer's /loaded.json
#: runs its memory map (about a second on a busy box); past this the local
#: answer goes out without that peer, which is listed as not answering.
PEER_LOADED_S = 2.5


#: Chat endpoints on peers, as the peers themselves last reported them:
#: {base: {"machine": name, "relay": the peer's page}}. `base` is the
#: model's address as seen from here (what the page shows and keys a chat
#: by); a peer's server listens on ITS loopback, so every request for it
#: goes to the peer's page relay (PEER_RELAY) instead. Refilled by every
#: peer survey.
PEER_TARGETS: dict = {}


#: each peer's last good survey, {address: (time, entry)}
_PEER_LAST: dict = {}


#: the cluster jobs peers last reported, {job: doc} (running, and the ones
#: that ended lately with why): what a dropped connection is explained by
PEER_JOBS: dict = {}


#: the peer page's relay prefix: /peer/v1/... reaches the model servers
#: that page itself started, by model name (peer_relay)
PEER_RELAY = "/peer"


#: the ONE route between pages: a POST of a protocol envelope
#: (cluster/transport.py); /peer/v1/ is otherwise the model relay above
MSG_PATH = "/peer/v1/msg"


def upstream(base: str, path: str) -> str:
    """The URL a request for `path` on the model at `base` goes to: the
    server itself when it is this machine's, the peer page's relay when it
    is a peer's."""
    t = PEER_TARGETS.get(base)
    if t:
        return t["relay"] + PEER_RELAY + path
    return base + path


# --- a MACHINE's own settings, set from any page ---------------------------
# The allowance and the strategy are each machine's own, kept in its own
# ~/.config. A page changes a peer's by asking that peer's page (the peer
# gate, a MachineSet message), which applies it to itself -- never by writing
# anything of the peer's here. The wired limit is not one of them: it is a
# `sudo sysctl` knurlogic never runs; a peer's command is read through /peek
# (wired_gib) and shown to run on that machine.

MACHINE_MAX = 1 << 10


def peer_machine(want) -> tuple:
    """A MachineSet message on the machine being changed (the caller has
    passed the peer gate): {"allowance_gib": N}, {"strategy": name} and/or
    {"settings": {...}} (the knurlogic-wide ones), applied here exactly as
    this machine's own page applies them. -> (status, doc): the machine's
    allowance and strategy after, with what applied, or the first
    refusal."""
    from knurlogic.interfaces.page import documents
    if isinstance(want, dict):
        want = {k: v for k, v in want.items() if v is not None}
    if not isinstance(want, dict) or not want or set(want) - {
            "allowance_gib", "strategy", "settings"}:
        return 400, {"error": "send {\"allowance_gib\": N}, "
                              "{\"strategy\": name} and/or "
                              "{\"settings\": {name: value}}"}
    applied = {}
    if "allowance_gib" in want:
        out = documents.set_allowance(json.dumps({"gib": want["allowance_gib"]}))
        if "error" in out:
            return 400, out
        applied.update(out["applied"])
    if "strategy" in want:
        out = documents.set_strategy(json.dumps({"preset": want["strategy"]}))
        if "error" in out:
            return 400, out
        applied.update(out["applied"])
    if "settings" in want:
        out = documents.set_knurlogic(json.dumps(want["settings"]))
        if "error" in out:
            return 400, out
        applied.update(out["applied"])
    return 200, {"allowance": documents.allowance_doc(),
                 "strategy": documents.strategy_doc(),
                 "knurlogic": documents.knurlogic_doc(), "applied": applied,
                 "machine": identity.identity().get("name") or ""}


def peer_pages() -> set:
    """The pages of peers answering this one, as http://host:port -- the
    only places a machine setting is sent, never an address from the
    request."""
    return {f"http://{p.key}" for p in (nodes.PEERS.all() if nodes.PEERS else [])
            if p.state == "answering"}


def machine_apply(where: str, body: bytes, post=None) -> tuple:
    """POST /machine.json?where=<peer page>: a machine setting for THAT
    machine, sent as a MachineSet message; where '' is this machine,
    applied here. -> (status, doc) as the machine answered."""
    base = (where or "").rstrip("/")
    if len(body or b"") > MACHINE_MAX:
        return 413, {"error": "a machine setting is small"}
    if not base:
        try:
            return peer_machine(json.loads(body or b""))
        except ValueError:
            return 400, {"error": "the body must be a JSON object"}
    if base not in peer_pages():
        return 403, {"error": f"not a machine answering this page: {base}"}
    try:
        want = json.loads(body or b"")
    except ValueError:
        want = None
    if not isinstance(want, dict):
        return 400, {"error": "the body must be a JSON object"}
    from knurlogic.cluster import transport
    post = post or transport.send
    try:
        out = post(base.removeprefix("http://"), "MachineSet", want)
    except (OSError, ValueError, http.client.HTTPException) as e:
        return 502, {"error": f"{base} did not take it: "
                              f"{type(e).__name__}: {e}"}
    if not isinstance(out, dict):
        return 502, {"error": f"{base} answered without a JSON object"}
    return (400 if out.get("error") else 200), out


def _peer_where(where: str, host: str) -> str:
    """A peer reports its models at ITS loopback; seen from here the same
    port is at the peer's address. An endpoint on any OTHER host is
    dropped (""): peers are found by Bonjour, which anything on the network
    can advertise into, and what a peer reports becomes an address this
    page's chat proxy will POST to -- so a peer may only offer itself."""
    u = urlparse(where or "")
    if not u.port or u.scheme not in ("http", ""):
        return ""
    if u.hostname in ("127.0.0.1", "localhost", "::1", host):
        return f"http://{host}:{u.port}"
    return ""


def peer_residency(peers, timeout: float = PEER_LOADED_S,
                   fetch=None) -> list:
    """What every answering peer says it is holding, one entry per machine.

    Asked in parallel with one shared deadline, so a slow or dead peer costs
    at most `timeout` and never the local answer. Peers are asked plain
    a Survey message (what its own /loaded.json says, never ?peers=1), so
    two pages asking each other cannot recurse. Each row is labelled with
    its machine and its address rewritten from the peer's loopback to the
    peer's address."""
    import threading

    from knurlogic.cluster import transport
    from knurlogic.cluster.peers import clean
    if fetch is None:
        def fetch(page, t):
            return transport.send(page, "Survey", {}, timeout=t)
    todo = [p for p in (peers.all() if peers else [])
            if p.state == "answering"]
    out: dict = {}

    def one(p):
        try:
            doc = fetch(p.key, timeout)
            rows = []
            for r in doc.get("resident") or []:
                if isinstance(r, dict):
                    rows.append(dict(r, machine=p.name or p.host,
                                     where=_peer_where(r.get("where"),
                                                       p.host)))
            js = [clean(j) for j in doc.get("jobs") or []
                  if isinstance(j, dict)]
            out[p.key] = {"machine": p.name or p.host, "address": p.key,
                          "id": getattr(p, "id", ""), "resident": rows,
                          "jobs": js, "loads": doc.get("loads") or []}
        # peer survey: one peer's failure is its row's error; the others are still
        # listed
        except Exception as e:
            out[p.key] = {"machine": p.name or p.host, "address": p.key,
                          "resident": [],
                          "error": f"{type(e).__name__}: {e}"}
    ts = [threading.Thread(target=one, args=(p,), daemon=True) for p in todo]
    for t in ts:
        t.start()
    end = time.time() + timeout
    for t in ts:
        t.join(max(end - time.time(), 0))
    res = []
    for p in todo:
        got = out.get(p.key)
        if got and "error" not in got:
            _PEER_LAST[p.key] = (time.time(), got)
        was = _PEER_LAST.get(p.key)
        if got is None or ("error" in got and was):
            # Nothing came back in OUR deadline, or the Survey itself timed
            # out on a peer busy loading (a 101 GiB pipeline load did): its
            # status probe still answers, so it is a slow peer, not a gone
            # one. Show what it last said, labelled with its age -- dropping
            # it made a cluster load's % fall to this machine's share alone
            # and jump back (54 -> 38 -> 54). "no reply yet" when it never did.
            got = (dict(was[1], heard_ago=round(time.time() - was[0]))
                   if was else {
                       "machine": p.name or p.host, "address": p.key,
                       "resident": [], "late": True})
        res.append(got)
    targets, pjobs = {}, {}
    for m in res:
        for r in m["resident"]:
            if r.get("runtime") == "knurlogic" and r.get("where"):
                c = r.get("cluster") if isinstance(r.get("cluster"),
                                                   dict) else {}
                targets[r["where"].rstrip("/")] = {
                    "machine": m["machine"],
                    "relay": f"http://{m['address']}",
                    "job": str(c.get("job") or "")}
        for j in m.get("jobs") or []:
            if j.get("job"):
                pjobs[str(j["job"])] = j
    PEER_TARGETS.clear()
    PEER_TARGETS.update(targets)
    # a job that ended stays explainable after its page stops listing it
    PEER_JOBS.update(pjobs)
    PEER_AT[0] = time.time()
    return res


#: when peers were last asked what they serve (peer_residency)
PEER_AT = [0.0]
