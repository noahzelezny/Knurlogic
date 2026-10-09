"""The machines this page sees: its peer store (PEERS), Bonjour discovery,
and /status.json -- the full document a browser reads and the light
liveness document every peer polls (`hot` makes it re-measure memory
often while memory moves)."""

from __future__ import annotations

import logging
import subprocess
import sys
import time
import uuid
from typing import TYPE_CHECKING

from knurlogic.interfaces import spawn
from knurlogic.interfaces.page import documents
from knurlogic.machine import identity, status
from knurlogic.machine.memory import footprint, wired

if TYPE_CHECKING:
    from knurlogic.cluster.peers import Peers

logger = logging.getLogger(__name__)


_MM: dict = {"doc": None, "at": 0.0}


#: The other machines this page knows (cluster/peers.py); None until the
#: page starts, so importing this module starts nothing.
PEERS: Peers | None = None


#: one per process start
_BOOT_ID = uuid.uuid4().hex


def _status_fn(_n=0):
    """A status for a box that is serving nothing.

    The node, its machine and its memory map are all still real -- that is
    the whole content of this mode. `artifact` is simply absent, and the
    page already handles that: it is the same shape `serve` emits.

    A peer's own memory map arrives only if a knurlogic there answers on the
    network.
    """
    # Reused for a few seconds, as `serve` does: the map runs `top`, which
    # takes over a second on a loaded box, and the page and every peer
    # asking for it would otherwise each pay that.
    now = time.time()
    if _MM["doc"] is None or now - _MM["at"] > 4.0:
        try:
            _MM["doc"] = footprint.memory_map()
        except (OSError, subprocess.SubprocessError, ValueError, KeyError,
                AttributeError):
            _MM["doc"] = None
        _MM["at"] = now
    mm = _MM["doc"]

    me = identity.identity()
    # This machine, always, under its own name.
    snaps = [status.snapshot(node=me["name"], role="local", memory_map=mm)]

    # Then every peer: a node that answers for itself is the best witness
    # of itself.
    peers = PEERS.all() if PEERS else []
    for p in peers:
        if p.state in ("answering", "version_mismatch") and p.node:
            snaps.append({**p.node, "role": "remote", "address": p.key,
                          "found_by": sorted(p.found_by), "state": p.state,
                          **({"problem": p.problem} if p.problem else {})})
        else:
            snaps.append({**status.snapshot(
                node=p.name or p.host, role="remote", reachable=False,
                memory_fn=lambda: {"available": False},
                machine_fn=lambda: {}),
                "found_by": sorted(p.found_by), "state": p.state,
                "problem": p.problem, "address": p.key})
    # what a coordinator page needs of this machine to place a rank on it
    try:
        from knurlogic.cluster import launch
        snaps[0]["cluster"] = launch.node_info(
            (snaps[0].get("memory") or {}).get("working_set_bytes") or 0)
    except (OSError, ValueError, AttributeError, KeyError, TypeError) as e:
        snaps[0]["cluster"] = {"error": f"{type(e).__name__}: {e}"}
    snap = status.aggregate(snaps)
    snap["wired"] = wired.advise(0)
    # liveness rides this GET: which process answered, and the control-plane
    # protocol it speaks (a peer that changed boot_id lost its ranks)
    from knurlogic.cluster import protocol
    snap["boot_id"] = _BOOT_ID
    snap["v"] = list(protocol.VERSION)
    # Who this machine is, and -- measured by the peers, since this machine
    # cannot see connections its own firewall drops -- whether they can
    # reach it.
    snap["me"] = {**me, "port": spawn.SERVE_PORT["ui"]}
    if DISCOVERY is not None:
        snap["discovery"] = DISCOVERY.status()
    if PEERS:
        snap["peers"] = [p.public() for p in peers]
        seen = PEERS.seen_by_peers()
        if seen:
            snap["me"]["seen_by"] = seen
        prob = PEERS.self_problem()
        if prob:
            snap["me"]["problem"] = prob
    return snap, status.render_cluster(snap)


def _status_light(_n=0):
    """The liveness document: what every peer asks of this page every
    couple of seconds (/status.json?light=1). This machine's own node
    entry -- its cluster block and the memory map as last measured, never a
    new one unless that is over LIGHT_MAP_S old -- the peers as this page
    sees them, `boot_id` and protocol `v`. No aggregate, no wired advice,
    no Bonjour state: a browser's full /status.json has those. The node
    entry is built at most once per LIGHT_TTL_S however many peers ask (it
    runs a few `sysctl`s), by one request at a time: the others are
    answered with the last one, so N peers cost one build, never N at
    once."""
    now = time.time()
    if _LIGHT["doc"] is None or now - _LIGHT["at"] > LIGHT_TTL_S:
        # the first build is waited for; a refresh is done by whoever
        # gets there first, the rest are served the last entry
        if _LIGHT["doc"] is None:
            with _LIGHT_LOCK:
                _build_light(now)
        elif _LIGHT_LOCK.acquire(blocking=False):
            # a refresh runs off the request: the memory map walks every
            # process (seconds while a rank maps tens of GiB), and the
            # peers ask with a 1.5 s timeout -- the liveness answer must
            # never wait on it
            def refresh():
                try:
                    _build_light(time.time())
                except Exception:
                    logger.debug("light status refresh", exc_info=True)
                finally:
                    _LIGHT_LOCK.release()
            __import__("threading").Thread(
                target=refresh, daemon=True,
                name="knurlogic-light-status").start()
    from knurlogic.cluster import protocol
    return {"schema": status.SCHEMA, "nodes": [_LIGHT["doc"]],
            "boot_id": _BOOT_ID, "v": list(protocol.VERSION),
            "peers": [p.public() for p in (PEERS.all() if PEERS else [])]}


def _build_light(now: float) -> None:
    if _LIGHT["doc"] is not None and now - _LIGHT["at"] <= LIGHT_TTL_S:
        return                      # built while this request waited
    if _MM["doc"] is None or now - _MM["at"] > _map_age_limit(now):
        try:
            _MM["doc"] = footprint.memory_map()
        except (OSError, subprocess.SubprocessError, ValueError, KeyError,
                AttributeError):
            _MM["doc"] = None
        _MM["at"] = time.time()
    me = identity.identity()
    own = status.snapshot(node=me["name"], role="local",
                          memory_map=_MM["doc"])
    try:
        from knurlogic.cluster import launch
        own["cluster"] = launch.node_info(
            (own.get("memory") or {}).get("working_set_bytes") or 0)
    except (OSError, ValueError, AttributeError, KeyError, TypeError) as e:
        own["cluster"] = {"error": f"{type(e).__name__}: {e}"}
    _LIGHT.update(doc=own, at=time.time())


_LIGHT_LOCK = __import__("threading").Lock()


#: the light node entry, and how long one build of it serves
_LIGHT: dict = {"doc": None, "at": 0.0}


LIGHT_TTL_S = 1.5


#: the liveness document reuses the memory map up to this old while the
#: machine is quiet, and only HOT_MAP_S old for HOT_S after anything that
#: moves memory (a load, an unload, a request, a message from a peer) or
#: while a model here is answering one -- a card that trails a load by half
#: a minute reads as the machine not reporting
LIGHT_MAP_S = 30.0


HOT_MAP_S = 3.0


HOT_S = 30.0


_HOT: dict = {"until": 0.0}


def hot() -> None:
    """Memory is about to move here: read it often for a while."""
    _HOT["until"] = time.time() + HOT_S


def _answering() -> bool:
    """A model on this machine has a request in flight or queued, by the
    residency document the page already keeps (never a fresh survey)."""
    doc = documents._LOADED.get("doc") or {}
    for r in doc.get("resident") or []:
        q = r.get("requests") or {}
        if q.get("in_flight") or q.get("pending"):
            return True
    return False


def _map_age_limit(now: float) -> float:
    return HOT_MAP_S if now < _HOT["until"] or _answering() else LIGHT_MAP_S


#: Bonjour (cluster/discovery.py); None when it could not start.
DISCOVERY = None


def _start_discovery(me: dict, host: str, port: int, reachable: bool):
    """Browse always -- a loopback page can still reach peers outbound --
    and advertise only when bound where others can reach it: advertising an
    address nobody can connect to is the silence this replaces."""
    global DISCOVERY
    try:
        from knurlogic import __version__
        from knurlogic.cluster import discovery as dsd

        def found(services):
            for svc in services:
                txt = svc.get("txt") or {}
                if txt.get("id") == me["id"] or not PEERS:
                    continue
                pid = txt.get("id", "") or PEERS.id_of_instance(
                    svc.get("name", ""))
                p = PEERS.add(svc["host"], svc["port"], "bonjour", id=pid)
                p.id = p.id or pid
                p.name = p.name or txt.get("name", "")
        d = dsd.Discovery(on_change=found)
        txt = {"id": me["id"], "name": me["name"], "ver": __version__,
               "schema": status.SCHEMA, "role": "ui"}
        if host == "cluster":
            # Advertised on the Thunderbolt links only: a peer that can
            # only see us over the cable cannot pick Wi-Fi.
            import socket as _s

            from knurlogic.cluster import links
            for i in links.thunderbolt():
                d.if_index = _s.if_nametoindex(i["iface"])
                d.register(f"{me['name']} {me['id'][:6]}", port, txt)
            d.if_index = 0
        elif reachable:
            d.if_index = dsd.interface_of(host)
            d.register(f"{me['name']} {me['id'][:6]}", port, txt)
            d.if_index = 0
        d.browse()
        DISCOVERY = d.start()
    # Bonjour is optional; the failure is printed and peers can still be named
    except Exception as e:
        print(f"bonjour unavailable ({type(e).__name__}: {e}); peers can "
              f"still be named with --peer", file=sys.stderr)
