"""exo as ONE source of which machines exist -- a witness, not the authority.

What exo's /state says about each node (name, RAM, swap, chip, address),
and the snapshot a page draws for a node only exo knows about. Nodes that
answer for themselves come from cluster/peers.py and cluster/discovery.py
and win over this: a knurlogic on the node measures it; exo reports
system RAM (docs/DISCOVERY.md, review item 9). Read-only: knurlogic
never launches, places on or proxies to exo.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from knurlogic.machine import status, wired

GIB = 1 << 30

#: Where exo answers, asked only WHO IS THERE. KNURLOGIC_EXO_URL overrides.
EXO_URL = os.environ.get("KNURLOGIC_EXO_URL", "http://127.0.0.1:52415")

def _get(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _bytes(v) -> int:
    """exo's Memory serializes as {"inBytes": N}; accept a bare int too."""
    if isinstance(v, dict):
        return int(v.get("inBytes", v.get("in_bytes", 0)) or 0)
    return int(v or 0)


def _pick(d: dict, *names):
    for n in names:
        if n in d:
            return d[n]
    return {}


@dataclass
class ExoNode:
    node_id: str
    name: str
    ram_total: int
    ram_available: int
    model_id: str = ""
    ip: str = ""
    chip: str = ""          # "M4 Max": exo's chipId, "Apple " dropped
    swap_used: int | None = None


def _node_ip(info: dict) -> str:
    """A routable address for a node, from exo's own interface list.

    Loopback is skipped: every node reports 127.0.0.1 and asking THAT for a
    peer's status would return this box's answer for all of them -- the same
    shape as the bare-`python3` version check that answered for the system
    interpreter every iteration.
    """
    best = ""
    for i in info.get("interfaces") or []:
        ip = i.get("ipAddress") or i.get("ip_address") or ""
        if not ip or ":" in ip or ip.startswith("127."):
            continue
        if not best or i.get("interfaceType") in ("thunderbolt", "ethernet"):
            best = ip
    return best


def _swap_used(m: dict):
    """Swap in use from exo's per-node memory, or None when it did not say
    -- a node that never reported swap is not a node with none."""
    tot = _pick(m, "swapTotal", "swap_total")
    free = _pick(m, "swapAvailable", "swap_available")
    if tot == {} or free == {}:           # _pick answers {} for absent
        return None
    return max(_bytes(tot) - _bytes(free), 0)


def inventory(exo_url: str) -> list:
    """The nodes exo can see, with the memory it reports for each.

    CAVEAT, and it is printed next to the numbers: exo reports SYSTEM RAM
    (psutil), not the Metal recommended working set that `resolve` wants.
    They are close on an Apple box and they are not the same number, which
    is why a node that answers for itself (cluster/peers.py) wins over this.
    """
    state = _get(f"{exo_url}/state")
    mem = _pick(state, "nodeMemory", "node_memory")
    ident = _pick(state, "nodeIdentities", "node_identities")
    net = _pick(state, "nodeNetwork", "node_network")
    out = []
    for node_id, m in sorted(mem.items()):
        who = ident.get(node_id, {}) or {}
        out.append(ExoNode(
            node_id=node_id,
            name=who.get("friendlyName") or who.get("friendly_name")
            or node_id[:12],
            ram_total=_bytes(_pick(m, "ramTotal", "ram_total")),
            ram_available=_bytes(_pick(m, "ramAvailable", "ram_available")),
            # exo's `modelId` is the PRODUCT NAME -- measured against the
            # live daemon, which reports "Mac Studio" and "MacBook Pro", not
            # `Mac15,14`. That is a stronger channel than anything this end
            # can infer, so it is used directly rather than guessed at.
            model_id=who.get("modelId") or who.get("model_id") or "",
            ip=_node_ip(net.get(node_id) or {}),
            chip=str(who.get("chipId") or who.get("chip_id") or "")
            .removeprefix("Apple ").strip(),
            swap_used=_swap_used(m),
        ))
    return out


#: A peer's knurlogic, asked for the one thing only that node can answer:
#: which of ITS processes is holding ITS memory. Cached per node, because the
#: page polls every two seconds and a cross-network fetch is not free.
_PEER: dict = {}


#: knurlogic's own default, tried after whatever port this front end is on.
#: exo cannot help here: it resolves exo's topology and has no idea knurlogic
#: exists, so there is nothing in its state that says where a peer knurlogic
#: listens. Asking a couple of likely ports is the honest substitute for
#: assuming exactly one, and the port that answers is remembered per node.
PEER_PORTS = (8080, 8899)     # serve's default, then the page's


#: The metrics a peer reported beside its memory map, by address. Filled by
#: the same request, so asking for them costs nothing extra.
_PEER_METRICS: dict = {}


def peer_memory_map(ip: str, port: int, ttl: float = 6.0) -> dict | None:
    """The memory map a knurlogic on another node reports for itself.

    This is the whole reason it is worth running knurlogic on every node:
    process footprints are true only of the machine they were read on, so
    the node has to answer for itself.
    exo reports RAM totals per node and nothing about who is spending it.

    Absent is a normal answer, and the COMMON cause is not a missing
    knurlogic: `serve` binds 127.0.0.1 by default, so a peer running one is
    reachable only on its own loopback. That default is deliberate -- a model
    endpoint should not appear on the network because somebody started it --
    so this does not work around it. A peer that should answer needs
    `--host 0.0.0.0` (or its own address) chosen on purpose.

    Without it the node is still drawn, from exo's own per-node RAM figures.
    What is lost is only the split by runtime, which nothing else can supply.
    """
    import time

    if not ip:
        return None
    now = time.time()
    hit = _PEER.get(ip)
    if hit and now - hit[0] < ttl:
        return hit[1]

    # A port that answered before is tried first, so the common case is one
    # request rather than a sweep every refresh.
    tried, doc = [], None
    known = hit[2] if hit and len(hit) > 2 else None
    for cand in ([known] if known else []) + [port, *PEER_PORTS]:
        if not cand or cand in tried:
            continue
        tried.append(cand)
        try:
            # 3 s, not 1: a busy peer building its map was measured at 1.4 s,
            # and a timeout shorter than that drops the node every time.
            # The answer is cached for `ttl`, so this is not paid per poll.
            d = _get(f"http://{ip}:{cand}/status.json", timeout=3.0)
        except Exception:
            continue
        # The peer's OWN entry: its status lists every node it knows, and
        # the first map in the list need not be the one it measured.
        for nd in (d or {}).get("nodes") or []:
            if nd.get("memory_map") and nd.get("role") in ("local", "server"):
                doc = nd["memory_map"]
                _PEER_METRICS[ip] = nd.get("metrics")
                break
        if doc is not None:
            _PEER[ip] = (now, doc, cand)
            return doc
    _PEER[ip] = (now, None, known)
    return None


def _snapshot_for(n: ExoNode, local_name: str | None,
                  peer_port: int = 0) -> dict:
    """A per-node status built from what exo reports about that node.

    Only `active` is knowable this way -- total minus available is everything
    on the box, not this runtime -- so it is labelled and the reclaimable
    cache, which is the number `serve` exists to separate out, is NOT faked
    as zero: it is absent, and the renderer shows what it has.
    """
    used = max(n.ram_total - n.ram_available, 0)
    ws = n.ram_total
    device = "reported by exo (system RAM, not the Metal working set)"
    # The LOCAL node can be asked directly; every other node gets the guess
    # made from what it reported. Handing them all wired.machine() would
    # label the whole cluster with this box.
    is_local = n.name == local_name
    mm = None
    if is_local:
        from knurlogic.machine import loaded
        try:
            mm = loaded.memory_map()
        except Exception:
            mm = None
    elif peer_port:
        # A peer is asked on the port this page's servers use: knurlogic
        # on every node is the assumption the feature rests on, and a node
        # that is not running one simply does not answer.
        mm = peer_memory_map(n.ip, peer_port)
    return status.snapshot(
        node=n.name,
        role="local" if is_local else "remote",
        machine_fn=(None if is_local
                    else lambda: {**wired.kind_from(n.name,
                                                    product=n.model_id),
                                  "chip": n.chip}),
        memory_map=mm,
        # A peer's own knurlogic sends its lines; without one, exo still
        # knows its swap, which is the reading that matters most.
        metrics=(None if is_local else
                 (mm and _PEER_METRICS.get(n.ip))
                 or ({"now": {"swap_bytes": n.swap_used}, "history": []}
                     if n.swap_used is not None else None)),
        memory_fn=lambda: {
            "available": ws > 0,
            "device": device,
            # Box-wide: total minus available is everything running on that
            # machine. exo reports the box; only a snapshot taken inside the
            # runtime can separate weights from reclaimable cache.
            "scope": "box",
            "active_bytes": used,
            "cache_bytes": 0,
            "peak_bytes": 0,
            "working_set_bytes": ws,
            "total_bytes": n.ram_total,
            "headroom_bytes": max(ws - used, 0),
            "process_rss_bytes": 0,
        })
