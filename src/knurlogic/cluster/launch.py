"""One model across several machines, launched page to page.

The coordinator is the page Launch was pressed on with two or more
machines picked. It places the ranks, picks the cable, and runs two
rounds so nothing starts unless everything can: Prepare to every page
(cluster/preparing), any refusal and nothing starts, then Start
(cluster/starting). It follows the job for a link-init failure to move to
the next cable on. Rank 0's HTTP port is where chat goes. Every message is
an envelope on /peer/v1/msg (cluster/transport.py), gated by
ui.peer_refusal; peer_step answers them.

Design: docs/design/cluster.md (launch).
"""
from __future__ import annotations

import logging
import re
import secrets
import threading
import time
from pathlib import Path

from knurlogic.cluster import (
    NET_ERRORS,
    here,
    links,
    preparing,
    recovery,
    starting,
    stopping,
    tables,
    transport,
    watching,
)
from knurlogic.cluster import jobs as J
from knurlogic.cluster.protocol import (
    FAILURE_KINDS,
    LINK_NAMES,
    backend,
    link_name,
)
from knurlogic.machine.memory.wired import available_now, budget_of
from knurlogic.machine.units import GIB, gib
from knurlogic.tuning.knobs import canonical_sets
from knurlogic.tuning.placement import placement, shape_of

logger = logging.getLogger(__name__)

#: the message kinds a page answers for a job (cluster/protocol.py)
CLUSTER_KINDS = ("Prepare", "Start", "Stop", "JobState", "Shape")
#: how long the coordinator watches a job it launched for a link-init
#: failure to move to the next cable on
FAILOVER_S = 300.0
FAILOVER_POLL_S = 2.0


# ------------------------------------------------------------ coordinator

#: Carried by every launch of RDMA across more than two Macs.
RDMA_N_NOTE = "RDMA across more than two Macs is experimental and untested"

CABLE_RX = re.compile(r"(\d{1,3}\.\d{1,3}\.\d{1,3})(?:\.(?:\d{1,3}|x|0/24))?")


def refresh_memory(infos: list, post, wait_s: float = tables.EXIT_WAIT_S,
                   sleep=time.sleep) -> bool:
    """Re-read every machine's memory now (a Survey to each peer, this
    machine directly), first waiting up to `wait_s` for ranks still exiting
    on any of them. Updates `infos` in place. -> True when any machine's
    available memory changed (a second placement may now differ)."""
    end = time.time() + wait_s
    changed = False
    while True:
        busy = False
        for m in infos:
            if m.get("page") is None:
                av, going = available_now(), len(here.exiting(J.registry()))
            else:
                try:
                    doc = post(m["page"], "Survey", {})
                except (*NET_ERRORS, transport.P.ProtocolError):
                    continue
                if not isinstance(doc, dict):
                    continue
                av = int(doc.get("available_bytes") or 0)
                going = int(doc.get("exiting") or 0)
            if going:
                busy = True
            if av and av != int(m.get("available_bytes") or 0):
                m["available_bytes"] = av
                changed = True
        if not busy or time.time() >= end:
            return changed
        sleep(1.0)


def machine_infos(ids: list, *, me: dict, peers: list,
                  local_info: dict) -> tuple:
    """([info per machine], "") for the picked node ids, in their order:
    this machine's cluster block or each answering peer's, with its id,
    name and page; ([], why) when one cannot take part."""
    by_id = {getattr(p, "id", ""): p for p in peers
             if p.state == "answering"}
    infos = []
    for nid in ids:
        if nid == me.get("id"):
            infos.append({**local_info, "id": nid, "name": me.get("name"),
                          "page": None, "links": {}})
            continue
        p = by_id.get(nid)
        if p is None:
            old = next((q for q in peers if getattr(q, "id", "") == nid
                        and q.state == "version_mismatch"), None)
            if old is not None:
                return [], (old.problem or f"{old.name or old.host} "
                            f"speaks another protocol: update knurlogic "
                            f"there")
            return [], f"{nid!r} is not a machine answering this page"
        c = (p.node or {}).get("cluster")
        if not isinstance(c, dict):
            return [], (f"{p.name or p.host} runs a knurlogic without "
                        f"cluster launch; update it")
        infos.append({**c, "id": nid, "name": p.name or p.host,
                      "page": p.key, "links": {}})
    return infos, ""


def placed_room(plan: dict, infos: list, cfg: dict, kv_bits=None) -> dict:
    """What a placement (`placement`) leaves its conversations
    (tuning/fit.split_room), each rank against its budget now."""
    from knurlogic.tuning import fit
    by = {m["name"]: m for m in infos}
    ranks = [{"name": s["machine"], "budget_bytes": budget_of(by[s["machine"]]),
              "holds_bytes": s["bytes"], "bounds": s.get("bounds")}
             for s in plan["shares"]]
    return fit.split_room(ranks, cfg, plan["split"], kv_bits)


class _Refused(Exception):
    """A launch step's final answer (an error or a refusal): `launch`
    returns `doc` as it is."""

    def __init__(self, doc: dict):
        super().__init__(doc)
        self.doc = doc


def _launch_request(req: dict) -> tuple:
    """(node ids, split, backend or None, link unnamed, identity, name) of
    a launch request, or _Refused naming what is wrong with it."""
    ids = req.get("nodes")
    if not isinstance(ids, list) or len(ids) < 2 or \
            len(set(map(str, ids))) != len(ids):
        raise _Refused({"error": "a cluster launch names two or more machines"})
    split = req.get("split") if req.get("split") in tables.SPLITS else None
    link = backend(req.get("link"))
    unnamed = req.get("link") in (None, "")
    if not split or not (link or unnamed):
        raise _Refused({"error": "split is tensor|pipeline, link is tcp|rdma"})
    ident = str(req.get("identity") or "")
    if not ident:
        raise _Refused({"error": "a cluster launch names the model by identity"})
    # what the requester called it: a directory name, never a path; each
    # rank prefers the artifact of that name among its identity's matches
    aname = Path(str(req.get("name") or "")).name[:255]
    return ids, split, link, unnamed, ident, aname


def _note_links(infos: list, me: dict, peers: list, link) -> None:
    """Each machine's `links`: the ones this page can see, its own to each
    peer, by kind (rank order follows them)."""
    by_id = {getattr(p, "id", ""): p for p in peers
             if p.state == "answering"}
    kinds = {"thunderbolt": "rdma" if link == "jaccl" else "tb4",
             "ethernet": "ethernet", "wifi": "wifi"}
    mine = next((m for m in infos if m["id"] == me.get("id")), None)
    for m in infos:
        p = by_id.get(m["id"])
        if p is not None and mine is not None:
            k = kinds.get(getattr(p, "link", ""), "")
            mine["links"][m["name"]] = k
            m["links"][mine["name"]] = k


def _unnamed_link(infos: list) -> str:
    """No link named: tcp, over the one cable the machines share; _Refused
    when they share more than one (the request must name it)."""
    shared: set | None = None
    for i, a in enumerate(infos):
        for b in infos[i + 1:]:
            nets = set(links.shared_subnets(a, b))
            shared = nets if shared is None else shared & nets
    if shared and len(shared) > 1:
        kinds = sorted({links.cable_name(links.link_gbps(infos[0], infos[1], n))
                        .split(" (")[0] for n in shared})
        raise _Refused({"refused": "link is tcp | rdma; the machines share "
                                  + ", ".join(sorted(shared)) + " over "
                                  + " and ".join(kinds) + ": name one"})
    return LINK_NAMES["tcp"]


def _launch_model(req: dict, infos: list, split: str, ident: str,
                  aname: str, post) -> tuple:
    """(request, identity, alert, launch settings, shape) of the model a
    launch runs: the shared copy over a local one that differs (the alert
    says so), its settings checked, and its shape read here when this
    machine has it, else from a peer. _Refused when it cannot run."""
    from knurlogic.machine.artifact import prefer_shared
    from knurlogic.tuning.checks import clean_sets
    from knurlogic.tuning.knobs import mtp_of, vision_of
    from knurlogic.tuning.presets import preset_or
    # a local copy beside a DIFFERENT shared (network) copy of the same
    # name: every rank loads the shared one, and the launch says so
    try:
        ident, alert = prefer_shared(ident, aname)
    except (OSError, ValueError):
        alert = ""
    if alert:
        logger.warning("cluster launch: %s", alert)
        req = dict(req, identity=ident)
    path = here.resolve(ident, aname)
    sets, bad = clean_sets(req.get("sets") or {})
    if bad:
        raise _Refused({"error": f"not a launch setting: {', '.join(bad)}"})
    if req.get("draft") is False:
        # "turn MTP off and launch": every rank's settings say so
        sets = dict(sets, KNURLOGIC_MTP="off")
    if path:
        why = preparing.sets_refusal(path, sets, preset_or(req.get("tune"), "default"))
        if why:
            raise _Refused({"refused": f"nothing started: {why}"})
    world = len(infos)
    try:
        if path:
            shape = shape_of(path, world, split, vision=vision_of(sets),
                             mtp=mtp_of(sets))
        else:
            first = next(m for m in infos if m["page"])
            shape = post(first["page"], "Shape",
                         {"identity": ident, "name": aname, "world": world,
                          "split": split, "vision": vision_of(sets),
                          "mtp": mtp_of(sets)})
            if shape.get("error"):
                raise _Refused({"error": f"{first['name']}: {shape['error']}"})
    except (*NET_ERRORS, LookupError, StopIteration, AttributeError) as e:
        raise _Refused({"error": f"could not read the model's shape: "
                                f"{type(e).__name__}: {e}"}) from e
    return req, ident, alert, sets, shape


def _place(infos: list, shape: dict, split: str, order, post) -> dict:
    """`placement`, placed once more on fresh memory before refusing: a
    peer's memory in its last status is stale right after an unload, so
    any rank still exiting is waited out and every machine read again."""
    try:
        try:
            return placement(infos, shape, split, order)
        except ValueError:
            if not refresh_memory(infos, post):
                raise
            return placement(infos, shape, split, order)
    except ValueError as e:
        raise _Refused({"refused": f"cannot place it: {e}"}) from e


def _kept_split(kept: dict, infos: list, post) -> dict:
    """A relaunch keeps the placement its first launch made (`kept`), so a
    machine still holding the old job's memory is refused, not handed a
    smaller share; memory is read once more before refusing."""
    from knurlogic.tuning import fit

    names = {m["name"] for m in infos}
    gone = next((s["machine"] for s in kept["shares"]
                 if s.get("machine") not in names), None)
    if gone:
        raise _Refused({
            "refused": f"{gone} has not answered; the split this job "
                       f"started with needs it",
            "placement": kept})

    def short() -> list:
        by = {m["name"]: m for m in infos}
        return [(s, by[s["machine"]]) for s in kept["shares"]
                if fit.share_refusal(
                    int(s["bytes"]), budget_of(by[s["machine"]]))]
    bad = short()
    if bad and refresh_memory(infos, post):
        bad = short()
    if bad:
        s, m = bad[0]
        ws, budget = int(m.get("working_set_bytes") or 0), budget_of(m)
        raise _Refused({
            "refused": f"{m['name']} still holds {gib(max(ws - budget, 0))} "
                       f"of its {gib(ws)} working set: its share of the "
                       f"split this job started with needs "
                       f"{gib(int(s['bytes']))} and it has {gib(budget)} "
                       f"now -- check {m['name']}, then load again",
            "placement": kept})
    return kept


def _auto_cable_note(order: list, net: str, nets: list, link) -> str:
    """Why a launch that named no cable runs on `net`."""
    a, b = order[0], order[1]
    sp = links.link_gbps(a, b, net)
    others = [n for n in links.shared_subnets(a, b) if n != net]
    if sp and any((links.link_gbps(a, b, n) or 0) < sp for n in others):
        note = (f"cable {net}: {links.cable_name(sp)}, the fastest "
                f"shared Thunderbolt link")
    elif sp:
        note = (f"cable {net}: {links.cable_name(sp)}, the lowest shared "
                f"Thunderbolt subnet")
    else:
        note = f"cable {net}: the lowest shared Thunderbolt subnet"
    skipped = [n for n in others if n not in nets]
    if skipped and link == "jaccl":
        note += "; not " + ", ".join(
            f"{n} ({links.cable_name(links.link_gbps(a, b, n))}: RDMA "
            f"needs Thunderbolt 5)" if links.link_gbps(a, b, n)
            else f"{n} (no RDMA at both ends)" for n in skipped)
    return note


def _choose_cable(order: list, link, cable, tried: tuple, moved,
                  plan: dict) -> tuple:
    """(subnet, note, cable ignored) for the job's link: two machines run
    on the cable the launch names, else the first shared one not yet
    `tried`; more than two pick per pair, and a named cable is ignored.
    _Refused when the named cable is not shared, or every one failed."""
    world = len(order)
    net, note, nets = "", "", []
    if world == 2:
        nets = links.shared_subnets(order[0], order[1], rdma=link == "jaccl")
    if cable not in (None, ""):
        if world > 2:
            return net, note, True
        c = CABLE_RX.fullmatch(str(cable).strip()) if isinstance(
            cable, str) else None
        if not c or c.group(1) not in nets:
            raise _Refused({"refused": f"cable {str(cable)[:40]!r}: a launch "
                                      f"names the Thunderbolt subnet both "
                                      f"machines share"
                                      + (" with RDMA up at both ends"
                                         if link == "jaccl" else "")
                                      + f" ({', '.join(nets) or 'none here'})",
                           "placement": plan})
        return c.group(1), f"cable {c.group(1)}: named by the launch", False
    if not nets:
        return net, note, False
    left = [n for n in nets if n not in tried]
    if not left:
        raise _Refused({"refused": f"every shared Thunderbolt cable failed "
                                  f"link init: {', '.join(tried)}",
                       "placement": plan})
    net = left[0]
    bad = links.BAD_CABLES.get(links.machine_pair(order[0], order[1])) or {}
    if moved:
        note = (f"cable {net}: moved from {moved['from']}, whose link "
                f"init failed ({moved['why']})")
    elif bad and net != min(nets):
        b = min(bad)
        note = (f"cable {net}: {b} failed link init earlier this "
                f"session ({bad[b]})")
    else:
        note = _auto_cable_note(order, net, nets, link)
    if left[1:]:
        note += f"; {left[1]} next if link init fails"
    return net, note, False


def _rdma_refusal(order: list, net: str, plan: dict) -> None:
    """_Refused when jaccl cannot join the machines: past two, every pair
    needs a Thunderbolt 5 cable with RDMA up; two need one shared subnet
    with it."""
    world = len(order)
    if world > 2:
        for i in range(world):
            for j in range(i + 1, world):
                why = links.rdma_pair_reason(order[i], order[j]) or (
                    "" if links.shared_subnets(order[i], order[j], rdma=True)
                    else "no Thunderbolt cable with RDMA up at both ends")
                if why:
                    raise _Refused({
                        "refused": f"RDMA across {world} Macs needs every "
                                   f"pair joined by a Thunderbolt 5 cable: "
                                   f"{order[i]['name']} and "
                                   f"{order[j]['name']}: {why}. Use the "
                                   f"TCP ring, or cable that pair.",
                        "placement": plan})
        return
    if net:
        return
    why = links.rdma_pair_reason(order[0], order[1])
    if why:
        raise _Refused({"refused": why + ". Use the TCP ring, or join them "
                                        "with a Thunderbolt 5 cable.",
                       "placement": plan})

    def up(m):
        return ", ".join((m.get("rdma") or {}).get("active") or []) or "none"
    raise _Refused({"refused": f"link rdma needs RDMA up at both ends of one "
                              f"Thunderbolt cable, and no Thunderbolt subnet "
                              f"has it ({order[0]['name']}: {up(order[0])}; "
                              f"{order[1]['name']}: {up(order[1])}). Use "
                              f"link tcp, or bring RDMA up on the shared "
                              f"cable.",
                   "placement": plan})


def _ring_layout(order: list, link, net: str, job: str, plan: dict) -> tuple:
    """(each rank's address, its host:port, the ibv device matrix, the
    jaccl coordinator) of the job's ring, on its ring-port slot."""
    world = len(order)
    try:
        ips = links.ring_ips(order, rdma=link == "jaccl", net=net)
        slot = tables.slot(job)
    except ValueError as e:
        raise _Refused({"refused": str(e), "placement": plan}) from e
    hosts = [f"{ip}:{tables.ring_port(slot, r)}" for r, ip in enumerate(ips)]
    if link != "jaccl":
        return ips, hosts, None, ""
    coord = f"{ips[0]}:{tables.ring_port(slot, tables.COORDINATOR_OFFSET)}"
    if world > 2:
        return ips, hosts, [[None if i == j else
                             links.rdma_device(order[i], order[j])
                             for j in range(world)] for i in range(world)], coord
    a0 = links.rdma_device(order[0], order[1], net)
    a1 = links.rdma_device(order[1], order[0], net)
    for m, d in ((order[0], a0), (order[1], a1)):
        if d is None:
            raise _Refused({"refused": f"{m['name']} has RDMA up on "
                                      f"{', '.join(m['rdma'].get('active') or [])}"
                                      f" and none of them is on "
                                      f"{order[1 - order.index(m)]['name']}'s "
                                      f"Thunderbolt subnet", "placement": plan})
    return ips, hosts, [[None, a0], [a1, None]], coord


def _saved_chunk(sets: dict) -> int:
    """The ring's prompt chunk before Prepare answers. The base model's
    saved chunk (Settings -> Models) is the ring's, like every launch set;
    unset, the ring runs the smallest of its ranks' room-based chunks
    (ring_chunk, once Prepare has answered; PREFILL_CHUNK until then).
    serve puts the ring's value over any --set, so the saved one must be
    resolved here or it is shown and never runs."""
    saved = canonical_sets(sets).get(tables.PREFILL_KNOB)
    if saved is None:
        return tables.PREFILL_CHUNK
    from knurlogic.tuning.checks import check_knob
    why = check_knob(tables.PREFILL_KNOB, saved)
    if why:
        raise _Refused({"error": why})
    return int(saved)


def _rank_specs(base: dict, order: list, port: int, auto_port: bool) -> list:
    """Each rank's Prepare: the job's spec and what is that rank's own."""
    return [{**base, "rank": r, "port": port if r == 0 else 0,
             "auto_port": auto_port,
             "working_set_gib": round(
                 int(m.get("working_set_bytes") or 0) / GIB, 3),
             "bandwidth_gbs": m.get("bandwidth_gbs") or 0}
            for r, m in enumerate(order)]


def _prepare_round(job: str, order: list, specs: list, ask, post,
                   auto_port: bool, plan: dict) -> tuple:
    """(answers, rank 0's port): a Prepare to every page at once, rank 0
    retried once on the free port its machine suggests when nobody named
    one. _Refused, every page told to forget the job, on any refusal."""
    got = transport.parallel(lambda r: ask("Prepare", r, specs[r]),
                             list(range(len(order))))
    port = specs[0]["port"]
    if auto_port and isinstance(got[0], dict) and got[0].get("free_port"):
        port = int(got[0]["free_port"])
        specs[0] = dict(specs[0], port=port)
        got[0] = ask("Prepare", 0, specs[0])
    bad = [(order[r]["name"], g) for r, g in enumerate(got)
           if not (isinstance(g, dict) and g.get("ok"))]
    if bad:
        _abandon(job, order, post)
        raise _Refused({"refused": "nothing started: " + "; ".join(
                           (g or {}).get("refused") or (g or {}).get("error")
                           or f"{nm} did not answer" for nm, g in bad),
                       "placement": plan})
    return got, port


def _start_round(job: str, order: list, start_doc: dict, ask, post,
                 plan: dict) -> None:
    """A Start to every page at once; _Refused, every rank stopped, when one
    did not start."""
    got = transport.parallel(lambda r: ask("Start", r, start_doc),
                             list(range(len(order))))
    bad = [(order[r]["name"], g) for r, g in enumerate(got)
           if not (isinstance(g, dict) and g.get("started"))]
    if bad:
        _abandon(job, order, post, reason="a rank did not start")
        raise _Refused({"error": "a rank did not start, so none run: "
                                + "; ".join(
                                    f"{nm}: {(g or {}).get('error') or 'no answer'}"
                                    for nm, g in bad), "placement": plan})


def launch(req: dict, *, me: dict, peers: list, local_info: dict,
           ui_port: int, serve_port: int, post=None, follow=None,
           tried: tuple = (), moved: dict | None = None,
           recovering: dict | None = None) -> dict:
    """The coordinator: plan, prepare everywhere, start everywhere.
    `me`: this machine's identity; `peers`: the answering peers
    (cluster/peers.Peer); `local_info`: this machine's cluster block.

    Two machines on more than one shared Thunderbolt cable: the job runs
    on one cable's subnet (`req["cable"]` names it, else the first of
    `links.shared_subnets`), and unless it was named the coordinator follows
    the job (`follow`, a thread by default): a rank whose link init fails
    on that cable has the job relaunched on the next one, the cable
    remembered as failing for the pair. `tried`/`moved`: that relaunch.
    `recovering`: this is auto-recovery's relaunch (cluster/recovery.py),
    carrying its report to rank 0's page; any other launch that starts is
    tracked there for recovery."""
    try:
        return _launch(req, me=me, peers=peers, local_info=local_info,
                       ui_port=ui_port, serve_port=serve_port,
                       post=post or transport.send, follow=follow,
                       tried=tried, moved=moved, recovering=recovering)
    except _Refused as a:
        return a.doc


def _launch(req: dict, *, me: dict, peers: list, local_info: dict,
            ui_port: int, serve_port: int, post, follow, tried: tuple,
            moved: dict | None, recovering: dict | None) -> dict:
    """`launch`'s steps, in order; a step that ends the launch raises
    _Refused with what it answers."""
    from knurlogic.tuning.presets import preset_or
    ids, split, link, unnamed, ident, aname = _launch_request(req)
    infos, why = machine_infos(ids, me=me, peers=peers,
                               local_info=local_info)
    if why:
        raise _Refused({"error": why})
    _note_links(infos, me, peers, link)
    if unnamed:
        link = _unnamed_link(infos)
    if link == "jaccl":
        for m in infos:
            rd = m.get("rdma") or {}
            if not rd.get("available"):
                raise _Refused({"error": f"RDMA on {m['name']}: "
                                         f"{rd.get('reason') or 'unknown'}"})
    world = len(infos)
    req, ident, alert, sets, shape = _launch_model(req, infos, split, ident,
                                                   aname, post)
    kept = req.get("placement") if recovering is not None else None
    plan = _kept_split(kept, infos, post) \
        if isinstance(kept, dict) and kept.get("shares") \
        else _place(infos, shape, split, req.get("order"), post)
    order = [next(m for m in infos if m["name"] == nm) for nm in plan["order"]]
    job = secrets.token_hex(8)
    cable = req.get("cable")
    net, note, cable_ignored = _choose_cable(order, link, cable, tried, moved,
                                             plan)
    if link == "jaccl":
        _rdma_refusal(order, net, plan)
    ips, hosts, ibv, coord = _ring_layout(order, link, net, job, plan)
    if cable_ignored:
        note = (note + "; " if note else "") + (
            "cable ignored: with more than two Macs each pair picks its own "
            "Thunderbolt link")
    my_tb = ips[plan["order"].index(me.get("name"))] \
        if me.get("name") in plan["order"] else None
    nodes = [{"rank": r, "id": m["id"], "name": m["name"],
              "page": m["page"] or (f"{my_tb}:{ui_port}" if my_tb else "")}
             for r, m in enumerate(order)]
    port = req.get("port")
    auto_port = not (isinstance(port, int) and 1024 <= port < 65536)
    if auto_port:
        # nobody named one: the first free port from the default up (the
        # other machines answer for theirs at prepare, below)
        from knurlogic.machine.servers import free_port
        port = free_port(serve_port)
    assert isinstance(port, int)        # the one named, or a free one
    base = {"job": job, "world": world, "split": split, "link": link,
            "identity": ident, "name": aname, "hosts": hosts, "ibv_devices": ibv,
            # the bell's nonce: random, in the spec the pages share, never
            # shown (the job id is on the page)
            "bell_nonce": starting.bell_nonce(),
            "coordinator": coord, "layers": plan["layers"],
            "prefill_chunk": _saved_chunk(sets),
            "tune": preset_or(req.get("tune"), "default"),
            "nodes": nodes, "versions": local_info.get("versions") or {},
            "jaccl_timeout_ms": J.JACCL_TIMEOUT_MS if link == "jaccl" else 0,
            "sets": sets, "cable": net, "cable_note": note[:400],
            # each machine's chip and GPU architecture, rank order: the
            # ranks name them when per-chip rounding is off
            "chips": [{"name": m.get("chip") or m.get("name") or "",
                       "arch": m.get("gpu_architecture") or ""}
                      for m in order],
            "recovery": recovering,
            # rank 0 binds loopback and its own link address
            "serve_hosts": ["127.0.0.1", ips[0]]}

    def ask(kind, r, doc):
        m = order[r]
        if m["page"] is None:
            return peer_step(kind, doc)[1]
        return post(m["page"], kind, doc)

    got, port = _prepare_round(job, order,
                               _rank_specs(base, order, port, auto_port),
                               ask, post, auto_port, plan)
    alerts = ([alert] if alert else []) + (
        [RDMA_N_NOTE] if link == "jaccl" and world > 2 else []) + [
        g["alert"] for g in got if isinstance(g, dict) and g.get("alert")]
    for a in alerts[1 if alert else 0:]:
        logger.warning("cluster job %s: %s", job, a)
    chunk, chunk_why = preparing.ring_chunk(sets, got)
    base["prefill_chunk"] = chunk
    # who reads from whose disk, and in what order: a rank on a network
    # mount fetches its share from a machine with a local copy, over the
    # job's link, and that machine reads its own share after
    from knurlogic.cluster import weights
    reads = weights.read_plan(
        [m["name"] for m in order],
        [bool(isinstance(g, dict) and g.get("local_copy")) for g in got],
        [f"{ips[r]}:{(n['page'] or '').rpartition(':')[2] or ui_port}"
         for r, n in enumerate(nodes)])
    start_doc = {"job": job, "prefill_chunk": chunk, "reads": reads}
    if chunk_why != "set":
        start_doc["prefill_why"] = chunk_why
    _start_round(job, order, start_doc, ask, post, plan)
    args = {"me": me, "peers": peers, "local_info": local_info,
            "ui_port": ui_port, "serve_port": serve_port,
            "post": post, "follow": follow}
    if net and cable in (None, ""):
        (follow or _follow_thread)(job, {
            "req": req, "net": net, "tried": tuple(tried) + (net,),
            "order": order, "link": link, "post": post, "args": dict(args)})
    if note:
        logger.warning("cluster job %s: %s", job, note)
    if recovering is None:
        # a relaunch is the same launch: this machine order and placement
        # (so the same split), this port, link, tune and settings
        recovery.track_cluster(
            job, req=dict(req, order=list(plan["order"]), port=port,
                          placement=plan),
            args=dict(args), order=order, port=port,
            leader_here=order[0]["page"] is None,
            previous=(moved or {}).get("job"))
    url = tables.leader_url(base | {"port": port})
    return {"job": job, "starting": ident, "placement": plan,
            "leader": plan["leader"], "port": port,
            "link": link_name(link), "url": url,
            "machines": plan["order"], "cable": net, "cable_note": note,
            **({"alerts": alerts} if alerts else {}),
            "note": f"rank 0 on {plan['leader']} serves on port {port} "
                    f"(loopback there, {url} from the job's machines) once "
                    f"every rank has loaded; poll /loaded.json"}


def job_end(job: str, order: list, post):
    """None while `job` runs (or loads) on every page, else why it ended:
    this page's record, or a page of the job saying it ended there."""
    e = tables.ENDED.get(job)
    if e:
        return e.get("reason") or "stopped"
    for m in order:
        if m.get("page") is None:
            continue
        doc = transport.answer(m["page"], "JobState", {"job": job}, post)
        if doc and doc.get("ended"):
            return str(doc["ended"])
    return None


def failover(job: str, ctx: dict, reason: str):
    """`job` ended for `reason`: when that is a rank's link init failing,
    remember the cable as failing for the pair and relaunch on the next
    one (-> the relaunch's answer); else None."""
    line = tables.link_init_failure(reason)
    if not line:
        return None
    order = ctx["order"]
    if len(order) != 2:
        # a cable is only chosen for two machines; at more, the ring names
        # its own addresses and there is no cable to move
        logger.warning("cluster job %s: link init failed on %d machines "
                       "(%s); no cable failover beyond two", job,
                       len(order), line)
        return None
    pair = links.machine_pair(order[0], order[1])
    links.BAD_CABLES.setdefault(pair, {})[ctx["net"]] = line
    out = launch(ctx["req"], tried=ctx["tried"],
                 moved={"from": ctx["net"], "why": line, "job": job},
                 **ctx["args"])
    to = out.get("job")
    msg = (f"; relaunched on cable {out.get('cable')} as job {to}" if to
           else f"; not relaunched: {out.get('refused') or out.get('error')}")
    with tables.LOCK:
        e = tables.ENDED.setdefault(job, {"reason": reason, "t": time.time(),
                                   "port": None, "machines": [
                                       m.get("name") for m in ctx["order"]]})
        e["reason"] = (str(e.get("reason") or reason) + msg)[:600]
        if to:
            e["relaunched"] = to
    logger.warning("cluster job %s: cable %s failed link init (%s)%s",
                   job, ctx["net"], line, msg)
    return out


def _follow(job: str, ctx: dict, clock=time.time, sleep=time.sleep):
    """The coordinator's watch on a job it launched, until it is serving
    (ready here) or FAILOVER_S: a link-init failure moves it to the next
    cable (`failover`)."""
    tables.FOLLOWING.add(job)
    try:
        end = clock() + FAILOVER_S
        while clock() < end:
            sleep(FAILOVER_POLL_S)
            why = job_end(job, ctx["order"], ctx["post"])
            if why is not None:
                return failover(job, ctx, why)
            recs = J.by_job().get(job)
            if recs and J.phase_of(job, recs) == "ready":
                return None
        return None
    finally:
        tables.FOLLOWING.discard(job)


def _follow_thread(job: str, ctx: dict) -> None:
    threading.Thread(target=_follow, args=(job, ctx), daemon=True,
                     name=f"knurlogic-cluster-follow-{job}").start()


def _abandon(job: str, order: list, post, reason: str = "refused") -> None:
    """Every page forgets a job that will not run (and stops any rank it
    already started)."""
    for m in order:
        try:
            if m["page"] is None:
                stopping.stop(job, reason=reason, propagate=False)
            else:
                post(m["page"], "Stop", {"job": job, "reason": reason})
        except NET_ERRORS:
            logger.debug("could not stop job %s on %s", job, m.get("page"),
                     exc_info=True)


# ------------------------------------------------------------ peer routes

PEER_MAX = 16 << 10


def peer_step(kind: str, req: dict) -> tuple:
    """A cluster message from another page (it has passed peer_refusal and
    the envelope check) -- or this page's own step, for its own rank:
    (status, doc)."""
    if not isinstance(req, dict):
        return 400, {"error": "the body must be a JSON object"}
    from knurlogic.tuning.checks import PATH_KEYS
    if any(k in req for k in PATH_KEYS):
        return 400, {"error": "a cluster job names its model by identity, "
                              "never by a path"}
    if kind == "Prepare":
        return preparing.prepare(req)
    if kind == "Start":
        return starting.start(req.get("job"), ring=req)
    if kind == "Stop":
        if not J.JOB_RX.fullmatch(str(req.get("job") or "")):
            return 400, {"error": "job is a hex nonce"}
        reason = str(req.get("reason") or "stopped by another machine")[:300]
        k = req.get("kind")
        return 200, stopping.stop(req["job"], reason=reason, propagate=False,
                         kind=k if k in (*FAILURE_KINDS, "requested")
                         else None, wait=False)
    if kind == "JobState":
        if not J.JOB_RX.fullmatch(str(req.get("job") or "")):
            return 400, {"error": "job is a hex nonce"}
        return 200, watching.job_state(req["job"])
    if kind == "Shape":
        p = here.resolve(req.get("identity"), str(req.get("name") or ""))
        if not p:
            return 404, {"error": "no artifact with that identity here"}
        w, s = req.get("world"), req.get("split")
        if not isinstance(w, int) or s not in tables.SPLITS:
            return 400, {"error": "world and split"}
        return 200, shape_of(p, w, s, req.get("vision") is not False,
                             req.get("mtp") is not False)
    return 404, {"error": "not a cluster message"}
