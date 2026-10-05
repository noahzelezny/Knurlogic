"""One model across several machines, launched page to page.

The coordinator is the page Launch was pressed on with two or more
machines picked. Every page starts only its own ranks, after checking the
request against its own disk, memory, software and links, in two phases so
nothing starts unless everything can: `prepare` (a Prepare message to
every page; any refusal and nothing starts) then `start` (a Start message).
Any rank dying or stalling stops the whole job on every page. A stop is
done when the ranks' processes are gone, not when they were signalled.
Rank 0's HTTP port is where chat goes. Every message is an envelope on
/peer/v1/msg (cluster/transport.py), gated by ui.peer_refusal.

Design: docs/design/cluster.md (launch).
"""
from __future__ import annotations

import json
import logging
import os
import re
import secrets
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from knurlogic.cluster import NET_ERRORS, PROC_ERRORS, transport
from knurlogic.cluster import jobs as J
from knurlogic.cluster.protocol import (
    FAILURE_KINDS,
    JobState,
    PrepareReply,
    Started,
    Stopped,
    typed,
)

logger = logging.getLogger(__name__)


def _no_status():
    raise LookupError("no page has set cluster_jobs.status_fn")


# What this module needs of the page that runs it, injected by that page at
# startup (interfaces/page/server.py) so this module never imports it.
# Unset, this machine is alone: no status snapshot (node_info() answers)
# and no peers.
#: () -> (status snapshot, _): the page's own status document
status_fn: Callable = _no_status
#: () -> [peer record]: the page's PEERS store
peers_fn: Callable[[], list] = list

GIB = 1 << 30
#: the message kinds a page answers for a job (cluster/protocol.py)
CLUSTER_KINDS = ("Prepare", "Start", "Stop", "JobState", "Shape")
#: a prepare's fields; nothing else is read, and a path is refused
SPEC_KEYS = ("job", "rank", "world", "split", "link", "identity", "hosts",
             "ibv_devices", "coordinator", "layers", "prefill_chunk", "tune",
             "port", "working_set_gib", "bandwidth_gbs", "nodes", "versions",
             "jaccl_timeout_ms", "sets", "chips", "cable", "cable_note",
             "recovery", "serve_hosts", "name", "auto_port", "bell_nonce",
             "prefill_why")
SPLITS = ("tensor", "pipeline")
LINKS = ("ring", "jaccl")
#: the link names people see (load(), state(), the page, the recovery
#: record) -> mlx's distributed backend. The ONE place they are mapped:
#: ring/jaccl are only ever the backend argument inside a job's spec.
LINK_NAMES = {"tcp": "ring", "rdma": "jaccl"}


def backend(link) -> str | None:
    """mlx's backend for a link named either way (tcp|rdma, or an older
    record's ring|jaccl); None when it is neither."""
    if link in LINK_NAMES:
        return LINK_NAMES[link]
    return link if link in LINKS else None


def link_name(link):
    """The name people see for a link named either way: tcp | rdma."""
    return {v: k for k, v in LINK_NAMES.items()}.get(link, link)


def leader_url(spec: dict) -> str:
    """The job's API as the other machines reach it: rank 0's link
    address (the leader binds loopback and that), or '' when unknown."""
    hs = [h for h in (spec.get("serve_hosts") or [])
          if h not in ("127.0.0.1", "::1", "localhost")]
    port = spec.get("port")
    return f"http://{hs[0]}:{int(port)}/v1" if hs and port else ""

#: the prompt chunk a ring runs when a rank cannot work out its own room's:
#: the floor. Otherwise it is the smallest of the ranks' room-based chunks
#: (each rank answers Prepare with its own; see ring_chunk)
PREFILL_CHUNK = 512
RING_CHUNK_WHY = "ring: the smallest rank's room"
#: first ring port; a job's ranks take RING_PORT + slot*20 + rank, and
#: the jaccl coordinator slot*20 + 19
RING_PORT = 47200
#: a prepared job not started within this long is forgotten
PREPARED_S = 120.0
#: how long a prepare waits for a stopping rank of an earlier job to be gone
EXIT_WAIT_S = 30.0
#: how often the page checks its ranks
WATCH_S = 2.0
#: how long start waits for a stopped rank's process on this machine to be
#: gone before it refuses: two jobs' shares in one working set is an OOM
#: that can reboot the machine (a 397B share loading beside the last job's
#: 50 GiB)
START_WAIT_S = 20.0

#: job -> prepared spec (+ resolved path), awaiting start
PREPARED: dict = {}
#: job -> {"reason", "t", "machines"} for jobs that stopped, shown a while
ENDED: dict = {}
#: job -> spec, for the jobs this page has ranks in (to stop the others)
SPECS: dict = {}
#: job -> Popen for ranks this page process started (exit codes, reaping)
_PROCS: dict = {}
_LOCK = threading.Lock()
_WATCH = J.Watch()
_WATCHER: list = []
#: one start at a time on this page: the loading check and the spawn are
#: one step, so two jobs' ranks never begin loading together
_START_LOCK = threading.Lock()
#: jobs a stop() in this process is tearing down right now
_STOPPING: set = set()
#: (job, peer id) -> when that peer's page last said it runs its rank
_PEER_OK: dict = {}
#: the boot_id a peer had when it held its rank of a job: {(job, id): id}
_PEER_BOOT: dict = {}
#: frozenset of two machine ids -> {subnet: why} -- a cable a rank's link
#: init failed on while this page runs (e.g. one cable failing jaccl QP
#: RTR with errno 96 while the other works): tried last from then on
BAD_CABLES: dict = {}
#: how long the coordinator watches a job it launched for a link-init
#: failure to move to the next cable on
FAILOVER_S = 300.0
FAILOVER_POLL_S = 2.0
#: jobs a coordinator's _follow is watching now (recovery leaves a link-init
#: failure of one of these to the cable failover)
FOLLOWING: set = set()
#: a rank's link init failing: its log line (mlx's jaccl / ring errors)
LINK_INIT_RX = re.compile(
    r"[^\n]*\[(?:jaccl|ring)\][^\n]*(?:RTR|RTS|queue pair|connect|"
    r"Connection)[^\n]*", re.I)


# ------------------------------------------------------------ this machine

_INFO: dict = {"doc": None, "at": 0.0}


def _chip() -> str:
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True,
                              timeout=5).stdout.strip()
    except PROC_ERRORS:
        return ""


def _mlx_version() -> str:
    from importlib import metadata
    try:
        return metadata.version("mlx")
    except metadata.PackageNotFoundError:
        return ""


_BUILD: dict = {}


def build_fingerprint(root=None, cache=None) -> str:
    """Which build of knurlogic this is, beyond its version string (every
    dev build says 0.1.0.dev0): a hash over the installed package's .py
    files -- relative path and bytes, in a fixed order, so a wheel install
    and a PYTHONPATH source tree of the same commit agree -- and the mlx
    version, as "<12 hex>+mlx<version>". Read once per process."""
    import hashlib
    memo = _BUILD if cache is None else cache
    if root is None:
        import knurlogic
        root = Path(knurlogic.__file__).resolve().parent
    root = Path(root)
    key = str(root)
    if key not in memo:
        h = hashlib.sha256()
        for f in sorted(root.rglob("*.py"),
                        key=lambda q: q.relative_to(root).as_posix()):
            if "__pycache__" in f.parts:
                continue
            h.update(f.relative_to(root).as_posix().encode() + b"\0")
            try:
                h.update(f.read_bytes())
            except OSError:
                pass
            h.update(b"\0")
        memo[key] = h.hexdigest()[:12]
    return f"{memo[key]}+mlx{_mlx_version() or 'none'}"


def _selfheal() -> bool:
    from knurlogic.machine import deps
    m = deps.probe(sys.executable).get("mlx") or {}
    return bool(m.get("jaccl_selfheal"))


def available_now() -> int:
    """Bytes of memory macOS would hand this machine's next allocation now
    (free plus file cache and purgeable pages, not inactive
    anonymous ones: machine/loaded.available_memory), 0 when
    it cannot be read. Read fresh each time, never cached: it is the
    difference between a working set the GPU is ALLOWED and memory that is
    actually there (a 96 GiB Mac's 84 GiB working set leaves the OS and
    every other program 12 GiB; a share that fills the working set swaps)."""
    from knurlogic.machine.loaded import available_memory
    try:
        return int(available_memory().get("available_bytes") or 0)
    except (OSError, ValueError, TypeError, AttributeError):
        return 0


def budget_of(m: dict) -> int:
    """What a rank on machine `m` may hold: its working set (under its
    allowance), lowered to the memory available now when the machine says
    it (`available_bytes`; a peer that predates it says nothing)."""
    ws = int(m.get("working_set_bytes") or 0)
    av = int(m.get("available_bytes") or 0)
    return min(ws, av) if ws and av else ws


def node_info(working_set_bytes: int = 0, ttl: float = 30.0) -> dict:
    """What a coordinator needs to know of THIS machine, for its status'
    `cluster` block. Cached: it runs a few subprocesses."""
    now = time.time()
    doc = _INFO["doc"]
    if doc is None or now - _INFO["at"] > ttl:
        from knurlogic import __version__
        from knurlogic.cluster import links
        from knurlogic.tuning.resolve import chip_bandwidth_gbs
        chip = _chip()
        try:
            tb = [{"iface": i["iface"], "ip": i["ip"],
                   "gbps": i.get("gbps"), "generation": i.get("generation")}
                  for i in links.thunderbolt()]
        except (*PROC_ERRORS, ValueError, KeyError):
            tb = []
        try:
            heal = _selfheal()
        except (*PROC_ERRORS, ValueError):
            heal = False
        from knurlogic.engine.crosschip import gpu_architecture
        doc = {"chip": chip, "gpu_architecture": gpu_architecture(),
               "p_core_ghz": None,
               "bandwidth_gbs": chip_bandwidth_gbs(chip),
               "thunderbolt": tb, "rdma": links.rdma(),
               "versions": {"knurlogic": __version__,
                            "mlx": _mlx_version(),
                            "build": build_fingerprint()},
               "jaccl_selfheal": heal}
        _INFO.update(doc=doc, at=now)
    from knurlogic.machine import allowance
    return dict(doc, working_set_bytes=allowance.cap(
        gpu_working_set(int(working_set_bytes or 0))),
        available_bytes=available_now())


def gpu_working_set(installed: int, wired_limit=None) -> int:
    """What a rank may use on this machine: the GPU's wired limit
    (iogpu.wired_limit_mb -- what Metal recommends, what one-machine `serve`
    guards), never the installed RAM the page's status reports. The M3
    Ultra's 96 GiB has 84 wired; placing a 90 GiB share there on RAM
    alone would pass prepare and fail at load."""
    if wired_limit is None:
        from knurlogic.machine import wired
        wired_limit = wired.read().limit_bytes
    known = [b for b in (int(installed or 0), int(wired_limit or 0)) if b > 0]
    return min(known) if known else 0


# ------------------------------------------------------------ plan

def _subnet(ip: str) -> str:
    return ".".join(str(ip).split(".")[:3])


def _pair(a: dict, b: dict) -> frozenset:
    return frozenset((str(a.get("id") or a.get("name") or ""),
                      str(b.get("id") or b.get("name") or "")))


def _link_gbps(a: dict, b: dict, net: str):
    """The speed of the cable on subnet `net` between two machines: the
    slower end's Gb/s, None when either end does not say (an older
    knurlogic)."""
    sp = []
    for m in (a, b):
        g = next((t.get("gbps") for t in m.get("thunderbolt") or []
                  if t.get("ip") and _subnet(t["ip"]) == net), None)
        if not isinstance(g, (int, float)) or isinstance(g, bool) or g <= 0:
            return None
        sp.append(g)
    return min(sp)


def _cable_name(gbps) -> str:
    from knurlogic.cluster.links import generation
    g = generation(gbps)
    return f"Thunderbolt {g} ({gbps:g} Gb/s)" if g else "speed unknown"


def _shared_subnets(a: dict, b: dict, rdma: bool = False) -> list:
    """Every Thunderbolt /24 two machines both sit on, the order a launch
    tries them: the fastest cable first (Thunderbolt 5 over Thunderbolt 4;
    a link whose speed an older peer does not report counts as slowest),
    then the lowest subnet, so every page picks the same -- except that a
    cable whose link init failed between these two this session goes last.
    `rdma`: only a subnet with RDMA up at both ends AND a Thunderbolt 5
    link at both ends: over a 40 Gb/s Thunderbolt 4 cable ibv_devinfo
    still says PORT_ACTIVE, and jaccl fails RTR with errno 96."""
    from knurlogic.cluster.links import TB5_GBPS

    def on(m):
        act = set((m.get("rdma") or {}).get("active") or [])
        return {_subnet(t["ip"]) for t in m.get("thunderbolt") or []
                if t.get("ip") and (not rdma
                                    or f"rdma_{t.get('iface')}" in act)}
    bad = BAD_CABLES.get(_pair(a, b)) or {}
    both = on(a) & on(b)
    if rdma:
        both = {n for n in both
                if (_link_gbps(a, b, n) or TB5_GBPS) >= TB5_GBPS}
    return sorted(both, key=lambda n: (n in bad, -(_link_gbps(a, b, n) or 0),
                                       n))


def rdma_pair_reason(a: dict, b: dict) -> str:
    """Why jaccl cannot join these two machines ("" when it can): RDMA off
    at either end, or no Thunderbolt 5 cable between them."""
    from knurlogic.cluster.links import TB5_GBPS
    for m in (a, b):
        rd = m.get("rdma") or {}
        if not rd.get("available"):
            return f"RDMA on {m.get('name')}: {rd.get('reason') or 'unknown'}"
    if _shared_subnets(a, b, rdma=True):
        return ""
    slow = [n for n in _shared_subnets(a, b)
            if (_link_gbps(a, b, n) or TB5_GBPS) < TB5_GBPS]
    if slow:
        return ("RDMA needs a Thunderbolt 5 cable between these Macs; "
                + "; ".join(f"the {n} link is "
                            f"{_cable_name(_link_gbps(a, b, n))}"
                            for n in slow))
    return ""


def link_init_failure(text: str) -> str:
    """The line of a rank's output saying its link init failed ("" if
    none): jaccl's queue pair not reaching RTR, a ring that could not
    connect."""
    m = LINK_INIT_RX.search(text or "")
    return m.group(0).strip()[:240] if m else ""


def _log_tail(path, n: int = 64 << 10) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(fh.tell() - n, 0))
            return fh.read().decode("utf-8", "replace")
    except (OSError, TypeError):
        return ""


def _shared_subnet(a: dict, b: dict, rdma: bool = False) -> str:
    """The one Thunderbolt /24 two machines both sit on ("" if none) --
    lowest first, so every page picks the same. Two Macs joined by two
    cables share two subnets; BOTH ends of a link must be on the same one
    (en2 on one /24 and en4 on another are different cables, and a jaccl queue
    pair across them fails RTR with errno 60).
    `rdma`: only a subnet whose interface has RDMA up on both ends. A
    cable that failed link init between the two this session goes last
    (`_shared_subnets`)."""
    both = _shared_subnets(a, b, rdma)
    return both[0] if both else ""


def _on_subnet(m: dict, net: str, key: str = "ip"):
    return next((t.get(key) for t in m.get("thunderbolt") or []
                 if t.get("ip") and _subnet(t["ip"]) == net), None)


def _ring_ips(infos: list, rdma: bool = False, net: str = "") -> list:
    """Each rank's address on the ring, in rank order. Two machines: both
    on the subnet they share (`_shared_subnet`). More: a Thunderbolt
    address sharing a /24 with a neighbour's when there is one, else its
    first. Raises ValueError naming a machine with none."""
    ips = []
    n = len(infos)
    for _r, m in enumerate(infos):
        mine = [t["ip"] for t in m.get("thunderbolt") or [] if t.get("ip")]
        if not mine:
            raise ValueError(f"{m['name']} has no Thunderbolt address: the "
                             f"ring runs over the Thunderbolt bridge")
    if n == 2:
        net = net or _shared_subnet(infos[0], infos[1], rdma) \
            or _shared_subnet(infos[0], infos[1])
        if net:
            return [_on_subnet(m, net) for m in infos]
    for r, m in enumerate(infos):
        mine = [t["ip"] for t in m.get("thunderbolt") or [] if t.get("ip")]
        near = {_subnet(t["ip"]) for k in ((r - 1) % n, (r + 1) % n)
                for t in infos[k].get("thunderbolt") or []}
        ips.append(next((ip for ip in mine if _subnet(ip) in near), mine[0]))
    return ips


def _rdma_device(m: dict, peer: dict, net: str = ""):
    """The rdma_<iface> device on `m` that reaches `peer`: the one on the
    Thunderbolt subnet both share with RDMA up at both ends -- the same
    subnet from either side --,
    else None: a device on another subnet is another cable, never a
    fallback."""
    net = net or _shared_subnet(m, peer, rdma=True)
    if net:
        return "rdma_" + str(_on_subnet(m, net, "iface"))
    return None


def _slot(job: str, used=None) -> int:
    """The job's ring-port slot: its nonce's, or the next one after it that
    no live job on this machine holds (`used`: default, the registry's)."""
    if used is None:
        used = _used_slots()
    start = int(job[:4], 16) % 100
    for i in range(100):
        s = (start + i) % 100
        if s not in used:
            return s
    raise ValueError("every ring-port slot is held by a live job here")


def _used_slots() -> set:
    """The ring-port slots of the live jobs in this machine's registry."""
    out = set()
    for job, recs in J.by_job().items():
        for r in recs:
            rp = r.get("ring_port")
            if isinstance(rp, int) and rp >= RING_PORT:
                out.add((rp - RING_PORT) // 20)
            else:
                out.add(int(job[:4], 16) % 100)
    return out


def placement(machines: list, shape: dict, split: str,
              order: list | None = None) -> dict:
    """Rank order and each rank's share. `machines`: [{"name", "chip",
    "p_core_ghz", "working_set_bytes", "bandwidth_gbs", "links"}];
    `shape`: the artifact's {"layer_bytes", "other_bytes", "leader_bytes"
    (rank 0's alone: the MTP head and the vision tower),
    "tensor_per_rank_bytes", "refusals"}. Pure: the same inputs give the
    same answer on every page.
    -> {"order": [names], "leader", "split", "shares": [{"rank", "machine",
        "bytes", "layers"?, "bounds"?}], "layers": [counts] | [],
        "reason"}. Raises ValueError with the arithmetic when it cannot."""
    from knurlogic.tuning import resolve as R
    if shape.get("refusals"):
        raise ValueError("; ".join(shape["refusals"]))
    names = R.rank_order([{**m, "free_bytes": m.get("working_set_bytes")}
                          for m in machines], order)
    by = {m["name"]: m for m in machines}
    n = len(names)
    if split == "tensor":
        per = int(shape["tensor_per_rank_bytes"])
        # the ranks' shares are equal; rank 0 alone adds the head and tower
        lead = int(shape.get("leader_bytes") or 0)
        shares, left = [], []
        for r, nm in enumerate(names):
            ws = budget_of(by[nm])
            floor = R.step_margin(ws)
            own = per + (lead if r == 0 else 0)
            if own > ws - floor:
                plus = (f" + {lead / GIB:.1f} GiB rank 0 alone holds (the "
                        f"MTP head and vision tower)" if r == 0 and lead
                        else "")
                raise ValueError(f"{nm}: its tensor share {per / GIB:.1f} "
                                 f"GiB{plus} does not fit its working set "
                                 f"{ws / GIB:.1f} GiB less the "
                                 f"{floor / GIB:.1f} GiB step margin: "
                                 f"more space required")
            shares.append({"rank": r, "machine": nm, "bytes": own})
            left.append(f"{nm} leaves {(ws - own) / GIB:.1f} GiB")
        reason = (f"tensor split {n} ways: every rank holds "
                  f"~{per / GIB:.1f} GiB"
                  + (f", rank 0 {lead / GIB:.1f} GiB more (the MTP head and "
                     f"vision tower)" if lead else "")
                  + f" ({', '.join(left)}); rank 0 "
                  f"{names[0]} leads (newest chip, then P-core clock, then "
                  f"free memory) and samples")
        return {"order": names, "leader": names[0], "split": split,
                "shares": shares, "layers": [], "reason": reason}
    ranks = [{"name": nm,
              "working_set_bytes": budget_of(by[nm]),
              "memory_bandwidth_gbs": by[nm].get("bandwidth_gbs")}
             for nm in names]
    lead = int(shape.get("leader_bytes") or 0)
    layers, other = list(shape["layer_bytes"]), int(shape.get("other_bytes") or 0)
    try:
        sh = R.pipeline_shares(layers, ranks, other, lead,
                               reserve=shape.get("reserve"))
    except ValueError:
        if not shape.get("reserve"):
            raise
        # the layers fit with only the step margin kept free: the fit
        # line is the step margin (resolve.single_fit_check), so it fits
        sh = R.pipeline_shares(layers, ranks, other, lead, reserve=None)
    shares = [{"rank": r, "machine": nm,
               "bytes": sh["bytes"][r] + int(shape.get("other_bytes") or 0)
               + (lead if r == 0 else 0),
               "layers": sh["layers"][r], "bounds": list(sh["bounds"][r])}
              for r, nm in enumerate(names)]
    return {"order": names, "leader": names[0], "split": split,
            "shares": shares, "layers": sh["layers"],
            "reason": f"rank 0 {names[0]} leads; " + sh["reason"]}


def shape_of(path: str, world: int, split: str,
             vision: bool = True, mtp: bool = True) -> dict:
    """What placement needs of an artifact, read off its headers.
    `vision`: KNURLOGIC_VISION; off, rank 0 holds no tower. `mtp`:
    KNURLOGIC_MTP; off, rank 0 holds no head."""
    from knurlogic.machine.artifact import Artifact
    from knurlogic.tuning import resolve as R
    a = Artifact.load(path)
    if split == "pipeline":
        per, other = R.pipeline_layer_bytes(a)
        refusals = R.pipeline_refusals(a.raw_config, world)
        return {"layer_bytes": per, "other_bytes": other,
                "leader_bytes": R.leader_bytes(a, vision=vision, mtp=mtp),
                "reserve": R.fit_reserve(a.raw_config),
                "tensor_per_rank_bytes": 0, "refusals": refusals}
    return {"layer_bytes": [], "other_bytes": 0,
            "leader_bytes": R.leader_bytes(a, vision=vision, mtp=mtp),
            "reserve": R.fit_reserve(a.raw_config),
            "tensor_per_rank_bytes":
                R.tensor_placement(a, world)["per_rank_bytes"],
            "refusals": R.tensor_split_refusals(a.path, world, a.raw_config)}


def viability_refusals(path: str, world: int) -> list:
    """A tensor split of a module whose layout no split rule knows is run
    whole and split on this machine (engine/runtime/viability) before any
    rank starts; rank 0's page asks, once."""
    from knurlogic.machine.artifact import Artifact
    from knurlogic.tuning import resolve as R
    todo = R.tensor_unverified(path)
    if not todo:
        return []
    from knurlogic.engine.runtime import viability
    a = Artifact.load(path)
    try:
        return viability.refusals(path, world, todo, a.raw_config,
                                  bool(a.model_file))
    except Exception as e:  # a module that cannot be built is not verified
        return [f"the tensor split of {len(todo)} module layout(s) no rule "
                f"knows could not be checked: {type(e).__name__}: {e}"]


# ------------------------------------------------------------ one page

def _resolve(identity: str | None, name: str = ""):
    """The local path for this identity: several copies of it are the same
    weights, and machine/artifact.resolve_identity picks one (the one
    called `name`, else one on this Mac's disk)."""
    from knurlogic.machine.artifact import resolve_identity
    return resolve_identity(identity, name=name)


def sets_refusal(path, sets: dict, tune: str = "default") -> str:
    """"" when a rank of `path` would start with these launch settings,
    else why not -- serve's own deterministic refusals (bad settings, a
    context past the model's maximum, a preset or KV precision it cannot
    take), asked BEFORE a rank starts: a refusal at startup would only be
    seen in a rank's log."""
    from knurlogic.interfaces.serve import launch_refusal
    from knurlogic.machine.artifact import Artifact
    try:
        a = Artifact.load(path)
    except (OSError, ValueError, AttributeError):
        return ""           # shape_of says why it cannot be read
    why = launch_refusal(a, sets or {}, tune)
    return f"its launch settings are refused: {why}" if why else ""


def check_spec(spec) -> str:
    """"" when `spec` has the prepare schema's shape, else why not."""
    if not isinstance(spec, dict):
        return "the body must be a JSON object"
    extra = sorted(set(spec) - set(SPEC_KEYS))
    if extra:
        return f"not part of a cluster job: {', '.join(extra)[:200]}"
    if not J.JOB_RX.fullmatch(str(spec.get("job") or "")):
        return "job is a hex nonce"
    for k in ("rank", "world", "prefill_chunk"):
        v = spec.get(k)
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            return f"{k} is a whole number"
    if not 2 <= spec["world"] <= 16 or spec["rank"] >= spec["world"]:
        return "rank and world do not make a ring"
    if spec.get("split") not in SPLITS or spec.get("link") not in LINKS:
        return "split is tensor|pipeline, link is ring|jaccl"
    nm = spec.get("name", "")
    if not isinstance(nm, str) or "/" in nm or nm in (".", "..") \
            or len(nm) > 255:
        return "name is a directory name, never a path"
    hosts = spec.get("hosts")
    if not isinstance(hosts, list) or not all(isinstance(h, str)
                                              for h in hosts):
        return "hosts is a list of address:port"
    sh = spec.get("serve_hosts")
    if sh is not None:
        import ipaddress
        try:
            ok = isinstance(sh, list) and 1 <= len(sh) <= 4 and all(
                isinstance(h, str) and ipaddress.ip_address(h) for h in sh)
        except ValueError:
            ok = False
        if not ok:
            return "serve_hosts is a list of IP addresses"
    if spec["link"] == "ring" and len(hosts) != spec["world"]:
        return "hosts names one address per rank"
    if spec["link"] == "jaccl":
        ibv = spec.get("ibv_devices")
        if not (isinstance(ibv, list) and len(ibv) == spec["world"] and all(
                isinstance(row, list) and len(row) == spec["world"] and all(
                    d is None or isinstance(d, str) for d in row)
                for row in ibv)) or not isinstance(spec.get("coordinator"),
                                                   str):
            return "jaccl needs an ibv_devices matrix and a coordinator"
    nodes = spec.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != spec["world"]:
        return "nodes names one machine per rank"
    for n in nodes:
        if not isinstance(n, dict) or not all(
                isinstance(n.get(k), str) for k in ("name", "id")) or not (
                n.get("page") is None or isinstance(n.get("page"), str)):
            return "each node is {name, id, page}"
    port = spec.get("port", 0)
    if not isinstance(port, int) or isinstance(port, bool) \
            or not 0 <= port <= 65535:
        return "port is 0..65535"
    for k in ("working_set_gib", "bandwidth_gbs"):
        v = spec.get(k, 0)
        if v is None:
            continue
        if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
            return f"{k} is a number >= 0"
    tune = spec.get("tune")
    if tune is not None:
        from knurlogic.tuning.settings import preset_of
        try:
            preset_of(tune)
        except ValueError as e:
            return str(e)
    sets = spec.get("sets")
    if sets is not None and not isinstance(sets, dict):
        return "sets is an object"
    chips = spec.get("chips")
    if chips is not None and not (
            isinstance(chips, list) and len(chips) <= 64 and all(
                isinstance(c, dict) and set(c) <= {"name", "arch"} and all(
                    isinstance(v, str) and len(v) <= 64 for v in c.values())
                for c in chips)):
        return "chips is [{name, arch}]"
    for k, most in (("cable", 32), ("cable_note", 400)):
        v = spec.get(k)
        if v is not None and not (isinstance(v, str) and len(v) <= most):
            return f"{k} is a short string"
    rv = spec.get("recovery")
    if rv is not None and not (isinstance(rv, dict) and len(rv) <= 8):
        return "recovery is an object"
    lay = spec.get("layers") or []
    if not isinstance(lay, list) or (lay and len(lay) != spec["world"]):
        return "layers is one count per rank"
    return ""


def held_here(job: str = "", reg=None, sreg=None, rss=None) -> list:
    """What knurlogic ranks and servers on this machine hold now, other
    than `job`'s: [(what, pid, bytes)]. A rank being stopped is left out
    -- start waits for it to be gone -- so this is what a new load would
    have to share the working set with."""
    rss = rss or J.rss_bytes
    reg = J.registry() if reg is None else reg
    out, seen = [], set()
    for rec in reg.values():
        j = str(rec.get("job") or "")
        pid = int(rec["pid"])
        if j == job or rec.get("stopping") or not _alive(j, pid):
            continue
        seen.add(pid)
        out.append((f"job {j} rank {rec.get('rank')}", pid, rss(pid)))
    from knurlogic.machine import servers
    for port, rec in sorted((servers.registry() if sreg is None
                             else sreg).items()):
        pid = int(rec["pid"])
        if pid in seen or (job and rec.get("job") == job) \
                or not servers.is_our_server(pid):
            continue
        out.append((f"the server on port {port}", pid, rss(pid)))
    return out


def _loading_elsewhere(job: str, reg: dict) -> str:
    """"" unless another job's rank here is still joining or loading --
    one load at a time on a machine."""
    for j, recs in J.by_job(reg).items():
        live = [r for r in recs if not r.get("stopping")]
        if j == job or not live:
            continue
        ph = J.phase_of(j, live)
        if ph != "ready":
            return (f"job {j}'s rank {live[0].get('rank')} is still "
                    f"{ph} here; one load at a time")
    return ""


def rank_room_chunk(path, spec: dict, ws: int, share: int):
    """The prompt chunk this rank's own room allows -- its share of the
    weights against its own budget, the resolver's rule -- or None when it
    cannot be worked out (the ring then runs PREFILL_CHUNK)."""
    try:
        from knurlogic.machine.artifact import Artifact
        from knurlogic.tuning import settings as S
        from knurlogic.tuning.resolve import preset_env, resolve
        a = Artifact.load(path)
        tune = spec.get("tune") or "default"
        sets = S.canonical_sets(dict(spec.get("sets") or {}))
        launch = S.engine_settings({**preset_env(a, tune),
                                    **{k: v for k, v in sets.items()
                                       if k in S.MODEL_KNOBS}})
        r = resolve(a, int(ws), tune=tune, holds_bytes=int(share),
                    kv_bits=launch.get("kv_bits"),
                    long_context=launch.get("long_context", "off"),
                    vision=launch.get("vision", True))
        v = S.engine_settings(r.env).get("prefill_step_size")
        return int(v) if v else None
    except Exception:  # a rank that cannot say leaves the ring on the floor
        logger.exception("could not work out this rank's prompt chunk")
        return None


def ring_chunk(sets: dict, got: list) -> tuple:
    """(chunk, why) for the ring: an explicit set wins; else the smallest
    of the ranks' room-based chunks, or the floor when one cannot say."""
    for k in ("KNURLOGIC_PREFILL_CHUNK", "VQLAB_PREFILL_CHUNK"):
        if k in sets:
            return int(sets[k]), "set"
    rooms = [(g or {}).get("prefill_chunk") for g in got]
    if rooms and all(isinstance(x, int) and x > 0 for x in rooms):
        return min(rooms), RING_CHUNK_WHY
    return PREFILL_CHUNK, "ring: a rank could not work out its room"


def prepare(spec: dict, *, resolve=None, info=None, shape=None,
            registry=None, held=None) -> tuple:
    """A Prepare message, on the page asked to run one rank:
    (status, doc). Checks, and remembers the job for start; starts
    nothing."""
    why = check_spec(spec)
    if why:
        # a spec this page cannot read is most often a newer page's (a key
        # added since): say the versions differ, which is the cause
        theirs = (spec.get("versions") or {}) if isinstance(
            spec.get("versions"), dict) else {}
        try:
            ours = (info if info is not None else _local_info()).get(
                "versions") or {}
        except (OSError, ValueError, AttributeError, TypeError):
            ours = {}
        k = "build" if theirs.get("build") and ours.get("build") \
            else "knurlogic"
        if theirs.get(k) and theirs.get(k) != ours.get(k):
            why = (f"{why} -- the coordinator runs {k} {theirs.get(k)}, "
                   f"this machine {ours.get(k) or 'missing'}: every rank "
                   f"runs the same build")
        return 400, {"error": why}
    from knurlogic.machine import identity
    me = identity.identity().get("name") or "this machine"
    path = resolve(spec.get("identity")) if resolve else \
        _resolve(spec.get("identity"), str(spec.get("name") or ""))
    if not path:
        return 200, {"ok": False, "machine": me,
                     "refused": f"not on {me}: no artifact with identity "
                                f"{str(spec.get('identity'))[:64]!r} in its "
                                f"model stores. Copy it there first."}
    info = info if info is not None else _local_info()
    refusals = []
    want = spec.get("versions") or {}
    have = info.get("versions") or {}
    # "same build" is the code hash (it names mlx too): when both sides
    # send one, the version string is information only -- a source copy
    # and a wheel of the same commit can disagree on it
    keys = ("mlx", "build") if want.get("build") and have.get("build") \
        else ("knurlogic", "mlx", "build")
    for k in keys:
        if want.get(k) != have.get(k):
            what = "build" if k == "build" else k
            refusals.append(f"{what} {have.get(k) or 'missing'} here, "
                            f"{want.get(k) or 'missing'} on the coordinator"
                            f": every rank runs the same build")
            if k == "mlx":
                break           # the build names mlx too; say it once
    from knurlogic.tuning.settings import clean_sets
    ok_sets, bad_sets = clean_sets(spec.get("sets") or {})
    if bad_sets:
        refusals.append(f"settings a rank does not take: "
                        f"{', '.join(bad_sets)}")
    spec = dict(spec, sets=ok_sets)
    why = sets_refusal(path, ok_sets, spec.get("tune") or "default")
    if why:
        refusals.append(why)
    rank, world = spec["rank"], spec["world"]
    try:
        from knurlogic.tuning.settings import mtp_of, vision_of
        sh = (shape or shape_of)(path, world, spec["split"],
                                 vision=vision_of(ok_sets),
                                 mtp=mtp_of(ok_sets))
    except Exception as e:  # a failed read is reported as the launch's refusal
        sh = {"refusals": [f"could not read the artifact: "
                           f"{type(e).__name__}: {e}"]}
    refusals += sh.get("refusals") or []
    if not refusals and spec["split"] == "tensor" and rank == 0:
        refusals += viability_refusals(path, world)
    ws = budget_of(info)
    if not refusals:
        if spec["split"] == "tensor":
            need = int(sh["tensor_per_rank_bytes"]) \
                + (int(sh.get("leader_bytes") or 0) if rank == 0 else 0)
        else:
            counts = spec.get("layers") or []
            start = sum(counts[rank + 1:])
            need = sum(sh["layer_bytes"][start:start + counts[rank]]) \
                + int(sh.get("other_bytes") or 0) \
                + (int(sh.get("leader_bytes") or 0) if rank == 0 else 0) \
                if counts else 0
        from knurlogic.tuning.resolve import step_margin
        # the weights and the minimum step margin are the refusal
        floor = step_margin(ws)
        busy = (held or held_here)(spec["job"]) if need and ws else []
        hold = sum(b for _, _, b in busy)
        if need and ws and need > ws - floor and not hold:
            refusals.append(f"rank {rank}'s share is {need / GIB:.1f} GiB "
                            f"and {me}'s working set (under its allowance) "
                            f"is {ws / GIB:.1f} GiB, which must leave the "
                            f"{floor / GIB:.1f} GiB step margin: more space "
                            f"required")
        elif need and ws and need > ws - hold - floor:
            refusals.append(
                f"rank {rank}'s share is {need / GIB:.1f} GiB and {me}'s "
                f"working set (under its allowance) is {ws / GIB:.1f} GiB, "
                f"{hold / GIB:.1f} GiB of it held now by "
                + ", ".join(f"{w} (pid {p}, {b / GIB:.1f} GiB)"
                            for w, p, b in busy if b)
                + f"; the {(ws - hold) / GIB:.1f} GiB left must also leave "
                  f"the {floor / GIB:.1f} GiB step margin. Unload that "
                  f"first")
    room_chunk = None
    if not refusals and need and ws:
        room_chunk = rank_room_chunk(path, spec, ws, need)
    if spec["link"] == "ring":
        ip = spec["hosts"][rank].rsplit(":", 1)[0]
        mine = {t.get("ip") for t in info.get("thunderbolt") or []}
        if ip not in mine and not ip.startswith("127."):
            refusals.append(f"{ip} is not one of {me}'s Thunderbolt "
                            f"addresses ({', '.join(sorted(mine)) or 'none'})")
    else:
        rd = info.get("rdma") or {}
        mine_devs = [d for d in (spec.get("ibv_devices") or [[]] * world)[rank]
                     if d] if isinstance(spec.get("ibv_devices"), list) else []
        if not rd.get("available"):
            refusals.append(f"RDMA on {me}: {rd.get('reason') or 'unknown'}")
        elif not mine_devs or any(d not in (rd.get("active") or [])
                                  for d in mine_devs):
            refusals.append(f"{', '.join(mine_devs) or 'no device'} is not an "
                            f"active RDMA device on {me} (active: "
                            f"{', '.join(rd.get('active') or []) or 'none'})")
    reg = registry() if registry else J.registry()
    if any(k.startswith(spec["job"] + "/") for k in reg):
        refusals.append(f"job {spec['job']} already runs here")
    why = _loading_elsewhere(spec["job"], reg)
    if why:
        refusals.append(why)
    stale = _wait_for_exit(reg, EXIT_WAIT_S)
    if stale:
        refusals.append("; ".join(
            f"rank {r} of job {j} is still exiting on {me}"
            for j, r, _ in stale))
    else:
        conflict = _slot_conflict(spec, reg)
        if conflict:
            j, r, pid = conflict
            left = J.wait_gone([pid], EXIT_WAIT_S, alive=lambda p: _alive(j, p))
            if left:
                refusals.append(f"rank {r} of job {j} is still exiting on "
                                f"{me} and holds this job's ring port; "
                                f"wait for it to finish")
    if rank == 0:
        from knurlogic.machine.servers import is_our_server
        from knurlogic.machine.servers import registry as sreg
        port = int(spec.get("port") or 0)
        rec = sreg().get(port)
        from knurlogic.machine.servers import port_free
        if rec and is_our_server(int(rec["pid"])):
            refusals.append(f"port {port} on {me} already serves "
                            f"{rec.get('artifact')}")
        elif port and not port_free(port):
            refusals.append(f"port {port} on {me} is in use")
        if refusals and spec.get("auto_port"):
            # a port nobody asked for: say which one is free instead
            from knurlogic.machine.servers import free_port
            try:
                return 200, typed(PrepareReply(
                    ok=False, machine=me, free_port=free_port(port + 1),
                    refused=f"{me} refuses rank {rank}: "
                    + "; ".join(refusals)))
            except OSError:
                pass
    if refusals:
        return 200, typed(PrepareReply(
            ok=False, machine=me, refused=f"{me} refuses rank {rank}: "
            + "; ".join(refusals)))
    with _LOCK:
        now = time.time()
        for j in [j for j, p in PREPARED.items() if now - p["t"] > PREPARED_S]:
            PREPARED.pop(j)
        PREPARED[spec["job"]] = {"spec": dict(spec), "path": path, "t": now}
    differs = _local_copy_differs(path, spec)
    note = None
    if want.get("knurlogic") != have.get("knurlogic"):
        note = (f"same build, but knurlogic says "
                f"{have.get('knurlogic') or 'missing'} here and "
                f"{want.get('knurlogic') or 'missing'} on the "
                f"coordinator")
    return 200, typed(PrepareReply(
        ok=True, machine=me, rank=rank, prefill_chunk=room_chunk,
        alert=f"on {me}: {differs}" if differs else None, note=note))


def _local_copy_differs(path, spec: dict) -> str:
    """"" unless this Mac also holds its own copy of the job's model (same
    name) that is NOT the weights the job runs -- then what to tell the
    person: the rank runs the job's copy, the local one differs."""
    from knurlogic.machine import artifact as A
    name = str(spec.get("name") or "")
    if not name:
        return ""
    try:
        from knurlogic.machine import discover
        for f in discover.find():
            p = str(f.path)
            if Path(p).name == name and p != str(path) \
                    and not A.on_network(p) \
                    and A.identity(p) not in ("", spec.get("identity")):
                return (f"the local copy of {name} ({p}) differs from the "
                        f"shared copy this job runs ({path}); this rank "
                        f"loads the shared one")
    except (OSError, ValueError):
        return ""
    return ""


def _local_info() -> dict:
    """This machine's cluster block, off the page's own status."""
    try:
        snap, _ = status_fn()
        own = next(n for n in snap.get("nodes") or []
                   if n.get("role") in ("local", "server"))
        return dict(own.get("cluster") or node_info(),
                    available_bytes=available_now())
    except (LookupError, StopIteration, AttributeError, TypeError, OSError, ValueError):
        return node_info()


def rank_argv(path: str, spec: dict, files: dict) -> list:
    """`knurlogic serve` for one rank, with the hidden ring flags."""
    cmd = [sys.executable, "-m", "knurlogic", "serve", path,
           "--rank", str(spec["rank"]), "--world", str(spec["world"]),
           "--split", spec["split"], "--link", spec["link"],
           "--job", spec["job"],
           "--prefill-chunk", str(spec["prefill_chunk"]),
           "--prefill-why", str(spec.get("prefill_why") or ""),
           "--working-set-gib", f"{float(spec.get('working_set_gib') or 0):.3f}",
           "--tune", spec.get("tune") or "default"]
    if spec.get("port"):
        cmd += ["--port", str(int(spec["port"]))]
    if spec.get("serve_hosts") and spec["rank"] == 0:
        # the leader answers the job's other machines on its link
        # address, and this one on loopback -- never every interface
        cmd += ["--host", ",".join(spec["serve_hosts"])]
    if spec["link"] == "ring":
        cmd += ["--hosts", ",".join(spec["hosts"])]
    else:
        cmd += ["--ibv-devices", files["ibv"],
                "--coordinator", spec["coordinator"]]
    if spec.get("layers"):
        cmd += ["--layers", ",".join(str(int(x)) for x in spec["layers"])]
    if spec.get("bandwidth_gbs"):
        cmd += ["--bandwidth-gbs", str(float(spec["bandwidth_gbs"]))]
    for k, v in sorted((spec.get("sets") or {}).items()):
        cmd += ["--set", f"{k}={v}"]
    if spec.get("chips"):
        # every rank's chip, so each resolves KNURLOGIC_CROSS_CHIP=auto
        # the same way (engine/crosschip.resolve)
        cmd += ["--ring-chips", json.dumps(spec["chips"])]
    return cmd


#: the argv builder; tests put a fake rank here
RANK_ARGV = [rank_argv]


#: the bell's port: rank 0's ring port + this (a slot's ranks take +0..+17
#: of it here, the jaccl coordinator +19)
BELL_OFFSET = 18


def _bell_nonce() -> int:
    """A job's bell nonce (bell_address): random, never derived from the
    job id the page shows."""
    return secrets.randbits(62)


def bell_address(spec: dict) -> str:
    """"host:port:nonce:world" of rank 0's bell (engine/runtime/tensor.init):
    every rank connects to it over TCP BEFORE the ring is joined, so the
    ranks enter jaccl's first collectives together -- a rank that arrives
    seconds late (a slower chip, a longer start) can have rank 0's first
    RDMA message dropped. "" when the job has no ring hosts or more ranks
    than a slot leaves room for (the bell's port then comes over the ring,
    as before)."""
    hosts = spec.get("hosts") or []
    world = int(spec.get("world") or len(hosts))
    if not hosts or world < 2 or world > BELL_OFFSET:
        return ""
    nonce = spec.get("bell_nonce")
    if not isinstance(nonce, int) or nonce <= 0:
        return ""                 # a spec from a page without it: as before
    host, port = hosts[0].rsplit(":", 1)
    return f"{host}:{int(port) + BELL_OFFSET}:{nonce}:{world}"


def rank_env(spec: dict, files: dict, selfheal: bool) -> dict:
    # MLX_METAL_FAST_SYNCH: the GPU hands each collective to the CPU (and
    # takes it back) by a spinning shared event, not a command-buffer
    # completion. Measured M4 + M3 Ultra over Thunderbolt, 35B-A3B VQ split
    # two ways, one decode step (80 all_sums): jaccl 71 -> 18.5 ms, ring
    # 94 -> 25 ms. exo sets it for every runner.
    env = {"PYTHONUNBUFFERED": "1", "MLX_RANK": str(spec["rank"]),
           "MLX_METAL_FAST_SYNCH": "1"}
    bell = bell_address(spec)
    if bell:
        env["KNURLOGIC_BELL"] = bell
    if spec["link"] == "ring":
        env["MLX_HOSTFILE"] = files["hostfile"]
    else:
        env["MLX_IBV_DEVICES"] = files["ibv"]
        env["MLX_JACCL_COORDINATOR"] = spec["coordinator"]
        if selfheal and spec.get("jaccl_timeout_ms"):
            # 0 while loading -- a cold read is not a hang -- and the
            # deadline once the model is in (cluster/jobs.after_load)
            env["JACCL_COLLECTIVE_TIMEOUT_MS"] = "0"
            env["KNURLOGIC_JACCL_TIMEOUT_MS"] = str(int(
                spec["jaccl_timeout_ms"]))
    return env


def _exiting(reg: dict) -> dict:
    """{pid: job} of the ranks here that were stopped and are not gone."""
    return {int(r["pid"]): str(r.get("job") or "") for r in reg.values()
            if r.get("stopping")}


def _wait_for_exit(reg: dict, wait_s: float) -> list:
    """Wait up to `wait_s` for every rank on this machine that was stopped
    (a previous job, mid-relaunch) to be gone. -> [(job, rank, pid)] still
    alive after the wait: a job being stopped counts as alive until its
    process is reaped, and its memory -- and its ring port -- are not
    free until then."""
    going = {int(r["pid"]): (str(r.get("job") or ""), r.get("rank"))
             for r in reg.values() if r.get("stopping")}
    if not going:
        return []
    left = J.wait_gone(list(going), wait_s,
                       alive=lambda p: _alive(going[p][0], p))
    return [(going[p][0], going[p][1], p) for p in left]


def _slot_conflict(spec: dict, reg: dict):
    """(job, rank, pid) of an alive rank of a DIFFERENT job on this
    machine whose ring port falls in this job's slot, else None. A
    coordinator's `_slot` only scans its own registry (`_used_slots`),
    so a quick relaunch can still hand a job a slot a previous job's
    still-live rank holds on a PEER machine; this is the peer-side net
    that catches it before two jobs' ranks talk over the same port."""
    try:
        port = int(spec["hosts"][spec["rank"]].rsplit(":", 1)[1])
    except (IndexError, ValueError, TypeError, AttributeError, KeyError):
        return None
    slot = (port - RING_PORT) // 20
    for rec in reg.values():
        j = str(rec.get("job") or "")
        if not j or j == spec["job"]:
            continue
        rp = rec.get("ring_port")
        if not isinstance(rp, int) or (rp - RING_PORT) // 20 != slot:
            continue
        pid = int(rec["pid"])
        if _alive(j, pid):
            return j, rec.get("rank"), pid
    return None


def start(job: str | None, *, spawn=None, wait_s: float | None = None,
          ring: dict | None = None) -> tuple:
    """A Start message: spawn this page's prepared rank -- once
    every stopped rank on this machine is gone (up to START_WAIT_S), and
    never beside another job's rank still loading."""
    with _LOCK:
        prep = PREPARED.pop(str(job or ""), None)
    c = (ring or {}).get("prefill_chunk")
    if prep is not None and isinstance(c, int) and not isinstance(c, bool) \
            and c > 0:
        # the ring's chunk, known only once every rank answered Prepare
        prep["spec"] = dict(prep["spec"], prefill_chunk=c,
                            prefill_why=str(ring.get("prefill_why") or "")[:120])
    if prep is None:
        # a retried Start (the answer was lost): the rank is already here
        again = _running_here(str(job or ""))
        if again:
            return 200, typed(Started(job=str(job), rank=again[0],
                                      pid=again[1], log=again[2]))
        return 404, {"error": f"no prepared job {str(job)[:40]!r} here"}
    with _START_LOCK:
        return _start(prep, spawn, START_WAIT_S if wait_s is None
                      else wait_s)


def _running_here(job: str):
    """(rank, pid, log) of this machine's live, not-stopping rank of `job`,
    or None."""
    for rec in J.by_job().get(job, []):
        try:
            if not rec.get("stopping") and _alive(job, int(rec["pid"])):
                return int(rec["rank"]), int(rec["pid"]), rec.get("log")
        except (KeyError, TypeError, ValueError):
            continue
    return None


def _start(prep: dict, spawn, wait_s: float) -> tuple:
    spec, path = prep["spec"], prep["path"]
    from knurlogic.machine import identity
    me = identity.identity().get("name") or "this machine"
    going = _exiting(J.registry())
    left = J.wait_gone(list(going), wait_s,
                       alive=lambda p: _alive(going[p], p)) if going else []
    if left:
        return 409, {"error": f"{me} refuses rank {spec['rank']}: "
                              + ", ".join(f"job {going[p]}'s rank (pid {p})"
                                          for p in left)
                              + f" was stopped and is still exiting after "
                                f"{wait_s:.0f} s; its memory is not free"}
    why = _loading_elsewhere(spec["job"], J.registry())
    if why:
        return 409, {"error": f"{me} refuses rank {spec['rank']}: {why}"}
    conflict = _slot_conflict(spec, J.registry())
    if conflict:
        j, r, pid = conflict
        left = J.wait_gone([pid], wait_s, alive=lambda p: _alive(j, p))
        if left:
            return 409, {"error": f"{me} refuses rank {spec['rank']}: rank "
                                  f"{r} of job {j} is still exiting on "
                                  f"{me} and holds this job's ring port; "
                                  f"its memory is not free"}
    d = J.job_dir(spec["job"])
    files = {"hostfile": str(d / f"hostfile-rank{spec['rank']}.json"),
             "ibv": str(d / "ibv-devices.json")}
    if spec["link"] == "ring":
        Path(files["hostfile"]).write_text(json.dumps(
            [[h] for h in spec["hosts"]]))
    else:
        Path(files["ibv"]).write_text(json.dumps(spec["ibv_devices"]))
    log = d / f"rank{spec['rank']}.log"
    try:
        heal = bool(_local_info().get("jaccl_selfheal"))
    except (OSError, ValueError, AttributeError, TypeError):
        heal = False
    cmd = RANK_ARGV[0](path, spec, files)
    env = {**os.environ, **rank_env(spec, files, heal)}
    try:
        with open(log, "w") as fh:
            proc = (spawn or subprocess.Popen)(
                cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                start_new_session=True)
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        return 500, {"error": f"{type(e).__name__}: {e}"}
    rec = {"job": spec["job"], "rank": spec["rank"], "world": spec["world"],
           "pid": proc.pid, "artifact": path, "log": str(log),
           # the model by identity: each rank's folder name may differ
           "identity": spec.get("identity") or "",
           "split": spec["split"], "link": spec["link"],
           "machines": [n.get("name") for n in spec["nodes"]],
           # the machine THIS rank runs on, so a stop reason names it --
           # the reason is propagated to every page of the job
           "machine": spec["nodes"][spec["rank"]].get("name")
           if spec["rank"] < len(spec["nodes"]) else None,
           "leader": spec["nodes"][0].get("name"),
           # every machine of the job by id, so the watcher can ask their
           # pages -- after a page restart too (SPECS is in memory)
           "nodes": [{"rank": n.get("rank"), "id": n.get("id"),
                      "name": n.get("name")} for n in spec["nodes"]],
           "started": time.strftime("%Y-%m-%d %H:%M:%S"), "t": time.time()}
    if spec.get("port") and spec["rank"] == 0:
        rec["port"] = int(spec["port"])
    try:
        rec["ring_port"] = int(spec["hosts"][spec["rank"]].rsplit(":", 1)[1])
    except (IndexError, ValueError, TypeError, AttributeError):
        pass
    with _LOCK:
        _PROCS[(spec["job"], spec["rank"])] = proc
        SPECS[spec["job"]] = spec
        reg = J.registry()
        reg[f"{spec['job']}/{spec['rank']}"] = rec
        J.save_registry(reg)
        if "port" in rec:
            from knurlogic.machine import servers
            sreg = servers.registry()
            sreg[rec["port"]] = {"pid": proc.pid, "artifact": path,
                                 "log": str(log), "job": spec["job"],
                                 "started": rec["started"], "t": rec["t"]}
            servers.save_registry(sreg)
    if "port" in rec:
        # this job's recovery row (a relaunch carries it; a launch by
        # somebody clears it), for rank 0's own /v1/residency
        from knurlogic.cluster import recovery
        recovery.write_port(rec["port"], spec.get("recovery"))
    _ensure_watcher()
    return 200, typed(Started(job=spec["job"], rank=spec["rank"],
                              pid=proc.pid, log=str(log)))


def _alive(job: str, pid: int) -> bool:
    proc = next((p for (j, _), p in _PROCS.items()
                 if j == job and p.pid == pid), None)
    if proc is not None:
        return proc.poll() is None
    return J.is_rank(pid, job)


def stop(job: str, reason: str = "unloaded", propagate: bool = True,
         post=None, grace: float = J.GRACE_S,
         reap: float = J.REAP_S, kind: str | None = None) -> dict:
    """Stop every rank of `job` on this machine (SIGTERM, then SIGKILL
    after `grace`), forget it, and -- when `propagate` -- tell every other
    page of the job to do the same."""
    job = str(job or "")
    from knurlogic.cluster import recovery
    cleared: list = []
    if recovery.kind(reason, kind) == "requested":
        cleared = recovery.cancel_job(job)   # asked for: never recovered
    # an unload carries no kind of its own: say "requested" from the start,
    # so a page polling while the ranks exit never shows it as a failure
    stop_kind = kind or ("requested" if recovery.kind(reason) == "requested"
                         else None)
    with _LOCK:
        PREPARED.pop(job, None)
        spec = SPECS.pop(job, None)
        _STOPPING.add(job)
        reg = J.registry()
        mine = {k: v for k, v in reg.items() if v.get("job") == job}
        # marked first: a start on this page waits for these to be gone,
        # and a prepare does not count them as staying
        for k in mine:
            reg[k] = dict(reg[k], stopping=reg[k].get("stopping")
                          or time.time(),
                          stop_reason=reg[k].get("stop_reason") or reason,
                          stop_kind=reg[k].get("stop_kind") or stop_kind)
        if mine:
            J.save_registry(reg)
    try:
        pids = [int(v["pid"]) for v in mine.values()
                if _alive(job, int(v["pid"]))]

        def alive(p):
            return _alive(job, p)

        killed = J.terminate(pids, grace=grace, alive=alive,
                             reap=reap) if pids else []
        left = [p for p in pids if alive(p)]
    finally:
        with _LOCK:
            _STOPPING.discard(job)
    with _LOCK:
        for (j, r) in [k for k in _PROCS if k[0] == job]:
            if _PROCS[(j, r)].pid in left:
                continue
            try:
                _PROCS.pop((j, r)).wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass                    # not gone yet; the watcher reaps it
        reg = J.registry()
        for k, v in mine.items():
            if int(v["pid"]) not in left:
                reg.pop(k, None)
        J.save_registry(reg)
        from knurlogic.machine import servers
        sreg = servers.registry()
        for port in [p for p, v in sreg.items() if v.get("job") == job
                     and int(v["pid"]) not in left]:
            sreg.pop(port)
        servers.save_registry(sreg)
        _WATCH.forget(job)
        for k in [k for k in _PEER_OK if k[0] == job]:
            _PEER_OK.pop(k)
        for k in [k for k in _PEER_BOOT if k[0] == job]:
            _PEER_BOOT.pop(k)
        # stopped means gone: a rank still exiting keeps its record
        # (phase "stopping"), and the watcher finishes the stop
        if (mine or spec) and not left:
            any_rec: dict = next(iter(mine.values()), {})
            port = next((int(v["port"]) for v in mine.values()
                         if v.get("port")), None) or (
                int((spec or {}).get("port") or 0) or None)
            # an unload says so, so the page drops its card instead of
            # showing "failed" (picker.js followLaunch)
            ENDED[job] = {"reason": reason, "kind": stop_kind,
                          "t": time.time(), "port": port,
                          "split": any_rec.get("split")
                          or (spec or {}).get("split"),
                          "link": link_name(any_rec.get("link")
                                            or (spec or {}).get("link")),
                          "machines": any_rec.get("machines")
                          or [n.get("name") for n in (spec or {}).get(
                              "nodes") or []]}
    told = []
    # the job's machines: its spec, or -- after a page restart, SPECS being
    # in memory -- the ids its rank records here keep
    nodes = (spec or {}).get("nodes") or next(
        (v["nodes"] for v in mine.values() if v.get("nodes")), [])
    if propagate and not nodes and not mine and not spec:
        # nothing of it here (a failed job, its ranks gone and recovery's
        # record all that is left): every peer page hears the stop, so the
        # record goes on whichever Mac holds it
        nodes = [{"id": i} for i in _peer_pages()]
    if propagate and nodes:
        from knurlogic.machine import identity
        me = identity.identity().get("id")
        known = _peer_pages()
        for n in nodes:
            if n.get("id") == me:
                continue
            # the address is this page's own record of that peer, never the
            # spec's: a prepare body cannot aim this page's stop elsewhere
            page = known.get(str(n.get("id") or ""))
            if not page:
                continue
            try:
                ans = (post or transport.send)(
                    page, "Stop", {"job": job, "reason": reason,
                                   **({"kind": kind} if kind else {})})
                told.append(n.get("name") or n.get("id"))
                # what that Mac's recovery cleared (a failed job's record
                # lives on whichever page started it)
                if isinstance(ans, dict):
                    cleared += [c for c in ans.get("cleared") or []
                                if c not in cleared]
            except NET_ERRORS:
                logger.debug("could not tell %s to stop job %s", page, job,
                             exc_info=True)
    return typed(Stopped(job=job, ranks_here=sorted(v["rank"] for v in
                                                    mine.values()),
                         killed=killed, exiting=left, told=told,
                         reason=reason, cleared=cleared))


def _peer_pages() -> dict:
    """{peer id: page address} from this page's PEERS store (an answering
    record first)."""
    out = {}
    peers: list = peers_fn()
    for p in sorted(peers, key=lambda p: p.state == "answering"):
        if getattr(p, "id", ""):
            out[p.id] = p.key
    return out


def watch_once(now: float | None = None) -> list:
    """Check every job with a rank on this machine; stop the failed ones.
    -> [(job, reason)] stopped."""
    out = []
    for job, recs in J.by_job().items():
        if job in _STOPPING:
            continue
        if all(r.get("stopping") for r in recs):
            # stopped, and a rank was still exiting: finish it
            stop(job, reason=recs[0].get("stop_reason") or "stopped",
                 propagate=False)
            continue
        why = _WATCH.verdict(job, recs, lambda pid, j=job: _alive(j, pid),
                             now=now)
        fkind = None
        if not why:
            why = peer_verdict(job, recs, now=now)
            if why and "stopped the job:" not in why:
                fkind = "machine"     # a page gone, or its rank gone
        if why and " exited" in why:
            line = next((x for x in (link_init_failure(_log_tail(
                r.get("log"))) for r in recs) if x), "")
            if line:
                why = f"{why}: link init failed: {line}"[:300]
            else:
                from knurlogic.cluster.recovery import memory_line, refusal_line
                tails = [_log_tail(r.get("log")) for r in recs]
                ref = next((x for x in map(refusal_line, tails) if x), "")
                mem = next((x for x in map(memory_line, tails) if x), "")
                if ref:
                    # a rank refused to start: the job fails with its words
                    why = f"{why}: {ref}"[:700]
                    fkind = "refusal"
                elif mem:
                    why = f"{why}: out of memory: {mem}"[:300]
                    fkind = "memory"
        if why:
            stop(job, reason=why, kind=fkind)
            out.append((job, why))
    return out


def job_state(job: str) -> dict:
    """JobState: whether this page still runs its ranks of `job`."""
    recs = J.by_job().get(job, [])
    live = sorted(int(r["rank"]) for r in recs if not r.get("stopping")
                  and _alive(job, int(r["pid"])))
    ended = ENDED.get(job) or {}
    stopping = [r for r in recs if r.get("stopping")]
    return typed(JobState(
        job=job, ranks_here=live, prepared=job in PREPARED,
        stopping=bool(stopping),
        phase=J.phase_of(job, [r for r in recs if not r.get("stopping")])
        if live else None,
        # by process, records or not: a relaunch waits for none left
        processes=J.pids_of_job(job),
        ended=ended.get("reason") or (
            stopping[0].get("stop_reason") if stopping else None),
        ended_kind=ended.get("kind") or (
            stopping[0].get("stop_kind") if stopping else None)))


def _ask_job(page: str, job: str) -> dict:
    return transport.send(page, "JobState", {"job": job})


def _record_of(nid: str):
    """This page's peer record of machine `nid` (an answering one first)."""
    recs = [p for p in peers_fn() if getattr(p, "id", "") == nid]
    return sorted(recs, key=lambda p: p.state != "answering")[0] \
        if recs else None


def _page_up(page: str, timeout: float = 2.0) -> bool:
    """Whether the machine at `page` (host:port) still accepts a TCP
    connection on its page port. The kernel accepts into the listen backlog
    while the page's Python is busy, so a slow page connects and a machine
    that is off, asleep or unplugged (or a page process that died) does
    not."""
    import socket
    host, _, port = page.rpartition(":")
    try:
        socket.create_connection((host.strip("[]"), int(port)),
                                 timeout=timeout).close()
        return True
    except (OSError, ValueError):
        return False


#: (job, peer id) pages slow but up, warned once per episode
_PEER_SLOW: set = set()


def peer_verdict(job: str, recs: list, now: float | None = None,
                 ask=None, pages=None, me=None, reach=None) -> str:
    """"" while every other machine of the job still runs its rank, else
    why not: its page says the job ended there, it restarted (a new
    boot_id), or it has been unreachable -- or answering without the rank --
    for PEER_GONE_S. Unreachable is the peer list's own clock
    (cluster/peers.py: the status GET every peer answers), not a second
    one. A rank blocked in a collective on a peer that vanished never exits
    and never counts as stalled (it is idle), so this is how its page
    learns.

    A page that does not answer while its machine still accepts a
    connection on the page port is slow (a blocked handler), not gone: the
    job is kept and a warning logged. Only a page this machine cannot reach
    at all counts toward PEER_GONE_S. A peer that powers off fails that
    probe too, and its rank's death fails the ring on this side as well."""
    now = time.time() if now is None else now
    reach = reach or _page_up
    nodes = next((r.get("nodes") for r in recs if r.get("nodes")), None) \
        or (SPECS.get(job) or {}).get("nodes") or []
    if not nodes:
        return ""
    if me is None:
        from knurlogic.machine import identity
        me = identity.identity().get("id")
    pages = _peer_pages() if pages is None else pages
    ask = ask or _ask_job
    for n in nodes:
        nid = str(n.get("id") or "")
        if not nid or nid == me:
            continue
        name = n.get("name") or nid
        key = (job, nid)
        page = pages.get(nid)
        rec = _record_of(nid)
        seen = float(getattr(rec, "last_seen", 0) or 0)
        boot = getattr(rec, "boot_id", "") or ""
        why, heard = "", 0.0
        try:
            if not page:
                raise ConnectionError("not a peer this page knows")
            doc = ask(page, job)
            if doc.get("ended"):
                return f"{name} stopped the job: {doc['ended']}"[:300]
            held = bool(doc.get("ranks_here") or doc.get("prepared"))
            if boot and _PEER_BOOT.get(key, boot) != boot:
                # the process that held the rank is not the one answering:
                # it restarted, and its ranks went with it
                return (f"{name} restarted (a new knurlogic process); "
                        f"its rank of the job is gone")
            if held:
                _PEER_SLOW.discard(key)
                _PEER_OK[key] = now
                if boot:
                    _PEER_BOOT.setdefault(key, boot)
                continue
            why = f"{name} no longer runs its rank of the job"
        except (*NET_ERRORS, AttributeError) as e:
            why = (f"{name}'s page has not answered "
                   f"({type(e).__name__})")
            if page and reach(page):
                # the machine is there; its page is only slow to answer
                if key not in _PEER_SLOW:
                    _PEER_SLOW.add(key)
                    logger.warning("cluster job %s: %s, but its machine "
                                   "accepts connections; the job is kept",
                                   job, why)
                _PEER_OK[key] = now
                continue
            # the peer list heard it lately though this ask failed
            heard = seen
        # the clock starts at the last good answer, or at first sight
        last = max(_PEER_OK.setdefault(key, now), min(heard, now))
        if now - last >= J.PEER_GONE_S:
            return f"{why} for {now - last:.0f} s; the job cannot run"
    return ""


def _ensure_watcher() -> None:
    with _LOCK:
        if _WATCHER:
            return
        _WATCHER.append(1)

    def loop():
        while True:
            time.sleep(WATCH_S)
            try:
                watch_once()
            # a watcher thread must survive any one failed pass (logged)
            except Exception as e:
                logger.warning("cluster watch: %s: %s", type(e).__name__, e)
    threading.Thread(target=loop, daemon=True,
                     name="knurlogic-cluster-watch").start()


def start_watching_existing() -> None:
    """At page start: ranks from an earlier page process are watched too,
    and the models an earlier page process was recovering are again."""
    if J.registry():
        _ensure_watcher()
    from knurlogic.cluster import recovery
    recovery.restore()


def jobs_document() -> list:
    """The jobs with a rank on this machine, and the ones that ended
    lately with why, for /loaded.json."""
    out = []
    for job, recs in J.by_job().items():
        r0 = min(recs, key=lambda r: r["rank"])
        if all(r.get("stopping") for r in recs):
            out.append({"job": job, "phase": "stopping",
                        "split": r0.get("split"),
                        "link": link_name(r0.get("link")),
                        "reason": r0.get("stop_reason"),
                        "kind": r0.get("stop_kind"),
                        "machines": r0.get("machines"),
                        "port": next((r.get("port") for r in recs
                                      if r.get("port")), None),
                        "exiting": sorted(int(r["pid"]) for r in recs)})
            continue
        out.append({"job": job, "split": r0.get("split"),
                    "link": link_name(r0.get("link")),
                    "machines": r0.get("machines"),
                    "leader": r0.get("leader"), "world": r0.get("world"),
                    "ranks_here": sorted(r["rank"] for r in recs),
                    "port": next((r.get("port") for r in recs
                                  if r.get("port")), None),
                    "artifact": Path(r0.get("artifact") or "").name,
                    "identity": r0.get("identity") or "",
                    "phase": J.phase_of(job, recs),
                    **{k: (SPECS.get(job) or {}).get(k) for k in
                       ("cable", "cable_note")
                       if (SPECS.get(job) or {}).get(k)},
                    **({"url": leader_url(dict(SPECS[job], port=next(
                        (r.get("port") for r in recs if r.get("port")),
                        None)))} if SPECS.get(job) else {})})
    now = time.time()
    for job, e in list(ENDED.items()):
        if now - e["t"] > 600:
            ENDED.pop(job, None)
        else:
            out.append({"job": job, "phase": "stopped", **e})
    return out


def failure_of_port(port) -> str:
    """Why the cluster job whose rank 0 serves on `port` here cannot
    answer -- "" when no job of this machine's serves there. A dropped
    connection to a live job's rank 0 has the watcher look at once, so the
    answer carries the job's real stop reason, not a guess."""
    try:
        port = int(port or 0)
    except (TypeError, ValueError):
        return ""
    if not port:
        return ""
    job = job_of_port(port)
    if job:
        try:
            watch_once()
        except Exception:  # an HTTP request must still answer; logged
            logger.debug("watch_once failed while reading job %s", job, exc_info=True)
    for j, e in sorted(ENDED.items(), key=lambda kv: -kv[1]["t"]):
        if (job and j == job) or (not job and e.get("port") == port):
            return e.get("reason") or "stopped"
    if job:
        return (f"rank 0 of cluster job {job} dropped the connection; the "
                f"job is failing")
    return ""


def job_of_port(port: int) -> str:
    for job, recs in J.by_job().items():
        if any(r.get("port") == port for r in recs):
            return job
    return ""


# ------------------------------------------------------------ coordinator

#: Carried by every launch of RDMA across more than two Macs.
RDMA_N_NOTE = "RDMA across more than two Macs is experimental and untested"

CABLE_RX = re.compile(r"(\d{1,3}\.\d{1,3}\.\d{1,3})(?:\.(?:\d{1,3}|x|0/24))?")


def refresh_memory(infos: list, post, wait_s: float = EXIT_WAIT_S,
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
                av, going = available_now(), len(_exiting(J.registry()))
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


def launch(req: dict, *, me: dict, peers: list, local_info: dict,
           ui_port: int, serve_port: int, post=None, follow=None,
           tried: tuple = (), moved: dict | None = None,
           recovering: dict | None = None) -> dict:
    """The coordinator: plan, prepare everywhere, start everywhere.
    `me`: this machine's identity; `peers`: the answering peers
    (cluster/peers.Peer); `local_info`: this machine's cluster block.

    Two machines on more than one shared Thunderbolt cable: the job runs
    on one cable's subnet (`req["cable"]` names it, else the first of
    `_shared_subnets`), and unless it was named the coordinator follows
    the job (`follow`, a thread by default): a rank whose link init fails
    on that cable has the job relaunched on the next one, the cable
    remembered as failing for the pair. `tried`/`moved`: that relaunch.
    `recovering`: this is auto-recovery's relaunch (cluster/recovery.py),
    carrying its report to rank 0's page; any other launch that starts is
    tracked there for recovery."""
    post = post or transport.send
    ids = req.get("nodes")
    if not isinstance(ids, list) or len(ids) < 2 or \
            len(set(map(str, ids))) != len(ids):
        return {"error": "a cluster launch names two or more machines"}
    split = req.get("split") if req.get("split") in SPLITS else None
    link = backend(req.get("link"))
    unnamed = req.get("link") in (None, "")
    if not split or not (link or unnamed):
        return {"error": "split is tensor|pipeline, link is tcp|rdma"}
    ident = str(req.get("identity") or "")
    if not ident:
        return {"error": "a cluster launch names the model by identity"}
    # what the requester called it: a directory name, never a path; each
    # rank prefers the artifact of that name among its identity's matches
    aname = Path(str(req.get("name") or "")).name[:255]
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
                return {"error": old.problem or f"{old.name or old.host} "
                        f"speaks another protocol: update knurlogic there"}
            return {"error": f"{nid!r} is not a machine answering this "
                             f"page"}
        c = (p.node or {}).get("cluster")
        if not isinstance(c, dict):
            return {"error": f"{p.name or p.host} runs a knurlogic without "
                             f"cluster launch; update it"}
        infos.append({**c, "id": nid, "name": p.name or p.host,
                      "page": p.key, "links": {}})
    # the links this page can see: its own to each peer
    kinds = {"thunderbolt": "rdma" if link == "jaccl" else "tb4",
             "ethernet": "ethernet", "wifi": "wifi"}
    mine = next((m for m in infos if m["id"] == me.get("id")), None)
    for m in infos:
        p = by_id.get(m["id"])
        if p is not None and mine is not None:
            k = kinds.get(getattr(p, "link", ""), "")
            mine["links"][m["name"]] = k
            m["links"][mine["name"]] = k
    if unnamed:
        # no link named: tcp, over the one cable the machines share
        shared: set | None = None
        for i, a_ in enumerate(infos):
            for b_ in infos[i + 1:]:
                n_ = set(_shared_subnets(a_, b_))
                shared = n_ if shared is None else shared & n_
        if shared and len(shared) > 1:
            kinds_ = sorted({_cable_name(_link_gbps(infos[0], infos[1], n))
                             .split(" (")[0] for n in shared})
            return {"refused": "link is tcp | rdma; the machines share "
                               + ", ".join(sorted(shared)) + " over "
                               + " and ".join(kinds_) + ": name one"}
        link = LINK_NAMES["tcp"]
    if link == "jaccl":
        for m in infos:
            rd = m.get("rdma") or {}
            if not rd.get("available"):
                return {"error": f"RDMA on {m['name']}: "
                                 f"{rd.get('reason') or 'unknown'}"}
    # the model's shape, from this machine when it has it, else a peer
    world = len(infos)
    # a local copy beside a DIFFERENT shared (network) copy of the same
    # name: every rank loads the shared one, and the launch says so
    from knurlogic.machine.artifact import prefer_shared
    try:
        ident, alert = prefer_shared(ident, aname)
    except (OSError, ValueError):
        alert = ""
    if alert:
        logger.warning("cluster launch: %s", alert)
        req = dict(req, identity=ident)
    path = _resolve(ident, aname)
    from knurlogic.tuning.settings import (clean_sets, mtp_of, preset_or,
                                           vision_of)
    sets, bad = clean_sets(req.get("sets") or {})
    if bad:
        return {"error": f"not a launch setting: {', '.join(bad)}"}
    if req.get("draft") is False:
        # "turn MTP off and launch": every rank's settings say so
        sets = dict(sets, KNURLOGIC_MTP="off")
    if path:
        why = sets_refusal(path, sets, preset_or(req.get("tune"), "default"))
        if why:
            return {"refused": f"nothing started: {why}"}
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
                return {"error": f"{first['name']}: {shape['error']}"}
    except (*NET_ERRORS, LookupError, StopIteration, AttributeError) as e:
        return {"error": f"could not read the model's shape: "
                         f"{type(e).__name__}: {e}"}
    def plan_once():
        return placement(infos, shape, split, req.get("order"))
    try:
        try:
            plan = plan_once()
        except ValueError:
            # a peer's memory in its last status is stale right after an
            # unload: wait out any rank still exiting, read every machine's
            # memory fresh, and place once more before refusing
            if not refresh_memory(infos, post):
                raise
            plan = plan_once()
    except ValueError as e:
        return {"refused": f"cannot place it: {e}"}
    order = [next(m for m in infos if m["name"] == nm) for nm in plan["order"]]
    job = secrets.token_hex(8)
    cable = req.get("cable")
    net, note, nets = "", "", []
    cable_ignored = False
    if world == 2:
        nets = _shared_subnets(order[0], order[1], rdma=link == "jaccl")
    if cable not in (None, ""):
        c = CABLE_RX.fullmatch(str(cable).strip()) if isinstance(
            cable, str) else None
        if world > 2:
            cable_ignored = True
        elif not c or c.group(1) not in nets:
            return {"refused": f"cable {str(cable)[:40]!r}: a launch names "
                               f"the Thunderbolt subnet both machines share"
                               + (" with RDMA up at both ends"
                                  if link == "jaccl" else "")
                               + f" ({', '.join(nets) or 'none here'})",
                    "placement": plan}
        else:
            net, note = c.group(1), f"cable {c.group(1)}: named by the launch"
    elif nets:
        left = [n for n in nets if n not in tried]
        if not left:
            return {"refused": f"every shared Thunderbolt cable failed link "
                               f"init: {', '.join(tried)}",
                    "placement": plan}
        net = left[0]
        bad = (BAD_CABLES.get(_pair(order[0], order[1])) or {}) \
            if len(order) == 2 else {}
        if moved:
            note = (f"cable {net}: moved from {moved['from']}, whose link "
                    f"init failed ({moved['why']})")
        elif bad and net != min(nets):
            b = min(bad)
            note = (f"cable {net}: {b} failed link init earlier this "
                    f"session ({bad[b]})")
        else:
            a_, b_ = order[0], order[1]
            sp = _link_gbps(a_, b_, net)
            others = [n for n in _shared_subnets(a_, b_) if n != net]
            if sp and any((_link_gbps(a_, b_, n) or 0) < sp
                          for n in others):
                note = (f"cable {net}: {_cable_name(sp)}, the fastest "
                        f"shared Thunderbolt link")
            elif sp:
                note = (f"cable {net}: {_cable_name(sp)}, the lowest shared "
                        f"Thunderbolt subnet")
            else:
                note = f"cable {net}: the lowest shared Thunderbolt subnet"
            skipped = [n for n in others if n not in nets]
            if skipped and link == "jaccl":
                note += "; not " + ", ".join(
                    f"{n} ({_cable_name(_link_gbps(a_, b_, n))}: RDMA "
                    f"needs Thunderbolt 5)" if _link_gbps(a_, b_, n)
                    else f"{n} (no RDMA at both ends)" for n in skipped)
        rest = [n for n in left[1:]]
        if rest:
            note += f"; {rest[0]} next if link init fails"
    if link == "jaccl" and world > 2:
        for i in range(world):
            for j in range(i + 1, world):
                why = rdma_pair_reason(order[i], order[j]) or (
                    "" if _shared_subnets(order[i], order[j], rdma=True)
                    else "no Thunderbolt cable with RDMA up at both ends")
                if why:
                    return {"refused": f"RDMA across {world} Macs needs "
                                       f"every pair joined by a Thunderbolt 5"
                                       f" cable: {order[i]['name']} and "
                                       f"{order[j]['name']}: {why}. Use the "
                                       f"TCP ring, or cable that pair.",
                            "placement": plan}
    if link == "jaccl" and not net and world == 2:
        why = rdma_pair_reason(order[0], order[1])
        if why:
            return {"refused": why + ". Use the TCP ring, or join them "
                                     "with a Thunderbolt 5 cable.",
                    "placement": plan}

        def up(m):
            return ", ".join((m.get("rdma") or {}).get("active") or []) \
                or "none"
        return {"refused": f"link rdma needs RDMA up at both ends of one "
                           f"Thunderbolt cable, and no Thunderbolt subnet "
                           f"has it ({order[0]['name']}: {up(order[0])}; "
                           f"{order[1]['name']}: {up(order[1])}). Use "
                           f"link tcp, or bring RDMA up on the shared cable.",
                "placement": plan}
    try:
        ips = _ring_ips(order, rdma=link == "jaccl", net=net)
    except ValueError as e:
        return {"refused": str(e), "placement": plan}
    try:
        slot = _slot(job)
    except ValueError as e:
        return {"refused": str(e), "placement": plan}
    hosts = [f"{ip}:{RING_PORT + slot * 20 + r}" for r, ip in enumerate(ips)]
    if cable_ignored:
        note = (note + "; " if note else "") + (
            "cable ignored: with more than two Macs each pair picks its own "
            "Thunderbolt link")
    ibv, coord = None, ""
    if link == "jaccl" and world > 2:
        ibv = [[None if i == j else _rdma_device(order[i], order[j])
                for j in range(world)] for i in range(world)]
        coord = f"{ips[0]}:{RING_PORT + slot * 20 + 19}"
    elif link == "jaccl":
        a0 = _rdma_device(order[0], order[1], net)
        a1 = _rdma_device(order[1], order[0], net)
        for m, d in ((order[0], a0), (order[1], a1)):
            if d is None:
                return {"refused": f"{m['name']} has RDMA up on "
                                   f"{', '.join(m['rdma'].get('active') or [])}"
                                   f" and none of them is on "
                                   f"{order[1 - order.index(m)]['name']}'s "
                                   f"Thunderbolt subnet", "placement": plan}
        ibv = [[None, a0], [a1, None]]
        coord = f"{ips[0]}:{RING_PORT + slot * 20 + 19}"
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
    # the base model's saved prompt chunk (Settings -> Models) is the
    # ring's, like every launch set; unset, the ring runs the smallest of
    # its ranks' room-based chunks (ring_chunk, once Prepare has answered;
    # PREFILL_CHUNK until then). serve puts the ring's value over any
    # --set, so the saved one must be resolved here or it is shown and
    # never runs.
    chunk = PREFILL_CHUNK
    for k in ("KNURLOGIC_PREFILL_CHUNK", "VQLAB_PREFILL_CHUNK"):
        if k in sets:
            from knurlogic.tuning.settings import check_knob
            why = check_knob(k, sets[k])
            if why:
                return {"error": why}
            chunk = int(sets[k])
            break
    base = {"job": job, "world": world, "split": split, "link": link,
            "identity": ident, "name": aname, "hosts": hosts, "ibv_devices": ibv,
            # the bell's nonce: random, in the spec the pages share, never
            # shown (the job id is on the page)
            "bell_nonce": _bell_nonce(),
            "coordinator": coord, "layers": plan["layers"],
            "prefill_chunk": chunk,
            "tune": preset_or(req.get("tune"), "default"),
            "nodes": nodes, "versions": local_info.get("versions") or {},
            "jaccl_timeout_ms": J.JACCL_TIMEOUT_MS if link == "jaccl" else 0,
            "sets": sets, "cable": net, "cable_note": note[:400],
            # each machine's chip and GPU architecture, rank order: the
            # ranks resolve KNURLOGIC_CROSS_CHIP=auto from it
            "chips": [{"name": m.get("chip") or m.get("name") or "",
                       "arch": m.get("gpu_architecture") or ""}
                      for m in order],
            "recovery": recovering,
            # rank 0 binds loopback and its own link address
            "serve_hosts": ["127.0.0.1", ips[0]]}
    specs = []
    for r, m in enumerate(order):
        specs.append({**base, "rank": r,
                      "port": port if r == 0 else 0,
                      "auto_port": auto_port,
                      "working_set_gib": round(
                          int(m.get("working_set_bytes") or 0) / GIB, 3),
                      "bandwidth_gbs": m.get("bandwidth_gbs") or 0})

    def ask(kind, r, doc):
        m = order[r]
        if m["page"] is None:
            return peer_step(kind, doc)[1]
        return post(m["page"], kind, doc)

    got = transport.parallel(lambda r: ask("Prepare", r, specs[r]),
                             list(range(world)))
    if auto_port and isinstance(got[0], dict) and got[0].get("free_port"):
        # rank 0's machine has that port taken: retry once on its suggestion
        port = int(got[0]["free_port"])
        specs[0] = dict(specs[0], port=port)
        got[0] = ask("Prepare", 0, specs[0])
    bad = [(order[r]["name"], g) for r, g in enumerate(got)
           if not (isinstance(g, dict) and g.get("ok"))]
    if bad:
        _abandon(job, order, post)
        return {"refused": "nothing started: " + "; ".join(
                    (g or {}).get("refused") or (g or {}).get("error")
                    or f"{nm} did not answer" for nm, g in bad),
                "placement": plan}
    alerts = ([alert] if alert else []) + (
        [RDMA_N_NOTE] if link == "jaccl" and world > 2 else []) + [
        g["alert"] for g in got if isinstance(g, dict) and g.get("alert")]
    for a in alerts[1 if alert else 0:]:
        logger.warning("cluster job %s: %s", job, a)
    chunk, chunk_why = ring_chunk(sets, got)
    base["prefill_chunk"] = chunk
    start_doc = {"job": job, "prefill_chunk": chunk}
    if chunk_why != "set":
        start_doc["prefill_why"] = chunk_why
    got = transport.parallel(lambda r: ask("Start", r, start_doc),
                             list(range(world)))
    bad = [(order[r]["name"], g) for r, g in enumerate(got)
           if not (isinstance(g, dict) and g.get("started"))]
    if bad:
        _abandon(job, order, post, reason="a rank did not start")
        return {"error": "a rank did not start, so none run: " + "; ".join(
            f"{nm}: {(g or {}).get('error') or 'no answer'}"
            for nm, g in bad), "placement": plan}
    if net and cable in (None, ""):
        (follow or _follow_thread)(job, {
            "req": req, "net": net, "tried": tuple(tried) + (net,),
            "order": order, "link": link, "post": post,
            "args": {"me": me, "peers": peers, "local_info": local_info,
                     "ui_port": ui_port, "serve_port": serve_port,
                     "post": post, "follow": follow}})
    if note:
        logger.warning("cluster job %s: %s", job, note)
    if recovering is None:
        from knurlogic.cluster import recovery
        # a relaunch is the same launch: this machine order (so the same
        # split), this port, link, tune and settings
        recovery.track_cluster(
            job, req=dict(req, order=list(plan["order"]), port=port),
            args={"me": me, "peers": peers, "local_info": local_info,
                  "ui_port": ui_port, "serve_port": serve_port,
                  "post": post, "follow": follow},
            order=order, port=port,
            leader_here=order[0]["page"] is None,
            previous=(moved or {}).get("job"))
    return {"job": job, "starting": ident, "placement": plan,
            "leader": plan["leader"], "port": port,
            "link": link_name(link), "url": leader_url(base | {"port": port}),
            "machines": plan["order"], "cable": net, "cable_note": note,
            **({"alerts": alerts} if alerts else {}),
            "note": f"rank 0 on {plan['leader']} serves on port {port} "
                    f"(loopback there, {leader_url(base | {'port': port})} "
                    f"from the job's machines) once every rank has loaded; "
                    f"poll /loaded.json"}


def _job_end(job: str, order: list, post):
    """None while `job` runs (or loads) on every page, else why it ended:
    this page's record, or a page of the job saying it ended there."""
    e = ENDED.get(job)
    if e:
        return e.get("reason") or "stopped"
    for m in order:
        if m.get("page") is None:
            continue
        try:
            doc = post(m["page"], "JobState", {"job": job})
        except NET_ERRORS:
            continue
        if isinstance(doc, dict) and doc.get("ended"):
            return str(doc["ended"])
    return None


def failover(job: str, ctx: dict, reason: str):
    """`job` ended for `reason`: when that is a rank's link init failing,
    remember the cable as failing for the pair and relaunch on the next
    one (-> the relaunch's answer); else None."""
    line = link_init_failure(reason)
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
    BAD_CABLES.setdefault(_pair(order[0], order[1]), {})[ctx["net"]] = line
    out = launch(ctx["req"], tried=ctx["tried"],
                 moved={"from": ctx["net"], "why": line, "job": job},
                 **ctx["args"])
    to = out.get("job")
    msg = (f"; relaunched on cable {out.get('cable')} as job {to}" if to
           else f"; not relaunched: {out.get('refused') or out.get('error')}")
    with _LOCK:
        e = ENDED.setdefault(job, {"reason": reason, "t": time.time(),
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
    FOLLOWING.add(job)
    try:
        end = clock() + FAILOVER_S
        while clock() < end:
            sleep(FAILOVER_POLL_S)
            why = _job_end(job, ctx["order"], ctx["post"])
            if why is not None:
                return failover(job, ctx, why)
            recs = J.by_job().get(job)
            if recs and J.phase_of(job, recs) == "ready":
                return None
        return None
    finally:
        FOLLOWING.discard(job)


def _follow_thread(job: str, ctx: dict) -> None:
    threading.Thread(target=_follow, args=(job, ctx), daemon=True,
                     name=f"knurlogic-cluster-follow-{job}").start()


def _abandon(job: str, order: list, post, reason: str = "refused") -> None:
    """Every page forgets a job that will not run (and stops any rank it
    already started)."""
    for m in order:
        try:
            if m["page"] is None:
                stop(job, reason=reason, propagate=False)
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
    from knurlogic.tuning.settings import PATH_KEYS
    if any(k in req for k in PATH_KEYS):
        return 400, {"error": "a cluster job names its model by identity, "
                              "never by a path"}
    if kind == "Prepare":
        return prepare(req)
    if kind == "Start":
        return start(req.get("job"), ring=req)
    if kind == "Stop":
        if not J.JOB_RX.fullmatch(str(req.get("job") or "")):
            return 400, {"error": "job is a hex nonce"}
        reason = str(req.get("reason") or "stopped by another machine")[:300]
        k = req.get("kind")
        return 200, stop(req["job"], reason=reason, propagate=False,
                         kind=k if k in (*FAILURE_KINDS, "requested")
                         else None)
    if kind == "JobState":
        if not J.JOB_RX.fullmatch(str(req.get("job") or "")):
            return 400, {"error": "job is a hex nonce"}
        return 200, job_state(req["job"])
    if kind == "Shape":
        from knurlogic.machine.artifact import AmbiguousIdentity
        try:
            p = _resolve(req.get("identity"), str(req.get("name") or ""))
        except AmbiguousIdentity as e:
            return 409, {"error": str(e)}
        if not p:
            return 404, {"error": "no artifact with that identity here"}
        w, s = req.get("world"), req.get("split")
        if not isinstance(w, int) or s not in SPLITS:
            return 400, {"error": "world and split"}
        return 200, shape_of(p, w, s, req.get("vision") is not False,
                             req.get("mtp") is not False)
    return 404, {"error": "not a cluster message"}
