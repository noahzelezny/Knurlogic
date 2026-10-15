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

from knurlogic.cluster import NET_ERRORS, every, links, recovery, transport
from knurlogic.cluster import jobs as J
from knurlogic.cluster.protocol import (
    FAILURE_KINDS,
    LINK_NAMES,
    JobState,
    PrepareReply,
    Started,
    Stopped,
    backend,
    link_name,
    typed,
)
from knurlogic.machine import identity, processes, servers
from knurlogic.machine.artifact import bytes_on_disk
from knurlogic.machine.memory.wired import available_now, budget_of, gpu_working_set
from knurlogic.machine.processes import PROC_ERRORS
from knurlogic.machine.units import GIB, gib
from knurlogic.tuning.knobs import canonical_sets
from knurlogic.tuning.placement import placement, rank_room_chunk, shape_of

logger = logging.getLogger(__name__)


def _no_status():
    raise LookupError("no page has set cluster_jobs.status_fn")


# What this module needs of the page that runs it, injected by that page at
# startup (interfaces/page/server.wire, from make_handler) so this module
# never imports it.
# Unset, this machine is alone: no status snapshot (node_info() answers)
# and no peers.
#: () -> (status snapshot, _): the page's own status document
status_fn: Callable = _no_status
#: () -> [peer record]: the page's PEERS store
peers_fn: Callable[[], list] = list

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
#: mlx's distributed backends a spec names (protocol.LINK_NAMES)
LINKS = tuple(LINK_NAMES.values())
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
#: the launch setting that names the ring's chunk outright
PREFILL_KNOB = "KNURLOGIC_PREFILL_CHUNK"
RING_CHUNK_WHY = "ring: the smallest rank's room"
#: first ring port; a job's ranks take ring_port(slot, rank) (+0..+17),
#: rank 0's bell +BELL_OFFSET and the jaccl coordinator +COORDINATOR_OFFSET
RING_PORT = 47200
#: ports per slot, and slots
SLOT_PORTS = 20
SLOTS = 100
COORDINATOR_OFFSET = 19
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
#: set once the watcher runs
_WATCHER = threading.Event()
#: one start at a time on this page: the loading check and the spawn are
#: one step, so two jobs' ranks never begin loading together
_START_LOCK = threading.Lock()
#: jobs a stop() in this process is tearing down right now
_STOPPING: set = set()
#: (job, peer id) -> when that peer's page last said it runs its rank
_PEER_OK: dict = {}
#: the boot_id a peer had when it held its rank of a job: {(job, id): id}
_PEER_BOOT: dict = {}
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


def node_info(working_set_bytes: int = 0, ttl: float = 30.0) -> dict:
    """What a coordinator needs to know of THIS machine, for its status'
    `cluster` block. Cached: it runs a few subprocesses."""
    now = time.time()
    doc = _INFO["doc"]
    if doc is None or now - _INFO["at"] > ttl:
        from knurlogic import __version__
        chip = identity.chip()
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
        from knurlogic.engine.matmul_padding import gpu_architecture
        doc = {"chip": chip, "gpu_architecture": gpu_architecture(),
               "bandwidth_gbs": identity.chip_bandwidth_gbs(chip),
               "thunderbolt": tb, "rdma": links.rdma(),
               "versions": {"knurlogic": __version__,
                            "mlx": _mlx_version(),
                            "build": build_fingerprint()},
               "jaccl_selfheal": heal}
        _INFO.update(doc=doc, at=now)
    from knurlogic.machine.memory import allowance
    return dict(doc, working_set_bytes=allowance.cap(
        gpu_working_set(int(working_set_bytes or 0))),
        available_bytes=available_now())


# ------------------------------------------------------------ plan

def link_init_failure(text: str) -> str:
    """The line of a rank's output saying its link init failed ("" if
    none): jaccl's queue pair not reaching RTR, a ring that could not
    connect."""
    m = LINK_INIT_RX.search(text or "")
    return m.group(0).strip()[:240] if m else ""


def _slot(job: str, used=None) -> int:
    """The job's ring-port slot: its nonce's, or the next one after it that
    no live job on this machine holds (`used`: default, the registry's)."""
    if used is None:
        used = _used_slots()
    start = _nonce_slot(job)
    for i in range(SLOTS):
        s = (start + i) % SLOTS
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
                out.add(_slot_of_port(rp))
            else:
                out.add(_nonce_slot(job))
    return out


def _nonce_slot(job: str) -> int:
    """The ring-port slot a job's nonce names (`_slot` moves off it when a
    live job holds it)."""
    return int(job[:4], 16) % SLOTS


def ring_port(slot: int, offset: int) -> int:
    """Port `offset` of ring-port slot `slot`."""
    return RING_PORT + slot * SLOT_PORTS + offset


def _slot_of_port(port: int) -> int:
    """The ring-port slot a port of it (ring_port) belongs to."""
    return (port - RING_PORT) // SLOT_PORTS


def viability_refusals(path: str, world: int) -> list:
    """A tensor split of a module whose layout no split rule knows is run
    whole and split on this machine (engine/split/viability) before any
    rank starts; rank 0's page asks, once."""
    from knurlogic.machine.artifact import Artifact
    from knurlogic.tuning import tensor_split
    todo = tensor_split.tensor_unverified(path)
    if not todo:
        return []
    from knurlogic.engine.split import viability
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
    from knurlogic.machine.artifact import Artifact
    from knurlogic.tuning.checks import launch_refusal
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
        from knurlogic.tuning.presets import preset_of
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
    rss = rss or processes.rss
    reg = J.registry() if reg is None else reg
    out, seen = [], set()
    for rec in reg.values():
        j = str(rec.get("job") or "")
        pid = int(rec["pid"])
        if j == job or rec.get("stopping") or not _alive(j, pid):
            continue
        seen.add(pid)
        out.append((f"job {j} rank {rec.get('rank')}", pid, rss(pid)))
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


def ring_chunk(sets: dict, got: list) -> tuple:
    """(chunk, why) for the ring: an explicit set wins; else the smallest
    of the ranks' room-based chunks, or the floor when one cannot say."""
    chunk = canonical_sets(sets).get(PREFILL_KNOB)
    if chunk is not None:
        return int(chunk), "set"
    rooms = [(g or {}).get("prefill_chunk") for g in got]
    if rooms and all(isinstance(x, int) and x > 0 for x in rooms):
        return min(rooms), RING_CHUNK_WHY
    return PREFILL_CHUNK, "ring: a rank could not work out its room"


def _unreadable_spec(spec, why: str, info) -> str:
    """`why` a spec failed check_spec, with the cause when it is the most
    common one: a newer page's spec (a key added since), the versions
    differing."""
    theirs = (spec.get("versions") or {}) if isinstance(
        spec.get("versions"), dict) else {}
    try:
        ours = (info if info is not None else _local_info()).get(
            "versions") or {}
    except (OSError, ValueError, AttributeError, TypeError):
        ours = {}
    k = "build" if theirs.get("build") and ours.get("build") else "knurlogic"
    if theirs.get(k) and theirs.get(k) != ours.get(k):
        why = (f"{why} -- the coordinator runs {k} {theirs.get(k)}, "
               f"this machine {ours.get(k) or 'missing'}: every rank "
               f"runs the same build")
    return why


def _version_refusals(want: dict, have: dict) -> list:
    """The coordinator's build (`want`) against this machine's (`have`).
    "Same build" is the code hash (it names mlx too): when both sides send
    one, the version string is information only -- a source copy and a
    wheel of the same commit can disagree on it."""
    keys = ("mlx", "build") if want.get("build") and have.get("build") \
        else ("knurlogic", "mlx", "build")
    out = []
    for k in keys:
        if want.get(k) != have.get(k):
            out.append(f"{k} {have.get(k) or 'missing'} here, "
                       f"{want.get(k) or 'missing'} on the coordinator"
                       f": every rank runs the same build")
            if k == "mlx":
                break           # the build names mlx too; say it once
    return out


def _model_refusals(spec: dict, path, shape) -> tuple:
    """(refusals, the spec with only the settings a rank takes, the
    artifact's shape): the launch settings, the shape's own refusals and,
    for rank 0 of a tensor split, the viability check."""
    from knurlogic.tuning.checks import clean_sets
    from knurlogic.tuning.knobs import mtp_of, vision_of
    refusals = []
    ok_sets, bad_sets = clean_sets(spec.get("sets") or {})
    if bad_sets:
        refusals.append(f"settings a rank does not take: "
                        f"{', '.join(bad_sets)}")
    spec = dict(spec, sets=ok_sets)
    why = sets_refusal(path, ok_sets, spec.get("tune") or "default")
    if why:
        refusals.append(why)
    try:
        sh = (shape or shape_of)(path, spec["world"], spec["split"],
                                 vision=vision_of(ok_sets),
                                 mtp=mtp_of(ok_sets))
    except Exception as e:  # a failed read is reported as the launch's refusal
        sh = {"refusals": [f"could not read the artifact: "
                           f"{type(e).__name__}: {e}"]}
    refusals += sh.get("refusals") or []
    if not refusals and spec["split"] == "tensor" and spec["rank"] == 0:
        refusals += viability_refusals(path, spec["world"])
    return refusals, spec, sh


def _share_check(spec: dict, shape: dict, budget: int, me: str,
                 held) -> tuple:
    """(this rank's share in bytes, why it does not fit or ""): the weights
    and the minimum step margin against the budget, less what other ranks
    and servers hold here now."""
    from knurlogic.tuning.fit import rank_share_bytes, share_refusal
    counts = spec.get("layers") or []
    if spec["split"] != "tensor" and not counts:
        return 0, ""
    need = rank_share_bytes(shape, spec["rank"], counts or None)
    if not (need and budget):
        return need, ""
    why = share_refusal(need, budget, (held or held_here)(spec["job"]))
    return need, (f"rank {spec['rank']}'s share is {gib(need)} and {me}'s "
                  + why if why else "")


def _link_refusals(spec: dict, info: dict, me: str) -> list:
    """This machine's end of the job's link: its ring address is one of its
    Thunderbolt addresses, or its RDMA devices are active here."""
    rank = spec["rank"]
    if spec["link"] == "ring":
        ip = spec["hosts"][rank].rsplit(":", 1)[0]
        mine = {t.get("ip") for t in info.get("thunderbolt") or []}
        if ip not in mine and not ip.startswith("127."):
            return [f"{ip} is not one of {me}'s Thunderbolt addresses "
                    f"({', '.join(sorted(mine)) or 'none'})"]
        return []
    rd = info.get("rdma") or {}
    ibv = spec.get("ibv_devices")
    mine_devs = [d for d in (ibv or [[]] * spec["world"])[rank]
                 if d] if isinstance(ibv, list) else []
    if not rd.get("available"):
        return [f"RDMA on {me}: {rd.get('reason') or 'unknown'}"]
    if not mine_devs or any(d not in (rd.get("active") or [])
                            for d in mine_devs):
        return [f"{', '.join(mine_devs) or 'no device'} is not an "
                f"active RDMA device on {me} (active: "
                f"{', '.join(rd.get('active') or []) or 'none'})"]
    return []


def _here_refusals(spec: dict, reg: dict, me: str) -> list:
    """What this machine's own ranks say: the job already here, another
    job still loading, a stopped rank not gone (waited for up to
    EXIT_WAIT_S), or one holding this job's ring port."""
    refusals = []
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
        return refusals
    conflict = _slot_conflict(spec, reg)
    if conflict:
        j, r, pid = conflict
        if J.wait_gone([pid], EXIT_WAIT_S, alive=lambda p: _alive(j, p)):
            refusals.append(f"rank {r} of job {j} is still exiting on "
                            f"{me} and holds this job's ring port; "
                            f"wait for it to finish")
    return refusals


def _port_refusals(port: int, me: str) -> list:
    """Rank 0's serving port: taken by a knurlogic server, or in use."""
    from knurlogic.machine.servers import is_our_server, port_free
    from knurlogic.machine.servers import registry as sreg
    rec = sreg().get(port)
    if rec and is_our_server(int(rec["pid"])):
        return [f"port {port} on {me} already serves {rec.get('artifact')}"]
    if port and not port_free(port):
        return [f"port {port} on {me} is in use"]
    return []


def _remember(spec: dict, path) -> None:
    """Hold a prepared job for its Start, forgetting ones never started."""
    with _LOCK:
        now = time.time()
        for j in [j for j, p in PREPARED.items() if now - p["t"] > PREPARED_S]:
            PREPARED.pop(j)
        PREPARED[spec["job"]] = {"spec": dict(spec), "path": path, "t": now}


def prepare(spec: dict, *, resolve=None, info=None, shape=None,
            registry=None, held=None) -> tuple:
    """A Prepare message, on the page asked to run one rank:
    (status, doc). Checks, and remembers the job for start; starts
    nothing."""
    why = check_spec(spec)
    if why:
        return 400, {"error": _unreadable_spec(spec, why, info)}
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
    want = spec.get("versions") or {}
    have = info.get("versions") or {}
    rank = spec["rank"]
    refusals = _version_refusals(want, have)
    more, spec, sh = _model_refusals(spec, path, shape)
    refusals += more
    budget = budget_of(info)
    need, room_chunk = 0, None
    if not refusals:
        need, why = _share_check(spec, sh, budget, me, held)
        if why:
            refusals.append(why)
        elif need and budget:
            room_chunk = rank_room_chunk(path, spec, budget, need)
    refusals += _link_refusals(spec, info, me)
    refusals += _here_refusals(spec, registry() if registry else J.registry(),
                               me)
    if rank == 0:
        port = int(spec.get("port") or 0)
        refusals += _port_refusals(port, me)
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
    _remember(spec, path)
    differs = _local_copy_differs(path, spec)
    note = None
    if want.get("knurlogic") != have.get("knurlogic"):
        note = (f"same build, but knurlogic says "
                f"{have.get('knurlogic') or 'missing'} here and "
                f"{want.get('knurlogic') or 'missing'} on the "
                f"coordinator")
    from knurlogic.machine.artifact import on_network
    return 200, typed(PrepareReply(
        ok=True, machine=me, rank=rank, prefill_chunk=room_chunk,
        alert=f"on {me}: {differs}" if differs else None, note=note,
        local_copy=not on_network(path)))


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
    from knurlogic.machine.servers import serve_argv
    cmd = serve_argv(
        path,
        "--rank", str(spec["rank"]), "--world", str(spec["world"]),
        "--split", spec["split"], "--link", spec["link"],
        "--job", spec["job"],
        "--prefill-chunk", str(spec["prefill_chunk"]),
        "--prefill-why", str(spec.get("prefill_why") or ""),
        "--working-set-gib", f"{float(spec.get('working_set_gib') or 0):.3f}",
        "--tune", spec.get("tune") or "default")
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
        # every rank's chip: per-chip rounding off names them in status
        # and the log (engine/matmul_padding.resolve)
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
    """"host:port:nonce:world" of rank 0's bell (engine/split/link.init):
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
    from knurlogic.cluster import weights
    env.update(weights.rank_env(spec.get("read")))
    if spec["link"] == "ring":
        env["MLX_HOSTFILE"] = files["hostfile"]
    else:
        env["MLX_IBV_DEVICES"] = files["ibv"]
        env["MLX_JACCL_COORDINATOR"] = spec["coordinator"]
        if selfheal and spec.get("jaccl_timeout_ms"):
            # 0 while loading -- a cold read is not a hang -- and the
            # deadline once the model is in (engine/split/marker.after_load)
            env["JACCL_COLLECTIVE_TIMEOUT_MS"] = "0"
            env["KNURLOGIC_JACCL_TIMEOUT_MS"] = str(int(
                spec["jaccl_timeout_ms"]))
    return env


def exiting(reg: dict) -> dict:
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
    slot = _slot_of_port(port)
    for rec in reg.values():
        j = str(rec.get("job") or "")
        if not j or j == spec["job"]:
            continue
        rp = rec.get("ring_port")
        if not isinstance(rp, int) or _slot_of_port(rp) != slot:
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
    reads = (ring or {}).get("reads")
    if prep is not None and isinstance(reads, list) and \
            len(reads) == prep["spec"]["world"]:
        # this rank's entry of the job's read plan (cluster/weights.py)
        prep["spec"] = dict(prep["spec"], read=reads[prep["spec"]["rank"]])
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
    going = exiting(J.registry())
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
    except (*PROC_ERRORS, ValueError) as e:
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
           # measured once at launch: the page's polls read this, not
           # the model folder
           "bytes": bytes_on_disk(path),
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
            sreg = servers.registry()
            sreg[rec["port"]] = {"pid": proc.pid, "artifact": path,
                                 "log": str(log), "job": spec["job"],
                                 "bytes": rec["bytes"],
                                 "started": rec["started"], "t": rec["t"]}
            servers.save_registry(sreg)
    if "port" in rec:
        # this job's recovery row (a relaunch carries it; a launch by
        # somebody clears it), for rank 0's own /v1/residency
        recovery.write_port(rec["port"], spec.get("recovery"))
    _ensure_watcher()
    return 200, typed(Started(job=spec["job"], rank=spec["rank"],
                              pid=proc.pid, log=str(log)))


def _alive(job: str, pid: int) -> bool:
    proc = next((p for (j, _), p in _PROCS.items()
                 if j == job and p.pid == pid), None)
    if proc is not None:
        # poll() answers None while another thread waits on the process
        # (Popen's wait lock), dead or not; a zombie is dead
        return proc.poll() is None and processes.running(pid)
    return J.is_rank(pid, job)


def stop(job: str, reason: str = "unloaded", propagate: bool = True,
         post=None, grace: float = J.GRACE_S,
         reap: float = J.REAP_S, kind: str | None = None) -> dict:
    """Stop every rank of `job` on this machine (SIGTERM, then SIGKILL
    after `grace`), forget it, and -- when `propagate` -- tell every other
    page of the job to do the same."""
    job = str(job or "")
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
        rank_of = {int(v["pid"]): int(v["rank"]) for v in mine.values()}

        def spare(p):
            return J.mid_step(job, rank_of[p], p)

        killed = J.terminate(pids, grace=grace, alive=alive,
                             reap=reap, spare=spare) if pids else []
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
            # a failover that already moved the job keeps saying where it
            # went: the failing rank's stop can land after it
            moved = ENDED.get(job) if (ENDED.get(job) or {}).get(
                "relaunched") else None
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
            if moved:
                ENDED[job].update(reason=moved["reason"],
                                  relaunched=moved["relaunched"])
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
            live = [r for r in recs if _alive(job, int(r["pid"]))]
            if live and all(J.mid_step(job, int(r["rank"]), int(r["pid"]))
                            for r in live):
                continue                  # spared mid-step: nothing to do
            # stopped, and a rank was still exiting: finish it
            stop(job, reason=recs[0].get("stop_reason") or "stopped",
                 propagate=False)
            continue
        why, fkind = _WATCH.check(job, recs,
                                  lambda pid, j=job: _alive(j, pid), now=now)
        if not why:
            why = peer_verdict(job, recs, now=now)
            if why and "stopped the job:" not in why:
                fkind = "machine"     # a page gone, or its rank gone
        if why and " exited" in why:
            line = next((x for x in (link_init_failure(J.log_tail(
                r.get("log"))) for r in recs) if x), "")
            if line:
                why = f"{why}: link init failed: {line}"[:300]
            else:
                tails = [J.log_tail(r.get("log")) for r in recs]
                ref = next((x for x in map(recovery.refusal_line, tails) if x), "")
                mem = next((x for x in map(recovery.memory_line, tails) if x), "")
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
    boot_id) without its rank, or it has been unreachable -- or answering
    without the rank -- for PEER_GONE_S. Unreachable is the peer list's own clock
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
            if held:
                # ranks run in their own session and outlive a restarted
                # page: one still holding its rank is believed, new process
                # or not (a page redeploy used to stop a healthy job here)
                _PEER_SLOW.discard(key)
                _PEER_OK[key] = now
                if boot:
                    _PEER_BOOT[key] = boot
                continue
            if boot and _PEER_BOOT.get(key, boot) != boot:
                # a new process answering, holding no rank: it restarted,
                # and its rank went with the old one
                return (f"{name} restarted (a new knurlogic process); "
                        f"its rank of the job is gone")
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
    """The page's watcher, started once per process: `watch_once` every
    WATCH_S. A test sets _WATCHER and drives watch_once by hand."""
    with _LOCK:
        if _WATCHER.is_set():
            return
        _WATCHER.set()
    every(WATCH_S, watch_once, "knurlogic-cluster-watch")


def start_watching_existing() -> None:
    """At page start: ranks from an earlier page process are watched too,
    and the models an earlier page process was recovering are again."""
    if J.registry():
        _ensure_watcher()
    recovery.restore()


def jobs_document() -> list:
    """The jobs with a rank on this machine, and the ones that ended
    lately with why, for /loaded.json."""
    out = []
    for job, recs in J.by_job().items():
        r0 = min(recs, key=lambda r: r["rank"])
        phase = J.phase_of(job, recs)
        # a rank stayed inside a step rather than leave (a stop spares it):
        # neither stopping forever nor loaded
        stuck = ({"stuck": servers.stuck_reason(r0.get("machines") or [])}
                 if phase == "stuck" else {})
        if all(r.get("stopping") for r in recs):
            out.append({"job": job, "phase": "stuck" if stuck else "stopping",
                        **stuck,
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
                    "phase": phase, **stuck,
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
                av, going = available_now(), len(exiting(J.registry()))
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
    split = req.get("split") if req.get("split") in SPLITS else None
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
    path = _resolve(ident, aname)
    sets, bad = clean_sets(req.get("sets") or {})
    if bad:
        raise _Refused({"error": f"not a launch setting: {', '.join(bad)}"})
    if req.get("draft") is False:
        # "turn MTP off and launch": every rank's settings say so
        sets = dict(sets, KNURLOGIC_MTP="off")
    if path:
        why = sets_refusal(path, sets, preset_or(req.get("tune"), "default"))
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
        slot = _slot(job)
    except ValueError as e:
        raise _Refused({"refused": str(e), "placement": plan}) from e
    hosts = [f"{ip}:{ring_port(slot, r)}" for r, ip in enumerate(ips)]
    if link != "jaccl":
        return ips, hosts, None, ""
    coord = f"{ips[0]}:{ring_port(slot, COORDINATOR_OFFSET)}"
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
    saved = canonical_sets(sets).get(PREFILL_KNOB)
    if saved is None:
        return PREFILL_CHUNK
    from knurlogic.tuning.checks import check_knob
    why = check_knob(PREFILL_KNOB, saved)
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
    base = {"job": job, "world": world, "split": split, "link": link,
            "identity": ident, "name": aname, "hosts": hosts, "ibv_devices": ibv,
            # the bell's nonce: random, in the spec the pages share, never
            # shown (the job id is on the page)
            "bell_nonce": _bell_nonce(),
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
    chunk, chunk_why = ring_chunk(sets, got)
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
    url = leader_url(base | {"port": port})
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
    pair = links.machine_pair(order[0], order[1])
    links.BAD_CABLES.setdefault(pair, {})[ctx["net"]] = line
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
            why = job_end(job, ctx["order"], ctx["post"])
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
    from knurlogic.tuning.checks import PATH_KEYS
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
        p = _resolve(req.get("identity"), str(req.get("name") or ""))
        if not p:
            return 404, {"error": "no artifact with that identity here"}
        w, s = req.get("world"), req.get("split")
        if not isinstance(w, int) or s not in SPLITS:
            return 400, {"error": "world and split"}
        return 200, shape_of(p, w, s, req.get("vision") is not False,
                             req.get("mtp") is not False)
    return 404, {"error": "not a cluster message"}
