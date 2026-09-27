"""One model across several machines, launched page to page.

The coordinator is the page someone pressed Launch on with two or more
machines picked. It never starts a rank on another machine itself: every
page starts its own ranks, after checking the request against its own
disk, memory, software and links. Two phases, so nothing starts unless
everything can:

  plan      gather each machine's facts (its status' `cluster` block:
            chip, working set under its allowance, memory bandwidth,
            Thunderbolt addresses, RDMA, versions), order the ranks
            (tuning/resolve.rank_order) and place the model (tensor: an
            equal share each; pipeline: tuning/resolve.pipeline_shares).
            Deterministic, and shown before anything loads.
  prepare   POST /peer/cluster/prepare to every page (this one directly):
            the job's fixed schema. Each checks the artifact by identity,
            the fit of ITS share, knurlogic/mlx versions against the
            coordinator's, and its link. Any refusal: nothing starts, the
            prepared pages are told to forget it, the refusal is shown.
  start     POST /peer/cluster/start: each page spawns its own rank
            (`knurlogic serve` with the hidden ring flags), records it,
            and watches it.

Every page that runs a rank watches it (cluster/jobs.py); any rank dying
or stalling stops the whole job: its own ranks SIGTERM then SIGKILL, and
/peer/cluster/stop to every other page of the job. Unloading the job from
any page does the same. Rank 0's HTTP port is where chat goes, through the
existing relay.

Peer routes are gated exactly like /peer/loaded.json (ui.peer_refusal).
"""
from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import threading
import time
from pathlib import Path

from knurlogic.cluster import jobs as J

GIB = 1 << 30
PREPARE_PATH = "/peer/cluster/prepare"
START_PATH = "/peer/cluster/start"
STOP_PATH = "/peer/cluster/stop"
SHAPE_PATH = "/peer/cluster/shape"
PEER_PATHS = (PREPARE_PATH, START_PATH, STOP_PATH, SHAPE_PATH)
#: a prepare's fields; nothing else is read, and a path is refused
SPEC_KEYS = ("job", "rank", "world", "split", "link", "identity", "hosts",
             "ibv_devices", "coordinator", "layers", "prefill_chunk", "tune",
             "port", "working_set_gib", "bandwidth_gbs", "nodes", "versions",
             "jaccl_timeout_ms", "sets")
SPLITS = ("tensor", "pipeline")
LINKS = ("ring", "jaccl")
#: the prompt chunk every rank runs (ring-wide): 512, as everywhere
PREFILL_CHUNK = 512
#: first ring port; a job's ranks take RING_PORT + slot*20 + rank, and
#: the jaccl coordinator slot*20 + 19
RING_PORT = 47200
#: a prepared job not started within this long is forgotten
PREPARED_S = 120.0
#: how long a peer page has to answer a cluster step
PEER_S = 30.0
#: how often the page checks its ranks
WATCH_S = 2.0

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


# ------------------------------------------------------------ this machine

_INFO = {"doc": None, "at": 0.0}


def _chip() -> str:
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True,
                              timeout=5).stdout.strip()
    except Exception:
        return ""


def _mlx_version() -> str:
    from importlib import metadata
    try:
        return metadata.version("mlx")
    except metadata.PackageNotFoundError:
        return ""


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
        from knurlogic.cluster import links
        from knurlogic.tuning.resolve import chip_bandwidth_gbs
        chip = _chip()
        try:
            tb = [{"iface": i["iface"], "ip": i["ip"]}
                  for i in links.thunderbolt()]
        except Exception:
            tb = []
        try:
            heal = _selfheal()
        except Exception:
            heal = False
        doc = {"chip": chip, "p_core_ghz": None,
               "bandwidth_gbs": chip_bandwidth_gbs(chip),
               "thunderbolt": tb, "rdma": links.rdma(),
               "versions": {"knurlogic": __version__,
                            "mlx": _mlx_version()},
               "jaccl_selfheal": heal}
        _INFO.update(doc=doc, at=now)
    from knurlogic.machine import allowance
    return dict(doc, working_set_bytes=allowance.cap(
        gpu_working_set(int(working_set_bytes or 0))))


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


def _shared_subnet(a: dict, b: dict, rdma: bool = False) -> str:
    """The one Thunderbolt /24 two machines both sit on ("" if none) --
    lowest first, so every page picks the same. Two Macs joined by two
    cables share two subnets; BOTH ends of a link must be on the same one
    (the M4's en2 at 198.51.100.2 and the M3's en4 at 192.0.2.1 are different
    cables, and a jaccl queue pair across them fails RTR with errno 60).
    `rdma`: only a subnet whose interface has RDMA up on both ends."""
    def on(m):
        act = set((m.get("rdma") or {}).get("active") or [])
        return {_subnet(t["ip"]) for t in m.get("thunderbolt") or []
                if t.get("ip") and (not rdma
                                    or f"rdma_{t.get('iface')}" in act)}
    both = sorted(on(a) & on(b))
    return both[0] if both else ""


def _on_subnet(m: dict, net: str, key: str = "ip"):
    return next((t.get(key) for t in m.get("thunderbolt") or []
                 if t.get("ip") and _subnet(t["ip"]) == net), None)


def _ring_ips(infos: list, rdma: bool = False) -> list:
    """Each rank's address on the ring, in rank order. Two machines: both
    on the subnet they share (`_shared_subnet`). More: a Thunderbolt
    address sharing a /24 with a neighbour's when there is one, else its
    first. Raises ValueError naming a machine with none."""
    ips = []
    n = len(infos)
    for r, m in enumerate(infos):
        mine = [t["ip"] for t in m.get("thunderbolt") or [] if t.get("ip")]
        if not mine:
            raise ValueError(f"{m['name']} has no Thunderbolt address: the "
                             f"ring runs over the Thunderbolt bridge")
    if n == 2:
        net = _shared_subnet(infos[0], infos[1], rdma) \
            or _shared_subnet(infos[0], infos[1])
        if net:
            return [_on_subnet(m, net) for m in infos]
    for r, m in enumerate(infos):
        mine = [t["ip"] for t in m.get("thunderbolt") or [] if t.get("ip")]
        near = {_subnet(t["ip"]) for k in ((r - 1) % n, (r + 1) % n)
                for t in infos[k].get("thunderbolt") or []}
        ips.append(next((ip for ip in mine if _subnet(ip) in near), mine[0]))
    return ips


def _rdma_device(m: dict, peer: dict):
    """The rdma_<iface> device on `m` that reaches `peer`: the one on the
    Thunderbolt subnet both share with RDMA up at both ends (en4 at
    192.0.2.1 reaches 192.0.2.2) -- the same subnet from either side --,
    else the only active one."""
    act = (m.get("rdma") or {}).get("active") or []
    net = _shared_subnet(m, peer, rdma=True)
    if net:
        return "rdma_" + str(_on_subnet(m, net, "iface"))
    return act[0] if len(act) == 1 else None


def _slot(job: str) -> int:
    return int(job[:4], 16) % 100


def placement(machines: list, shape: dict, split: str,
              order: list | None = None) -> dict:
    """Rank order and each rank's share. `machines`: [{"name", "chip",
    "p_core_ghz", "working_set_bytes", "bandwidth_gbs", "links"}];
    `shape`: the artifact's {"layer_bytes", "other_bytes",
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
        shares = []
        for r, nm in enumerate(names):
            ws = int(by[nm].get("working_set_bytes") or 0)
            if per > ws:
                raise ValueError(f"{nm}: its tensor share {per / GIB:.1f} "
                                 f"GiB does not fit its working set "
                                 f"{ws / GIB:.1f} GiB")
            shares.append({"rank": r, "machine": nm, "bytes": per})
        reason = (f"tensor split {n} ways: every rank holds "
                  f"~{per / GIB:.1f} GiB; rank 0 {names[0]} leads (newest "
                  f"chip, then P-core clock, then free memory) and samples")
        return {"order": names, "leader": names[0], "split": split,
                "shares": shares, "layers": [], "reason": reason}
    ranks = [{"name": nm,
              "working_set_bytes": int(by[nm].get("working_set_bytes") or 0),
              "memory_bandwidth_gbs": by[nm].get("bandwidth_gbs")}
             for nm in names]
    sh = R.pipeline_shares(list(shape["layer_bytes"]), ranks,
                           int(shape.get("other_bytes") or 0))
    shares = [{"rank": r, "machine": nm,
               "bytes": sh["bytes"][r] + int(shape.get("other_bytes") or 0),
               "layers": sh["layers"][r], "bounds": list(sh["bounds"][r])}
              for r, nm in enumerate(names)]
    return {"order": names, "leader": names[0], "split": split,
            "shares": shares, "layers": sh["layers"],
            "reason": f"rank 0 {names[0]} leads; " + sh["reason"]}


def shape_of(path: str, world: int, split: str) -> dict:
    """What placement needs of an artifact, read off its headers."""
    from knurlogic.machine.artifact import Artifact
    from knurlogic.tuning import resolve as R
    a = Artifact.load(path)
    if split == "pipeline":
        per, other = R.pipeline_layer_bytes(a)
        refusals = R.pipeline_refusals(a.raw_config, world)
        return {"layer_bytes": per, "other_bytes": other,
                "tensor_per_rank_bytes": 0, "refusals": refusals}
    return {"layer_bytes": [], "other_bytes": 0,
            "tensor_per_rank_bytes":
                R.tensor_placement(a, world)["per_rank_bytes"],
            "refusals": R.tensor_refusals(a.raw_config, world)}


# ------------------------------------------------------------ one page

def _resolve(identity: str):
    from knurlogic.machine.artifact import resolve_identity
    return resolve_identity(identity)


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
    hosts = spec.get("hosts")
    if not isinstance(hosts, list) or not all(isinstance(h, str)
                                              for h in hosts):
        return "hosts is a list of address:port"
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
    lay = spec.get("layers") or []
    if not isinstance(lay, list) or (lay and len(lay) != spec["world"]):
        return "layers is one count per rank"
    return ""


def prepare(spec: dict, *, resolve=None, info=None, shape=None,
            registry=None) -> tuple:
    """POST /peer/cluster/prepare, on the page asked to run one rank:
    (status, doc). Checks, and remembers the job for start; starts
    nothing."""
    why = check_spec(spec)
    if why:
        return 400, {"error": why}
    from knurlogic.machine import identity
    me = identity.identity().get("name") or "this machine"
    path = (resolve or _resolve)(spec.get("identity"))
    if not path:
        return 200, {"ok": False, "machine": me,
                     "refused": f"not on {me}: no artifact with identity "
                                f"{str(spec.get('identity'))[:64]!r} in its "
                                f"model stores. Copy it there first."}
    info = info if info is not None else _local_info()
    refusals = []
    want = spec.get("versions") or {}
    have = info.get("versions") or {}
    for k in ("knurlogic", "mlx"):
        if want.get(k) != have.get(k):
            refusals.append(f"{k} {have.get(k) or 'missing'} here, "
                            f"{want.get(k) or 'missing'} on the coordinator"
                            f": every rank runs the same build")
    rank, world = spec["rank"], spec["world"]
    try:
        sh = (shape or shape_of)(path, world, spec["split"])
    except Exception as e:
        sh = {"refusals": [f"could not read the artifact: "
                           f"{type(e).__name__}: {e}"]}
    refusals += sh.get("refusals") or []
    ws = int(info.get("working_set_bytes") or 0)
    if not refusals:
        if spec["split"] == "tensor":
            need = int(sh["tensor_per_rank_bytes"])
        else:
            counts = spec.get("layers") or []
            start = sum(counts[rank + 1:])
            need = sum(sh["layer_bytes"][start:start + counts[rank]]) \
                + int(sh.get("other_bytes") or 0) if counts else 0
        if need and ws and need > ws:
            refusals.append(f"rank {rank}'s share is {need / GIB:.1f} GiB "
                            f"and {me}'s working set (under its allowance) "
                            f"is {ws / GIB:.1f} GiB")
    if spec["link"] == "ring":
        ip = spec["hosts"][rank].rsplit(":", 1)[0]
        mine = {t.get("ip") for t in info.get("thunderbolt") or []}
        if ip not in mine and not ip.startswith("127."):
            refusals.append(f"{ip} is not one of {me}'s Thunderbolt "
                            f"addresses ({', '.join(sorted(mine)) or 'none'})")
    else:
        rd = info.get("rdma") or {}
        mine = [d for d in (spec.get("ibv_devices") or [[]] * world)[rank]
                if d] if isinstance(spec.get("ibv_devices"), list) else []
        if not rd.get("available"):
            refusals.append(f"RDMA on {me}: {rd.get('reason') or 'unknown'}")
        elif not mine or any(d not in (rd.get("active") or [])
                             for d in mine):
            refusals.append(f"{', '.join(mine) or 'no device'} is not an "
                            f"active RDMA device on {me} (active: "
                            f"{', '.join(rd.get('active') or []) or 'none'})")
    reg = registry() if registry else J.registry()
    if any(k.startswith(spec["job"] + "/") for k in reg):
        refusals.append(f"job {spec['job']} already runs here")
    if rank == 0:
        from knurlogic.machine.servers import is_our_server
        from knurlogic.machine.servers import registry as sreg
        port = int(spec.get("port") or 0)
        rec = sreg().get(port)
        if rec and is_our_server(int(rec["pid"])):
            refusals.append(f"port {port} on {me} already serves "
                            f"{rec.get('artifact')}")
    if refusals:
        return 200, {"ok": False, "machine": me,
                     "refused": f"{me} refuses rank {rank}: "
                                + "; ".join(refusals)}
    with _LOCK:
        now = time.time()
        for j in [j for j, p in PREPARED.items() if now - p["t"] > PREPARED_S]:
            PREPARED.pop(j)
        PREPARED[spec["job"]] = {"spec": dict(spec), "path": path, "t": now}
    return 200, {"ok": True, "machine": me, "rank": rank}


def _local_info() -> dict:
    """This machine's cluster block, off the page's own status."""
    from knurlogic.interfaces import ui
    try:
        snap, _ = ui._status_fn()
        own = next(n for n in snap.get("nodes") or []
                   if n.get("role") in ("local", "server"))
        return own.get("cluster") or node_info()
    except Exception:
        return node_info()


def rank_argv(path: str, spec: dict, files: dict) -> list:
    """`knurlogic serve` for one rank, with the hidden ring flags."""
    cmd = [sys.executable, "-m", "knurlogic", "serve", path,
           "--rank", str(spec["rank"]), "--world", str(spec["world"]),
           "--split", spec["split"], "--link", spec["link"],
           "--job", spec["job"],
           "--prefill-chunk", str(spec["prefill_chunk"]),
           "--working-set-gib", f"{float(spec.get('working_set_gib') or 0):.3f}",
           "--tune", spec.get("tune") or "balanced"]
    if spec.get("port"):
        cmd += ["--port", str(int(spec["port"]))]
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
    return cmd


#: the argv builder; tests put a fake rank here
RANK_ARGV = [rank_argv]


def rank_env(spec: dict, files: dict, selfheal: bool) -> dict:
    env = {"PYTHONUNBUFFERED": "1", "MLX_RANK": str(spec["rank"])}
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


def start(job: str, *, spawn=None) -> tuple:
    """POST /peer/cluster/start: spawn this page's prepared rank."""
    with _LOCK:
        prep = PREPARED.pop(str(job or ""), None)
    if prep is None:
        return 404, {"error": f"no prepared job {str(job)[:40]!r} here"}
    spec, path = prep["spec"], prep["path"]
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
    except Exception:
        heal = False
    cmd = RANK_ARGV[0](path, spec, files)
    env = {**os.environ, **rank_env(spec, files, heal)}
    try:
        with open(log, "w") as fh:
            proc = (spawn or subprocess.Popen)(
                cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                start_new_session=True)
    except Exception as e:
        return 500, {"error": f"{type(e).__name__}: {e}"}
    rec = {"job": spec["job"], "rank": spec["rank"], "world": spec["world"],
           "pid": proc.pid, "artifact": path, "log": str(log),
           "split": spec["split"], "link": spec["link"],
           "machines": [n.get("name") for n in spec["nodes"]],
           "leader": spec["nodes"][0].get("name"),
           "started": time.strftime("%Y-%m-%d %H:%M:%S"), "t": time.time()}
    if spec.get("port") and spec["rank"] == 0:
        rec["port"] = int(spec["port"])
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
    _ensure_watcher()
    return 200, {"started": spec["job"], "rank": spec["rank"],
                 "pid": proc.pid, "log": str(log)}


def _alive(job: str, pid: int) -> bool:
    proc = next((p for (j, _), p in _PROCS.items()
                 if j == job and p.pid == pid), None)
    if proc is not None:
        return proc.poll() is None
    return J.is_rank(pid, job)


def stop(job: str, reason: str = "unloaded", propagate: bool = True,
         post=None, grace: float = J.GRACE_S) -> dict:
    """Stop every rank of `job` on this machine (SIGTERM, then SIGKILL
    after `grace`), forget it, and -- when `propagate` -- tell every other
    page of the job to do the same."""
    job = str(job or "")
    with _LOCK:
        PREPARED.pop(job, None)
        spec = SPECS.pop(job, None)
        reg = J.registry()
        mine = {k: v for k, v in reg.items() if v.get("job") == job}
    pids = [int(v["pid"]) for v in mine.values()
            if _alive(job, int(v["pid"]))]
    killed = J.terminate(pids, grace=grace,
                         alive=lambda p: _alive(job, p)) if pids else []
    with _LOCK:
        for (j, r) in [k for k in _PROCS if k[0] == job]:
            try:
                _PROCS.pop((j, r)).wait(timeout=2)
            except Exception:
                pass
        reg = J.registry()
        for k in mine:
            reg.pop(k, None)
        J.save_registry(reg)
        from knurlogic.machine import servers
        sreg = servers.registry()
        for port in [p for p, v in sreg.items() if v.get("job") == job]:
            sreg.pop(port)
        servers.save_registry(sreg)
        _WATCH.forget(job)
        if mine or spec:
            any_rec = next(iter(mine.values()), {})
            ENDED[job] = {"reason": reason, "t": time.time(),
                          "machines": any_rec.get("machines")
                          or [n.get("name") for n in (spec or {}).get(
                              "nodes") or []]}
    told = []
    if propagate and spec:
        from knurlogic.machine import identity
        me = identity.identity().get("id")
        for n in spec.get("nodes") or []:
            if n.get("id") == me or not n.get("page"):
                continue
            try:
                (post or _post)(f"http://{n['page']}{STOP_PATH}",
                                {"job": job, "reason": reason})
                told.append(n.get("name"))
            except Exception:
                pass
    return {"stopped": job, "ranks_here": sorted(v["rank"] for v in
                                                 mine.values()),
            "killed": killed, "told": told, "reason": reason}


def watch_once(now: float | None = None) -> list:
    """Check every job with a rank on this machine; stop the failed ones.
    -> [(job, reason)] stopped."""
    out = []
    for job, recs in J.by_job().items():
        why = _WATCH.verdict(job, recs, lambda pid, j=job: _alive(j, pid),
                             now=now)
        if why:
            stop(job, reason=why)
            out.append((job, why))
    return out


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
            except Exception as e:
                print(f"cluster watch: {type(e).__name__}: {e}",
                      file=sys.stderr)
    threading.Thread(target=loop, daemon=True,
                     name="knurlogic-cluster-watch").start()


def start_watching_existing() -> None:
    """At page start: ranks from an earlier page process are watched too."""
    if J.registry():
        _ensure_watcher()


def jobs_document() -> list:
    """The jobs with a rank on this machine, and the ones that ended
    lately with why, for /loaded.json."""
    out = []
    for job, recs in J.by_job().items():
        r0 = min(recs, key=lambda r: r["rank"])
        out.append({"job": job, "split": r0.get("split"),
                    "link": r0.get("link"), "machines": r0.get("machines"),
                    "leader": r0.get("leader"), "world": r0.get("world"),
                    "ranks_here": sorted(r["rank"] for r in recs),
                    "port": next((r.get("port") for r in recs
                                  if r.get("port")), None),
                    "artifact": Path(r0.get("artifact") or "").name,
                    "phase": J.phase_of(job, recs)})
    now = time.time()
    for job, e in list(ENDED.items()):
        if now - e["t"] > 600:
            ENDED.pop(job, None)
        else:
            out.append({"job": job, "phase": "stopped", **e})
    return out


def job_of_port(port: int) -> str:
    for job, recs in J.by_job().items():
        if any(r.get("port") == port for r in recs):
            return job
    return ""


# ------------------------------------------------------------ coordinator

def _post(url: str, doc: dict, timeout: float = PEER_S) -> dict:
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, data=json.dumps(doc).encode(),
                                 method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        raw = e.read()
    out = json.loads(raw)
    if not isinstance(out, dict):
        raise ValueError("not a JSON object")
    from knurlogic.cluster.peers import clean
    return clean(out)


def _parallel(fn, items: list) -> list:
    out = [None] * len(items)

    def one(i, x):
        try:
            out[i] = fn(x)
        except Exception as e:
            out[i] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    ts = [threading.Thread(target=one, args=(i, x), daemon=True)
          for i, x in enumerate(items)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(PEER_S + 5)
    return out


def launch(req: dict, *, me: dict, peers: list, local_info: dict,
           ui_port: int, serve_port: int, post=None) -> dict:
    """The coordinator: plan, prepare everywhere, start everywhere.
    `me`: this machine's identity; `peers`: the answering peers
    (cluster/peers.Peer); `local_info`: this machine's cluster block."""
    post = post or _post
    ids = req.get("nodes")
    if not isinstance(ids, list) or len(ids) < 2 or \
            len(set(map(str, ids))) != len(ids):
        return {"error": "a cluster launch names two or more machines"}
    split = req.get("split") if req.get("split") in SPLITS else None
    link = req.get("link") if req.get("link") in LINKS else None
    if not split or not link:
        return {"error": "split is tensor|pipeline, link is ring|jaccl"}
    ident = str(req.get("identity") or "")
    if not ident:
        return {"error": "a cluster launch names the model by identity"}
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
    if link == "jaccl":
        if len(infos) != 2:
            return {"error": "jaccl here joins exactly two machines; use "
                             "the TCP ring for more"}
        for m in infos:
            rd = m.get("rdma") or {}
            if not rd.get("available"):
                return {"error": f"RDMA on {m['name']}: "
                                 f"{rd.get('reason') or 'unknown'}"}
    # the model's shape, from this machine when it has it, else a peer
    world = len(infos)
    path = _resolve(ident)
    try:
        if path:
            shape = shape_of(path, world, split)
        else:
            first = next(m for m in infos if m["page"])
            shape = post(f"http://{first['page']}{SHAPE_PATH}",
                         {"identity": ident, "world": world, "split": split})
            if shape.get("error"):
                return {"error": f"{first['name']}: {shape['error']}"}
    except Exception as e:
        return {"error": f"could not read the model's shape: "
                         f"{type(e).__name__}: {e}"}
    try:
        plan = placement(infos, shape, split, req.get("order"))
    except ValueError as e:
        return {"refused": f"cannot place it: {e}"}
    order = [next(m for m in infos if m["name"] == nm) for nm in plan["order"]]
    job = secrets.token_hex(8)
    try:
        ips = _ring_ips(order, rdma=link == "jaccl")
    except ValueError as e:
        return {"refused": str(e), "placement": plan}
    slot = _slot(job)
    hosts = [f"{ip}:{RING_PORT + slot * 20 + r}" for r, ip in enumerate(ips)]
    ibv, coord = None, ""
    if link == "jaccl":
        a0 = _rdma_device(order[0], order[1])
        a1 = _rdma_device(order[1], order[0])
        for m, d in ((order[0], a0), (order[1], a1)):
            if d is None:
                return {"refused": f"{m['name']} has RDMA up on "
                                   f"{', '.join(m['rdma'].get('active') or [])}"
                                   f" and none of them is on the other Mac's "
                                   f"Thunderbolt subnet", "placement": plan}
        ibv = [[None, a0], [a1, None]]
        coord = f"{ips[0]}:{RING_PORT + slot * 20 + 19}"
    my_tb = ips[plan["order"].index(me.get("name"))] \
        if me.get("name") in plan["order"] else None
    nodes = [{"rank": r, "id": m["id"], "name": m["name"],
              "page": m["page"] or (f"{my_tb}:{ui_port}" if my_tb else "")}
             for r, m in enumerate(order)]
    port = req.get("port")
    port = port if isinstance(port, int) and 1024 <= port < 65536 \
        else serve_port
    from knurlogic.interfaces.ui import clean_sets
    sets, bad = clean_sets(req.get("sets") or {})
    if bad:
        return {"error": f"not a launch setting: {', '.join(bad)}"}
    base = {"job": job, "world": world, "split": split, "link": link,
            "identity": ident, "hosts": hosts, "ibv_devices": ibv,
            "coordinator": coord, "layers": plan["layers"],
            "prefill_chunk": PREFILL_CHUNK,
            "tune": req.get("tune") if req.get("tune") in
            ("safe", "balanced", "fast") else "balanced",
            "nodes": nodes, "versions": local_info.get("versions") or {},
            "jaccl_timeout_ms": J.JACCL_TIMEOUT_MS if link == "jaccl" else 0,
            "sets": sets}
    specs = []
    for r, m in enumerate(order):
        specs.append({**base, "rank": r,
                      "port": port if r == 0 else 0,
                      "working_set_gib": round(
                          int(m.get("working_set_bytes") or 0) / GIB, 3),
                      "bandwidth_gbs": m.get("bandwidth_gbs") or 0})

    def ask(path_, r, doc):
        m = order[r]
        if m["page"] is None:
            fn = {PREPARE_PATH: lambda: prepare(doc)[1],
                  START_PATH: lambda: start(doc["job"])[1]}[path_]
            return fn()
        return post(f"http://{m['page']}{path_}", doc)

    got = _parallel(lambda r: ask(PREPARE_PATH, r, specs[r]),
                    list(range(world)))
    bad = [(order[r]["name"], g) for r, g in enumerate(got)
           if not (isinstance(g, dict) and g.get("ok"))]
    if bad:
        _abandon(job, order, post)
        return {"refused": "nothing started: " + "; ".join(
                    (g or {}).get("refused") or (g or {}).get("error")
                    or f"{nm} did not answer" for nm, g in bad),
                "placement": plan}
    got = _parallel(lambda r: ask(START_PATH, r, {"job": job}),
                    list(range(world)))
    bad = [(order[r]["name"], g) for r, g in enumerate(got)
           if not (isinstance(g, dict) and g.get("started"))]
    if bad:
        _abandon(job, order, post, reason="a rank did not start")
        return {"error": "a rank did not start, so none run: " + "; ".join(
            f"{nm}: {(g or {}).get('error') or 'no answer'}"
            for nm, g in bad), "placement": plan}
    return {"job": job, "starting": ident, "placement": plan,
            "leader": plan["leader"], "port": port,
            "machines": plan["order"],
            "note": f"rank 0 on {plan['leader']} serves on port {port} once "
                    f"every rank has loaded; poll /loaded.json"}


def _abandon(job: str, order: list, post, reason: str = "refused") -> None:
    """Every page forgets a job that will not run (and stops any rank it
    already started)."""
    for m in order:
        try:
            if m["page"] is None:
                stop(job, reason=reason, propagate=False)
            else:
                post(f"http://{m['page']}{STOP_PATH}",
                     {"job": job, "reason": reason})
        except Exception:
            pass


# ------------------------------------------------------------ peer routes

PEER_MAX = 16 << 10


def peer_route(path: str, body: bytes) -> tuple:
    """A cluster step from another page, which has passed peer_refusal:
    (status, doc)."""
    if len(body or b"") > PEER_MAX:
        return 413, {"error": "a cluster request is small"}
    try:
        req = json.loads(body or b"")
    except ValueError:
        req = None
    if not isinstance(req, dict):
        return 400, {"error": "the body must be a JSON object"}
    from knurlogic.interfaces.ui import PATH_KEYS
    if any(k in req for k in PATH_KEYS):
        return 400, {"error": "a cluster job names its model by identity, "
                              "never by a path"}
    if path == PREPARE_PATH:
        return prepare(req)
    if path == START_PATH:
        return start(req.get("job"))
    if path == STOP_PATH:
        if not J.JOB_RX.fullmatch(str(req.get("job") or "")):
            return 400, {"error": "job is a hex nonce"}
        reason = str(req.get("reason") or "stopped by another machine")[:300]
        return 200, stop(req["job"], reason=reason, propagate=False)
    if path == SHAPE_PATH:
        p = _resolve(req.get("identity"))
        if not p:
            return 404, {"error": "no artifact with that identity here"}
        w, s = req.get("world"), req.get("split")
        if not isinstance(w, int) or s not in SPLITS:
            return 400, {"error": "world and split"}
        return 200, shape_of(p, w, s)
    return 404, {"error": "not a cluster route"}
