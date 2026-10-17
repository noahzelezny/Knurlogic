"""Prepare: one page checks a rank of a job against itself.

The request's fields, the model on its disk, its version, memory, links,
ports and the ranks it already runs; any refusal and nothing starts.
A prepared rank waits for Start for PREPARED_S.
"""
from __future__ import annotations

import time
from pathlib import Path

from knurlogic.cluster import here, tables
from knurlogic.cluster import jobs as J
from knurlogic.cluster.protocol import (
    PrepareReply,
    typed,
)
from knurlogic.machine.memory.wired import budget_of
from knurlogic.machine.units import gib
from knurlogic.tuning.knobs import canonical_sets
from knurlogic.tuning.placement import rank_room_chunk, shape_of

#: a prepare's fields; nothing else is read, and a path is refused
SPEC_KEYS = ("job", "rank", "world", "split", "link", "identity", "hosts",
             "ibv_devices", "coordinator", "layers", "prefill_chunk", "tune",
             "port", "working_set_gib", "bandwidth_gbs", "nodes", "versions",
             "jaccl_timeout_ms", "sets", "chips", "cable", "cable_note",
             "recovery", "serve_hosts", "name", "auto_port", "bell_nonce",
             "prefill_why")


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
    if spec.get("split") not in tables.SPLITS or spec.get("link") not in tables.LINKS:
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


def ring_chunk(sets: dict, got: list) -> tuple:
    """(chunk, why) for the ring: an explicit set wins; else the smallest
    of the ranks' room-based chunks, or the floor when one cannot say."""
    chunk = canonical_sets(sets).get(tables.PREFILL_KNOB)
    if chunk is not None:
        return int(chunk), "set"
    rooms = [(g or {}).get("prefill_chunk") for g in got]
    sized = [x for x in rooms if isinstance(x, int) and x > 0]
    if rooms and len(sized) == len(rooms):
        return min(sized), tables.RING_CHUNK_WHY
    return tables.PREFILL_CHUNK, "ring: a rank could not work out its room"


def _unreadable_spec(spec, why: str, info) -> str:
    """`why` a spec failed check_spec, with the cause when it is the most
    common one: a newer page's spec (a key added since), the versions
    differing."""
    theirs = (spec.get("versions") or {}) if isinstance(
        spec.get("versions"), dict) else {}
    try:
        ours = (info if info is not None else here.local_info()).get(
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


def _share_check(spec: dict, shape: dict, info: dict, me: str,
                 held) -> tuple:
    """(this rank's share in bytes, why it does not fit or ""): the weights
    and the minimum step margin against `info`'s working set less what
    other ranks and servers hold here now, and against the memory
    available now (fit.share_refusal)."""
    from knurlogic.tuning.fit import rank_share_bytes, share_refusal
    counts = spec.get("layers") or []
    if spec["split"] != "tensor" and not counts:
        return 0, ""
    need = rank_share_bytes(shape, spec["rank"], counts or None)
    ws = int(info.get("working_set_bytes") or 0)
    if not (need and ws):
        return need, ""
    why = share_refusal(need, ws, (held or here.held_here)(spec["job"]),
                        int(info.get("available_bytes") or 0))
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
    why = here.loading_elsewhere(spec["job"], reg)
    if why:
        refusals.append(why)
    stale = here.wait_for_exit(reg, tables.EXIT_WAIT_S)
    if stale:
        refusals.append("; ".join(
            f"rank {r} of job {j} is still exiting on {me}"
            for j, r, _ in stale))
        return refusals
    conflict = here.slot_conflict(spec, reg)
    if conflict:
        j, r, pid = conflict
        if J.wait_gone([pid], tables.EXIT_WAIT_S, alive=lambda p: here.alive(j, p)):
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
    with tables.LOCK:
        now = time.time()
        for j in [j for j, p in tables.PREPARED.items()
                  if now - p["t"] > tables.PREPARED_S and not p.get("starting")]:
            tables.PREPARED.pop(j)
        tables.PREPARED[spec["job"]] = {"spec": dict(spec), "path": path, "t": now}


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
        here.resolve(spec.get("identity"), str(spec.get("name") or ""))
    if not path:
        return 200, {"ok": False, "machine": me,
                     "refused": f"not on {me}: no artifact with identity "
                                f"{str(spec.get('identity'))[:64]!r} in its "
                                f"model stores. Copy it there first."}
    info = info if info is not None else here.local_info()
    want = spec.get("versions") or {}
    have = info.get("versions") or {}
    rank = spec["rank"]
    refusals = _version_refusals(want, have)
    more, spec, sh = _model_refusals(spec, path, shape)
    refusals += more
    budget = budget_of(info)
    need, room_chunk = 0, None
    if not refusals:
        need, why = _share_check(spec, sh, info, me, held)
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
