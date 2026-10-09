"""What is resident on this machine right now, across every runtime.

`discover` answers what is on disk; this answers what is in memory. One
machine has one pool of memory, so one page shows every runtime's models:

  knurlogic   `/status.json`: the artifact and its resolved settings
  exo         `GET /state` -> `instances` and `runners` (both matter)
  ollama      `GET /api/ps` (resident), not `/api/tags` (downloaded)
  mlx-lm/vlm  `GET /v1/models`: what a port offers, marked as such

Every read is HTTP with a short timeout; a runtime that is not running
reports as absent, not as an error.

Design: docs/design/memory.md (loaded).
"""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path

from knurlogic.machine.memory import footprint

GIB = 1 << 30

#: Ports worth asking. exo and ollama have fixed defaults; the rest is where
#: these servers are conventionally started, including mlx-lm's own default.
DEFAULT_PORTS = {
    "exo": [52415],
    "ollama": [11434],
    "openai": [8080, 8081, 8000, 1234, 11435],
}

#: exo runner states that mean the weights are actually in memory. Taken from
#: exo/shared/types/worker/runners.py -- Loading and WarmingUp are on the way
#: there and are reported as such, ShuttingDown/Shutdown/Failed are not.
EXO_UP = {"RunnerLoaded", "RunnerReady", "RunnerRunning"}
EXO_COMING = {"RunnerLoading", "RunnerWarmingUp", "RunnerConnecting",
              "RunnerConnected"}


@dataclass
class Resident:
    """One model, in one runtime, right now."""
    runtime: str
    name: str
    where: str                      # the URL it was read from
    state: str = "loaded"           # loaded | loading | warming | failed | offered
    bytes_resident: int = 0         # 0 when the runtime does not say
    detail: str = ""
    can_unload: bool = False
    #: knurlogic only: the part of bytes_resident the OS has swapped out
    swapped_bytes: int = 0
    ident: str = ""                 # what an unload would name
    #: knurlogic only: the 16-hex id given at launch (a single-Mac server),
    #: or a cluster job's id (its instance is its job id); "" elsewhere
    instance: str = ""
    #: knurlogic only: in_flight, pending, capacity, oldest_pending_s,
    #: holding from the server's /status.json; None elsewhere
    requests: dict | None = None
    extra: dict = field(default_factory=dict, repr=False)

    @property
    def gib(self) -> float:
        return self.bytes_resident / GIB


def _get(url: str, timeout: float = 1.5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode() or "null")
    except (OSError, ValueError, http.client.HTTPException):
        return None


def exo_placement(inst, names: dict, local_id=None) -> dict:
    """{model, nodes: [{node, layers, bytes_est, local}]} for one instance.

    exo tags its instances by kind -- {"MlxRingInstance": {...}} -- so the
    fields are one level down. Reading them at the top found no model name
    and reported a UUID, which is how a model held entirely by ANOTHER node
    showed up in this machine's list as resident here with 0 bytes.
    """
    body = inst
    if isinstance(inst, dict) and len(inst) == 1:
        (only,) = inst.values()
        if isinstance(only, dict) and ("shardAssignments" in only
                                       or "shard_assignments" in only):
            body = only
    body = body if isinstance(body, dict) else {}
    sa = body.get("shardAssignments") or body.get("shard_assignments") or {}
    model = sa.get("modelId") or sa.get("model_id") or ""
    runner_node = {r: n for n, r in (sa.get("nodeToRunner") or {}).items()}
    nodes = []
    for runner, shard in (sa.get("runnerToShard") or {}).items():
        meta: dict = (next(iter(shard.values()), {})
                      if isinstance(shard, dict) else {})
        card = meta.get("modelCard") or {}
        n_layers = int(card.get("nLayers") or 0)
        lo, hi = meta.get("startLayer"), meta.get("endLayer")
        size = int((card.get("storageSize") or {}).get("inBytes") or 0)
        share = ((hi - lo) / n_layers) if (n_layers and lo is not None
                                           and hi is not None) else 0
        nid = runner_node.get(runner, "")
        nodes.append({"node": names.get(nid, nid[:12] or "?"),
                      "layers": f"{lo}-{hi} of {n_layers}",
                      "bytes_est": int(size * share),
                      "local": bool(local_id) and nid == local_id})
    return {"model": model, "nodes": nodes}


def _exo(base: str) -> list:
    st = _get(f"{base}/state", timeout=2.5)
    if st is None:
        return []
    runners = st.get("runners") or {}
    instances = st.get("instances") or {}

    def state_of(blob) -> str:
        if isinstance(blob, dict) and blob:
            return next(iter(blob))
        return str(blob)

    # An instance is only as loaded as its runners. exo reports the two
    # separately and the instance list alone will happily describe a model
    # whose every runner is shutting down.
    live = [state_of(v) for v in runners.values()]
    up = sum(1 for s in live if s in EXO_UP)
    coming = sum(1 for s in live if s in EXO_COMING)

    local_id = _get(f"{base}/node_id", timeout=1.0)
    names = {k: (v or {}).get("friendlyName") or k[:12]
             for k, v in (st.get("nodeIdentities") or {}).items()}
    out = []
    for iid, inst in instances.items():
        placed = exo_placement(inst, names, local_id)
        here = [p for p in placed["nodes"] if p["local"]]
        where_txt = ", ".join(f"{p['node']} layers {p['layers']}"
                              for p in placed["nodes"]) or "placement unknown"
        out.append(Resident(
            runtime="exo", name=placed["model"] or iid[:12], where=base,
            state="loaded" if up else ("loading" if coming else "offered"),
            detail=(f"{up} runner{'' if up == 1 else 's'} up; {where_txt}"
                    if up else f"no runner up; {where_txt}"),
            bytes_resident=sum(p["bytes_est"] for p in here),
            ident=iid,
            extra={"placement": placed["nodes"],
                   "on_this_machine": bool(here),
                   "bytes_note": "bytes_resident counts only shards on THIS "
                                 "machine, estimated as the model's size "
                                 "times its share of layers"}))
    # Runners can be up before exo has recorded an instance for them --
    # measured against the live daemon, which held 2 WarmingUp and 1 Loading
    # against an empty instance map. Reporting "nothing loaded" there would
    # be wrong in the direction that matters: something IS taking memory.
    if not out and (up or coming):
        n = up + coming
        out.append(Resident(
            runtime="exo", name=f"{n} runner{'' if n == 1 else 's'} "
                                f"{'up' if up else 'starting'}",
            where=base, state="loading" if not up else "loaded",
            detail="no instance recorded yet"))
    return out


def _ollama(base: str) -> list:
    ps = _get(f"{base}/api/ps")
    if not isinstance(ps, dict):
        return []
    out = []
    for m in ps.get("models") or []:
        out.append(Resident(
            runtime="ollama", name=m.get("name") or m.get("model") or "?",
            where=base, bytes_resident=int(m.get("size_vram")
                                           or m.get("size") or 0),
            detail=(m.get("details") or {}).get("parameter_size", ""),
            can_unload=True, ident=m.get("model") or m.get("name") or ""))
    return out


def _openai_port(base: str) -> list:
    """An mlx-lm / mlx-vlm style server.

    `/v1/models` is what it can serve, NOT what it has loaded -- mlx-lm loads
    on demand and exposes nothing that says which one is resident. So this is
    reported as `offered`, and the distinction is kept rather than smoothed
    over: claiming a port has a model loaded because it listed one would be
    inventing the only fact anybody came here for.
    """
    d = _get(f"{base}/v1/models")
    if not isinstance(d, dict):
        return []
    rows = d.get("data") or []
    if not rows:
        return []
    return [Resident(runtime="mlx", name=str(r.get("id") or "?"), where=base,
                     state="offered", detail="served on demand")
            for r in rows[:12]]


#: the last /status.json row each of our ports answered with: a server busy
#: generating can miss the 1.5 s read while /v1/models still answers, and
#: the card fell back to an anonymous "mlx ... offered" until it caught up
_LAST: dict = {}


def _knurlogic(base: str) -> list:
    d = _get(f"{base}/status.json")
    if not isinstance(d, dict) or "schema" not in d:
        return []
    a = d.get("artifact") or {}
    if not a:
        return []
    m = d.get("memory") or {}
    # The full model_type (Artifact.load prefers it over the nested TEXT
    # config's), plus a VISION tag only when this instance actually serves
    # images -- a ring with vision=False gets neither tag, since claiming
    # text-only would be as wrong as calling it "_text".
    detail = a.get("model_type") or ""
    if bool((d.get("vision") or {}).get("served")):
        detail = f"{detail} · VISION" if detail else "VISION"
    # MTP likewise: only when a head is bound and drafting
    if bool((d.get("drafting") or {}).get("drafts_now")):
        detail = f"{detail} · MTP" if detail else "MTP"
    held = _held(d, m)
    # The artifact is in /status.json from the moment the server starts, so
    # an answering port is not a loaded model: the host's own state is.
    host = str((d.get("load") or {}).get("state") or "")
    state = host if host in ("loading", "warming", "failed") else "loaded"
    if state == "loading":
        size = int(a.get("size_bytes") or 0)
        if size and held:
            detail += (" · " if detail else "") + \
                f"loading {min(100 * held // size, 99)}%"
    return [Resident(
        runtime="knurlogic", name=a.get("name") or "?", where=base,
        state=state, bytes_resident=held,
        detail=detail, can_unload=True,
        ident=a.get("path") or a.get("name") or "",
        instance=_instance_of(base),
        requests=d.get("requests") if isinstance(d.get("requests"), dict)
        else None)]


def _held(d: dict, m: dict) -> int:
    """What the model holds: a cluster's rank 0 reports every rank, so the
    sum of them (rank 0 alone read 104 GiB of a 178 GiB model)."""
    ranks = [r for r in d.get("ranks") or [] if isinstance(r, dict)]
    if len(ranks) > 1:
        return sum(int(r.get("active_bytes") or 0) for r in ranks)
    return int(m.get("active_bytes") or 0)


def _instance_of(base: str) -> str:
    """The instance id of the knurlogic server at `base`: a cluster job's
    id (rank 0 writes `job` into the same registry a single-Mac load
    writes `instance` into), read from THIS box's own registry -- a peer's
    row already carries whatever its own survey put there."""
    from urllib.parse import urlparse

    from knurlogic.machine import servers
    port = urlparse(base).port
    if not port:
        return ""
    rec = servers.registry().get(port) or {}
    return str(rec.get("job") or rec.get("instance") or "")


def _from_the_map(out: list, ours: dict, mm: dict) -> None:
    """Put a single-Mac knurlogic instance's bytes on the SAME footing as
    the memory map beside it: its process footprint from that one sample,
    not the allocator's own count (a different measure, taken at a
    different moment: the card read 109.2 GiB beside a machine that was
    105.8 used). Where the footprint exceeds what is resident, the
    swapped part is said (`swapped_bytes`) instead of dropped."""
    procs = {r["pid"]: r["bytes"] for r in mm.get("processes") or []}
    foot = mm.get("footprint_by_runtime") or {}
    swapped = (mm.get("swapped_by_runtime") or {}).get("knurlogic", 0)
    total = foot.get("knurlogic", 0)
    for r in out:
        if r.runtime != "knurlogic":
            continue
        from urllib.parse import urlparse
        rec = ours.get(urlparse(r.where).port) or {}
        if rec.get("job") or not str(rec.get("pid", "")).isdigit():
            continue
        b = procs.get(int(rec["pid"]))
        if not b:
            continue
        r.bytes_resident = b
        if swapped and total:
            r.swapped_bytes = int(swapped * b / total)


def survey(ports: dict | None = None, self_url: str = "") -> dict:
    """Every runtime on this box, and what each one is holding."""
    p = {**DEFAULT_PORTS, **(ports or {})}
    out: list = []
    seen_ports = set()

    for port in p.get("exo", []):
        seen_ports.add(port)
        out += _exo(f"http://127.0.0.1:{port}")
    for port in p.get("ollama", []):
        seen_ports.add(port)
        out += _ollama(f"http://127.0.0.1:{port}")
    # knurlogic's own servers, wherever they were started. A fixed list of
    # guessed ports missed two models holding 33 GiB on 8092 and 8093.
    from knurlogic.machine import servers
    ours = {}
    for port, rec in servers.registry().items():
        if servers.is_our_server(int(rec["pid"])):
            ours[port] = rec
    # and serves started by hand (a shell, another agent): shown, and
    # unloaded like the page's own children (interfaces/spawn.stop finds
    # them by port)
    listening = servers.listening_serves()
    by_hand = {port for port in listening if port not in ours}
    for port in list(p.get("openai", [])) + sorted(ours) + sorted(by_hand):
        if port in seen_ports:
            continue
        seen_ports.add(port)
        base = f"http://127.0.0.1:{port}"
        # Ours answers /status.json; anything else gets read as a plain
        # OpenAI port. Asking ours first stops knurlogic listing itself as
        # an anonymous mlx server.
        rows = _knurlogic(base)
        if rows:
            _LAST[base] = rows
        elif base in _LAST and port in ours and port not in listening:
            # alive but its port closed: an unload's SIGTERM, on its way out
            # -- not busy (shown READY · busy, a dead model looked live)
            rows = [replace(r, state="stopping", detail="stopping",
                            can_unload=False) for r in _LAST[base]]
        elif base in _LAST and (port in ours or port in by_hand):
            # ours, slow to answer while it works: still the same model
            rows = [replace(r, detail=(r.detail + " · " if r.detail else "")
                            + "busy") for r in _LAST[base]]
        else:
            rows = _openai_port(base)
        if port in by_hand:
            for r in rows:
                r.can_unload = False
        if not rows and port in ours:
            # Alive, registered, not answering yet: it is loading, and saying
            # so is the difference between "nothing here" and "wait".
            rows = [Resident(runtime="knurlogic",
                             name=Path(ours[port].get("artifact", "")).name,
                             where=base, state="loading",
                             detail="process up, port not answering yet",
                             can_unload=True, ident=str(port),
                             instance=str(ours[port].get("job")
                                         or ours[port].get("instance")
                                         or ""))]
        out += rows

    doc = {
        "resident": [r.__dict__ for r in out],
        "runtimes": sorted({r.runtime for r in out}),
        "bytes_resident": sum(r.bytes_resident for r in out),
    }
    # What each runtime is COSTING, from the OS, next to what each runtime
    # SAYS it is holding. They disagree, and the disagreement is the useful
    # part: a runtime reporting no model while its processes hold 40 GiB is
    # exactly the situation that is hard to get to the bottom of.
    try:
        doc["memory"] = footprint.memory_map()
    except Exception as e:  # the status document must still answer; the error is in it
        doc["memory"] = {"error": str(e)}
    _from_the_map(out, ours, doc["memory"])
    doc["resident"] = [r.__dict__ for r in out]
    doc["bytes_resident"] = sum(r.bytes_resident for r in out)
    return doc


def render(doc: dict) -> str:
    L = []
    rows = doc.get("resident") or []
    if not rows:
        L.append("nothing reports a loaded model. Asked: knurlogic, exo, "
                 "ollama, and any OpenAI-style port.")
    else:
        L.append(f"{'RUNTIME':<11}{'STATE':<10}{'SIZE':>8}  MODEL")
        for r in rows:
            size = (f"{r['bytes_resident'] / GIB:.1f}G"
                    if r["bytes_resident"] else "--")
            L.append(f"{r['runtime']:<11}{r['state']:<10}{size:>8}  "
                     f"{r['name'][:46]}"
                     + (f"   ({r['detail']})" if r["detail"] else ""))
        L.append("")
        L.append("A runtime that does not report its own size shows --, "
                 "never 0.")

    m = doc.get("memory") or {}
    if m.get("installed_bytes"):
        L.append("")
        L.append("WHERE THE MEMORY IS, per the OS rather than per runtime")
        L.append(f"  {m['installed_bytes'] / GIB:.0f} GiB installed")
        for rt, b in sorted(m.get("by_runtime", {}).items(),
                            key=lambda kv: -kv[1]):
            L.append(f"  {b / GIB:8.2f} GiB  {rt}")
        L.append(f"  {m.get('other_bytes', 0) / GIB:8.2f} GiB  "
                 f"everything else above "
                 f"{m.get('floor_bytes', 0) / GIB:.2f} GiB")
        L.append(f"  {m.get('seen_bytes', 0) / GIB:8.2f} GiB  seen in total")
        L.append("")
        L.append(f"  {m.get('metric', '')}")
        big = [r for r in m.get("processes", []) if r.get("runtime")][:8]
        if big:
            L.append("")
            for r in big:
                L.append(f"  {r['bytes'] / GIB:8.2f} GiB  {r['runtime']:<10} "
                         f"pid {r['pid']:<7} {r['name'][:32]}")
    return "\n".join(L)


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        prog="knurlogic loaded",
        description="what is in memory right now, in every runtime on this "
                    "machine")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    doc = survey()
    print(json.dumps(doc, indent=1) if a.json else render(doc))
    return 0


# --- acting on them ---------------------------------------------------------
# Reading is half of it. ollama's models can be let go from the page too;
# exo is only read, never driven.

def _post(url: str, payload=None, method: str = "POST", timeout: float = 30.0):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode()
    return json.loads(body) if body.strip() else {}


def ollama_unload(base: str, model: str) -> dict:
    """ollama has no unload endpoint; `keep_alive: 0` is the documented way.

    An empty prompt keeps it from generating anything on the way out.
    """
    return _post(f"{base}/api/generate",
                 {"model": model, "prompt": "", "keep_alive": 0})
