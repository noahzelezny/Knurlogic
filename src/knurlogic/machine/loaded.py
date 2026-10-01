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
from dataclasses import dataclass, field
from pathlib import Path

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
    # and serves started by hand: shown, but only the page's own children
    # are offered for unloading (it stops only what it started)
    by_hand = {port for port in servers.listening_serves() if port not in ours}
    for port in list(p.get("openai", [])) + sorted(ours) + sorted(by_hand):
        if port in seen_ports:
            continue
        seen_ports.add(port)
        base = f"http://127.0.0.1:{port}"
        # Ours answers /status.json; anything else gets read as a plain
        # OpenAI port. Asking ours first stops knurlogic listing itself as
        # an anonymous mlx server.
        rows = _knurlogic(base) or _openai_port(base)
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
        doc["memory"] = memory_map()
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


# --- where the RAM went -----------------------------------------------------
# The runtimes are unreliable narrators about their own size: exo reports none
# per instance, and mlx-lm reports none at all. The OS knows, so ask it.
#
# WHICH NUMBER, and the obvious one is wrong: `ps -o rss` undercounts badly.
# Measured on one live process here, 1.93 GiB RSS against a 4.40 GiB phys
# footprint -- 2.3x, and that is a process holding no Metal buffers. RSS
# misses what a model runtime mostly IS. `top -l 1 -stats pid,mem` reports
# phys_footprint, the same number Activity Monitor shows as Memory, and costs
# 0.31s.

#: command-line fragment -> runtime. Order matters: knurlogic before the
#: plain mlx names, because ours is an mlx server too.
_RUNTIME_MARKS = (
    ("knurlogic", "knurlogic"),
    ("exo", "exo"),
    ("ollama", "ollama"),
    ("mlx_vlm", "mlx-vlm"),
    ("mlx_lm", "mlx-lm"),
    ("vqlab", "vqlab"),
)


_UNIT = {"B": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}


def available_memory() -> dict:
    """How much memory a model could actually have, from `vm_stat`.

    Used is what cannot be handed back without swapping: anonymous pages
    (a model's weights, active OR inactive -- an MoE's idle experts sit in
    inactive anonymous pages), wired pages and what the compressor
    occupies. Available is what can: free, file-backed pages (cache,
    droppable; speculative read-ahead is among them) and purgeable ones.

    Earlier versions counted `inactive` as available because psutil does.
    On a 128 GiB Mac holding a 109 GiB model that read 66.9 GiB used when
    real use was ~125: inactive anonymous pages are reclaimable only by
    swapping, and the fit check then admitted a model that did not fit.
    File-backed pages (active or inactive) are ONE bucket here, so nothing
    is counted twice; purgeable pages are anonymous and are moved from used
    to available.
    """
    import re
    import subprocess
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True,
                             timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    m = re.search(r"page size of (\d+)", out)
    if not m:
        return {}
    page = int(m.group(1))
    st = {}
    for line in out.splitlines()[1:]:
        g = re.match(r'"?([^":]+)"?:\s+(\d+)', line.strip())
        if g:
            st[g.group(1).strip()] = int(g.group(2)) * page

    free = st.get("Pages free", 0)
    # Speculative pages (read-ahead) are already inside "File-backed pages":
    # active + inactive + speculative == file-backed + anonymous. Adding
    # them again overstated what a launch could have.
    cache = st.get("File-backed pages", 0)
    purgeable = st.get("Pages purgeable", 0)
    wired = st.get("Pages wired down", 0)
    comp = st.get("Pages occupied by compressor", 0)
    res = {
        "available_bytes": free + cache + purgeable,
        "free_bytes": free,
        "cached_bytes": cache,
        "purgeable_bytes": purgeable,
        "wired_bytes": wired,
        "compressed_bytes": comp,
    }
    if "Anonymous pages" in st:
        res["used_bytes"] = max(
            st["Anonymous pages"] - purgeable, 0) + wired + comp
    return res


def _footprints() -> tuple:
    """({pid: bytes}, physmem) from `top`, which reports phys_footprint."""
    import re
    import subprocess
    try:
        out = subprocess.run(
            # `-o mem` is load-bearing: without it `-n` takes the first N
            # processes in top's default order, not the N biggest, and the
            # multi-gigabyte ones are simply not in the output. The first
            # pass here saw 1.9 GiB on a box holding 15.
            #
            # No `-n` at all: the whole table costs the same 0.31s as sixty
            # rows, and a cut loses exactly the case that matters -- an IDLE
            # runtime sits near the bottom, and "exo is holding 163 MiB
            # because nothing is loaded" is an answer, not noise.
            ["top", "-l", "1", "-o", "mem", "-stats", "pid,mem"],
            capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return {}, {}
    found, phys = {}, available_memory()
    for line in out.splitlines():
        if line.startswith("PhysMem:"):
            continue
        m = re.match(r"\s*(\d+)\s+([\d.]+)([BKMGT])\s*$", line)
        if m:
            found[int(m.group(1))] = int(float(m.group(2)) * _UNIT[m.group(3)])
    return found, phys


def _commands() -> dict:
    """{pid: command line}. `ps` is the only place the full argv lives, and
    the argv is what says which runtime a bare `python3.12` belongs to."""
    import subprocess
    try:
        out = subprocess.run(["ps", "-Ao", "pid=,command="],
                             capture_output=True, text=True,
                             timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    cmds = {}
    for line in out.splitlines():
        line = line.strip()
        pid, _, rest = line.partition(" ")
        if pid.isdigit():
            cmds[int(pid)] = rest.strip()
    return cmds


#: Never a model runtime, however the command line reads. A shell sitting in
#: a directory named after a runtime would match, and so would a `tail`
#: watching an exo log -- both reported as runtimes holding memory.
_NOT_A_RUNTIME = {
    "zsh", "bash", "sh", "fish", "tail", "head", "less", "more", "grep",
    "cat", "vim", "nvim", "nano", "code", "git", "ssh", "tmux", "screen",
    "make", "watch", "sed", "awk", "find", "rg", "fd", "top", "ps",
}


def _runtime_of(cmd: str) -> str:
    """Which runtime a process belongs to, from its EXECUTABLE and its
    `-m module`, never from the command line as a whole.

    Matching anywhere in the line over-attributes badly: it catches a
    shell whose working directory is named after a project and a `tail`
    following a log, and reports both as runtimes holding memory. The
    executable path
    and the module being run are the two places that actually say what the
    process IS.
    """
    if not cmd:
        return ""
    from knurlogic.machine.servers import is_test_process
    if is_test_process(cmd):
        return ""             # a test's fake rank is no runtime of this Mac
    parts = cmd.split()
    exe = parts[0]
    base = exe.rsplit("/", 1)[-1].lstrip("-").lower()
    if base in _NOT_A_RUNTIME:
        return ""

    # The module after `-m`, which is what names a python process -- or,
    # for an interpreter running a script, the script: a console-script
    # launch (`.../Python .../venv/bin/knurlogic serve`) names itself only
    # there, and would otherwise count as "everything else".
    module = ""
    for i, tok in enumerate(parts[:-1]):
        if tok == "-m":
            module = parts[i + 1].lower()
            break
    places = [exe]
    if not module and base.startswith("python") and len(parts) > 1 \
            and not parts[1].startswith("-"):
        places.append(parts[1])

    # The module wins over the interpreter's path: `envs/exo/bin/python -m
    # vqlab.cli` is vqlab borrowing exo's env, not exo.
    for mark, name in _RUNTIME_MARKS:
        if module == mark or module.startswith(mark + "."):
            return name
    if module:
        places = places[1:]
    # The script before the interpreter, as the module is: `envs/exo/bin/
    # python .../vqlab/bench/speed_pair.py` is vqlab borrowing exo's env.
    for place in reversed(places):
        low = place.lower()
        for mark, name in _RUNTIME_MARKS:
            if low.rsplit("/", 1)[-1].lstrip("-") == mark:
                return name
            # A path component, i.e. an env or install directory belonging
            # to it -- `.../envs/exo/bin/python3.13` is exo's interpreter.
            if f"/{mark}/" in low:
                return name
    return ""


def memory_map(floor: int = 256 << 20) -> dict:
    """Every process above `floor`, attributed to a runtime where possible.

    The unattributed remainder is REPORTED, not hidden. "Where did the RAM
    go" is not answered by a list that sums to less than the machine and
    does not say so.
    """
    (foot, phys), cmds = _footprints(), _commands()
    rows: list = []
    by_runtime: dict = {}
    for pid, b in foot.items():
        cmd = cmds.get(pid, "")
        rt = _runtime_of(cmd)
        # The floor is for everything else. A runtime process is reported
        # whatever it weighs: the question is where the memory went, and
        # "exo has four processes totalling 200 MiB" answers it -- nothing
        # is loaded. Dropping it under a floor answers nothing and looks
        # identical to exo not running.
        if not rt and b < floor:
            continue
        name = (cmd.split(" ")[0].rsplit("/", 1)[-1] or f"pid {pid}")[:40]
        rows.append({"pid": pid, "bytes": b, "runtime": rt, "name": name,
                     "cmd": cmd[:200]})
        if rt:
            by_runtime[rt] = by_runtime.get(rt, 0) + b
    rows.sort(key=lambda r: -r["bytes"])

    total = 0
    try:
        import subprocess
        total = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                   capture_output=True, text=True,
                                   timeout=5).stdout.strip() or 0)
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    seen = sum(r["bytes"] for r in rows)
    avail = phys.get("available_bytes")
    used = phys.get("used_bytes")
    if used is None:
        used = (total - avail) if (total and avail is not None) else seen
    # A footprint counts pages that were swapped out; `used` does not. What
    # the footprints add up to beyond `used` is therefore in swap (at most
    # what the OS says is swapped), shared among the runtimes by size.
    # Each runtime's row is then its RESIDENT part, so the rows add up to
    # the machine from this one sample.
    from knurlogic.machine import metrics as _metrics
    swap = _metrics._swap() or 0
    foot_rt = dict(by_runtime)
    fp = sum(foot_rt.values())
    in_swap = min(swap, max(sum(foot.values()) - used, 0), fp) \
        if phys.get("used_bytes") is not None else 0
    swapped = {k: int(in_swap * v / fp) for k, v in foot_rt.items()} \
        if fp and in_swap else {}
    by_runtime = {k: v - swapped.get(k, 0) for k, v in foot_rt.items()}
    rt = sum(by_runtime.values())
    # "Everything else" is what the OS says is spent MINUS what we could put
    # a name to -- the kernel, the file cache, compressed pages and every
    # process under the floor. Deriving it from the footprints instead made
    # the free figure wrong by 74 GiB on this machine.
    return {
        "installed_bytes": total,
        "seen_bytes": seen,
        "used_bytes": used,
        # What a model could have: the free pages PLUS the file cache macOS
        # will hand over on demand. Not top's "unused", which is only the
        # first of those.
        "free_bytes": phys.get("available_bytes", max(total - used, 0)),
        "truly_free_bytes": phys.get("free_bytes", 0),
        "cached_bytes": phys.get("cached_bytes", 0),
        "wired_bytes": phys.get("wired_bytes", 0),
        # RESIDENT bytes per runtime (footprint minus what is swapped out);
        # the footprints themselves and the swapped part ride beside it
        "by_runtime": by_runtime,
        "footprint_by_runtime": foot_rt,
        "swapped_by_runtime": swapped,
        "swap_bytes": swap,
        "runtime_bytes": rt,
        "other_bytes": max(used - rt, 0),
        "processes": rows[:25],
        "floor_bytes": floor,
        "from_os": bool(phys),
        "metric": "phys_footprint (what Activity Monitor calls Memory), not "
                  "RSS -- measured 2.3x apart on one process here. Used and "
                  "free come from the OS, not from summing these.",
    }
