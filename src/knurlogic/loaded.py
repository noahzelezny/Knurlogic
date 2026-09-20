"""What is RESIDENT on this machine right now, across every runtime.

`discover` answers what is on the disk. This answers what is in memory, and
they are different questions with different answers: forty artifacts on a
volume, none of them loaded, is the normal state of an afternoon.

Nobody should have to open exo to see exo's models, or run `ollama ps` in a
terminal to see ollama's. A machine has one pool of memory and every runtime
is spending from it, so one page should show all of them.

FOUR RUNTIMES, THREE DIFFERENT CHANNELS, and none of them is a guess:

  knurlogic   our own `/status.json`, which names the artifact and its
              resolved settings.
  exo         `GET /state` -> `instances` (what was asked for) and `runners`
              (what is actually up). Both matter: an instance with every
              runner in `RunnerShuttingDown` is not loaded, and reading only
              the instance list would report it as though it were.
  ollama      `GET /api/ps`, which is precisely "what is resident", as
              distinct from `/api/tags` which is what is downloaded. Using
              tags here would list a disk again.
  mlx-lm      `GET /v1/models` on a port that answers it. mlx-lm's server has
  mlx-vlm     no "what is loaded" endpoint at all -- `/v1/models` lists what
              it COULD serve -- so what is reported is the port and what it
              offers, marked as such rather than dressed up as residency.

Everything here is an HTTP read with a short timeout. A runtime that is not
running is not an error; it is the ordinary case and it reports as absent.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
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
    state: str = "loaded"           # loaded | loading | offered
    bytes_resident: int = 0         # 0 when the runtime does not say
    detail: str = ""
    can_unload: bool = False
    ident: str = ""                 # what an unload would name
    extra: dict = field(default_factory=dict, repr=False)

    @property
    def gib(self) -> float:
        return self.bytes_resident / GIB


def _get(url: str, timeout: float = 1.5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode() or "null")
    except Exception:
        return None


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

    out = []
    for iid, inst in instances.items():
        model = ""
        if isinstance(inst, dict):
            sa = inst.get("shard_assignments") or inst.get("shardAssignments") or {}
            model = sa.get("model_id") or sa.get("modelId") or ""
        out.append(Resident(
            runtime="exo", name=str(model) or iid[:12], where=base,
            state="loaded" if up else ("loading" if coming else "offered"),
            detail=f"{up} runner{'' if up == 1 else 's'} up" if up
                   else "no runner up",
            can_unload=True, ident=iid))
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
    return [Resident(
        runtime="knurlogic", name=a.get("name") or "?", where=base,
        bytes_resident=int(m.get("active_bytes") or 0),
        detail=a.get("model_type") or "", can_unload=True,
        ident=a.get("path") or a.get("name") or "")]


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
    for port in p.get("openai", []):
        if port in seen_ports:
            continue
        base = f"http://127.0.0.1:{port}"
        # Ours answers /status.json; anything else gets read as a plain
        # OpenAI port. Asking ours first stops knurlogic listing itself as
        # an anonymous mlx server.
        rows = _knurlogic(base) or _openai_port(base)
        out += rows

    return {
        "resident": [r.__dict__ for r in out],
        "runtimes": sorted({r.runtime for r in out}),
        "bytes_resident": sum(r.bytes_resident for r in out),
    }


def render(doc: dict) -> str:
    rows = doc.get("resident") or []
    if not rows:
        return ("nothing is loaded on this machine. Four runtimes asked: "
                "knurlogic, exo, ollama, and any OpenAI-style port.")
    L = [f"{'RUNTIME':<11}{'STATE':<10}{'SIZE':>8}  MODEL"]
    for r in rows:
        size = f"{r['bytes_resident'] / GIB:.1f}G" if r["bytes_resident"] else "--"
        L.append(f"{r['runtime']:<11}{r['state']:<10}{size:>8}  {r['name'][:46]}"
                 + (f"   ({r['detail']})" if r["detail"] else ""))
    total = doc.get("bytes_resident") or 0
    if total:
        L.append("")
        L.append(f"{total / GIB:.1f} GiB accounted for. A runtime that does "
                 f"not report its own size is shown as --, never as zero.")
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
# Reading is half of it. Nobody should have to open exo's own page to put a
# model on the cluster or take it off, so the two calls that do it live here.

def _post(url: str, payload=None, method: str = "POST", timeout: float = 30.0):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode()
    return json.loads(body) if body.strip() else {}


def exo_load(base: str, model_id: str, min_nodes: int = 1,
             sharding: str = "Pipeline") -> dict:
    """Put a model on the exo cluster: placement, then instance.

    Two calls, and the first one is why this is not guesswork --
    `GET /instance/placement` returns the very `Instance` object that
    `POST /instance` takes as `payload.instance`, so the shard assignment is
    exo's own decision rather than something assembled here. Verified live
    against the running daemon (a 0.7 GB Llama rung), which answered with a
    full MlxRingInstance including runnerToShard.

    exo refuses with a 400 when the model will not fit, and that refusal is
    worth passing through verbatim: it is the same check knurlogic's resolver
    does, made by the thing that will actually hold the weights.
    """
    q = urllib.parse.urlencode({"model_id": model_id, "sharding": sharding,
                                "min_nodes": min_nodes})
    inst = _get(f"{base}/instance/placement?{q}", timeout=30.0)
    if not inst:
        raise RuntimeError(f"exo would not place {model_id!r}: no placement "
                           f"returned (is it downloaded?)")
    return _post(f"{base}/instance", {"instance": inst})


def exo_unload(base: str, instance_id: str) -> dict:
    return _post(f"{base}/instance/{instance_id}", method="DELETE")


def ollama_unload(base: str, model: str) -> dict:
    """ollama has no unload endpoint; `keep_alive: 0` is the documented way.

    An empty prompt keeps it from generating anything on the way out.
    """
    return _post(f"{base}/api/generate",
                 {"model": model, "prompt": "", "keep_alive": 0})
