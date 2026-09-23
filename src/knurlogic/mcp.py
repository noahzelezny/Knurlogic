#!/usr/bin/env python3
"""knurlogic mcp -- the interface an agent gets to the local models.

    knurlogic mcp            # serve on stdio
    knurlogic mcp --list     # print the tool table

Stdio JSON-RPC, stdlib only -- same rule as vqlab's server, for the same
reason: this has to run inside the exo envs and those must not have anything
pip-installed into them.

WHAT THIS IS FOR. Managing local models by hand through exo means guessing:
whether the ring has settled, whether a model fits, what the knobs are and
why they are set that way. An agent guesses worse than a person, and faster.
Every tool here answers deterministically, reports what it looked at, and
refuses rather than gambles.

DESIGN RULES, each one paid for:

* `ready` is a gate, not a status line. Loading into an unsettled ring is the
  failure the maintainer hits repeatedly: runners still shutting down, downloads in
  flight, a node not yet seen. `ready` names every reason it is not, and
  `load` calls it first and REFUSES rather than trying anyway.

* `fit` refuses to be optimistic. It measures available memory as free +
  inactive (the file cache macOS hands over on demand), because both other
  definitions are wrong in opposite directions -- the footprint sum and
  top's "unused" were 75.9 and 1.6 GiB on a box with 70 available. A model
  that does not fit is a refusal with the arithmetic attached, not a warning
  somebody scrolls past.

* `settings` returns every knob WITH its measurement. A number without its
  provenance is a number an agent will change for no reason. This is the
  whole point of the tool: it is why a knob is what it is, not just what.

* `load` never switches a model inside a running server. The settings that
  matter are read at import and compiled into kernel source, so they can
  only be chosen before the process starts. Loading spawns a fresh one.

* Nothing here deletes an artifact or writes to one. Reading a machine's
  state and starting a server on it are reversible; removing weights is not.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List

SERVER_NAME = "knurlogic"
SERVER_VERSION = "0"
GIB = 1 << 30

EXO_URL = "http://127.0.0.1:52415"


def S(desc: str, typ: str = "string") -> Dict[str, Any]:
    return {"type": typ, "description": desc}


def _schema(props: Dict[str, Any], required: List[str] | None = None):
    return {"type": "object", "properties": props,
            "required": required or []}


# --- the answers ------------------------------------------------------------

def _exo_state():
    from .loaded import _get
    return _get(f"{EXO_URL}/state", timeout=4.0) or {}


def ready(**_) -> Dict[str, Any]:
    """Is the cluster in a state where loading will work?

    Every reason it is not, named. exo reports runners, instances, downloads
    and last-seen separately and a caller that checks one of them loads into
    a ring that is still moving.
    """
    st = _exo_state()
    if not st:
        return {"ready": True, "exo": False,
                "blockers": [],
                "note": "exo is not running. Nothing to settle -- knurlogic "
                        "serves a single box without it."}

    def state_of(v):
        return next(iter(v)) if isinstance(v, dict) and v else str(v)

    runners = [state_of(v) for v in (st.get("runners") or {}).values()]
    settling = [r for r in runners
                if r in ("RunnerLoading", "RunnerWarmingUp",
                         "RunnerConnecting", "RunnerShuttingDown")]
    downloads = [k for k, v in (st.get("downloads") or {}).items() if v]
    nodes = (st.get("topology") or {}).get("nodes") or []
    seen = st.get("lastSeen") or {}
    missing = [n for n in nodes if n not in seen]

    blockers = []
    if settling:
        from collections import Counter
        blockers.append({"what": "runners in transition",
                         "detail": dict(Counter(settling))})
    if downloads:
        blockers.append({"what": "downloads in flight",
                         "detail": len(downloads)})
    if missing:
        blockers.append({"what": "nodes in the topology not seen recently",
                         "detail": len(missing)})
    return {
        "ready": not blockers,
        "exo": True,
        "blockers": blockers,
        "nodes": len(nodes),
        "instances": len(st.get("instances") or {}),
        "checked": ["runners", "downloads", "topology vs lastSeen"],
    }


def fit(artifact: str = "", **_) -> Dict[str, Any]:
    """Will this artifact fit, with the arithmetic shown."""
    from .artifact import Artifact
    from .loaded import available_memory
    from . import wired

    a = Artifact.load(artifact)
    mem = available_memory()
    avail = mem.get("available_bytes", 0)
    adv = wired.advise(a.bytes_on_disk)
    fits = bool(avail) and a.bytes_on_disk <= avail
    return {
        "artifact": a.path.name,
        "size_gib": round(a.gib, 1),
        "available_gib": round(avail / GIB, 1),
        "free_now_gib": round(mem.get("free_bytes", 0) / GIB, 1),
        "reclaimable_cache_gib": round(mem.get("cached_bytes", 0) / GIB, 1),
        "fits": fits,
        "wired_limit_gib": round(adv.get("limit_bytes", 0) / GIB, 1),
        "wired_action": adv.get("action"),
        "wired_note": adv.get("note", ""),
        "how": "available = free + inactive, which is what macOS will hand "
               "over on demand and what exo reports. The other two "
               "definitions read 75.9 and 1.6 GiB on a box with 70 free.",
    }


def state(**_) -> Dict[str, Any]:
    """What is loaded on this machine, in every runtime, and where the RAM went."""
    from . import loaded
    doc = loaded.survey()
    m = doc.get("memory") or {}
    return {
        "resident": doc.get("resident", []),
        "runtimes": doc.get("runtimes", []),
        "memory": {
            "installed_gib": round(m.get("installed_bytes", 0) / GIB, 1),
            "used_gib": round(m.get("used_bytes", 0) / GIB, 1),
            "available_gib": round(m.get("free_bytes", 0) / GIB, 1),
            "by_runtime_gib": {k: round(v / GIB, 2)
                               for k, v in (m.get("by_runtime") or {}).items()},
            "unattributed_gib": round(m.get("other_bytes", 0) / GIB, 1),
        },
    }


def models(fits_only: bool = False, **_) -> Dict[str, Any]:
    """Every model on this machine, with what can actually run."""
    from . import discover, mtp
    from .artifact import Artifact
    from .loaded import available_memory

    avail = available_memory().get("available_bytes", 0)
    out = []
    for f in discover.find():
        row = {"name": f.name, "path": str(f.path), "store": f.store,
               "size_gib": round(f.bytes_on_disk / GIB, 1),
               "model_type": f.model_type, "is_vq": f.is_vq,
               "servable": f.servable, "why_not": f.why,
               "fits": bool(avail) and f.bytes_on_disk <= avail,
               "drafting_head": bool(f.extra.get("mtp_head"))}
        if fits_only and not (row["fits"] and row["servable"]):
            continue
        out.append(row)
    out.sort(key=lambda r: -r["size_gib"])
    return {"models": out, "available_gib": round(avail / GIB, 1),
            "count": len(out)}


def settings(artifact: str = "", tune: str = "balanced", **_) -> Dict[str, Any]:
    """The knobs for this artifact, each with the measurement behind it.

    The `why` is the point. A knob without its provenance is one an agent
    changes for no reason, and these were expensive to establish.
    """
    from . import web
    doc = web._preview(artifact, tune)
    for k in doc.get("knobs", []):
        k["change_at"] = ("runtime" if k.get("reach") == "live"
                          else "launch only")
    doc["note"] = ("`launch only` knobs are read at import and compiled into "
                   "kernel source. They can be set with `load(sets={...})` "
                   "and not afterwards.")
    return doc


def drafting(artifact: str = "", **_) -> Dict[str, Any]:
    """Does this artifact have a multi-token-prediction head, and will it run?"""
    from . import mtp
    from .artifact import Artifact
    a = Artifact.load(artifact)
    st = mtp.status(a)
    return {"artifact": a.path.name, "state": st.state,
            "head": (st.head.describe() if st.head else ""),
            "family": (st.head.family if st.head else ""),
            "explanation": st.render(),
            "note": "mlx-lm has no MTP path and neither does upstream exo. "
                    "knurlogic runs a packed head by default; --no-draft "
                    "turns it off."}


def load(artifact: str = "", port: int = 8080, tune: str = "balanced",
         sets: Dict[str, str] | None = None, force: bool = False,
         **_) -> Dict[str, Any]:
    """Start a server for this artifact, after checking it can work.

    REFUSES rather than gambles: an unsettled ring or a model that does not
    fit is a refusal with the reason attached. `force` overrides the ring
    check only -- it will not make a model fit.
    """
    from . import ui

    f = fit(artifact=artifact)
    if not f["fits"]:
        return {"loaded": False, "refused": "will not fit",
                "detail": f,
                "note": "no flag overrides this; it is arithmetic."}
    r = ready()
    if not r["ready"] and not force:
        return {"loaded": False, "refused": "cluster is not settled",
                "detail": r,
                "note": "pass force=true to load anyway. The failure this "
                        "prevents is a load into a ring that is still moving."}
    out = ui._spawn(artifact, int(port), tune, dict(sets or {}))
    out["fit"] = f
    out["ready"] = r
    return out


def unload(port: int = 8080, **_) -> Dict[str, Any]:
    from . import ui
    return ui._stop(int(port))


# --- the table --------------------------------------------------------------

def deps() -> Dict[str, Any]:
    """Which build of each piece every interpreter has, read off the fix
    itself rather than a version string."""
    from . import deps as D
    return D.survey()


TOOLS: Dict[str, Dict[str, Any]] = {
    "ready": {
        "fn": ready,
        "description": "Is the cluster settled enough to load? Names every "
                       "reason it is not. Call this before loading.",
        "schema": _schema({}),
    },
    "state": {
        "fn": state,
        "description": "What is loaded on this machine in every runtime "
                       "(knurlogic, exo, ollama, any OpenAI port) and where "
                       "the memory went.",
        "schema": _schema({}),
    },
    "models": {
        "fn": models,
        "description": "Every model on this machine, with whether it fits "
                       "and whether it has a drafting head.",
        "schema": _schema({"fits_only": {
            "type": "boolean",
            "description": "only models that can actually run here"}}),
    },
    "fit": {
        "fn": fit,
        "description": "Will this artifact fit, with the arithmetic and the "
                       "wired limit shown.",
        "schema": _schema({"artifact": S("path to the artifact")},
                          ["artifact"]),
    },
    "settings": {
        "fn": settings,
        "description": "The resolved knobs for an artifact, each with the "
                       "measurement behind it and whether it can be changed "
                       "at runtime or only at launch.",
        "schema": _schema({"artifact": S("path to the artifact"),
                           "tune": S("safe | balanced | fast")},
                          ["artifact"]),
    },
    "drafting": {
        "fn": drafting,
        "description": "Whether this artifact has a multi-token-prediction "
                       "head and what will happen to it.",
        "schema": _schema({"artifact": S("path to the artifact")},
                          ["artifact"]),
    },
    "load": {
        "fn": load,
        "description": "Start a server for an artifact. Refuses if it will "
                       "not fit or the ring is unsettled.",
        "schema": _schema({
            "artifact": S("path to the artifact"),
            "port": S("port to serve on", "integer"),
            "tune": S("safe | balanced | fast"),
            "sets": {"type": "object",
                     "description": "launch-only knob overrides, KEY: VALUE"},
            "force": {"type": "boolean",
                      "description": "load despite an unsettled ring"},
        }, ["artifact"]),
    },
    "deps": {
        "fn": deps,
        "description": "What this stack stands on, per interpreter "
                       "(knurlogic's and exo's): mlx, mlx-lm, mlx-vlm, exo, "
                       "each marked stock or fork by what is installed, not "
                       "by version -- plus what each fork carries and why it "
                       "is or is not ported. Call this when something works "
                       "in exo and not here, or the reverse.",
        "schema": _schema({}),
    },
    "unload": {
        "fn": unload,
        "description": "Stop a server this tool started.",
        "schema": _schema({"port": S("port it was started on", "integer")}),
    },
}


def tool_list() -> List[Dict[str, Any]]:
    return [{"name": n, "description": t["description"],
             "inputSchema": t["schema"]} for n, t in TOOLS.items()]


def _call(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    t = TOOLS.get(name)
    if t is None:
        return {"error": f"unknown tool {name!r}",
                "available": sorted(TOOLS)}
    try:
        return t["fn"](**(args or {}))
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "tool": name}


def _serve_stdio() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue
        rid, method = req.get("id"), req.get("method")
        params = req.get("params") or {}
        if method == "initialize":
            result = {"protocolVersion": "2024-11-05",
                      "capabilities": {"tools": {}},
                      "serverInfo": {"name": SERVER_NAME,
                                     "version": SERVER_VERSION}}
        elif method == "tools/list":
            result = {"tools": tool_list()}
        elif method == "tools/call":
            out = _call(params.get("name", ""), params.get("arguments") or {})
            result = {"content": [{"type": "text",
                                   "text": json.dumps(out, indent=1)}]}
        elif method in ("notifications/initialized", "initialized"):
            continue
        else:
            result = {}
        if rid is not None:
            sys.stdout.write(json.dumps(
                {"jsonrpc": "2.0", "id": rid, "result": result}) + "\n")
            sys.stdout.flush()
    return 0


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        prog="knurlogic mcp",
        description="the interface an agent gets to the local models")
    p.add_argument("--list", action="store_true",
                   help="print the tool table and exit")
    a = p.parse_args(argv)
    if a.list:
        for t in tool_list():
            print(f"{t['name']:<10} {t['description']}")
        return 0
    return _serve_stdio()
