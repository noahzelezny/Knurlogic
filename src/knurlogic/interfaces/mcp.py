#!/usr/bin/env python3
"""knurlogic mcp -- the interface an agent gets to the local models.

    knurlogic mcp            # serve on stdio
    knurlogic mcp --list     # print the tool table

Stdio JSON-RPC, stdlib only: it reads the machine and starts servers, and
imports nothing that could load a model into this process.

WHAT THIS IS FOR. Managing local models by hand means guessing: whether
memory has settled, whether a model fits, what the knobs are and why they
are set that way. An agent guesses worse than a person, and faster.
Every tool here answers deterministically, reports what it looked at, and
refuses rather than gambles.

DESIGN RULES, each one paid for:

* `ready` is a gate, not a status line. Loading while another load is still
  moving memory is the failure Noah hits repeatedly. `ready` names every
  reason it is not, and `load` calls it first and REFUSES rather than trying
  anyway.

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


def S(desc: str, typ: str = "string") -> Dict[str, Any]:
    return {"type": typ, "description": desc}


def _schema(props: Dict[str, Any], required: List[str] | None = None):
    return {"type": "object", "properties": props,
            "required": required or []}


# --- the answers ------------------------------------------------------------

def ready(**_) -> Dict[str, Any]:
    """Is this machine in a state where loading will work?

    Every reason it is not, named: a load still reading weights moves memory,
    so a fit measured now would be stale.
    """
    from . import ui
    # A second load let through while the first was still reading weights,
    # on a budget that did not yet count them, is the race this gate ends.
    blockers = [{"what": "a knurlogic server is still loading",
                 "detail": {"port": c["port"],
                            "artifact": (c["artifact"] or "").split("/")[-1],
                            "phase": c["phase"],
                            "seconds": c["seconds_since_start"],
                            "last_log_line": c.get("last_log_line", "")},
                 "why": "memory is about to change; a fit measured now is "
                        "stale"}
                for c in ui.loading()]
    # The load lock (P0/`machine/loadlock.py`) is a second source of the same
    # blocker: a load started by a DIFFERENT process (another agent, `serve`
    # run by hand) holds it and would not otherwise show up in `ui.loading()`,
    # which only knows about children this MCP process itself started.
    from knurlogic.machine import loadlock
    held = loadlock.holder()
    if held is not None:
        blockers.append({
            "what": "another process holds the model-load lock",
            "detail": held,
            "why": "memory is about to change; a fit measured now is stale"})
    return {"ready": not blockers, "blockers": blockers,
            "checked": ["servers this MCP started", "the model-load lock"]}


def fit(artifact: str = "", **_) -> Dict[str, Any]:
    """Will this artifact fit, with the arithmetic shown.

    Against `wired.load_budget()` -- the same number `settings` resolves
    against and `load` starts a server with, so the three cannot disagree.
    """
    from knurlogic.machine.artifact import Artifact
    from knurlogic.machine.loaded import available_memory
    from knurlogic.tuning import settings as S
    from knurlogic.machine import wired

    from knurlogic.engine.vision import registry as vision_registry

    a = Artifact.load(artifact)
    mem = available_memory()
    b = wired.load_budget()
    budget = b["bytes"]
    from knurlogic.tuning.resolve import vision_budget
    adv = wired.advise(a.bytes_on_disk)
    # A vision rung also holds its tower, its image store and its images'
    # KV -- the resolver's terms, so `fit` and `settings` agree.
    vb = vision_budget(a)
    extra = vb["extra_bytes"] if vb else 0
    headroom = budget - a.bytes_on_disk - extra
    fits = bool(budget) and headroom > 0
    tight = fits and headroom < S.TIGHT_HEADROOM_GIB * GIB
    verdict = ("will not fit" if not fits else
               "tight" if tight else "fits")
    return {
        "artifact": a.path.name,
        "verdict": verdict,
        "fits": fits,
        # Static: whether a family for this model_type exists at all, not
        # whether THIS config.json has a vision_config (that needs the build
        # step -- registry.build -- which does not run before a load). Good
        # enough for "would this be worth attaching an image to".
        "vision_capable": vision_registry.registered(a.model_type),
        "vision_budget": _vision_terms(vb),
        "size_gib": round(a.gib, 1),
        "budget_gib": round(budget / GIB, 1),
        "headroom_gib": round(headroom / GIB, 1),
        "limited_by": b["limited_by"],
        "what_tight_means": (
            f"under {S.TIGHT_HEADROOM_GIB:g} GiB left after the weights, "
            f"so a load now narrows the prompt chunk to "
            f"{S.PREFILL_CHUNK_TIGHT} tokens and prefills one prompt at a "
            f"time. It loads; long prompts are slower to start."
            if tight else ""),
        "available_now_gib": round(b["available_bytes"] / GIB, 1),
        "working_set_gib": round(b["working_set_bytes"] / GIB, 1),
        "free_now_gib": round(mem.get("free_bytes", 0) / GIB, 1),
        "reclaimable_cache_gib": round(mem.get("cached_bytes", 0) / GIB, 1),
        "wired_limit_gib": round(adv.get("limit_bytes", 0) / GIB, 1),
        "wired_action": adv.get("action"),
        "wired_note": adv.get("note", ""),
        "how": "budget = the smaller of the GPU working set and memory "
               "available now (free + inactive: what macOS hands over on "
               "demand).",
    }


def _vision_terms(vb) -> Dict[str, Any] | None:
    """The resolver's vision terms (tuning.resolve.vision_budget) in GiB,
    each with its note; None for a text-only artifact."""
    if not vb:
        return None
    g = lambda n: round(n / GIB, 2)
    return {"tower_gib": g(vb["tower_bytes"]),
            "tower_added_gib": g(vb["tower_outside_bytes"]),
            "image_store_gib": g(vb["store_bytes"]),
            "image_kv_allowance_gib": g(vb["kv_allowance_bytes"]),
            "added_to_size_gib": g(vb["extra_bytes"]),
            "notes": list(vb["notes"])}


def state(**_) -> Dict[str, Any]:
    """What is loaded on this machine, in every runtime, and where the RAM went."""
    from knurlogic.machine import loaded
    doc = loaded.survey()
    m = doc.get("memory") or {}
    from knurlogic.interfaces import ui
    from knurlogic.engine.vision import served_vision
    spec = served_vision()
    return {
        "resident": doc.get("resident", []),
        "runtimes": doc.get("runtimes", []),
        "started_here": ui.children(),
        # None when nothing served has vision, matching the served_path()
        # pattern the rest of `state()` follows -- absence is a fact, not
        # an omission.
        "vision": spec.to_json() if spec else None,
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
    from knurlogic.machine import discover
    from knurlogic.engine import mtp
    from knurlogic.machine.artifact import Artifact
    from knurlogic.machine.loaded import available_memory

    from knurlogic.machine import wired
    from knurlogic.engine.vision import registry as vision_registry
    from knurlogic.engine.serve import thinking
    avail = wired.load_budget()["bytes"]
    out = []
    for f in discover.find():
        row = {"name": f.name, "path": str(f.path), "store": f.store,
               "size_gib": round(f.bytes_on_disk / GIB, 1),
               "model_type": f.model_type, "is_vq": f.is_vq,
               "servable": f.servable, "why_not": f.why,
               "fits": bool(avail) and f.bytes_on_disk <= avail,
               "drafting_head": bool(f.extra.get("mtp_head")),
               "vision_capable": vision_registry.registered(f.model_type),
               # what reasoning_effort does on this model: its template's
               # dialect, native levels and default (engine/serve/thinking)
               "thinking": thinking.levels(thinking.template_of(f.path))}
        if fits_only and not (row["fits"] and row["servable"]):
            continue
        out.append(row)
    out.sort(key=lambda r: -r["size_gib"])
    return {"models": out, "budget_gib": round(avail / GIB, 1),
            "fits_means": "fits the load budget -- see `fit` for headroom "
                          "and whether it is tight",
            "count": len(out)}


def settings(artifact: str = "", tune: str = "balanced", **_) -> Dict[str, Any]:
    """The knobs for this artifact, each with the measurement behind it.

    The `why` is the point. A knob without its provenance is one an agent
    changes for no reason, and these were expensive to establish.
    """
    from knurlogic.interfaces import web
    doc = web._preview(artifact, tune)
    from knurlogic.machine.artifact import Artifact
    from knurlogic.tuning.resolve import vision_budget
    try:
        doc["vision_budget"] = _vision_terms(vision_budget(
            Artifact.load(artifact)))
    except (OSError, ValueError):
        doc["vision_budget"] = None
    for k in doc.get("knobs", []):
        k["change_at"] = ("runtime" if k.get("reach") == "live"
                          else "launch only")
    doc["note"] = ("`launch only` knobs are read at import and compiled into "
                   "kernel source. They can be set with `load(sets={...})` "
                   "and not afterwards.")
    return doc


def drafting(artifact: str = "", **_) -> Dict[str, Any]:
    """Does this artifact have a multi-token-prediction head, and will it run?"""
    from knurlogic.engine import mtp
    from knurlogic.machine.artifact import Artifact
    a = Artifact.load(artifact)
    st = mtp.status(a)
    return {"artifact": a.path.name, "state": st.state,
            "head": (st.head.describe() if st.head else ""),
            "family": (st.head.family if st.head else ""),
            "explanation": st.render(),
            "note": "mlx-lm has no MTP path. knurlogic drafts with a packed head by default, on "
                    "single requests and batches alike: load(draft=false) "
                    "or `serve --no-draft` turns it off. Drafting preserves "
                    "the output distribution, so off is for troubleshooting."}


def load(artifact: str = "", port: int = 8080, tune: str = "balanced",
         sets: Dict[str, str] | None = None, force: bool = False,
         draft: bool = True, **_) -> Dict[str, Any]:
    """Start a server for this artifact, after checking it can work.

    REFUSES rather than gambles: memory still moving or a model that does
    not fit is a refusal with the reason attached. `force` overrides the
    moving-memory check only -- it will not make a model fit.
    """
    from knurlogic.interfaces import ui

    f = fit(artifact=artifact)
    if not f["fits"]:
        return {"loaded": False, "refused": "will not fit",
                "detail": f,
                "note": "no flag overrides this; it is arithmetic."}
    r = ready()
    if not r["ready"] and not force:
        return {"loaded": False, "refused": "memory is about to move",
                "detail": r["blockers"],
                "note": "something is loading or unloading on this box, so "
                        "the fit above is stale. Poll `state` -- each server "
                        "says loading, serving or stalled -- then call "
                        "`ready` again. force=true loads anyway."}
    out = ui._spawn(artifact, int(port), tune, dict(sets or {}),
                    draft=bool(draft))
    out["fit"] = f
    out["ready"] = r
    return out


def unload(port: int = 8080, **_) -> Dict[str, Any]:
    from knurlogic.interfaces import ui
    return ui._stop(int(port))


# --- the table --------------------------------------------------------------

def deps() -> Dict[str, Any]:
    """Which build of each piece every interpreter has, read off the fix
    itself rather than a version string."""
    from knurlogic.machine import deps as D
    return D.survey()


TOOLS: Dict[str, Dict[str, Any]] = {
    "ready": {
        "fn": ready,
        "description": "Is it safe to load now? Not while another load "
                       "(a server `load` started, or any process holding "
                       "the model-load lock) is still moving memory. Every "
                       "blocker is named. Call this before loading.",
        "schema": _schema({}),
    },
    "state": {
        "fn": state,
        "description": "What is loaded on this machine in every runtime "
                       "(knurlogic, ollama, exo, any OpenAI port) and where "
                       "the memory went. `started_here` gives each server `load` "
                       "started a phase -- serving, loading (with elapsed "
                       "time and the last log line), stalled (stop waiting, "
                       "read the log), or exited (exit code, log tail).",
        "schema": _schema({}),
    },
    "models": {
        "fn": models,
        "description": "Every model on this machine, with whether it fits, "
                       "whether it has a drafting head, and what "
                       "reasoning_effort (none minimal low medium high "
                       "xhigh) maps to on it.",
        "schema": _schema({"fits_only": {
            "type": "boolean",
            "description": "only models that can actually run here"}}),
    },
    "fit": {
        "fn": fit,
        "description": "Will this artifact fit NOW: verdict fits | tight "
                       "| will not fit, with headroom and what tight "
                       "changes. Uses the same budget `settings` and `load` "
                       "use.",
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
        "description": "Start a local server for an artifact, with its "
                       "settings resolved against the load budget. Refuses "
                       "if it will not fit (no override) or another load is "
                       "still moving memory (force overrides).",
        "schema": _schema({
            "artifact": S("path to the artifact"),
            "port": S("port to serve on", "integer"),
            "tune": S("safe | balanced | fast"),
            "sets": {"type": "object",
                     "description": "launch-only knob overrides, KEY: VALUE"},
            "force": {"type": "boolean",
                      "description": "load while another load is still "
                                     "moving memory"},
            "draft": {"type": "boolean",
                      "description": "use a packed drafting head "
                                     "(default true)"},
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
        "description": "Stop a local server knurlogic started (any "
                       "session).",
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


def _reply(rid, result=None, error=None) -> None:
    msg = {"jsonrpc": "2.0", "id": rid}
    msg.update({"error": error} if error is not None else {"result": result})
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


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
            # isError is how a client tells an answer from a failure. A
            # refusal (`refused`) is an ANSWER -- the tool did its job.
            result = {"content": [{"type": "text",
                                   "text": json.dumps(out, indent=1)}],
                      "isError": "error" in out}
        elif method == "ping":
            result = {}
        elif rid is None or method.startswith("notifications/") \
                or method == "initialized":
            continue
        else:
            _reply(rid, error={"code": -32601,
                               "message": f"method not found: {method}"})
            continue
        _reply(rid, result=result)
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
