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

from knurlogic.machine.exo import EXO_URL  # noqa: E402  one home


def S(desc: str, typ: str = "string") -> Dict[str, Any]:
    return {"type": typ, "description": desc}


def _schema(props: Dict[str, Any], required: List[str] | None = None):
    return {"type": "object", "properties": props,
            "required": required or []}


# --- the answers ------------------------------------------------------------

def _exo_state():
    from knurlogic.machine.loaded import _get
    return _get(f"{EXO_URL}/state", timeout=4.0) or {}


def ready(**_) -> Dict[str, Any]:
    """Is the cluster in a state where loading will work?

    Every reason it is not, named. exo reports runners, instances, downloads
    and last-seen separately and a caller that checks one of them loads into
    a ring that is still moving.
    """
    from . import ui
    # knurlogic's OWN loads move memory too. Checking only exo let a second
    # load through while the first was still reading weights, on a budget
    # that did not yet count them -- the same race as placing on an
    # unsettled exo ring, reproduced here before this check existed.
    mine = [{"what": "a knurlogic server is still loading",
             "detail": {"port": c["port"],
                        "artifact": (c["artifact"] or "").split("/")[-1],
                        "phase": c["phase"],
                        "seconds": c["seconds_since_start"],
                        "last_log_line": c.get("last_log_line", "")},
             "why": "memory is about to change; a fit measured now is stale"}
            for c in ui.loading()]
    st = _exo_state()
    if not st:
        return {"ready": not mine, "exo": False,
                "blockers": mine, "exo_placement_blockers": [],
                "ready_for_exo_placement": not mine,
                "note": "exo is not running. Nothing to settle there -- "
                        "knurlogic serves a single box without it."}

    def state_of(v):
        return next(iter(v)) if isinstance(v, dict) and v else str(v)

    from knurlogic.machine import exo
    transit, ghosts = exo.runner_activity(st)
    ongoing, failed, pending = _downloads(st)
    nodes = (st.get("topology") or {}).get("nodes") or []
    seen = st.get("lastSeen") or {}
    missing = [n for n in nodes if n not in seen]

    # Two different questions, because `load` starts a LOCAL server and exo
    # placement is a ring operation. A runner loading anywhere on this box
    # moves memory, so `fit`'s number is about to be wrong: that blocks both.
    # A download or a missing node blocks placing a model on the ring and
    # has nothing to do with a server on this box.
    local, ring = list(mine) + exo.moving(st=st), []
    if transit:
        local.append({"what": "exo runners loading or unloading",
                      "detail": transit,
                      "why": "memory is about to change; a fit measured now "
                             "is stale"})
    if ongoing:
        ring.append({"what": "downloads in progress", "detail": ongoing})
    if missing:
        ring.append({"what": "nodes in the topology not seen recently",
                     "detail": len(missing)})
    return {
        "ready": not local,
        "ready_for_exo_placement": not (local or ring),
        "exo": True,
        "blockers": local,
        "exo_placement_blockers": ring,
        "nodes": len(nodes),
        "instances": len(st.get("instances") or {}),
        "downloads": {"in_progress": len(ongoing), "failed": failed,
                      "pending_not_started": pending},
        "checked": ["runners", "downloads (DownloadOngoing only)",
                    "topology vs lastSeen", "placements not yet published"],
        **({"stale_runner_records": ghosts,
            "stale_note": "runners no instance references, unchanged for "
                          f"{exo.GHOST_S}s+: exo never marked them shut down. "
                          "Not counted as memory in motion."}
           if ghosts else {}),
    }


def _downloads(st: dict) -> tuple:
    """(ongoing [named], failed [named], pending count) from exo's state.

    Only DownloadOngoing is in flight. exo lists every model card it knows
    as DownloadPending on every node, so counting any entry at all made a
    two-node cluster with nothing downloading report 'not ready' forever --
    and an agent told that forever learns to pass force=true every time,
    which is worse than no check.
    """
    ongoing, failed, pending = [], [], 0
    for node, items in (st.get("downloads") or {}).items():
        for it in (items if isinstance(items, list) else [items]):
            if not isinstance(it, dict) or not it:
                continue
            kind, body = next(iter(it.items()))
            if not isinstance(body, dict):
                body = {}
            card = (((body or {}).get("shardMetadata") or {})
                    .get("PipelineShardMetadata") or
                    next(iter(((body or {}).get("shardMetadata") or {})
                              .values()), {}) or {}).get("modelCard") or {}
            name = card.get("modelId", "?")
            if kind == "DownloadOngoing":
                prog = (body or {}).get("downloadProgress") or {}
                done = prog.get("downloadedBytes", {}).get("inBytes") \
                    if isinstance(prog.get("downloadedBytes"), dict) else None
                total = prog.get("totalBytes", {}).get("inBytes") \
                    if isinstance(prog.get("totalBytes"), dict) else None
                ongoing.append({"model": name, "node": node[:12],
                                "progress": (f"{done / total:.0%}"
                                             if done and total else "?")})
            elif kind == "DownloadFailed":
                failed.append({"model": name, "node": node[:12]})
            elif kind == "DownloadPending":
                pending += 1
    return ongoing, failed, pending


def fit(artifact: str = "", **_) -> Dict[str, Any]:
    """Will this artifact fit, with the arithmetic shown.

    Against `wired.load_budget()` -- the same number `settings` resolves
    against and `load` starts a server with, so the three cannot disagree.
    """
    from knurlogic.machine.artifact import Artifact
    from knurlogic.machine.loaded import available_memory
    from knurlogic.tuning import settings as S
    from knurlogic.machine import wired

    a = Artifact.load(artifact)
    mem = available_memory()
    b = wired.load_budget()
    budget = b["bytes"]
    adv = wired.advise(a.bytes_on_disk)
    headroom = budget - a.bytes_on_disk
    fits = bool(budget) and headroom > 0
    tight = fits and headroom < S.TIGHT_HEADROOM_GIB * GIB
    verdict = ("will not fit" if not fits else
               "tight" if tight else "fits")
    return {
        "artifact": a.path.name,
        "verdict": verdict,
        "fits": fits,
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
               "demand, and what exo reports).",
    }


def state(**_) -> Dict[str, Any]:
    """What is loaded on this machine, in every runtime, and where the RAM went."""
    from knurlogic.machine import loaded
    doc = loaded.survey()
    m = doc.get("memory") or {}
    from knurlogic.interfaces import ui
    from knurlogic.machine import exo
    return {
        "exo_instances": exo.phases(),
        "resident": doc.get("resident", []),
        "runtimes": doc.get("runtimes", []),
        "started_here": ui.children(),
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
    avail = wired.load_budget()["bytes"]
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
            "note": "mlx-lm has no MTP path and neither does upstream exo. "
                    "knurlogic drafts with a packed head by default, on "
                    "single requests and batches alike: load(draft=false) "
                    "or `serve --no-draft` turns it off. Drafting preserves "
                    "the output distribution, so off is for troubleshooting."}


def load(artifact: str = "", port: int = 8080, tune: str = "balanced",
         sets: Dict[str, str] | None = None, force: bool = False,
         draft: bool = True, **_) -> Dict[str, Any]:
    """Start a server for this artifact, after checking it can work.

    REFUSES rather than gambles: an unsettled ring or a model that does not
    fit is a refusal with the reason attached. `force` overrides the ring
    check only -- it will not make a model fit.
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


def _holders() -> List[Dict[str, Any]]:
    """What is holding memory right now, by name -- the answer to 'what
    would I unload to make room', which a bare refusal never gives."""
    from knurlogic.machine import exo
    from knurlogic.interfaces import ui
    out = [{"what": r["model"],
            "where": "exo, " + ", ".join(x["node"] for x in r["runners"]),
            "phase": r["phase"],
            "remove_with": f"unplace(instance_id='{r['instance_id']}')"}
           for r in exo.phases()]
    out += [{"what": (c["artifact"] or "").split("/")[-1],
             "where": f"knurlogic, this machine, port {c['port']}",
             "gib": round(c.get("bytes_resident", 0) / GIB, 1),
             "phase": c["phase"], "remove_with": f"unload(port={c['port']})"}
            for c in ui.children() if c["phase"] != "exited"]
    return out


def place(model: str = "", sharding: str = "Pipeline", min_nodes: int = 1,
          force: bool = False, **_) -> Dict[str, Any]:
    """Put a model on the exo cluster -- only on a settled ring, only if it
    fits every node it lands on.

    The failure this replaces: an agent places on a ring still moving, then
    waits for something to happen and nothing does. So it refuses first and
    says why, and afterwards every poll of `state` gives the instance a
    phase, including `stalled` when nothing has moved for minutes.
    """
    from knurlogic.machine import exo
    from knurlogic.tuning import settings as S

    if not model:
        return {"error": "pass model: an exo model id (org/name) or a path "
                         "from `models` in exo's store"}
    r = ready()
    if not r["ready_for_exo_placement"] and not force:
        return {"placed": False, "refused": "the cluster is not settled",
                "blockers": r["blockers"] + r["exo_placement_blockers"],
                "note": "placing now is how an instance ends up waiting on a "
                        "rank that never connects. Poll `ready`; force=true "
                        "places anyway."}
    p = exo.plan(model, sharding=sharding, min_nodes=min_nodes)
    if p.get("error"):
        out = {"placed": False, "refused": p["error"],
               "detail": {k: v for k, v in p.items()
                          if k not in ("instance", "error")},
               "holding_memory": _holders()}
        if p.get("short_by_gib"):
            out["note"] = (f"{p['short_by_gib']} GiB short across the "
                           f"cluster. `holding_memory` is what could be "
                           f"unloaded to make room.")
        return out
    if not p["fits"]:
        return {"placed": False, "refused": "a shard does not fit its node",
                "nodes": p["nodes"], "holding_memory": _holders(),
                "note": "no override: it is arithmetic. Unload something from "
                        "`holding_memory`, or raise min_nodes."}
    tight = [n for n in p["nodes"] if n["headroom_gib"] < S.TIGHT_HEADROOM_GIB]
    placed = exo.place(p)
    out = {"placed": True, "instance_id": placed["instance_id"],
           "model_id": p["model_id"], "nodes": p["nodes"],
           "next": "poll `state` (exo_instances) until this instance is "
                   "serving; it reports downloading, loading (layers), "
                   "warming, failed, or stalled -- never silence."}
    if tight:
        out["warning"] = (f"tight on {', '.join(n['node'] for n in tight)}: "
                          f"under {S.TIGHT_HEADROOM_GIB:g} GiB left after the "
                          f"shard, and long prompts need room to prefill")
    return out


def unplace(instance_id: str = "", **_) -> Dict[str, Any]:
    """Remove an instance from the exo cluster."""
    from knurlogic.machine import exo
    if not instance_id:
        return {"error": "pass instance_id; `state` lists them"}
    return exo.unplace(instance_id)


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
        "description": "Is it safe to load now? `ready` covers a local "
                       "load (exo runners moving memory on this box); "
                       "`ready_for_exo_placement` adds ring-wide blockers "
                       "(downloads in progress, unseen nodes). Every "
                       "blocker is named. Call this before loading.",
        "schema": _schema({}),
    },
    "state": {
        "fn": state,
        "description": "What is loaded on this machine in every runtime "
                       "(knurlogic, exo, ollama, any OpenAI port) and where "
                       "the memory went. `started_here` gives each server `load` "
                       "started a phase -- serving, loading (with elapsed "
                       "time and the last log line), stalled (stop waiting, "
                       "read the log), or exited (exit code, log tail).",
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
                       "if it will not fit (no override) or exo is moving "
                       "memory on this box (force overrides).",
        "schema": _schema({
            "artifact": S("path to the artifact"),
            "port": S("port to serve on", "integer"),
            "tune": S("safe | balanced | fast"),
            "sets": {"type": "object",
                     "description": "launch-only knob overrides, KEY: VALUE"},
            "force": {"type": "boolean",
                      "description": "load while exo runners are moving "
                                     "memory"},
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
    "place": {
        "fn": place,
        "description": "Put a model on the exo cluster. Refuses on an "
                       "unsettled ring (force overrides) and when any "
                       "node's shard will not fit (no override), naming "
                       "what is holding memory. Then poll `state`: the "
                       "instance reports downloading, loading (layers), "
                       "warming, serving, failed or stalled.",
        "schema": _schema({
            "model": S("exo model id (org/name) or a path from `models`"),
            "sharding": S("Pipeline (default) or Tensor"),
            "min_nodes": S("at least this many nodes", "integer"),
            "force": {"type": "boolean",
                      "description": "place on an unsettled ring anyway"},
        }, ["model"]),
    },
    "unplace": {
        "fn": unplace,
        "description": "Remove an instance from the exo cluster.",
        "schema": _schema({"instance_id": S("from `state` exo_instances")},
                          ["instance_id"]),
    },
    "unload": {
        "fn": unload,
        "description": "Stop a local server knurlogic started (any "
                       "session). For exo instances use `unplace`.",
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
