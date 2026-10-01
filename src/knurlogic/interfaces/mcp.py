#!/usr/bin/env python3
"""knurlogic mcp -- the interface an agent gets to the local models.

    knurlogic mcp            # serve on stdio
    knurlogic mcp --list     # print the tool table

Stdio JSON-RPC, stdlib only: it reads the machine and starts servers, and
imports nothing that could load a model into this process. Every tool
answers deterministically, reports what it looked at, and refuses rather
than gambles: `ready` is a gate `load` checks first; `fit` counts free +
file-cache memory and refuses with the arithmetic; `settings` returns every
knob with its measurement; `load` always spawns a fresh server; nothing
deletes or writes an artifact.

Design: docs/design/mcp.md.
"""

from __future__ import annotations

import http.client
import json
import sys
from typing import Any

SERVER_NAME = "knurlogic"
SERVER_VERSION = "0"
#: sent in the initialize result: how to drive these tools, for a model
#: that has nothing else to read
INSTRUCTIONS = (
    "knurlogic manages local MLX models on this Mac and the Macs its page "
    "sees. The loop: `models` (what is on disk and whether it fits) -> "
    "`fit` (will this artifact fit now, with the arithmetic) -> "
    "`settings` (the resolved knobs and why) -> `ready` (is it safe to "
    "load now) -> `load` -> `state` (poll until the phase is serving) -> "
    "`unload` when done. Never call `load` while `ready` is false: wait "
    "and call `ready` again, or unload something; nothing is evicted for "
    "you. A refusal is an answer with its reason, not an error to retry "
    "blindly. Never wait on silence: every load has a phase in `state` -- "
    "loading, warming, serving, stalled (stop waiting and read the log "
    "shown), or exited (exit code and log tail). Once serving, the model "
    "answers OpenAI and Anthropic Messages requests at "
    "http://<machine>:<port>/v1 (the port and leader `load` returned; "
    "127.0.0.1 for this Mac). `deps` says "
    "which mlx builds are installed. No tool deletes or modifies model "
    "files. The levers, when a load does not fit or runs short of memory: "
    "`load`'s tune (`default`, fastest; `lean`, 512-token prompt chunks, MTP "
    "off, 8-bit KV -- the most context in the least memory), or single "
    "settings in `load`'s sets: KNURLOGIC_CONTEXT_LENGTH (less context, "
    "less KV memory; past the native window the Qwen families use YaRN, "
    "up to 1,048,576), KNURLOGIC_KV_BITS=8, KNURLOGIC_MTP=off (frees the "
    "draft head's memory), KNURLOGIC_PREFILL_CHUNK=512 (a smaller spike "
    "while reading a prompt). `settings` shows what each resolves to."
)
GIB = 1 << 30


def S(desc: str, typ: str = "string") -> dict[str, Any]:
    return {"type": typ, "description": desc}


def _schema(props: dict[str, Any], required: list[str] | None = None):
    return {"type": "object", "properties": props,
            "required": required or []}


# --- the answers ------------------------------------------------------------

def ready(**_) -> dict[str, Any]:
    """Is this machine in a state where loading will work?

    Every reason it is not, named: a load still reading weights moves memory,
    so a fit measured now would be stale.
    """
    from knurlogic.interfaces.page import server as page_server
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
                for c in page_server.loading()]
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


def fit(artifact: str = "", **_) -> dict[str, Any]:
    """Will this artifact fit, with the arithmetic shown.

    Against `wired.load_budget()` -- the same number `settings` resolves
    against and `load` starts a server with, so the three cannot disagree.
    """
    from knurlogic.engine.vision import registry as vision_registry
    from knurlogic.machine import wired
    from knurlogic.machine.artifact import Artifact
    from knurlogic.machine.loaded import available_memory
    from knurlogic.tuning import settings as S

    a = Artifact.load(artifact)
    mem = available_memory()
    b = wired.load_budget()
    budget = b["bytes"]
    from knurlogic.tuning.resolve import room_for, vision_budget
    adv = wired.advise(a.bytes_on_disk)
    # A vision rung also holds its tower, its image store and its images'
    # KV -- the resolver's terms, so `fit` and `settings` agree.
    vb = vision_budget(a)
    extra = vb["extra_bytes"] if vb else 0
    headroom = budget - a.bytes_on_disk - extra
    fits = bool(budget) and headroom > 0
    tight = fits and headroom < S.tight_headroom_bytes(budget)
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
            f"under {S.tight_headroom_bytes(budget) / GIB:.0f} GiB left "
            f"after the weights, "
            f"so a load now narrows the prompt chunk to "
            f"{S.PREFILL_CHUNK_TIGHT} tokens and prefills one prompt at a "
            f"time. It loads; long prompts are slower to start."
            if tight else ""),
        # what a fit leaves to talk in (tuning/resolve.context_room)
        "room": room_for(a.bytes_on_disk + extra, a.raw_config),
        "available_now_gib": round(b["available_bytes"] / GIB, 1),
        "working_set_gib": round(b["working_set_bytes"] / GIB, 1),
        "allowance_gib": round(b.get("allowance_bytes", 0) / GIB, 1),
        "free_now_gib": round(mem.get("free_bytes", 0) / GIB, 1),
        "reclaimable_cache_gib": round(mem.get("cached_bytes", 0) / GIB, 1),
        "wired_limit_gib": round(adv.get("limit_bytes", 0) / GIB, 1),
        "wired_action": adv.get("action"),
        "wired_note": adv.get("note", ""),
        "how": "budget = the smaller of the GPU working set and memory "
               "available now (free + file cache + purgeable: what macOS hands "
               "over without swapping).",
    }


def _vision_terms(vb) -> dict[str, Any] | None:
    """The resolver's vision terms (tuning.resolve.vision_budget) in GiB,
    each with its note; None for a text-only artifact."""
    if not vb:
        return None

    def g(n):
        return round(n / GIB, 2)

    return {"tower_gib": g(vb["tower_bytes"]),
            "tower_added_gib": g(vb["tower_outside_bytes"]),
            "image_store_gib": g(vb["store_bytes"]),
            "image_kv_allowance_gib": g(vb["kv_allowance_bytes"]),
            "added_to_size_gib": g(vb["extra_bytes"]),
            "notes": list(vb["notes"])}


def state(**_) -> dict[str, Any]:
    """What is loaded on this machine, in every runtime, and where the RAM
    went -- plus `models`: every model resident on this Mac and the peers
    answering its page, a cluster job once."""
    from knurlogic.machine import loaded
    doc = loaded.survey()
    m = doc.get("memory") or {}
    from knurlogic.engine.vision import served_vision
    from knurlogic.interfaces.page import server as page_server
    spec = served_vision()
    # across machines: what the page on this Mac sees (its own residency and
    # each answering peer's). Without the page, this Mac's survey alone.
    try:
        page = _page_get("/loaded.json?peers=1")
    except PageDown as e:
        page, page_note = None, str(e)
    else:
        page_note = ""
    here = _me_name()
    everywhere = (models_across(page, here) if page is not None else
                  models_across({"resident": doc.get("resident", [])}, here))
    return {
        # one entry per model: a cluster job once, on its leader, with its
        # machines, split, link, job and rank 0's `requests`
        "models": everywhere,
        "machines": _machines_of(page, here),
        **({"page": page_note} if page_note else {}),
        "resident": doc.get("resident", []),
        "runtimes": doc.get("runtimes", []),
        "started_here": page_server.children(),
        # per knurlogic model: in_flight, pending, capacity,
        # oldest_pending_s, holding (its server's /status.json `requests`)
        "requests": [dict(r["requests"], model=r["name"], where=r["where"],
                          machine=r["machine"])
                     for r in everywhere if r.get("requests")],
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


# --- across machines: through the page on this Mac ---------------------------
# The page (`knurlogic ui`) is each Mac's node agent. It holds the peers it
# can see, the cluster jobs it coordinates and the watcher that tears a
# failed one down -- state that lives in ITS process, not this one. So every
# tool that reaches past this Mac asks that page over loopback, with the
# same POST /loaded.json its Launch and Unload buttons send: one code path,
# and a job the MCP started is watched and failed over like any other.

#: the page on this Mac, host:port (`knurlogic ui --port`, default 8899)
PAGE_ENV = "KNURLOGIC_PAGE"
#: a cluster launch answers once every machine has prepared and started
PAGE_LOAD_S = 300.0
PAGE_READ_S = 15.0
#: the MCP's link names, and the page's (cluster/launch.LINK_NAMES)
LINKS = ("tcp", "rdma")
SPLITS = ("tensor", "pipeline")


class PageDown(Exception):
    pass


def _page_addr() -> str:
    import os
    return os.environ.get(PAGE_ENV) or "127.0.0.1:8899"


def _page_call(path: str, doc=None, timeout: float = PAGE_READ_S):
    import urllib.error
    import urllib.request
    url = f"http://{_page_addr()}{path}"
    req = urllib.request.Request(
        url, data=None if doc is None else json.dumps(doc).encode(),
        method="GET" if doc is None else "POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        raw = e.read()
    except (OSError, ValueError, http.client.HTTPException) as e:
        raise PageDown(f"the page on this Mac ({url.split(path)[0]}) did "
                       f"not answer: {type(e).__name__}: {e}. Across "
                       f"machines goes through it: start `knurlogic ui` "
                       f"(set {PAGE_ENV}=host:port if it is not on 8899)") from e
    try:
        out = json.loads(raw)
    except ValueError:
        raise PageDown(f"the page at {url} answered with something other "
                       f"than JSON: {raw[:200]!r}") from None
    if not isinstance(out, dict):
        raise PageDown(f"the page at {url} answered {str(out)[:200]}")
    return out


def _page_get(path: str) -> dict:
    return _page_call(path)


def _page_post(doc: dict, timeout: float = PAGE_LOAD_S) -> dict:
    return _page_call("/loaded.json", doc, timeout)


def _me_name() -> str:
    from knurlogic.machine import identity
    return identity.identity().get("name") or "this Mac"


def _machines_of(page, here: str) -> list:
    """This Mac and each peer its page asked, with whether it answered."""
    out = [{"machine": here, "here": True}]
    for p in (page or {}).get("peers") or []:
        if isinstance(p, dict):
            out.append({"machine": p.get("machine"), "here": False,
                        **({"error": p["error"]} if p.get("error") else {})})
    return out


def _link_name(link):
    from knurlogic.cluster.launch import link_name
    return link_name(link)


def models_across(page: dict, here: str) -> list:
    """Every resident model in a page's /loaded.json?peers=1, one entry per
    model: a cluster job ONCE (rank 0's row, which carries its `requests`;
    or the job itself while rank 0 has no row yet), never one per rank."""
    rows: list = [(here, r) for r in page.get("resident") or []
            if isinstance(r, dict)]
    jobs: list = [(here, j) for j in page.get("jobs") or []
                  if isinstance(j, dict)]
    down: list = [(here, d) for d in page.get("recovery") or []
            if isinstance(d, dict)]
    for p in page.get("peers") or []:
        if not isinstance(p, dict):
            continue
        rows += [(p.get("machine"), r) for r in p.get("resident") or []
                 if isinstance(r, dict)]
        jobs += [(p.get("machine"), j) for j in p.get("jobs") or []
                 if isinstance(j, dict)]
        down += [(p.get("machine"), d) for d in p.get("recovery") or []
                 if isinstance(d, dict)]
    # a job's recovery report: the coordinator's copy is the live one (a
    # rank 0 page only has what the relaunch told it)
    recs: dict = {}
    for _, j in jobs:
        v = j.get("recovery")
        if j.get("job") and isinstance(v, dict):
            recs[j["job"]] = _newer(recs.get(j["job"]), v)
    live = {}
    for machine, j in jobs:
        if j.get("job") and j.get("phase") != "stopped":
            # the leader's copy knows the port; any copy knows the rest
            if j["job"] not in live or j.get("port"):
                live[j["job"]] = dict(j, _on=machine)
    out, seen = [], set()
    for machine, r in rows:
        c: Any = (r.get("cluster") if isinstance(r.get("cluster"), dict)
                  else {})
        job = str(c.get("job") or "")
        # a cluster job's instance is its job id; a single-Mac server's is
        # the 16-hex id its page gave it at launch (loaded.py _instance_of)
        instance = str(r.get("instance") or job or "")
        if instance and instance in seen:
            continue
        j = live.get(job, {})
        if instance:
            seen.add(instance)
        out.append({
            "name": r.get("name"), "runtime": r.get("runtime"),
            "machine": machine, "where": r.get("where"),
            "port": _port_of(r.get("where")),
            "state": r.get("state"),
            "requests": r.get("requests"),
            "machines": list(c.get("machines") or j.get("machines")
                             or [machine]),
            "split": c.get("split") or j.get("split"),
            "link": _link_name(c.get("link") or j.get("link")),
            "job": job or None,
            **({"url": c.get("url") or j.get("url")}
               if c.get("url") or j.get("url") else {}),
            "instance": instance or None,
            "leader": c.get("leader") or j.get("leader"),
            **({"phase": c.get("phase") or j.get("phase")} if job else {}),
            **({k: j[k] for k in ("cable", "cable_note") if j.get(k)}),
            "recovery": _newer(r.get("recovery") if isinstance(
                r.get("recovery"), dict) else None, recs.get(job)),
        })
    for job, j in live.items():
        if job in seen:
            continue
        # no rank 0 row yet (still loading) or its page did not answer
        out.append({
            "name": j.get("artifact"), "runtime": "knurlogic",
            "machine": j.get("leader") or j["_on"], "where": None,
            "port": j.get("port"), "state": None, "requests": None,
            "machines": list(j.get("machines") or []),
            "split": j.get("split"), "link": _link_name(j.get("link")),
            "job": job, "instance": job, "leader": j.get("leader"),
            **({"url": j["url"]} if j.get("url") else {}),
            "phase": j.get("phase"),
            **({k: j[k] for k in ("cable", "cable_note") if j.get(k)}),
            "recovery": recs.get(job)})
    # tracked by a page and not serving now: waiting to be relaunched, or
    # failed until someone loads it again
    for machine, d in down:
        if (d.get("job") and d["job"] in seen) or any(
                o["job"] and o["job"] == d.get("job") for o in out):
            continue
        ms = list(d.get("machines") or [machine])
        out.append({
            "name": d.get("name"), "runtime": "knurlogic",
            "machine": ms[0] if d.get("job") else machine, "where": None,
            "port": d.get("port"), "state": d.get("state"),
            "requests": None, "machines": ms, "split": d.get("split"),
            "link": _link_name(d.get("link")), "job": d.get("job"),
            "instance": d.get("job") or None,
            "leader": ms[0] if d.get("job") else None,
            "recovery": d.get("recovery")})
    return out


def _newer(a, b):
    """Of two recovery reports of one model, the later; at a tie the
    settled one (recovered / failed) over recovering."""
    if not a or not b:
        return a or b
    ka = (a.get("last_at") or 0, a.get("state") != "recovering")
    kb = (b.get("last_at") or 0, b.get("state") != "recovering")
    return a if ka >= kb else b


def _port_of(where):
    from urllib.parse import urlparse
    try:
        return urlparse(where or "").port
    except ValueError:
        return None


def _node_ids(names: list[str]) -> tuple:
    """([page node ids], refusal or None): machine names as this Mac's page
    knows them -- itself and the peers answering it. An id is accepted too."""
    st = _page_get("/status.json")
    me = st.get("me") or {}
    known = [(me.get("id"), me.get("name"), "answering")] + [
        (p.get("id"), p.get("name"), p.get("state"))
        for p in st.get("peers") or [] if isinstance(p, dict)]
    ids = []
    for n in names:
        hit = [k for k in known if n in (k[0], k[1])] or [
            k for k in known if str(k[1] or "").lower() == str(n).lower()]
        if not hit:
            return [], {"loaded": False,
                        "refused": f"{n!r} is not a machine this Mac's page "
                                   f"knows",
                        "machines": [k[1] for k in known
                                     if k[2] == "answering"]}
        if hit[0][2] != "answering":
            return [], {"loaded": False,
                        "refused": f"{hit[0][1]} is not answering "
                                   f"({hit[0][2]})"}
        ids.append(hit[0][0])
    return ids, None


def _refusal(out: dict) -> dict[str, Any] | None:
    """The page's refusal, as a refusal: an answer, never a crash."""
    why = out.get("refused") or out.get("error")
    if not why:
        return None
    return {"loaded": False, "refused": why,
            **{k: out[k] for k in ("note", "detail", "placement", "machine")
               if k in out}}


def models(fits_only: bool = False, **_) -> dict[str, Any]:
    """Every model on this machine, with what can actually run."""
    from knurlogic.engine.serve import thinking
    from knurlogic.engine.vision import registry as vision_registry
    from knurlogic.machine import discover, wired
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


def settings(artifact: str = "", tune: str = "default", **_) -> dict[str, Any]:
    """The knobs for this artifact, each with the measurement behind it.

    The `why` is the point. A knob without its provenance is one an agent
    changes for no reason, and these were expensive to establish.
    """
    from knurlogic.interfaces.page import documents
    from knurlogic.tuning.settings import preset_of
    try:
        tune = preset_of(tune)
    except ValueError as e:
        return {"error": str(e)}
    doc = documents._preview(artifact, tune)
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


def drafting(artifact: str = "", **_) -> dict[str, Any]:
    """Does this artifact have a multi-token-prediction head, and will it run?"""
    from knurlogic.engine import mtp
    from knurlogic.machine.artifact import Artifact
    a = Artifact.load(artifact)
    st = mtp.status(a)
    return {"artifact": a.path.name, "state": st.state,
            "head": (st.head.describe() if st.head else ""),
            "family": (st.head.family if st.head else ""),
            "explanation": st.render(),
            "note": "mlx-lm has no MTP path. knurlogic drafts with a "
                    "packed head by default, on "
                    "single requests and batches alike: load(draft=false) "
                    "or `serve --no-draft` turns it off. Drafting preserves "
                    "the output distribution, so off is for troubleshooting."}


def load(artifact: str = "", port: int = 0, tune: str = "default",
         sets: dict[str, str] | None = None, force: bool = False,
         draft: bool = True, machines: list[str] | None = None,
         split: str = "", link: str = "", cable: str = "",
         **_) -> dict[str, Any]:
    """Start a server for this artifact, after checking it can work.

    REFUSES rather than gambles: memory still moving or a model that does
    not fit is a refusal with the reason attached. `force` overrides the
    moving-memory check only -- it will not make a model fit.

    `machines` names where: empty is this Mac. Another Mac, or several,
    go through the page on this Mac -- its Launch, the same request.
    """
    from knurlogic.tuning.settings import preset_of
    try:
        tune = preset_of(tune)
    except ValueError as e:
        return {"loaded": False, "refused": str(e),
                "note": "the tune is default or lean; nothing was started"}
    names = [str(m) for m in (machines or []) if str(m)]
    if names:
        return _load_on(names, artifact, port, tune, sets, force, draft,
                        split, link, cable)
    from knurlogic.interfaces.loading import NotLoadable, resolve_name
    from knurlogic.interfaces.page import server as page_server

    # a model named, never a directory (interfaces/loading.py): the same
    # rule a switch on a running server follows
    try:
        artifact = resolve_name(artifact, None)
    except NotLoadable as e:
        return {"loaded": False, "refused": "not a known artifact",
                "note": str(e)}
    # serve's deterministic refusals (bad settings, a context past the
    # model's maximum, ...), asked before a process is started: a server
    # that prints REFUSING and exits is a reason nobody sees
    from knurlogic.interfaces.serve import launch_refusal
    from knurlogic.machine.artifact import Artifact
    try:
        why = launch_refusal(Artifact.load(artifact), dict(sets or {}),
                            tune)
    except (OSError, ValueError, AttributeError, KeyError) as e:
        why = f"could not read the artifact: {type(e).__name__}: {e}"
    if why:
        return {"loaded": False, "refused": why,
                "note": "the launch settings (Settings -> Models) or the "
                        "artifact; nothing was started"}
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
    if not port:
        # none asked for: the first free one from the serve default up
        from knurlogic.machine.servers import free_port
        port = free_port(page_server._SERVE_PORT["n"])
    out = page_server._spawn(artifact, int(port), tune, dict(sets or {}),
                    draft=bool(draft))
    out["fit"] = f
    out["ready"] = r
    return out


def _identity_of(artifact: str) -> tuple:
    """(identity, refusal): a model on other machines is named by identity
    (machine/artifact.identity), read from this Mac's copy; a 16-hex
    identity is taken as given, for a model this Mac does not hold."""
    import re

    from knurlogic.interfaces.loading import NotLoadable, resolve_name
    from knurlogic.machine.artifact import identity
    try:
        ident = identity(resolve_name(artifact, None))
    except NotLoadable as e:
        if re.fullmatch(r"[0-9a-f]{16}", str(artifact or "")):
            return artifact, None
        return "", {"loaded": False, "refused": "not a known artifact",
                    "note": f"{e}. A model this Mac does not hold is named "
                            f"by its identity (the page's /models.json on a "
                            f"Mac that has it)."}
    if not ident:
        return "", {"loaded": False, "refused": "no identity",
                    "note": f"{artifact} has no config.json"}
    return ident, None


def _artifact_name(artifact: str) -> str:
    """The directory name `load` was given the artifact by ("" for a bare
    identity): the name the other machines resolve it by."""
    import re
    a = str(artifact or "").rstrip("/")
    if re.fullmatch(r"[0-9a-f]{16}", a):
        return ""
    from pathlib import Path

    from knurlogic.interfaces.loading import NotLoadable, resolve_name
    try:
        # a path (or a pin's real path) -> the store's own name for it
        return Path(resolve_name(a, None)).name
    except NotLoadable:
        return Path(a).name


def _load_on(names, artifact, port, tune, sets, force, draft, split, link,
             cable) -> dict[str, Any]:
    """`load` on other machines: the page's Launch request, sent to the
    page on this Mac (page_server._load_fn), which forwards a one-peer load and
    coordinates a cluster (cluster/launch.launch)."""
    if len(set(names)) != len(names):
        return {"loaded": False, "refused": "a machine is named twice"}
    if len(names) >= 2:
        if split not in SPLITS:
            return {"loaded": False,
                    "refused": f"split is tensor | pipeline, not {split!r}"}
        if link not in LINKS:
            return {"loaded": False,
                    "refused": f"link is tcp | rdma, not {link!r}"}
        if not draft:
            return {"loaded": False,
                    "refused": "draft=false is a one-machine option; a "
                               "cluster job's ranks take no --no-draft"}
    ident, no = _identity_of(artifact)
    if no:
        return no
    try:
        ids, no = _node_ids(names)
        if no:
            return no
        from knurlogic.machine import identity
        if len(ids) == 1 and ids[0] == identity.identity().get("id"):
            # this Mac, named: the same as naming none
            return load(artifact=artifact, port=port, tune=tune, sets=sets,
                        force=force, draft=draft)
        # the name too: a machine holding two artifacts with one identity
        # loads the one called this, or refuses -- never picks
        req = {"action": "load", "identity": ident,
               "name": _artifact_name(artifact), "tune": tune,
               "sets": dict(sets or {})}
        if port:
            req["port"] = int(port)
        if len(ids) == 1:
            req.update(node=ids[0], force=bool(force))
        else:
            req.update(nodes=ids, split=split, link=link)
            if cable:
                req["cable"] = cable
        out = _page_post(req)
    except PageDown as e:
        return {"error": str(e)}
    no = _refusal(out)
    if no:
        return no
    if len(ids) == 1:
        return dict(out, machines=names)
    plan = dict(out.get("placement") or {})
    plan.update(cable=out.get("cable"), cable_note=out.get("cable_note"))
    return {"starting": out.get("starting"), "artifact": artifact,
            "job": out.get("job"), "port": out.get("port"),
            "url": out.get("url"),
            "leader": out.get("leader"), "machines": out.get("machines"),
            "split": split, "link": _link_name(out.get("link") or link),
            "placement": plan,
            **({"alerts": out["alerts"]} if out.get("alerts") else {}),
            "note": out.get("note", "") + " -- or `state`: the job is one "
                    "entry in `models`, with its phase."}


def unload(port: int | None = None, model: str = "", job: str = "",
           instance: str = "", machine: str = "", **_) -> dict[str, Any]:
    """Stop a model knurlogic started: by port on this Mac (as before), or
    by model name, job id or instance id on any machine this Mac's page
    sees. A cluster job stops on every machine -- the page's Unload, the
    same request. `instance` is the id `load` returned, or `state`'s
    `models[].instance` -- a single-Mac server's own 16-hex id, or a
    cluster job's id (its instance is its job id, so `instance` and `job`
    both find it)."""
    from knurlogic.interfaces.page import server as page_server
    if not (port or model or job or instance):
        return {"error": "name the port, the model, the job or the instance"}
    try:
        page = _page_get("/loaded.json?peers=1")
    except PageDown as e:
        if port and not (model or job or instance or machine):
            return page_server._stop(int(port))      # this Mac, without its page
        return {"error": str(e)}
    here = _me_name()
    rows = [r for r in models_across(page, here)
            if r.get("runtime") == "knurlogic"]
    if instance:
        hit = [r for r in rows if r.get("instance") == str(instance)]
    elif job:
        hit = [r for r in rows if r.get("job") == str(job)]
    else:
        hit = rows
        if model:
            hit = [r for r in hit if r.get("name") == model] or [
                r for r in hit if str(r.get("name") or "").lower().endswith(
                    str(model).lower())]
        if machine:
            hit = [r for r in hit if r.get("machine") == machine
                   or machine in (r.get("machines") or [])]
        elif port and not model:
            hit = [r for r in hit if r.get("machine") == here]
        if port:
            hit = [r for r in hit if r.get("port") == int(port)]
    if not hit:
        if port and not (model or job or instance or machine):
            return page_server._stop(int(port))      # e.g. a server still loading
        return {"error": "no knurlogic model matches that",
                "resident": [{k: r.get(k) for k in
                              ("name", "machine", "port", "job", "instance")}
                             for r in rows]}
    if len(hit) > 1:
        return {"error": "more than one model matches; name the job, the "
                         "instance, or the machine and port",
                "matches": [{k: r.get(k) for k in
                             ("name", "machine", "port", "job", "instance")}
                            for r in hit]}
    r = hit[0]
    try:
        if r.get("job") and any(j.get("job") == r["job"]
                                for j in page.get("jobs") or []):
            # a rank of it runs here: this page stops it everywhere
            out = _page_post({"action": "unload", "job": r["job"]}, 60)
        elif r.get("machine") == here:
            out = _page_post({"action": "unload",
                              "target": str(r.get("port"))}, 60)
        else:
            if not r.get("port"):
                return {"error": f"{r.get('name')} on {r.get('machine')} "
                                 f"has no port yet; stop it there"}
            ids, no = _node_ids([r["machine"]])
            if no:
                return {"error": no["refused"]}
            # the peer's rank 0 port: that page stops its job everywhere
            out = _page_post({"action": "unload", "node": ids[0],
                              "port": int(r["port"])}, 60)
    except PageDown as e:
        return {"error": str(e)}
    return dict(out, model=r.get("name"), machine=r.get("machine"),
                machines=r.get("machines"), job=r.get("job"),
                instance=r.get("instance"))


# --- the table --------------------------------------------------------------

def deps() -> dict[str, Any]:
    """Which build of each piece is installed, read off the fix itself
    rather than a version string."""
    from knurlogic.machine import deps as D
    return D.survey()


TOOLS: dict[str, dict[str, Any]] = {
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
        "description": "What is loaded, here and on every machine this "
                       "Mac's page (`knurlogic ui`) sees. `models`: one "
                       "entry per resident model -- name, machine, port, "
                       "machines, split (tensor | pipeline), link (tcp | "
                       "rdma), job, instance (a single-Mac server's own "
                       "16-hex id, or a cluster job's id), leader, phase, "
                       "and `requests` "
                       "(in_flight, pending, capacity, oldest_pending_s, "
                       "holding; null when its server does not report "
                       "them). A cluster job is ONE entry, on its leader, "
                       "never one per rank. `machines`: the peers asked, "
                       "with any that did not answer. The rest is this Mac "
                       "alone: every runtime (knurlogic, ollama, exo, any "
                       "OpenAI port), where the memory went, and "
                       "`started_here` -- each server `load` started with "
                       "its phase: serving, loading (elapsed time, last log "
                       "line), stalled (stop waiting, read the log), or "
                       "exited (exit code, log tail).",
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
                           "tune": S("default | lean")},
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
        "description": "Start a model. With no `machines`, a server on "
                       "this Mac, its settings resolved against the load "
                       "budget; refused if it will not fit (no override) "
                       "or another load is still moving memory (force "
                       "overrides). With `machines`, through this Mac's "
                       "page (`knurlogic ui` must run), exactly as its "
                       "Launch button: one other Mac is a load there, "
                       "checked by that Mac; two or more is a cluster job "
                       "-- placement, leader and cable are chosen, never "
                       "asked for. Returns job, port (rank 0 serves the "
                       "model there, on the leader), leader, machines and "
                       "placement {order, leader, layers, cable, "
                       "cable_note}; the job loads in the background -- "
                       "poll `state`. Refused, with the reason and nothing "
                       "started, when a share does not fit, a machine is "
                       "not answering, the model is not on a machine, rdma "
                       "has no Thunderbolt 5 cable with RDMA up, or a "
                       "machine is already loading (one load at a time). "
                       "Nothing is ever evicted to make room: unload first.",
        "schema": _schema({
            "artifact": S("the model's name (or, for machines that are "
                          "not this Mac, its 16-hex identity)"),
            "port": S("port to serve on (a cluster job: rank 0's port on "
                      "the leader); omitted: the first free port from "
                      "8080 up", "integer"),
            "tune": S("default | lean"),
            "sets": {"type": "object",
                     "description": "launch-only knob overrides, KEY: VALUE"},
            "force": {"type": "boolean",
                      "description": "load while another load is still "
                                     "moving memory (one machine only)"},
            "draft": {"type": "boolean",
                      "description": "use a packed drafting head "
                                     "(default true; one machine only)"},
            "machines": {"type": "array", "items": {"type": "string"},
                         "description": "machine names, as `state` lists "
                                        "them; empty: this Mac only"},
            "split": S("with two or more machines: tensor (every layer "
                       "split, same share each) | pipeline (layers in "
                       "runs, sized to each machine)"),
            "link": S("with two or more machines: tcp (the ring, any "
                      "link) | rdma (jaccl over Thunderbolt 5, every pair "
                      "cabled; beyond two machines experimental)"),
            "cable": S("optional, two machines: the Thunderbolt subnet to "
                       "use (e.g. 10.0.1); by default the fastest shared "
                       "one, moving to the next if link init fails; "
                       "ignored with three or more (each pair picks its own)"),
        }, ["artifact"]),
    },
    "deps": {
        "fn": deps,
        "description": "What this stack stands on: mlx, mlx-lm, mlx-vlm, "
                       "each marked stock or fork by what is installed, not "
                       "by version -- plus what each fork carries and why it "
                       "is or is not ported.",
        "schema": _schema({}),
    },
    "unload": {
        "fn": unload,
        "description": "Stop a model knurlogic started. `port` alone: the "
                       "server on that port of this Mac. `model`, `job` or "
                       "`instance`: on any machine this Mac's page sees "
                       "(`machine` and `port` narrow it when a name matches "
                       "twice). A cluster job stops on every machine it "
                       "runs on.",
        "schema": _schema({
            "port": S("port it serves on (this Mac, unless `machine`)",
                      "integer"),
            "model": S("its name, as `state` lists it"),
            "job": S("a cluster job's id, from `load` or `state`"),
            "instance": S("its instance id, from `load` or `state`'s "
                          "`models[].instance` -- a single-Mac server's own "
                          "16-hex id, or a cluster job's id"),
            "machine": S("the machine it runs on")}),
    },
}


def tool_list() -> list[dict[str, Any]]:
    return [{"name": n, "description": t["description"],
             "inputSchema": t["schema"]} for n, t in TOOLS.items()]


def _call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    t = TOOLS.get(name)
    if t is None:
        return {"error": f"unknown tool {name!r}",
                "available": sorted(TOOLS)}
    try:
        return t["fn"](**(args or {}))
    # a tool's failure is the tool call's error reply, never a dead server
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
        except ValueError:
            continue
        rid, method = req.get("id"), req.get("method")
        params = req.get("params") or {}
        if method == "initialize":
            result: dict = {"protocolVersion": "2024-11-05",
                      "capabilities": {"tools": {}},
                      "serverInfo": {"name": SERVER_NAME,
                                     "version": SERVER_VERSION},
                      "instructions": INSTRUCTIONS}
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
