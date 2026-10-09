"""The MCP's read-only tools: what this Mac can do and what it holds.

ready (is it safe to load now), fit (will this artifact fit, with the
arithmetic), state (what is loaded, here and on the peers the page sees),
models, model_folders, settings, drafting and deps. Nothing here starts or
stops a server; that is lifecycle.py.
"""

from __future__ import annotations

from typing import Any

from knurlogic.interfaces.mcp import page_client

GIB = 1 << 30


def ready(**_) -> dict[str, Any]:
    """Is this machine in a state where loading will work?

    Every reason it is not, named: a load still reading weights moves memory,
    so a fit measured now would be stale.
    """
    from knurlogic.interfaces import spawn
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
                for c in spawn.loading()]
    # The load lock (P0/`machine/loadlock.py`) is a second source of the same
    # blocker: a load started by a DIFFERENT process (another agent, `serve`
    # run by hand) holds it and would not otherwise show up in `spawn.loading()`,

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


def _named(artifact: str):
    """(path, None) for a model name or path `load` would accept, else
    (None, the refusal): fit, settings and drafting resolve a name the way
    load does, instead of raising on a bare name. A path stays a path
    (these tools read one; load does not)."""
    from knurlogic.interfaces.load_checks import NotLoadable, resolve_name
    try:
        return resolve_name(artifact, None), None
    except NotLoadable as e:
        if "/" in artifact:
            return artifact, None
        return None, {"refused": "not a known artifact", "note": str(e)}


def fit(artifact: str = "", draft: bool = True, vision: bool = True,
        **_) -> dict[str, Any]:
    """Will this artifact fit, with the arithmetic shown.

    Against `wired.load_budget()` -- the same number `settings` resolves
    against and `load` starts a server with, so the three cannot disagree.
    """
    from knurlogic.engine.vision import registry as vision_registry
    from knurlogic.machine.artifact import Artifact
    from knurlogic.machine.memory import wired
    from knurlogic.machine.memory.footprint import available_memory
    from knurlogic.tuning import measured

    artifact, refused = _named(artifact)
    if refused:
        return refused
    a = Artifact.load(artifact)
    mem = available_memory()
    b = wired.load_budget()
    budget = b["bytes"]
    from knurlogic.tuning.fit import room_for, vision_budget
    adv = wired.advise(a.bytes_on_disk)
    # A vision rung also holds its tower, its image store and its images'
    # KV -- the resolver's terms, so `fit` and `settings` agree.
    vb = vision_budget(a)
    extra = vb["extra_bytes"] if vb else 0
    from knurlogic.tuning.fit import mtp_head_bytes, vision_freed_bytes
    # what the load will really hold: MTP off takes the head out, vision
    # off the tower, image store and image KV (single_fit_check's terms)
    holds = (a.bytes_on_disk + extra
             - (0 if draft else mtp_head_bytes(a))
             - (0 if vision else vision_freed_bytes(a)))
    headroom = budget - holds
    from knurlogic.tuning.checks import launch_fit
    chk = launch_fit(a, {} if vision else {"KNURLOGIC_VISION": "off"},
                     "default", draft, budget)
    fits = bool(budget) and headroom > 0 and chk["state"] != "cannot"
    low_headroom = fits and headroom < measured.low_headroom_bytes(budget)
    verdict = ("will not fit" if not fits else
               "fits, low headroom" if low_headroom else "fits")
    return {
        "artifact": a.path.name,
        "verdict": verdict,
        "fits": fits,
        # a family for this model_type and a tower in this config.json
        "vision_capable": vision_registry.registered(a.model_type, a.path),
        # why a vision family's conversion has no vision ("" otherwise)
        "vision_unavailable": vision_registry.unavailable_why(
            a.model_type, a.path),
        "vision_budget": _vision_terms(vb) if vision else None,
        "vision": bool(vision),
        "size_gib": round(a.gib, 1),
        "budget_gib": round(budget / GIB, 1),
        "headroom_gib": round(headroom / GIB, 1),
        "limited_by": b["limited_by"],
        "what_low_headroom_means": (
            f"under {measured.low_headroom_bytes(budget) / GIB:.0f} GiB left "
            f"after the weights, "
            f"so a load now narrows the prompt chunk to "
            f"{measured.PREFILL_CHUNK_LOW_HEADROOM} tokens and prefills one "
            f"prompt at a time. It loads; long prompts are slower to start."
            if low_headroom else ""),
        # what a fit leaves to talk in (tuning/fit.context_room)
        "room": room_for(holds, a.raw_config),
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
    """The resolver's vision terms (tuning.fit.vision_budget) in GiB,
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
    from knurlogic.interfaces import spawn
    spec = served_vision()
    # across machines: what the page on this Mac sees (its own residency and
    # each answering peer's). Without the page, this Mac's survey alone.
    try:
        page = page_client.page_get("/loaded.json?peers=1")
    except page_client.PageDown as e:
        page, page_note = None, str(e)
    else:
        page_note = ""
    here = page_client.me_name()
    everywhere = page_client.models_across(
        page if page is not None else {"resident": doc.get("resident", [])},
        here)
    return {
        # one entry per model: a cluster job once, on its leader, with its
        # machines, split, link, job and rank 0's `requests`
        "models": everywhere,
        "machines": page_client.machines_of(page, here),
        **({"page": page_note} if page_note else {}),
        "resident": doc.get("resident", []),
        "runtimes": doc.get("runtimes", []),
        "started_here": spawn.children(),
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


def models(fits_only: bool = False, **_) -> dict[str, Any]:
    """Every model on this machine, with what can actually run."""
    from knurlogic.engine.model import thinking
    from knurlogic.engine.vision import registry as vision_registry
    from knurlogic.machine import discover
    from knurlogic.machine.memory import wired
    avail = wired.load_budget()["bytes"]
    out = []
    for f in discover.find():
        row = {"name": f.name, "path": str(f.path), "store": f.store,
               "size_gib": round(f.bytes_on_disk / GIB, 1),
               "model_type": f.model_type, "is_vq": f.is_vq,
               "servable": f.servable, "why_not": f.why,
               "fits": bool(avail) and f.bytes_on_disk <= avail,
               "drafting_head": bool(f.extra.get("mtp_head")),
               "vision_capable": vision_registry.registered(f.model_type, f.path),
               "vision_unavailable": vision_registry.unavailable_why(
                   f.model_type, f.path),
               # what reasoning_effort does on this model: its template's
               # dialect, native levels and default (engine/model/thinking)
               "thinking": thinking.levels(thinking.template_of(f.path))}
        if fits_only and not (row["fits"] and row["servable"]):
            continue
        out.append(row)
    out.sort(key=lambda r: -r["size_gib"])
    return {"models": out, "budget_gib": round(avail / GIB, 1),
            "fits_means": "fits the load budget -- see `fit` for headroom "
                          "and whether its headroom is low",
            "count": len(out)}


def model_folders(add: str = "", remove: str = "", **_) -> dict[str, Any]:
    """The model folders this Mac remembers; `add` or `remove` one."""
    from pathlib import Path

    from knurlogic.machine import folders
    out: dict[str, Any] = {}
    if add:
        out = folders.add(add)
    elif remove:
        out = folders.remove(remove)
    rows = [{"path": f, "mounted": Path(f).is_dir()} for f in folders.saved()]
    return {**({"error": out["error"]} if "error" in out else {}),
            "folders": rows}


def settings(artifact: str = "", tune: str = "default", **_) -> dict[str, Any]:
    """The knobs for this artifact, each with the measurement behind it.

    The `why` is the point. A knob without its provenance is one an agent
    changes for no reason, and these were expensive to establish.
    """
    from knurlogic.interfaces.page import documents
    from knurlogic.tuning.presets import preset_of
    artifact, refused = _named(artifact)
    if refused:
        return refused
    try:
        tune = preset_of(tune)
    except ValueError as e:
        return {"error": str(e)}
    doc = documents.preview(artifact, tune)
    from knurlogic.machine.artifact import Artifact
    from knurlogic.tuning.fit import vision_budget
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
    artifact, refused = _named(artifact)
    if refused:
        return refused
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
                    "the output distribution (not bit-identical to plain "
                    "decoding: a near-tie can go either way), so off is for "
                    "troubleshooting."}


def deps() -> dict[str, Any]:
    """Which build of each piece is installed, read off the fix itself
    rather than a version string."""
    from knurlogic.machine import deps as D
    return D.survey()
