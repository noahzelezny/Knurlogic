"""The exo cluster, read and placed on -- the one home for talking to exo.

exo places and shards well, and says very little about what it is doing
while it does it. The failure this module exists for: an agent places an
instance on a ring that has not settled, then waits for something to happen,
and nothing ever does. So there are three rules here.

  1. PLACE ONLY ON A SETTLED RING, and only what fits. `plan()` asks exo for
     its own placement (GET /instance/placement creates nothing) and checks
     every node's shard against THAT node's free memory, before anything is
     posted. exo checks too; this says which node and by how much.
  2. EVERY INSTANCE HAS A PHASE, read off exo's own evidence -- downloading
     (bytes), loading (layers loaded of total), warming, serving, failed
     (exo's error message), shutting down, gone.
  3. A PHASE THAT STOPS MOVING IS NAMED. exo will sit in "loading" forever
     if a rank never connects. Each observation is fingerprinted (runner
     states, layers loaded, bytes downloaded) and the time it last changed
     is kept on disk, so a poll can say "nothing has moved for 4 minutes:
     stop waiting" instead of letting an agent wait.

Stdlib only; reads exo over HTTP. It is a fact about the machines, so it
lives in machine/, and the MCP, the page and the CLI all come here.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from knurlogic.machine.loaded import _get, _post

#: Where exo answers. One home: the MCP, the page and `serve --cluster` read
#: it from here. KNURLOGIC_EXO_URL overrides.
EXO_URL = os.environ.get("KNURLOGIC_EXO_URL", "http://127.0.0.1:52415")

#: An instance whose evidence has not changed for this long is stalled.
#: Longer than a local load's: exo downloads over the network and loads
#: shards on several boxes, and a big rung can be slow without being stuck.
STALL_S = 240

GIB = 1 << 30


def state(base: str = EXO_URL) -> dict:
    return _get(f"{base}/state", timeout=4.0) or {}


def _body(tagged):
    """exo tags unions by kind -- {"MlxRingInstance": {...}}. Unwrap one."""
    if isinstance(tagged, dict) and len(tagged) == 1:
        (kind, body), = tagged.items()
        if isinstance(body, dict):
            return kind, body
    return "", tagged if isinstance(tagged, dict) else {}


def _bytes(v) -> int:
    return int((v or {}).get("inBytes") or 0) if isinstance(v, dict) else 0


def node_names(st: dict) -> dict:
    return {k: (v or {}).get("friendlyName") or k[:12]
            for k, v in (st.get("nodeIdentities") or {}).items()}


def model_id_for(model: str) -> str:
    """An exo model id from what an agent has: an id, or a path in exo's
    store. exo names a model's directory `org--name`; the id is `org/name`."""
    p = Path(model)
    if p.is_absolute() or os.sep in model and "--" in p.name:
        return p.name.replace("--", "/", 1)
    return model


# --- 1. the plan, checked before anything is posted -------------------------

def plan(model: str, sharding: str = "Pipeline", min_nodes: int = 1,
         base: str = EXO_URL) -> dict:
    """exo's own placement for this model, with every node's shard checked
    against that node's free memory. Creates nothing.

    Returns {model_id, instance, nodes: [{node, layers, bytes, available,
    fits}], fits} or {error}. exo refuses a model it will not place with a
    400; that detail is passed through as the error.
    """
    mid = model_id_for(model)
    q = urllib.parse.urlencode({"model_id": mid, "sharding": sharding,
                                "min_nodes": int(min_nodes)})
    try:
        with urllib.request.urlopen(f"{base}/instance/placement?{q}",
                                    timeout=30.0) as r:
            inst = json.loads(r.read().decode() or "null")
    except urllib.error.HTTPError as e:
        try:
            j = json.loads(e.read().decode())
            detail = ((j.get("error") or {}).get("message")
                      if isinstance(j.get("error"), dict) else None) \
                or j.get("detail") or str(j)
        except Exception:
            detail = str(e)
        out = {"model_id": mid, "refused_by_exo": detail,
               "error": f"exo will not place it: {detail}"}
        out.update(_memory_arithmetic(mid, base))
        return out
    except Exception as e:
        return {"model_id": mid, "error": f"exo did not answer: "
                                          f"{type(e).__name__}: {e}"}
    if not inst:
        return {"model_id": mid, "error": "exo returned no placement"}

    st = state(base)
    names = node_names(st)
    mem = st.get("nodeMemory") or {}
    _, body = _body(inst)
    sa = body.get("shardAssignments") or {}
    runner_node = {r: n for n, r in (sa.get("nodeToRunner") or {}).items()}
    nodes = []
    for runner, shard in (sa.get("runnerToShard") or {}).items():
        _, meta = _body(shard)
        card = meta.get("modelCard") or {}
        n_layers = int(card.get("nLayers") or 0)
        lo, hi = meta.get("startLayer"), meta.get("endLayer")
        size = _bytes(card.get("storageSize"))
        share = ((hi - lo) / n_layers) if n_layers and lo is not None \
            and hi is not None else 1.0
        nid = runner_node.get(runner, "")
        need = int(size * share)
        avail = _bytes((mem.get(nid) or {}).get("ramAvailable"))
        nodes.append({"node": names.get(nid, nid[:12]), "node_id": nid,
                      "layers": f"{lo}-{hi} of {n_layers}",
                      "gib": round(need / GIB, 1),
                      "available_gib": round(avail / GIB, 1),
                      "headroom_gib": round((avail - need) / GIB, 1),
                      "fits": bool(avail) and need < avail})
    return {"model_id": mid, "instance": inst, "sharding": sharding,
            "nodes": nodes, "fits": bool(nodes) and all(n["fits"]
                                                        for n in nodes)}


def _memory_arithmetic(mid: str, base: str) -> dict:
    """What the model needs against what every node has free, for when exo
    refuses with a sentence that says neither -- 'No cycles found with
    sufficient memory' is true and leaves an agent nothing to act on."""
    st = state(base)
    names = node_names(st)
    size = 0
    try:
        with urllib.request.urlopen(f"{base}/models", timeout=10) as r:
            for m in (json.loads(r.read().decode()).get("data") or []):
                if m.get("id") == mid:
                    size = int(m.get("storage_size_megabytes") or 0) << 20 \
                        or _bytes(m.get("storageSize"))
    except Exception:
        pass
    free = {names.get(n, n[:12]): round(_bytes((v or {}).get("ramAvailable"))
                                        / GIB, 1)
            for n, v in (st.get("nodeMemory") or {}).items()}
    out = {"available_gib_by_node": free,
           "available_gib_total": round(sum(free.values()), 1)}
    if size:
        out["model_gib"] = round(size / GIB, 1)
        out["short_by_gib"] = round(max(size / GIB - sum(free.values()), 0), 1)
    return out


def place(plan_doc: dict, base: str = EXO_URL) -> dict:
    """Post exo's own placement object back verbatim, and start watching it."""
    _, body = _body(plan_doc["instance"])
    iid = body.get("instanceId", "")
    out = _post(f"{base}/instance", {"instance": plan_doc["instance"]})
    _watch_note(iid, fingerprint=None)       # the clock starts now
    w = _watch_load()
    w.setdefault(iid, {})["placed_at"] = time.time()
    w[iid]["model"] = plan_doc.get("model_id", "")
    _watch_path().write_text(json.dumps(w))
    return {"instance_id": iid, "exo_said": out}


#: A runner no instance references, whose record has not changed for this
#: long, is a stale record rather than memory in motion.
GHOST_S = 30

_TRANSIT = ("RunnerIdle", "RunnerConnecting", "RunnerConnected",
            "RunnerLoading", "RunnerLoaded", "RunnerWarmingUp",
            "RunnerShuttingDown")


def runner_activity(st: dict) -> tuple:
    """(runners genuinely in transit by kind, stale records) from exo's state.

    Measured live: after two instances were removed, their runners sat in
    RunnerShuttingDown indefinitely, referenced by no instance, while the
    node's memory was back to exactly what it was before they were placed.
    exo never marks them shut down. A readiness check that counted them
    would itself wait forever -- the failure it exists to end. So a runner
    in transit blocks when an instance references it, or while its record
    is still changing; an orphan that has not changed for GHOST_S is
    reported by name and does not block.
    """
    used = set()
    for tagged in (st.get("instances") or {}).values():
        _, body = _body(tagged)
        used |= set((body.get("shardAssignments") or {})
                    .get("runnerToShard") or {})
    transit, ghosts = {}, []
    for rid, tagged in (st.get("runners") or {}).items():
        kind, _ = _body(tagged)
        if kind not in _TRANSIT:
            continue
        if rid in used:
            transit[kind] = transit.get(kind, 0) + 1
            continue
        quiet = _watch_note(f"runner:{rid}", fingerprint=kind)
        if quiet < GHOST_S:
            transit[kind] = transit.get(kind, 0) + 1
        else:
            ghosts.append({"runner": rid[:8], "state": kind,
                           "unchanged_seconds": round(quiet)})
    return transit, ghosts


#: A placement exo has not published within this long is itself an answer:
#: it was dropped. Until then it counts as memory about to move.
UNPUBLISHED_S = 60


def moving(base: str = EXO_URL, st: dict | None = None) -> list:
    """Instances whose memory is still in motion, each named -- the check
    that has to pass before anything else is placed or loaded.

    Measured live: a second placement posted one call after the first was
    accepted, because exo had not yet PUBLISHED the first -- no instance, no
    runners in /state, memory still reading free. A placement knurlogic has
    posted counts as moving from the moment it is posted, not from the
    moment exo gets round to saying so.
    """
    st = st if st is not None else state(base)
    rows = phases(base, st) if st else []
    listed = {r["instance_id"] for r in rows}
    out = [{"what": "an exo instance is still coming up",
            "detail": {"model": r["model"], "phase": r["phase"],
                       "runners": r["runners"],
                       **({"downloads": r["downloads"]} if r.get("downloads")
                          else {})},
            "why": "memory on its nodes is about to change"}
           for r in rows if r["phase"] not in ("serving", "failed")]
    now = time.time()
    for iid, rec in _watch_load().items():
        at = rec.get("placed_at")
        if not at or iid in listed:
            continue
        age = now - at
        if age < UNPUBLISHED_S:
            out.append({"what": "a placement exo has not published yet",
                        "detail": {"instance_id": iid,
                                   "model": rec.get("model", ""),
                                   "seconds_since_posted": round(age)},
                        "why": "posted, not yet in exo's state: its memory "
                               "is not counted anywhere yet"})
    return out


def unplace(instance_id: str, base: str = EXO_URL) -> dict:
    out = _post(f"{base}/instance/{instance_id}", method="DELETE")
    _watch_forget(instance_id)
    return {"removed": instance_id, "exo_said": out}


# --- 2. what phase every instance is in -------------------------------------

_ORDER = ["failed", "downloading", "loading", "warming", "shutting down",
          "serving"]


def phases(base: str = EXO_URL, st: dict | None = None) -> list:
    """Every exo instance, with its phase and the evidence for it."""
    st = st if st is not None else state(base)
    if not st:
        return []
    names = node_names(st)
    runners = st.get("runners") or {}
    downloads = st.get("downloads") or {}
    out = []
    for iid, tagged in (st.get("instances") or {}).items():
        _, body = _body(tagged)
        sa = body.get("shardAssignments") or {}
        mid = sa.get("modelId", "")
        node_of = {r: n for n, r in (sa.get("nodeToRunner") or {}).items()}
        per = []
        for rid in (sa.get("runnerToShard") or {}):
            kind, rb = _body(runners.get(rid) or {})
            per.append({"node": names.get(node_of.get(rid, ""), "?"),
                        "runner": kind or "not reported yet", **_evidence(kind, rb)})
        dl = _downloading(downloads, mid, set(node_of.values()), names)
        phase = _phase([p["runner"] for p in per], dl)
        row = {"instance_id": iid, "model": mid, "phase": phase,
               "runners": per}
        if dl:
            row["downloads"] = dl
        fp = json.dumps([per, dl], sort_keys=True)
        quiet = _watch_note(iid, fingerprint=fp)
        row["unchanged_seconds"] = round(quiet)
        if phase not in ("serving", "failed") and quiet > STALL_S:
            row["phase"] = "stalled"
            row["stalled_in"] = phase
            row["advice"] = (
                f"nothing has changed for {round(quiet)}s while {phase}. Stop "
                f"waiting: a rank that never connects leaves exo in this "
                f"state indefinitely. unplace(instance_id='{iid}') and check "
                f"`ready` before placing again.")
        if phase == "failed":
            row["advice"] = "exo reported the failure above; unplace it."
        out.append(row)
    return out


def _evidence(kind: str, rb: dict) -> dict:
    if kind == "RunnerLoading":
        done, total = rb.get("layersLoaded", rb.get("layers_loaded")), \
            rb.get("totalLayers", rb.get("total_layers"))
        return {"layers_loaded": done, "total_layers": total}
    if kind == "RunnerFailed":
        return {"error": rb.get("errorMessage") or rb.get("error_message")}
    return {}


def _downloading(downloads: dict, mid: str, node_ids: set, names: dict) -> list:
    rows = []
    for node, items in downloads.items():
        if node_ids and node not in node_ids:
            continue
        for it in (items if isinstance(items, list) else [items]):
            kind, body = _body(it)
            if kind != "DownloadOngoing":
                continue
            _, meta = _body(body.get("shardMetadata") or {})
            if (meta.get("modelCard") or {}).get("modelId") != mid:
                continue
            prog = body.get("downloadProgress") or {}
            done, total = _bytes(prog.get("downloadedBytes")), \
                _bytes(prog.get("totalBytes"))
            rows.append({"node": names.get(node, node[:12]),
                         "gib_done": round(done / GIB, 1),
                         "gib_total": round(total / GIB, 1),
                         "progress": f"{done / total:.0%}" if total else "?"})
    return rows


def _phase(runner_kinds: list, downloading: list) -> str:
    if any(k == "RunnerFailed" for k in runner_kinds):
        return "failed"
    if downloading:
        return "downloading"
    if any(k in ("RunnerShuttingDown", "RunnerShutdown") for k in runner_kinds):
        return "shutting down"
    if runner_kinds and all(k in ("RunnerReady", "RunnerRunning")
                            for k in runner_kinds):
        return "serving"
    if any(k in ("RunnerLoaded", "RunnerWarmingUp") for k in runner_kinds) \
            and not any(k in ("RunnerIdle", "RunnerConnecting",
                              "RunnerConnected", "RunnerLoading",
                              "not reported yet") for k in runner_kinds):
        return "warming"
    return "loading"


def phase_of(instance_id: str, base: str = EXO_URL) -> dict:
    """One instance's phase, or `gone` when exo no longer lists it -- which
    is its own answer: removed, or it crashed out of the state."""
    for row in phases(base):
        if row["instance_id"] == instance_id:
            return row
    return {"instance_id": instance_id, "phase": "gone",
            "advice": "exo no longer lists this instance: it was removed, or "
                      "it failed out of the state. Check `ready` and place "
                      "again if it is still wanted."}


# --- 3. how long since anything moved ---------------------------------------

def _watch_path() -> Path:
    root = Path(os.environ.get("XDG_CACHE_HOME",
                               Path.home() / ".cache")) / "knurlogic"
    root.mkdir(parents=True, exist_ok=True)
    return root / "exo_watch.json"


def _watch_load() -> dict:
    try:
        return json.loads(_watch_path().read_text())
    except Exception:
        return {}


def _watch_note(iid: str, fingerprint) -> float:
    """Record what an instance looks like now; return seconds since it last
    looked different. On disk, because the question spans MCP sessions."""
    now = time.time()
    w = _watch_load()
    rec = w.get(iid)
    if rec is None or (fingerprint is not None and rec.get("fp") != fingerprint):
        w[iid] = {"fp": fingerprint, "since": now}
        quiet = 0.0
    else:
        quiet = now - float(rec.get("since", now))
    _watch_path().write_text(json.dumps(w))
    return quiet


def _watch_forget(iid: str) -> None:
    w = _watch_load()
    if w.pop(iid, None) is not None:
        _watch_path().write_text(json.dumps(w))
