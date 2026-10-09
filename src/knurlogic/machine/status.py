"""What is loaded, and what it is actually using.

A runtime appears as one opaque number in Activity Monitor; this separates
weights, reclaimable cache and peaks. `ps` RSS and the framework's own
accounting diverge under pressure, so both are shown; cache memory is
reclaimable and never counted as usage. `snapshot()` answers for this
process; `aggregate()` is the cluster shape `/status.json` always serves
(single-node keys stay at the top level). A rollup sums memory and reports
how many nodes answered.

Design: docs/design/memory.md (status).
"""

from __future__ import annotations

import logging
import os
import subprocess
import time

from knurlogic.machine import metrics as _metrics

GIB = 1 << 30
_STARTED = time.time()


def _rss_bytes() -> int:
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                             capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip()) * 1024
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0


def memory() -> dict:
    """Delegated: how memory is accounted is an ENGINE question, and the
    machine/ never imports mlx (a tripwire test checks)."""
    from knurlogic.engine.serve import memory as _m

    d = _m()
    d["process_rss_bytes"] = _rss_bytes()
    d["scope"] = "process"      # see `scope` in aggregate/render_cluster
    return d


#: Bumped when the wire shape of /status.json changes incompatibly.
SCHEMA = 2


def snapshot(artifact=None, arch_rows=None, env=None, requests=0,
             node="local", role="server", reachable=True,
             memory_fn=None, machine_fn=None, memory_map=None,
             metrics=None) -> dict:
    """One node's answer. `memory_fn` exists so a snapshot can be BUILT from
    numbers that came off another node rather than only from this process.

    `machine` says what the box IS -- a desktop, a mini, a laptop. It is
    reported per node for the same reason memory is: in a cluster the
    interesting fact is that these two are DIFFERENT machines, and a page
    that draws them as identical boxes throws that away."""
    d = {
        "node": node,
        "role": role,
        "reachable": reachable,
        "uptime_seconds": round(time.time() - _STARTED, 1),
        "requests_served": requests,
        "memory": (memory_fn or memory)(),
    }
    if machine_fn is not None:
        d["machine"] = machine_fn()
    else:
        from knurlogic.machine import identity
        from knurlogic.machine.memory import wired
        d["machine"] = wired.machine()
        # Only a node answering for ITSELF has an id: it is what peers
        # deduplicate on, and a guessed one would merge two machines.
        d["id"] = identity.identity()["id"]
    # Which runtime is holding what, ON THIS NODE. It travels inside the
    # node's own snapshot rather than being computed centrally, because
    # process footprints are only true of the machine they were read on --
    # and knurlogic runs on every node, so every node can answer for itself.
    # A node that cannot say leaves the key out; nothing downstream fills it
    # in from somewhere else.
    if memory_map:
        d["memory_map"] = memory_map
    # How hard the box is working (GPU, CPU, pressure, swap, thermal) is read
    # the same way memory is: here for this box, off the node for a peer.
    # A peer that did not say leaves the key out, and the page draws no line.
    if metrics is None and machine_fn is None:
        try:
            metrics = _metrics.metrics(memory_map)
        except Exception:  # the status document must still answer without metrics
            logging.getLogger(__name__).debug("metrics failed", exc_info=True)
            metrics = None
    if metrics:
        d["metrics"] = metrics
    if artifact is not None:
        d["artifact"] = {
            "name": artifact.path.name,
            "path": str(artifact.path),
            "model_type": artifact.model_type,
            "size_bytes": artifact.bytes_on_disk,
            "is_vq": artifact.is_vq,
            "bundled_runtime": artifact.model_file,
            "geometry": {f"d{d_}-K{k}": n
                         for (d_, k), n in sorted(artifact.geometries.items())},
        }
    if arch_rows is not None:
        d["architectures"] = [
            {"module": r.module, "origin": r.origin, "state": r.state,
             "path": str(r.path) if r.path else None}
            for r in arch_rows]
    if env is not None:
        d["settings"] = dict(sorted(env.items()))
    return d


_SUMMED = ("active_bytes", "cache_bytes", "peak_bytes", "working_set_bytes",
           "total_bytes", "headroom_bytes", "process_rss_bytes")


def _num(v) -> int:
    """A summed figure from a node's snapshot: a number, or 0 -- a remote
    node's snapshot is its own word, not a guarantee of shape."""
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def aggregate(snapshots, artifact=None) -> dict:
    """The cluster shape. One node in, one node out -- with the rollup.

    The single node's own keys are also spread at the top level so that
    `/status.json` keeps the contract it already had; `nodes` is where a
    second node arrives, and `cluster` is the only place a total lives.
    """
    snaps = list(snapshots)
    up = [s for s in snaps if s.get("reachable", True)]
    mem: dict = {k: sum(_num(s.get("memory", {}).get(k)) for s in up)
           for k in _SUMMED}
    mem["available"] = any(s.get("memory", {}).get("available") for s in up)
    # A rollup is only as precise as its least precise node: one box-wide
    # number in the sum makes the sum box-wide, and the label has to follow
    # or the total quietly claims to be weights.
    mem["scope"] = ("box" if any(s.get("memory", {}).get("scope") == "box"
                                 for s in up) else "process")
    cluster = {
        "nodes_total": len(snaps),
        "nodes_reachable": len(up),
        "requests_served": sum(s.get("requests_served", 0) for s in snaps),
        "uptime_seconds": max([s.get("uptime_seconds", 0) for s in snaps]
                              or [0]),
        "memory": mem,
    }
    d = {"schema": SCHEMA, "cluster": cluster, "nodes": snaps}
    if len(snaps) == 1:
        # The one-box contract, unchanged: artifact/memory/settings at the top.
        d = {**snaps[0], **d}
    else:
        # Several nodes still answer the top-level keys, because the page and
        # every other client already read them -- with the CLUSTER's memory,
        # which is the only number that means anything across boxes.
        art = next((s["artifact"] for s in snaps if s.get("artifact")), None)
        d = {**{k: v for k, v in (("artifact", art),) if v},
             "memory": mem, "uptime_seconds": cluster["uptime_seconds"],
             "requests_served": cluster["requests_served"], **d}
    return d


def render_cluster(d: dict) -> str:
    """A cluster reads as its nodes plus one total line. A per-node render
    that hid the total would make you add GiB in your head; a total that hid
    the nodes would hide the node that did not answer."""
    c = d.get("cluster", {})
    if c.get("nodes_total", 1) <= 1:
        return render(d)
    L = []
    a = (d.get("nodes") or [{}])[0].get("artifact")
    if a:
        L.append(f"artifact   {a['name']}  ({a['model_type']}, "
                 f"{a['size_bytes'] / GIB:.1f} GiB on disk)")
        L.append("")
    for s in d.get("nodes", []):
        m = s.get("memory", {})
        if not s.get("reachable", True) or not m.get("available"):
            L.append(f"{s.get('node','?'):<12s} [{s.get('role','?')}]  "
                     f"NO ANSWER -- its memory is not in the total below")
            continue
        L.append(f"{s.get('node','?'):<12s} [{s.get('role','?')}]  "
                 f"{_num(m.get('active_bytes')) / GIB:6.1f} GiB live  "
                 f"{_num(m.get('cache_bytes')) / GIB:5.1f} cache  "
                 f"{_num(m.get('headroom_bytes')) / GIB:6.1f} free of "
                 f"{_num(m.get('working_set_bytes')) / GIB:.0f}")
    m = c["memory"]
    # Where the numbers came from changes what they MEAN. A per-process
    # snapshot separates weights from reclaimable cache; a number reported
    # for the whole box cannot, and saying "weights" over it would be a
    # smaller lie than it looks -- it is what makes a runtime get blamed for
    # everything else running on the machine.
    # `scope` says whether a node's numbers came from the runtime process or
    # from the whole machine. It is a field rather than a guess at the device
    # string, because a label that drifts is worse than no label.
    # aggregate() decided it from the nodes that answered, whose numbers
    # the total holds; a silent node's stale scope must not relabel them
    boxwide = m.get("scope") == "box"
    L.append("")
    L.append(f"cluster      {c['nodes_reachable']}/{c['nodes_total']} nodes "
             f"answering")
    L.append(f"  {'in use on the box' if boxwide else 'weights + live':<16s} "
             f"{m['active_bytes'] / GIB:8.2f} GiB"
             + ("   (everything on that machine, not just this runtime -- "
                "the box is reported)" if boxwide else ""))
    L.append(f"  cache            {m['cache_bytes'] / GIB:8.2f} GiB   "
             f"(RECLAIMABLE -- not usage)")
    L.append(f"  headroom         {m['headroom_bytes'] / GIB:8.2f} GiB   "
             f"of {m['working_set_bytes'] / GIB:.1f} GiB usable, summed over "
             f"the nodes that answered")
    L.append("")
    L.append(f"{c['requests_served']} requests served")
    return "\n".join(L)


def render(d: dict) -> str:
    """Plain text, because this is read in a terminal as often as a browser."""
    L = []
    a = d.get("artifact")
    if a:
        L.append(f"artifact   {a['name']}")
        L.append(f"           {a['model_type']}, "
                 f"{a['size_bytes'] / GIB:.1f} GiB on disk"
                 + (f", runtime {a['bundled_runtime']}"
                    if a["bundled_runtime"] else ""))
        if a["geometry"]:
            L.append("           " + ", ".join(
                f"{g} x{n}" for g, n in a["geometry"].items()))
    for r in d.get("architectures", []):
        L.append(f"arch       {r['module']}  [{r['origin']}]  {r['state']}")

    m = d.get("memory", {})
    if m.get("available"):
        L.append("")
        L.append(f"memory     {m['device']}")
        L.append(f"  weights + live   {m['active_bytes'] / GIB:8.2f} GiB")
        L.append(f"  cache            {m['cache_bytes'] / GIB:8.2f} GiB   "
                 f"(RECLAIMABLE -- not usage)")
        L.append(f"  peak this run    {m['peak_bytes'] / GIB:8.2f} GiB")
        L.append(f"  headroom         {m['headroom_bytes'] / GIB:8.2f} GiB   "
                 f"of {m['working_set_bytes'] / GIB:.1f} GiB usable "
                 f"({m['total_bytes'] / GIB:.0f} GiB installed)")
        L.append(f"  process RSS      {m['process_rss_bytes'] / GIB:8.2f} GiB"
                 f"   (what Activity Monitor shows; diverges under pressure)")
    if d.get("settings"):
        L.append("")
        L.append("settings")
        for k, v in d["settings"].items():
            L.append(f"  {k}={v}")
    L.append("")
    L.append(f"uptime {d['uptime_seconds']:.0f}s, "
             f"{d['requests_served']} requests served")
    return "\n".join(L)
