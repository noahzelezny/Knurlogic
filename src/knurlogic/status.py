"""What is loaded, and what it is actually using.

The complaint this answers: a runtime shows up as one opaque number in
Activity Monitor -- "python3.13, 45 GB" -- and you cannot tell weights from
reclaimable cache from a transient peak, or see which architecture actually
loaded. Every one of those is available; nothing surfaces it.

Two cautions, both measured and both worth printing next to the numbers:

* `ps` RSS and the framework's own accounting agree on a small resident model
  (12.09 GiB vs 11.61 on a 27B) and DIVERGE under pressure -- a probe once
  read 11.7 GiB from `ps` while the process held ~60. Neither number alone is
  trustworthy, so both are shown.
* Cache memory is RECLAIMABLE. Counting it as usage is what makes a runtime
  look like it is eating the machine when it is holding freed buffers it will
  hand back.

ONE PROCESS OR SEVERAL. `snapshot()` answers for the process it runs in and
that is all it can honestly do -- a remote node's numbers have to come off
that node. `aggregate()` is the shape a cluster arrives in: a list of those
per-node snapshots plus the rollup, and it is what `/status.json` serves even
for one node, so a client written against one box does not have to be
rewritten when a second appears. The single-node keys stay at the top level
for exactly the same reason.

A rollup SUMS memory and does not average it: two nodes each 40 GiB active
are 80 GiB of weights held, not 40. It also reports how many nodes answered,
because a sum over nodes that did not reply is a smaller number that looks
like good news.
"""

from __future__ import annotations

import os
import subprocess
import time

GIB = 1 << 30
_STARTED = time.time()


def _rss_bytes() -> int:
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                             capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip()) * 1024
    except Exception:
        return 0


def memory() -> dict:
    """Delegated: how memory is accounted is an ENGINE question, and the
    tripwire test caught this module importing mlx to answer it."""
    from .engine import memory as _m

    d = _m()
    d["process_rss_bytes"] = _rss_bytes()
    d["scope"] = "process"      # see `scope` in aggregate/render_cluster
    return d


#: Bumped when the wire shape of /status.json changes incompatibly.
SCHEMA = 2


def snapshot(artifact=None, arch_rows=None, env=None, requests=0,
             node="local", role="server", reachable=True,
             memory_fn=None) -> dict:
    """One node's answer. `memory_fn` exists so a snapshot can be BUILT from
    numbers that came off another node (exo reports them for every node in
    the cluster) rather than only from this process."""
    d = {
        "node": node,
        "role": role,
        "reachable": reachable,
        "uptime_seconds": round(time.time() - _STARTED, 1),
        "requests_served": requests,
        "memory": (memory_fn or memory)(),
    }
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


def aggregate(snapshots, artifact=None) -> dict:
    """The cluster shape. One node in, one node out -- with the rollup.

    The single node's own keys are also spread at the top level so that
    `/status.json` keeps the contract it already had; `nodes` is where a
    second node arrives, and `cluster` is the only place a total lives.
    """
    snaps = list(snapshots)
    up = [s for s in snaps if s.get("reachable", True)]
    mem = {k: sum(int(s.get("memory", {}).get(k, 0) or 0) for s in up)
           for k in _SUMMED}
    mem["available"] = any(s.get("memory", {}).get("available") for s in up)
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
                 f"{m['active_bytes'] / GIB:6.1f} GiB live  "
                 f"{m['cache_bytes'] / GIB:5.1f} cache  "
                 f"{m['headroom_bytes'] / GIB:6.1f} free of "
                 f"{m['working_set_bytes'] / GIB:.0f}")
    m = c["memory"]
    # Where the numbers came from changes what they MEAN. A per-process
    # snapshot separates weights from reclaimable cache; a number reported
    # for the whole box cannot, and saying "weights" over it would be a
    # smaller lie than it looks -- it is what makes a runtime get blamed for
    # everything else running on the machine.
    # `scope` says whether a node's numbers came from the runtime process or
    # from the whole machine. It is a field rather than a guess at the device
    # string, because a label that drifts is worse than no label.
    boxwide = any(s.get("memory", {}).get("scope") == "box"
                  for s in d.get("nodes", []))
    L.append("")
    L.append(f"cluster      {c['nodes_reachable']}/{c['nodes_total']} nodes "
             f"answering")
    L.append(f"  {'in use on the box' if boxwide else 'weights + live':<16s} "
             f"{m['active_bytes'] / GIB:8.2f} GiB"
             + ("   (everything on that machine, not just this runtime -- "
                "exo reports the box)" if boxwide else ""))
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
