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
    return d


def snapshot(artifact=None, arch_rows=None, env=None, requests=0) -> dict:
    d = {
        "uptime_seconds": round(time.time() - _STARTED, 1),
        "requests_served": requests,
        "memory": memory(),
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
