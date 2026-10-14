"""GET /running.json: every model running on this machine and the peers
its page sees, one entry per model -- a cluster job once, on its leader,
with its machines, split, link, job and rank 0's `requests` -- and which
machines answered. Read from the page's own /loaded.json?peers=1 (what
loads.with_jobs and peers.peer_residency make); the MCP's models and
unload read it."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from knurlogic.machine import identity

#: the page's route for document()
RUNNING_PATH = "/running.json"


def route(loaded_fn):
    """The page's RUNNING_PATH route over its /loaded.json handler."""
    from knurlogic.interfaces.documents import json_reply
    return lambda _q, _n=0: json_reply(document(loaded_fn({"peers": ["1"]})))


def document(loaded: dict) -> dict:
    """{"models", "machines", "jobs"} of a /loaded.json?peers=1 document.
    `jobs`: {"here": the cluster jobs with a rank on this machine, "live":
    the ones not stopped on any machine} -- what an unload decides by
    (this page stops a job it has a rank of, or one running nowhere)."""
    here = identity.identity().get("name") or "this machine"
    mine = [j for j in loaded.get("jobs") or [] if isinstance(j, dict)]
    every = mine + [j for p in loaded.get("peers") or [] if isinstance(p, dict)
                    for j in p.get("jobs") or [] if isinstance(j, dict)]
    return {"models": models_across(loaded, here),
            "machines": machines_of(loaded, here),
            "jobs": {"here": sorted({str(j["job"]) for j in mine
                                     if j.get("job")}),
                     "live": sorted({str(j["job"]) for j in every
                                     if j.get("job")
                                     and j.get("phase") != "stopped"})}}


def machines_of(page, here: str) -> list:
    """This machine and each peer its page asked, with whether it answered."""
    out = [{"machine": here, "here": True}]
    for p in (page or {}).get("peers") or []:
        if isinstance(p, dict):
            out.append({"machine": p.get("machine"), "here": False,
                        **({"error": p["error"]} if p.get("error") else {})})
    return out


def models_across(page: dict, here: str) -> list:
    """Every resident model in a page's /loaded.json?peers=1, one entry per
    model: a cluster job ONCE (rank 0's row, which carries its `requests`;
    or the job itself while rank 0 has no row yet), never one per rank."""
    from knurlogic.cluster.launch import link_name
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
        # a cluster job's instance is its job id; a one-machine server's is
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
            "link": link_name(c.get("link") or j.get("link")),
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
            "split": j.get("split"), "link": link_name(j.get("link")),
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
            "link": link_name(d.get("link")), "job": d.get("job"),
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
    try:
        return urlparse(where or "").port
    except ValueError:
        return None
