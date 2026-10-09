"""The MCP's client for the page on this Mac (`knurlogic ui`): every tool
that reaches past this Mac asks it over loopback, and models_across reads
its /loaded.json?peers=1 into one entry per model."""

from __future__ import annotations

import http.client
import json
from typing import Any

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


def page_call(path: str, doc=None, timeout: float = PAGE_READ_S):
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


def page_get(path: str) -> dict:
    return page_call(path)


def page_post(doc: dict, timeout: float = PAGE_LOAD_S) -> dict:
    return page_call("/loaded.json", doc, timeout)


def me_name() -> str:
    from knurlogic.machine import identity
    return identity.identity().get("name") or "this Mac"


def machines_of(page, here: str) -> list:
    """This Mac and each peer its page asked, with whether it answered."""
    out = [{"machine": here, "here": True}]
    for p in (page or {}).get("peers") or []:
        if isinstance(p, dict):
            out.append({"machine": p.get("machine"), "here": False,
                        **({"error": p["error"]} if p.get("error") else {})})
    return out


def link_name(link):
    from knurlogic.cluster import launch
    return launch.link_name(link)


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
    from urllib.parse import urlparse
    try:
        return urlparse(where or "").port
    except ValueError:
        return None


def node_ids(names: list[str]) -> tuple:
    """([page node ids], refusal or None): machine names as this Mac's page
    knows them -- itself and the peers answering it. An id is accepted too."""
    st = page_get("/status.json")
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


def refusal(out: dict) -> dict[str, Any] | None:
    """The page's refusal, as a refusal: an answer, never a crash."""
    why = out.get("refused") or out.get("error")
    if not why:
        return None
    return {"loaded": False, "refused": why,
            **{k: out[k] for k in ("note", "detail", "placement", "machine")
               if k in out}}
