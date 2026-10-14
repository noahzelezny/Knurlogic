"""The MCP's client for the page on this machine (`knurlogic ui`): every
tool that reaches past this machine asks it over loopback (page_get,
page_post); the models running across machines are its /running.json
(page/running.py)."""

from __future__ import annotations

import json
from typing import Any

from knurlogic.machine.http_client import NET_ERRORS, request
from knurlogic.machine.servers import PAGE_PORT

# --- across machines: through the page on this machine ---------------------------
# The page (`knurlogic ui`) is each Mac's node agent. It holds the peers it
# can see, the cluster jobs it coordinates and the watcher that tears a
# failed one down -- state that lives in ITS process, not this one. So every
# tool that reaches past this machine asks that page over loopback, with the
# same POST /loaded.json its Launch and Unload buttons send: one code path,
# and a job the MCP started is watched and failed over like any other.

#: the page on this machine, host:port (`knurlogic ui --port`, default 8899)
PAGE_ENV = "KNURLOGIC_PAGE"
#: a cluster launch answers once every machine has prepared and started
PAGE_LOAD_S = 300.0
PAGE_READ_S = 15.0
#: the MCP's link names, and the page's (cluster/protocol.LINK_NAMES)
LINKS = ("tcp", "rdma")
SPLITS = ("tensor", "pipeline")


class PageDown(Exception):
    pass


def _page_addr() -> str:
    import os
    return os.environ.get(PAGE_ENV) or f"127.0.0.1:{PAGE_PORT}"


def page_call(path: str, doc=None, timeout: float = PAGE_READ_S):
    url = f"http://{_page_addr()}{path}"
    try:
        _, raw = request(url, None if doc is None else json.dumps(doc).encode(),
                         timeout=timeout)
    except NET_ERRORS as e:
        raise PageDown(f"the page on this machine ({url.split(path)[0]}) did "
                       f"not answer: {type(e).__name__}: {e}. Across "
                       f"machines goes through it: start `knurlogic ui` "
                       f"(set {PAGE_ENV}=host:port if it is not on "
                       f"{PAGE_PORT})") from e
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
    return identity.identity().get("name") or "this machine"


def node_ids(names: list[str]) -> tuple:
    """([page node ids], refusal or None): machine names as this machine's page
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
                        "refused": f"{n!r} is not a machine this machine's page "
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
