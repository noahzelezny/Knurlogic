"""The protocol messages a page answers on MSG_PATH (`peer_table`): the
cluster kinds, Load/Unload, MachineSet, Settings, Survey and Read."""

from __future__ import annotations

import json

from knurlogic.interfaces.page import documents, loads, peek, peers, router


def survey_here(routes: dict) -> tuple:
    """A Survey: what this page's own /loaded.json says (never with
    ?peers=1: pages asking each other must not recurse)."""
    h = routes.get("/loaded.json")
    if h is None:
        return 404, {"error": "this page lists nothing resident"}
    body, _ctype = h({}, 0)
    doc = json.loads(body)
    if isinstance(doc, dict):
        # fresh memory and ranks still exiting: what a coordinator placing
        # a launch right after an unload needs, not the last status
        from knurlogic.cluster import launch
        doc["available_bytes"] = launch.available_now()
        doc["exiting"] = len(launch.exiting(launch.J.registry()))
    return 200, doc


#: the documents a Read may fetch from a page, by path (the allow-list;
#: anything else is refused): /peek's page-to-peer reads
READ_PAGE_PATHS = ("/settings.json", "/models.json", "/v1/models")


def read_here(routes: dict, req: dict) -> tuple:
    """A Read on this machine: a page document (READ_PAGE_PATHS), or --
    with `port` -- a model server this machine started, through
    peer_settings. Reads only."""
    path = req.get("path")
    q: dict = req["query"] if isinstance(req.get("query"), dict) else {}
    if req.get("port") is not None:
        if path != peek.PEER_SETTINGS:
            return 403, {"error": f"not a readable path: {str(path)[:60]!r}"}
        return peek.peer_settings("GET", req.get("port"), q)
    if path not in READ_PAGE_PATHS or path not in peek.PEEK_PATHS:
        return 403, {"error": f"not a readable path: {str(path)[:60]!r}"}
    if path == "/v1/models":
        return 200, router.route_models_document()
    h = routes.get(path)
    if h is None:
        return 404, {"error": "no such document here"}
    body, _ctype = h({k: [str(v)] for k, v in q.items()
                      if k in peek.peek_keys(path)}, 0)
    try:
        return 200, json.loads(body)
    except ValueError:
        return 502, {"error": "that document is not JSON"}


def peer_table(routes: dict) -> dict:
    """Message kind -> handler(body) -> (status, doc): every kind a page
    answers on MSG_PATH. Anything not here (Hello, Heartbeat, ...) is
    refused by the dispatcher as a typed Failure."""
    from knurlogic.cluster import launch

    def changes(fn):
        def run(body):
            out = fn(body)
            documents.LOADED["doc"] = None     # residency may have changed
            return out
        return run
    t: dict = {k: changes(lambda b, k=k: launch.peer_step(k, b))
               for k in launch.CLUSTER_KINDS}
    t["Load"] = changes(lambda b: loads.peer_launch(dict(b, action="load")))
    t["Unload"] = changes(lambda b: loads.peer_launch(dict(b, action="unload")))
    t["MachineSet"] = changes(peers.peer_machine)
    t["Settings"] = lambda b: peek.peer_settings(
        "POST", b.get("port"), b.get("values"))
    t["Survey"] = lambda b: survey_here(routes)
    t["Read"] = lambda b: read_here(routes, b)
    return t
