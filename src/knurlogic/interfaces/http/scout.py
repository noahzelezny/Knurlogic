"""What the harness's ingest asked of the server (docs/PLAN.md, "Requirements
for knurlogic's own server"), in OpenAI's shapes and nothing custom:

  /v1/models         capabilities (text / vision / thinking) and size_bytes
  /v1/residency      one flat list: model, capabilities, memory_bytes,
                     nodes, state (loading / ready / unloading / failed)
  /v1/ensure         {model, wait}: idempotent; a different model is a
                     switch -- only to an artifact this machine's stores
                     hold, through the same checks as startup
                     (interfaces/loading.py); 409 while requests are
                     running, unless force
  413                an image over the decode limit (judged from its
                     header before decoding: engine/vision/images.py), or
                     a request whose images together exceed the image
                     store's memory budget
  X-Knurlogic-Concurrency: rows=N, more=?1|?0  (RFC 8941) -- the batch's
                     width now, and whether one more row measured faster
                     per token; `more` is left out until both widths have
                     been timed. Ingest is prefill- and image-bound, where
                     the hint says less than it does for decoding.
"""

from __future__ import annotations

from pathlib import Path

from .openai import ApiError


def _capabilities() -> list:
    from knurlogic.engine.serve import state, thinking
    caps = ["text"]
    if state.VISION.get("serve") is not None:
        caps.append("vision")
    try:
        if thinking.status().get("dialect"):
            caps.append("thinking")
    except Exception:
        pass
    return caps


def served(artifact, host):
    """-> a function giving /v1/models' one entry for what is served now
    (the artifact changes on a switch)."""
    from knurlogic.machine.artifact import Artifact
    memo = {"path": str(artifact.path), "art": artifact}

    def current() -> dict:
        path = host.path or memo["path"]
        if path != memo["path"]:
            try:
                memo.update(path=path, art=Artifact.load(path))
            except Exception:
                memo.update(path=path, art=None)
        art = memo["art"]
        return {"id": Path(path).name, "path": path,
                "created": int(host.loaded_at or 0),
                "capabilities": _capabilities(),
                "size_bytes": int(getattr(art, "bytes_on_disk", 0) or 0)}
    return current


def ensure(body: dict) -> dict:
    """POST /v1/ensure {model, wait?, timeout?, force?} -- see
    interfaces/http.switch."""
    from knurlogic.interfaces import http
    from knurlogic.interfaces.loading import NotLoadable
    if not isinstance(body, dict):
        raise ApiError(400, "the body must be a JSON object")
    try:
        timeout = float(body.get("timeout", 3600))
    except (TypeError, ValueError):
        raise ApiError(400, "timeout must be a number of seconds",
                       param="timeout")
    try:
        return http.switch(str(body.get("model") or ""),
                           force=bool(body.get("force")),
                           wait=bool(body.get("wait", False)),
                           timeout=timeout)
    except NotLoadable as e:
        raise ApiError(e.status, str(e), param="model", code=e.code or None)


def residency(host, sched, port: int = 0) -> dict:
    from knurlogic.cluster import recovery
    from knurlogic.machine import identity, servers
    st = host.status()
    if st["state"] == "empty":
        return {"object": "list", "data": []}
    try:
        rec = recovery.served_view(port, st["state"] == "ready") \
            if port else None
    except Exception:
        rec = None
    row = {"model": Path(st["model"] or "").name,
           "capabilities": _capabilities() if st["state"] == "ready"
           else ["text"],
           "memory_bytes": int(st.get("memory_bytes") or 0),
           "nodes": [identity.identity().get("name") or "local"],
           "state": st["state"],
           "rows": sched.width,
           "requests": sched.requests(),
           # relaunched by its page after dying unasked (interfaces/
           # recovery.py): attempts, last_reason, last_at, next_at, state
           "recovery": rec}
    if port:
        # a 16-hex id (a single-Mac load) or a cluster job's id (rank 0's
        # own registry row), from this box's own registry -- known only
        # when a page (or `serve` itself, for a cluster rank) wrote it
        try:
            srec = servers.registry().get(int(port)) or {}
            inst = str(srec.get("job") or srec.get("instance") or "")
        except Exception:
            inst = ""
        if inst:
            row["instance"] = inst
    if st.get("error"):
        row["error"] = st["error"]
    return {"object": "list", "data": [row]}




def concurrency(sched) -> str:
    rows = sched.width
    out = f"rows={rows}"
    more = sched.more_helps(rows)
    if more is not None:
        out += f", more={'?1' if more else '?0'}"
    return out
