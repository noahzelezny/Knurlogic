"""What the harness's ingest asked of the server (docs/PLAN.md, "Requirements
for knurlogic's own server"), in OpenAI's shapes and nothing custom:

  /v1/models         capabilities (text / vision / thinking) and size_bytes
  /v1/residency      one flat list: model, capabilities, memory_bytes,
                     nodes, state (loading / ready / unloading / failed)
  /v1/ensure         {model, wait}: idempotent; a different model is a
                     switch (409 while requests are running, unless force)
  413                an image over the decode limit, from its header,
                     before anything is decoded
  X-Knurlogic-Concurrency: rows=N, more=?1|?0  (RFC 8941) -- the batch's
                     width now, and whether one more row measured faster
                     per token; `more` is left out until both widths have
                     been timed. Ingest is prefill- and image-bound, where
                     the hint says less than it does for decoding.
"""

from __future__ import annotations

import base64
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


def residency(artifact, host, sched) -> dict:
    from knurlogic.machine import identity
    st = host.status()
    if st["state"] == "empty":
        return {"object": "list", "data": []}
    row = {"model": Path(st["model"] or str(artifact.path)).name,
           "capabilities": _capabilities() if st["state"] == "ready"
           else ["text"],
           "memory_bytes": int(st.get("memory_bytes") or 0),
           "nodes": [identity.identity().get("name") or "local"],
           "state": st["state"],
           "rows": sched.width}
    if st.get("error"):
        row["error"] = st["error"]
    return {"object": "list", "data": [row]}


def _resolve(model: str, artifact, host) -> str:
    """A model name or path -> the artifact directory to load."""
    cur = host.path or str(artifact.path)
    if not model or model in (Path(cur).name, cur):
        return cur
    p = Path(model).expanduser()
    if p.is_dir() and (p / "config.json").is_file():
        return str(p.resolve())
    raise ApiError(404, f"no artifact named {model!r} here: pass the "
                        f"served model's id ({Path(cur).name}) or a path to "
                        f"an artifact directory on this machine",
                   param="model", code="model_not_found")


def ensure(body: dict, artifact, host, sched) -> dict:
    if not isinstance(body, dict):
        raise ApiError(400, "the body must be a JSON object")
    path = _resolve(str(body.get("model") or ""), artifact, host)
    wait = bool(body.get("wait", False))
    try:
        timeout = float(body.get("timeout", 3600))
    except (TypeError, ValueError):
        raise ApiError(400, "timeout must be a number of seconds",
                       param="timeout")
    same = host.path == path
    if not (same and host.state in ("ready", "loading")):
        if not same and sched.width and not body.get("force"):
            raise ApiError(409, f"{sched.width} request(s) are running on "
                                f"{Path(host.path or '').name}; switching "
                                f"would fail them. Retry when idle, or "
                                f"send force: true",
                           code="model_busy")
        sched.load(path)
    if wait:
        host.wait_ready(timeout)
    st = host.status()
    if st["state"] == "failed":
        raise ApiError(500, f"loading {Path(path).name} failed: "
                            f"{st['error']}", type_="server_error")
    return {"model": Path(path).name, "state": st["state"],
            "memory_bytes": int(st.get("memory_bytes") or 0)}


def _image_bytes(url: str):
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    try:
        return base64.b64decode(url.split(",", 1)[1], validate=False)
    except Exception:
        return None


def image_limit(messages) -> None:
    """413 for an image the server would refuse to decode, judged from its
    encoded size and its header (PIL reads the PNG IHDR / JPEG SOF without
    decoding pixels)."""
    import io

    from knurlogic.engine.vision import images
    from knurlogic.engine.vision import request as vreq
    for part in vreq.image_parts(messages):
        data = _image_bytes(vreq.image_source(part))
        if data is None:
            continue
        if len(data) > images.MAX_BYTES:
            raise ApiError(413, f"an image is {len(data)} bytes; the maximum "
                                f"is {images.MAX_BYTES} bytes",
                           param="messages", code="image_too_large")
        try:
            from PIL import Image
            with Image.open(io.BytesIO(data)) as im:
                w, h = im.size
        except Exception:
            continue            # not readable: decode will say why (400)
        if w * h > images.BOMB_PIXELS:
            edge = int(images.BOMB_PIXELS ** 0.5)
            raise ApiError(413, f"an image is {w}x{h} = {w * h} pixels; the "
                                f"maximum is {images.BOMB_PIXELS} pixels "
                                f"(about {edge}x{edge}). Larger images "
                                f"under it are downscaled, not refused.",
                           param="messages", code="image_too_large")


def concurrency(sched) -> str:
    rows = sched.width
    out = f"rows={rows}"
    more = sched.more_helps(rows)
    if more is not None:
        out += f", more={'?1' if more else '?0'}"
    return out
