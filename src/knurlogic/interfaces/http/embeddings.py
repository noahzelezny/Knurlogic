"""An embedding model's server: POST /v1/embeddings in OpenAI's shape,
with the /v1/models, /v1/residency, /health and page routes every
knurlogic server answers, so the page lists it, its router reaches it and
its unload stops it like any other model.

  {"input": "text" | ["text", ...], "model": ..., "encoding_format"?}
  -> {"object": "list", "data": [{"object": "embedding", "index": i,
      "embedding": [...]}], "model": ..., "usage": {...}}

The server answers at once while the weights load; they go after they
have been idle and come back with the next request
(engine/embeddings.Embedder), so the ~320 MiB model never stands in
the way of a big one on the same Mac.

Design: docs/design/server.md (embeddings).
"""

from __future__ import annotations

import base64
import json
import logging
import struct
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlparse

from . import openai as O
from .server import DEFAULT_MAX_BODY, BodyError, Handler

if TYPE_CHECKING:
    from knurlogic.engine.embeddings import Embedder

logger = logging.getLogger(__name__)

EMBEDDINGS_PATH = "/v1/embeddings"

#: the most texts one request may carry: OpenAI's own limit
MAX_INPUTS = 2048


def texts_of(body) -> list:
    """The texts a request embeds: `input` as one string or a list of
    them. Token arrays (OpenAI's other input forms) are refused: the
    tokenizer here is the model's own."""
    if not isinstance(body, dict):
        raise O.ApiError(400, "the body must be a JSON object")
    got = body.get("input")
    if isinstance(got, str):
        got = [got]
    if not isinstance(got, list) or not got:
        raise O.ApiError(400, "input must be a string or a non-empty list "
                              "of strings", param="input")
    if not all(isinstance(t, str) for t in got):
        raise O.ApiError(400, "input must be text: token arrays are not "
                              "accepted", param="input")
    if len(got) > MAX_INPUTS:
        raise O.ApiError(400, f"input has {len(got)} texts; the most is "
                              f"{MAX_INPUTS}", param="input")
    return got


def encoding_of(body: dict) -> str:
    """`encoding_format`: "float" (the default) or "base64" (little-endian
    float32, what the OpenAI SDKs ask for unless told otherwise)."""
    fmt = body.get("encoding_format") or "float"
    if fmt not in ("float", "base64"):
        raise O.ApiError(400, "encoding_format must be float or base64",
                         param="encoding_format")
    if body.get("dimensions") is not None:
        raise O.ApiError(400, "dimensions is not supported: the vectors "
                              "are the model's full size", param="dimensions")
    return fmt


def _encoded(vector: list, fmt: str):
    if fmt == "float":
        return vector
    return base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode()


def embeddings_document(vectors: list, tokens: int, model: str,
                        fmt: str = "float") -> dict:
    """The /v1/embeddings answer for `vectors`, in request order."""
    return {"object": "list",
            "data": [{"object": "embedding", "index": i,
                      "embedding": _encoded(v, fmt)}
                     for i, v in enumerate(vectors)],
            "model": model,
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens}}


@dataclass
class EmbeddingApp:
    """What the handler needs: the embedder, what is served, the page's
    routes and the guards' settings (http/server.App's names)."""
    embedder: Embedder
    served: Callable[[], dict]
    routes: dict = field(default_factory=dict)
    gate: object = None
    max_body: int = DEFAULT_MAX_BODY
    allow_origins: set = field(default_factory=set)
    allow_hosts: set = field(default_factory=set)
    requests: int = 0
    _count_lock: threading.Lock = field(default_factory=threading.Lock)

    def embed(self, body: dict) -> dict:
        """POST /v1/embeddings; a refusal or a failed load is an ApiError."""
        texts = texts_of(body)
        fmt = encoding_of(body)
        with self._count_lock:
            self.requests += 1
        try:
            got = self.embedder.embed(texts)
        except Exception as e:      # a load or a pass that failed: a 500
            logger.exception("embedding failed")
            raise O.ApiError(500, f"{type(e).__name__}: {e}") from e
        return embeddings_document(got.vectors, got.tokens,
                                   self.served()["id"], fmt)

    def load_state(self) -> str:
        """/v1/models' state: ready unless the last load failed -- a model
        not held is still served (its next request loads it)."""
        return "failed" if self.embedder.state == "failed" else "ready"

    def residency(self) -> dict:
        from knurlogic.machine import identity
        st = self.embedder.status()
        row = {"model": self.served()["id"], "capabilities": ["embeddings"],
               "memory_bytes": self.served()["size_bytes"]
               if st["state"] == "ready" else 0,
               "nodes": [identity.identity().get("name") or "local"],
               "state": st["state"], "rows": 1, "requests": None}
        if st["error"]:
            row["error"] = st["error"]
        return {"object": "list", "data": [row]}


class EmbeddingHandler(Handler):
    """http/server.Handler's guards, body limit and replies, with an
    embedding model's routes in place of a chat model's."""
    app: EmbeddingApp = None  # type: ignore[assignment]  # set by make_server

    def _get(self):
        u = urlparse(self.path)
        path = u.path.rstrip("/") or "/"
        if path == "/v1/models":
            return self._json(200, O.models_document(
                self.app.served(), load_state=self.app.load_state()))
        if path == "/health":
            return self._json(200, {"status": "ok", "server": "knurlogic",
                                    "model": self.app.embedder.state})
        if path == "/v1/residency":
            return self._json(200, self.app.residency())
        h = self.app.routes.get(path)
        if h is None:
            return self._json(404, {"error": {"message": f"no route {path}",
                                              "type": "not_found"}})
        body, ctype = h(parse_qs(u.query), self.app.requests)
        return self._send(200, body, ctype)

    def _post(self):
        path = urlparse(self.path).path.rstrip("/") or "/"
        try:
            raw = self._body()
        except BodyError as e:
            self.close_connection = True       # the unread body is left
            return self._error(O.ApiError(e.status, str(e)))
        if path != EMBEDDINGS_PATH:
            return self._error(O.ApiError(
                404, f"{path}: this server holds an embedding model and "
                     f"answers only {EMBEDDINGS_PATH}", type_="not_found"))
        try:
            body = json.loads(raw or b"{}")
        except ValueError as e:
            return self._error(O.ApiError(400, f"invalid JSON: {e}"))
        try:
            return self._json(200, self.app.embed(body))
        except O.ApiError as e:
            return self._error(e)


def make_server(app: EmbeddingApp, host: str, port: int):
    from http.server import ThreadingHTTPServer
    H = type("KnurlogicEmbeddingHandler", (EmbeddingHandler,), {"app": app})
    srv = ThreadingHTTPServer((host, port), H)
    srv.daemon_threads = True
    return srv


def served(artifact, embedder) -> Callable[[], dict]:
    """/v1/models' one entry for the embedding model."""
    def current() -> dict:
        return {"id": artifact.path.name, "path": str(artifact.path),
                "created": int(embedder.loaded_at or 0),
                "capabilities": ["embeddings"],
                "size_bytes": int(artifact.bytes_on_disk or 0)}
    return current


def status_fn(artifact, embedder) -> Callable:
    """The /status.json document (snapshot, text): the artifact and the
    embedder's load state, which the page's survey reads."""
    from knurlogic.engine import model as engine
    from knurlogic.machine import status

    def status_doc(requests):
        snap = status.aggregate([status.snapshot(
            artifact=artifact, requests=requests, node="local",
            memory_fn=engine.process_memory)])
        snap["load"] = embedder.status()
        return snap, (f"{artifact.path.name}: embedding model, "
                      f"{embedder.state}")
    return status_doc


def serve(artifact, host: str, port: int, *,
          settings: dict | None = None) -> int:
    """Serve `artifact`'s embeddings until SIGTERM (the page's unload)."""
    import signal

    from knurlogic.engine.embeddings import Embedder
    from knurlogic.interfaces import documents

    from .lifecycle import bind_gated

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    settings = dict(settings or {})
    embedder = Embedder(str(artifact.path))
    stop = threading.Event()
    threading.Thread(target=embedder.warm, daemon=True).start()
    threading.Thread(target=embedder.watch, args=(stop,),
                     daemon=True).start()
    app = EmbeddingApp(
        embedder, served(artifact, embedder),
        routes=documents.routes(status_fn=status_fn(artifact, embedder)),
        max_body=settings.get("max_body", DEFAULT_MAX_BODY),
        allow_origins={o.rstrip("/")
                       for o in settings.get("allow_origins") or ()},
        allow_hosts={h.lower() for h in settings.get("allow_hosts") or ()})

    def on_term(_sig, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_term)
    servers = bind_gated(app, host, port, make_server)
    for srv in servers[1:]:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"knurlogic embeddings on http://{host.split(',', 1)[0]}:{port}"
          f"{EMBEDDINGS_PATH} ({Path(artifact.path).name}, dropped after "
          f"{embedder.idle_s:.0f} s idle, loaded again on demand)",
          flush=True)
    try:
        servers[0].serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        for srv in servers:
            srv.server_close()
    return 0
