"""The engine seam -- the one place that knows what runs a model.

Today the engine is mlx-lm (and mlx-vlm for multimodal architectures). That
may not always be true, and the cost of keeping the option open is exactly
this file: every other module asks here instead of importing mlx directly.
Knurlogic's own code is ~1000 lines against ~3900 lines of vendored
architectures and a 4500-line runtime shipped inside each artifact -- so the
package itself is nearly uncoupled, and the job is to keep it that way rather
than to abstract anything clever.

WHAT A REPLACEMENT WOULD HAVE TO HONOUR, because these are not this file's
choices -- they are the ecosystem's:

  * `mlx_lm.models.<type>` / `mlx_vlm.models.<type>` is where a model class
    is looked up, by `importlib.import_module`. That is how `register` gets
    vendored architectures in front of installed ones without writing to
    anyone's site-packages.
  * An artifact may ship its OWN runtime and name it in `config.json`
    (`model_file`). That file is executed, and it is where a VQ artifact's
    kernels live. It is also, usefully, a PER-ARTIFACT runtime boundary: a
    new artifact can bundle a runtime for a new engine while every already
    published artifact keeps running the one it shipped with. An engine
    migration is therefore per-rung, not global.
  * The vendored architectures are written against the mlx array API. An
    engine that mirrors that API runs them unchanged; one that does not
    rewrites 3900 lines and every architecture after.

VERSION SKEW IS THIS FILE'S JOB. mlx-lm 0.31.3 executes `model_file`
unconditionally; 0.32.0 put it behind `trust_remote_code=` and raises without
it. Passing the kwarg blindly is a TypeError on one, omitting it a ValueError
on the other, so a VQ artifact cannot load on both unless something inspects
the signature. Nobody downstream should ever learn that.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass

#: Packages that can host a model architecture, in lookup order.
HOST_PACKAGES = ("mlx_lm", "mlx_vlm")


@dataclass
class EngineInfo:
    name: str
    version: str
    available: bool

    def __str__(self) -> str:
        return (f"{self.name} {self.version}" if self.available
                else f"{self.name} (not installed)")


def info(host: str = "mlx_lm") -> EngineInfo:
    try:
        m = importlib.import_module(host)
        return EngineInfo(host, getattr(m, "__version__", "unknown"), True)
    except Exception:
        return EngineInfo(host, "-", False)


def describe() -> str:
    return " | ".join(str(info(h)) for h in HOST_PACKAGES)


def memory() -> dict:
    try:
        import mlx.core as mx
    except Exception:
        return {"available": False}
    try:
        info = (mx.device_info() if hasattr(mx, "device_info")
                else mx.metal.device_info())
        ws = int(info.get("max_recommended_working_set_size", 0))
        total = int(info.get("memory_size", ws))
        device = info.get("device_name", "unknown")
    except Exception:
        ws = total = 0
        device = "unknown"
    active = int(mx.get_active_memory())
    cache = int(mx.get_cache_memory())
    return {
        "available": True,
        "device": device,
        "active_bytes": active,
        "cache_bytes": cache,
        "peak_bytes": int(mx.get_peak_memory()),
        "working_set_bytes": ws,
        "total_bytes": total,
        "headroom_bytes": max(ws - active, 0),
    }


def models_module(host: str = "mlx_lm"):
    """The package a model architecture is looked up in."""
    return importlib.import_module(f"{host}.models")


def load(path: str, executes_artifact_code: bool = False):
    """Load a model and tokenizer.

    `executes_artifact_code` says the artifact ships its own runtime, which
    WILL be executed. It is a separate argument rather than something inferred
    quietly, so a caller has to state it and can say so to the user.
    """
    from mlx_lm.utils import load as _load

    kw = {}
    if executes_artifact_code and \
            "trust_remote_code" in inspect.signature(_load).parameters:
        kw["trust_remote_code"] = True
    return _load(path, **kw)


def serve(model_path: str, host: str, port: int,
          executes_artifact_code: bool = False, extra: list | None = None,
          routes: dict | None = None):
    """Hand off to the engine's own OpenAI-compatible server.

    The engine ships a complete one -- request schema, streaming, chat
    templates, stop sequences -- and it takes its configuration through
    argv, so this rewrites argv and calls it. Knurlogic's job ended when the
    architectures were registered and the environment was resolved.
    """
    import sys

    from mlx_lm import server as srv

    # PIN THE SERVED MODEL. The engine's server treats a request's `model`
    # field as something to load, and anything it does not recognise it tries
    # to fetch from the Hub -- so a client configured with a different name
    # (Cline, Continue, Zed all send whatever the user typed) gets a 404 from
    # a server that is sitting on a loaded model. We serve exactly one
    # artifact, so every request resolves to it.
    _real = srv.ModelProvider.load

    def _pinned(self, model_path_req=None, *a, **k):
        return _real(self, model_path, *a, **k)

    srv.ModelProvider.load = _pinned

    # Knurlogic's own routes, added to the engine's own handler. The engine
    # serves the model; these answer what is loaded, what it is using and
    # what the knobs are -- none of which the engine's server has an opinion
    # about. What they CONTAIN is not this file's business: it takes a path
    # -> handler mapping so the seam stays a seam.
    if routes:
        _real_get = srv.APIHandler.do_GET
        _count = {"n": 0}
        _real_post = srv.APIHandler.do_POST

        def _post(self):
            _count["n"] += 1
            return _real_post(self)

        def _get(self):
            from urllib.parse import parse_qs, urlparse

            u = urlparse(self.path)
            handler = routes.get(u.path.rstrip("/") or "/")
            if handler is None:
                return _real_get(self)
            body, ctype = handler(parse_qs(u.query), _count["n"])
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        srv.APIHandler.do_GET = _get
        srv.APIHandler.do_POST = _post

    argv = [sys.argv[0], "--model", model_path, "--host", host,
            "--port", str(port)]
    if executes_artifact_code:
        argv.append("--trust-remote-code")
    argv += list(extra or [])
    sys.argv = argv
    return srv.main()


def generate(model, tokenizer, prompt: str, max_tokens: int = 8) -> str:
    from mlx_lm.generate import generate as _generate

    return _generate(model, tokenizer, prompt=prompt,
                     max_tokens=max_tokens, verbose=False) or ""
