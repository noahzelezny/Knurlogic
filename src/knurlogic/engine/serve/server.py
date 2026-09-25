"""Serve, switch, unload: hand the model to mlx-lm's own server.

The engine ships a complete OpenAI-compatible server -- request schema,
streaming, chat templates, stop sequences. knurlogic pins it to the one
model this process serves, installs its changes to that server (one module
each: vq_runtime, cache_report, drafting, vision), adds its own routes, and
hands over.
"""

from __future__ import annotations

from . import (cache_report, drafting, sampling, state, thinking, vision,
               vq_runtime)
# By name, not as `load.`: the package re-exports a FUNCTION called `load`,
# so `from . import load` hands back the function, not this module -- and
# `serve` died on its first real start after the move (2026-09-25).
from .load import server_argv, set_cache_limit


def serve(model_path: str, host: str, port: int,
          executes_artifact_code: bool = False, extra: list | None = None,
          routes: dict | None = None, draft: bool = True,
          settings: dict | None = None):
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
    # a server that is sitting on a loaded model. Every request resolves to
    # whatever this process is currently serving.
    #
    # The pin is a VARIABLE, not a constant, which is what makes switching
    # possible at all: `ModelProvider.load` already takes any path and swaps
    # (it drops the old model first -- see `_load`), so the only thing
    # standing between one artifact and another is which path this dict
    # holds. A client still cannot steer it; only `switch()` can.
    state.SERVED["path"] = model_path
    state.SERVED["draft"] = draft
    vq_runtime.install(srv)
    sampling.install()
    cache_report.install(srv)
    thinking.install(srv)
    _real = srv.ModelProvider.load
    _real_init = srv.ModelProvider.__init__

    def _pinned(self, model_path_req=None, *a, **k):
        out = _real(self, state.SERVED["path"], *a, **k)
        # The Family binds to a LOADED model, like the head below, and for
        # the same reason is re-run on every load: a switch may gain or lose
        # vision. Only when the model actually changed -- `load` is called
        # per request batch and returns the cached model otherwise.
        if state.VISION.get("model") is not self.model:
            try:
                vision.bind(state.SERVED["path"], self)
            except Exception as e:
                # A vision build that fails must not take the text model down
                # with it -- but it is said, never swallowed: /status.json
                # shows `vision_error`, and images get the 400 of a model
                # without vision rather than a guess.
                state.VISION.update(serve=None, model=self.model,
                               error=f"{type(e).__name__}: {e}")
                vision.set_spec(None)
        # A drafting head binds to a LOADED model, and the provider loads in
        # the generator thread -- so this is the first moment one can exist.
        # Re-run on every load, because switching artifacts changes the
        # answer: the new one may ship a head, or may not.
        if state.SERVED.get("draft", True) and state.SERVED["path"]:
            try:
                if drafting.load_head(state.SERVED["path"]) is not None:
                    drafting.install(srv)
            except Exception as e:
                state.DRAFT.update(on=False, why=f"{type(e).__name__}: {e}")
        return out

    def _init(self, *a, **k):
        _real_init(self, *a, **k)
        # Captured here rather than on first request: the provider is built
        # during server construction and `load_default()` runs inside the
        # generator thread, so waiting for a request means `switch()` has
        # nothing to talk to until someone has already used the server.
        state.SERVED["provider"] = self

    srv.ModelProvider.load = _pinned
    srv.ModelProvider.__init__ = _init
    vision.install(srv)

    # Knurlogic's own routes, added to the engine's own handler. The engine
    # serves the model; these answer what is loaded, what it is using and
    # what the knobs are -- none of which the engine's server has an opinion
    # about. What they CONTAIN is not this file's business: it takes a path
    # -> handler mapping so serving stays a thin layer over mlx-lm.
    if routes:
        _real_get = srv.APIHandler.do_GET
        _count = {"n": 0}
        _real_post = srv.APIHandler.do_POST

        def _post(self):
            from urllib.parse import parse_qs, urlparse

            u = urlparse(self.path)
            handler = routes.get("POST " + (u.path.rstrip("/") or "/"))
            if handler is not None:
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else b""
                if getattr(handler, "raw", False):
                    # Writes its own response: an event stream has no length
                    # to declare, so the response ends when the socket does.
                    def _start(code, ctype):
                        self.send_response(code)
                        self.send_header("Content-Type", ctype)
                        self.send_header("Cache-Control", "no-store")
                        self.send_header("Connection", "close")
                        self.end_headers()

                    def _write(b):
                        self.wfile.write(b)
                        self.wfile.flush()

                    self.close_connection = True
                    handler(body, _write, _start)
                    return
                out, ctype = handler(parse_qs(u.query), 0, body)
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
                return
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

    settings = settings or {}
    if "cache_limit_gb" in settings:
        print(f"cache limit {set_cache_limit(settings['cache_limit_gb'])}")
    sys.argv = [sys.argv[0]] + server_argv(
        model_path, host, port, executes_artifact_code, settings, extra)
    return srv.main()


def switch(path: str) -> dict:
    """Load a different artifact into the running server.

    The weights genuinely change: `ModelProvider._load` drops the old model
    and tokenizer before loading, so the previous artifact's memory is
    released rather than accumulated.

    WHAT THIS CANNOT DO, and the caller has to say so out loud: the
    environment was resolved and set BEFORE this process started, and the
    eight GEMM/numerics flags are read at import and compiled into Metal
    kernel source. A new artifact that resolves to different values for
    those gets the OLD process's values. `LIVE_KNOBS` is the set that does
    not have this problem.

    UNVERIFIED (needs a box with the memory free): whether a second artifact
    with its own bundled `model.py` re-reads its module-level settings on
    load, or picks up the first one's module out of `sys.modules`. If it is
    cached, even a fresh artifact's kernel flags are the first one's. Do not
    quote a speed number across a switch until that is measured.
    """
    prov = state.SERVED["provider"]
    if prov is None:
        raise RuntimeError("no model provider yet; the server is still "
                           "starting")
    from knurlogic.machine import loadlock

    before = state.SERVED["path"]
    state.SERVED["path"] = path
    try:
        with loadlock.model_load(str(path), "serve.switch"):
            prov.load(path)
    except BaseException:
        state.SERVED["path"] = before          # leave the pin pointing at what is
        raise                             # actually loaded, not at a wish
    return {"loaded": path, "was": before}


def unload() -> dict:
    """Drop the weights, keep the server up.

    mlx-lm has no unload, so this is done through the provider's own fields:
    clearing `model_key` is what makes the next `load` actually reload rather
    than return the cached model. Dropping the references without clearing
    the key would leave a server that says it has a model and has none.
    """
    prov = state.SERVED["provider"]
    if prov is None:
        raise RuntimeError("no model provider yet")
    had = state.SERVED["path"]
    prov.model_key = None
    prov.model = None
    prov.tokenizer = None
    prov.draft_model = None
    vision.clear()
    try:
        import gc

        import mlx.core as mx
        gc.collect()
        mx.clear_cache()
    except Exception:
        pass
    return {"unloaded": had}
