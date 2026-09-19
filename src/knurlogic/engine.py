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


#: Knobs that can be changed on a RUNNING process, and how.
#:
#: Measured by reading a real bundled runtime (4523 lines) rather than
#: assuming. Of eleven knobs the resolver emits:
#:
#:   * VQ_DECODE_CHUNK is captured into a module global on FIRST PREFILL
#:     (`_DECODE_CHUNK = _default_decode_chunk()`) and then read inside the
#:     expert loop as a global. Rebinding that global takes effect on the next
#:     prefill -- no reload.
#:   * VQLAB_CACHE_LIMIT_GB is applied through the framework's own live API.
#:   * the eight GEMM/numerics flags are read into module globals AT IMPORT and
#:     baked into Metal kernel source that is compiled once. Those genuinely
#:     need a restart, or an override module that reads them per dispatch.
LIVE_KNOBS = ("VQ_DECODE_CHUNK", "VQLAB_CACHE_LIMIT_GB")


def _artifact_runtime_modules():
    """Loaded modules that look like a bundled VQ runtime.

    `vars(mod)` and NOT `hasattr`. A hasattr sweep over sys.modules invokes
    every lazy module's `__getattr__`, which in this environment reached into
    transformers' lazy-import machinery and raised from inside a package that
    has nothing to do with any of this. Reading __dict__ asks the question
    without running anybody else's code.
    """
    import sys
    out = []
    for name, mod in list(sys.modules.items()):
        try:
            d = vars(mod)
        except TypeError:
            continue
        if "_DECODE_CHUNK" in d:
            out.append((name, mod))
    return out


def apply_live(env: dict) -> dict:
    """Apply what can be applied without reloading the model.

    Returns {knob: what happened}. A knob that cannot be applied is REPORTED,
    never silently skipped: a settings panel that says "applied" over a value
    that did not move is the same lie as an env file sourced after the one
    that overwrites it.
    """
    import os

    done = {}
    for k, v in env.items():
        if k == "VQLAB_CACHE_LIMIT_GB":
            try:
                import mlx.core as mx
                nbytes = int(float(v) * (1 << 30))
                setter = getattr(mx, "set_cache_limit", None) or \
                    getattr(getattr(mx, "metal", None), "set_cache_limit", None)
                if setter is None:
                    done[k] = "no live setter in this engine build"
                    continue
                setter(nbytes)
                os.environ[k] = str(v)
                done[k] = f"applied now ({v} GiB)"
            except Exception as e:
                done[k] = f"failed: {e}"
        elif k == "VQ_DECODE_CHUNK":
            mods = _artifact_runtime_modules()
            if not mods:
                # Before the first prefill the global does not exist yet, but
                # the environment is still what the runtime will read.
                os.environ[k] = str(v)
                done[k] = "set for the next prefill (runtime not resolved yet)"
                continue
            for name, mod in mods:
                mod._DECODE_CHUNK = int(v)
            os.environ[k] = str(v)
            done[k] = (f"applied to {len(mods)} loaded runtime"
                       f"{'s' if len(mods) > 1 else ''}; takes effect on the "
                       f"next prefill")
        else:
            done[k] = "needs a restart: read at import and compiled into the "\
                      "kernel"
    return done


def tool_support(chat_template: str) -> dict:
    """Which tool-call dialect an artifact speaks, and whether we can read it.

    Tool calling is not one format. This template asks for

        <tool_call>\n<function=NAME>\n<parameter=P>value</parameter>...

    which is the Qwen3-Coder / agentic-harness dialect, while other models
    emit JSON inside the same <tool_call> tags, and others use
    [TOOL_CALLS] or <|tool_calls_section_begin|>. The engine picks a parser by
    INFERRING it from the template, and when the inference misses it returns
    None -- at which point tool calls come back as prose. A harness then looks
    like a model that keeps describing the function it would call instead of
    calling it, which is a mystifying failure to debug from the outside and a
    one-line answer from here.

    The inference rule is the engine's, deliberately: reimplementing it here
    would drift from the parser actually used at serve time. It is a private
    function, so this is version-skew surface, which is this file's job.
    """
    out = {"has_template": bool(chat_template), "parser": None,
           "mentions_tools": "tool" in (chat_template or "").lower()}
    if not chat_template:
        return out
    try:
        from mlx_lm.tokenizer_utils import _infer_tool_parser
    except Exception:
        out["parser"] = "unknown (this engine exposes no inference rule)"
        return out
    try:
        out["parser"] = _infer_tool_parser(chat_template)
    except Exception:
        out["parser"] = None
    return out
