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

    from knurlogic.machine import loadlock

    kw = {}
    if executes_artifact_code and \
            "trust_remote_code" in inspect.signature(_load).parameters:
        kw["trust_remote_code"] = True
    # The box is shared: one model load at a time, across processes
    # (machine/loadlock.py; vision-contracts.md "Load lock" names this caller).
    with loadlock.model_load(str(path), "seam.load"):
        return _load(path, **kw)


def server_argv(model_path: str, host: str, port: int,
                executes_artifact_code: bool = False,
                settings: dict | None = None,
                extra: list | None = None) -> list:
    """The engine server's argv, with the resolved settings IN it.

    The prompt chunk and prompt concurrency are argv to mlx-lm's server and
    nothing else: an environment variable with the same meaning is read by
    nobody. Anything the caller passes through `extra` comes last and wins,
    because a flag somebody typed is a decision, and the resolver's value is
    a default.
    """
    settings = settings or {}
    extra = list(extra or [])
    argv = ["--model", model_path, "--host", host, "--port", str(port)]
    if executes_artifact_code:
        argv.append("--trust-remote-code")
    for key, flag in (("prefill_step_size", "--prefill-step-size"),
                      ("prompt_concurrency", "--prompt-concurrency")):
        if key in settings and flag not in extra:
            argv += [flag, str(int(settings[key]))]
    return argv + extra


def set_cache_limit(gib: float) -> str:
    """Bound mlx's freed-buffer cache for this process. mlx-lm never sets
    it, so without this the knob is a number on a page."""
    import mlx.core as mx
    setter = getattr(mx, "set_cache_limit", None) or \
        getattr(getattr(mx, "metal", None), "set_cache_limit", None)
    if setter is None:
        return "no cache-limit setter in this engine build"
    setter(int(float(gib) * (1 << 30)))
    return f"applied ({gib} GiB)"


def _install_vq_runtime(srv) -> None:
    """Route the server's `load` through knurlogic's VQ runtime for a rung
    rungs.json lists as VERIFIED (G-VQ: bit-identical to its published
    model.py). Every other artifact loads exactly as before, bundled
    model.py and all. The server imported `load` by name, so the name in
    its module is what gets swapped."""
    if getattr(srv.load, "_knurlogic", False):
        return
    real = srv.load

    def load(path_or_hf_repo, tokenizer_config=None, model_config=None,
             adapter_path=None, lazy=False, return_config=False, **kw):
        from pathlib import Path
        from knurlogic.engine.vq import runtime
        p = Path(str(path_or_hf_repo))
        if (adapter_path is None and not model_config and p.is_dir()
                and runtime.serves(p)):
            from mlx_lm.utils import load_tokenizer
            model, config = runtime.load_model(p, lazy=lazy)
            tok = load_tokenizer(p, tokenizer_config,
                                 eos_token_ids=config.get("eos_token_id"))
            _SERVED["runtime"] = "knurlogic"
            return (model, tok, config) if return_config else (model, tok)
        _SERVED["runtime"] = "bundled"
        return real(path_or_hf_repo, tokenizer_config=tokenizer_config,
                    model_config=model_config, adapter_path=adapter_path,
                    lazy=lazy, return_config=return_config, **kw)

    load._knurlogic = True
    srv.load = load


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
    _SERVED["path"] = model_path
    _SERVED["draft"] = draft
    _install_vq_runtime(srv)
    _real = srv.ModelProvider.load
    _real_init = srv.ModelProvider.__init__

    def _pinned(self, model_path_req=None, *a, **k):
        out = _real(self, _SERVED["path"], *a, **k)
        # The Family binds to a LOADED model, like the head below, and for
        # the same reason is re-run on every load: a switch may gain or lose
        # vision. Only when the model actually changed -- `load` is called
        # per request batch and returns the cached model otherwise.
        if _VISION.get("model") is not self.model:
            try:
                bind_vision(_SERVED["path"], self)
            except Exception as e:
                # A vision build that fails must not take the text model down
                # with it -- but it is said, never swallowed: /status.json
                # shows `vision_error`, and images get the 400 of a model
                # without vision rather than a guess.
                _VISION.update(serve=None, model=self.model,
                               error=f"{type(e).__name__}: {e}")
                _set_spec(None)
        # A drafting head binds to a LOADED model, and the provider loads in
        # the generator thread -- so this is the first moment one can exist.
        # Re-run on every load, because switching artifacts changes the
        # answer: the new one may ship a head, or may not.
        if _SERVED.get("draft", True) and _SERVED["path"]:
            try:
                if load_draft_head(_SERVED["path"]) is not None:
                    install_drafting(srv)
            except Exception as e:
                _DRAFT.update(on=False, why=f"{type(e).__name__}: {e}")
        return out

    def _init(self, *a, **k):
        _real_init(self, *a, **k)
        # Captured here rather than on first request: the provider is built
        # during server construction and `load_default()` runs inside the
        # generator thread, so waiting for a request means `switch()` has
        # nothing to talk to until someone has already used the server.
        _SERVED["provider"] = self

    srv.ModelProvider.load = _pinned
    srv.ModelProvider.__init__ = _init
    install_vision(srv)

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

    settings = settings or {}
    if "cache_limit_gb" in settings:
        print(f"cache limit {set_cache_limit(settings['cache_limit_gb'])}")
    sys.argv = [sys.argv[0]] + server_argv(
        model_path, host, port, executes_artifact_code, settings, extra)
    return srv.main()


#: The one model this process is serving, and the provider holding it. Only
#: `serve()` and `switch()` write here.
_SERVED: dict = {"path": None, "provider": None}


def served_path() -> str:
    return _SERVED["path"] or ""


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
    prov = _SERVED["provider"]
    if prov is None:
        raise RuntimeError("no model provider yet; the server is still "
                           "starting")
    from knurlogic.machine import loadlock

    before = _SERVED["path"]
    _SERVED["path"] = path
    try:
        with loadlock.model_load(str(path), "seam.switch"):
            prov.load(path)
    except BaseException:
        _SERVED["path"] = before          # leave the pin pointing at what is
        raise                             # actually loaded, not at a wish
    return {"loaded": path, "was": before}


def unload() -> dict:
    """Drop the weights, keep the server up.

    mlx-lm has no unload, so this is done through the provider's own fields:
    clearing `model_key` is what makes the next `load` actually reload rather
    than return the cached model. Dropping the references without clearing
    the key would leave a server that says it has a model and has none.
    """
    prov = _SERVED["provider"]
    if prov is None:
        raise RuntimeError("no model provider yet")
    had = _SERVED["path"]
    prov.model_key = None
    prov.model = None
    prov.tokenizer = None
    prov.draft_model = None
    clear_vision()
    try:
        import gc

        import mlx.core as mx
        gc.collect()
        mx.clear_cache()
    except Exception:
        pass
    return {"unloaded": had}


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
LIVE_KNOBS = ("VQ_DECODE_CHUNK", "VQLAB_CACHE_LIMIT_GB",
              "KNURLOGIC_CACHE_LIMIT_GB")


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
        if k in ("VQLAB_CACHE_LIMIT_GB", "KNURLOGIC_CACHE_LIMIT_GB"):
            try:
                said = set_cache_limit(v)
                if said.startswith("no "):
                    done[k] = said
                    continue
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


def keeps_mtp_weights(model_type: str) -> bool | None:
    """Does the architecture that will load this artifact keep its MTP head?

    Read off the module that will actually run, because the answer is a line
    in `sanitize()`:

        # Multi-token-prediction head ... not implemented by this text-only
        # port, and absent from the module tree -> drop them.
        if k.startswith(("mtp.", "model.mtp.")): continue

    So the head is in the checkpoint and is discarded at load. Nothing warns,
    and the weights were still downloaded. None means the module could not be
    located, which is not the same as "it keeps them".
    """
    import inspect

    from knurlogic.engine.arch import required_modules

    mods = required_modules(model_type)
    if not mods:
        return None
    try:
        from knurlogic.engine.register import source_for
        src_path, _is_pkg = source_for(mods[0])
        if src_path is None:
            from knurlogic.engine.arch import locate
            _host, src_path = locate(mods[0])
        if src_path is None:
            return None
        text = src_path.read_text()
    except Exception:
        return None
    dropped = 'k.startswith(("mtp.", "model.mtp."))' in text or \
        ('"mtp."' in text and "continue" in text)
    return not dropped


# --- drafting ---------------------------------------------------------------
# An artifact that ships a multi-token-prediction head carries weights mlx-lm
# will never run: it has no MTP path at all, and neither does upstream exo.
# knurlogic does, so a head beside the weights is simply used -- no flag, no
# environment variable, no mode file.

_DRAFT: dict = {"head": None, "spec": None, "why": "", "on": False,
                "requests": 0, "steps": 0, "accepted": 0,
                "batch_installed": False}


def drafting_status() -> dict:
    """What drafting is doing, for `/status.json` and the startup line.

    `engine_path` is here because it is the thing that explains a head that
    loaded and then never drafted. mlx-lm has two generators: a sequential
    one that calls `stream_generate`, and a BATCH one that does not. It picks
    the batch path whenever `is_batchable` -- which an MTP head does not
    affect, since that rule only asks about a separate draft MODEL. So an
    artifact whose caches can merge goes down a path the sequential swap
    never sees, and the only symptom is `requests: 0` next to `on: True`.
    """
    d = dict(_DRAFT)
    d.pop("head", None)
    spec = d.pop("spec", None)
    d["family"] = getattr(spec, "name", "")
    d["acceptance"] = (d["accepted"] / d["steps"]) if d["steps"] else None
    prov = _SERVED.get("provider")
    batchable = getattr(prov, "is_batchable", None) if prov else None
    d["batchable"] = batchable
    d["engine_path"] = ("batch" if batchable
                        else "sequential" if batchable is False else "")
    batch_on = bool(d.pop("batch_installed", False))
    if d["on"] and batchable and not batch_on:
        d["drafts_now"] = False
        d["blocked"] = ("this artifact's caches merge, so mlx-lm serves it "
                        "with the batch generator, and the drafting batch "
                        "generator was not installed. The head is loaded and "
                        "idle.")
    else:
        d["drafts_now"] = bool(d["on"])
        d["blocked"] = ""
    return d


def load_draft_head(model_path: str):
    """Load the drafting head beside this artifact, if there is one.

    Absent is the ordinary case and not an error: sidecars are named outside
    mlx-lm's `model*.safetensors` glob precisely so a directory carrying one
    still loads normally through the stock loader.
    """
    from knurlogic.engine.mtp import find_head

    found = find_head(model_path)
    if found is None:
        _DRAFT.update(on=False, why="no drafting head beside the weights")
        return None
    prov = _SERVED.get("provider")
    model = getattr(prov, "model", None) if prov else None
    if model is None:
        _DRAFT.update(on=False, why="model not loaded yet")
        return None
    try:
        from knurlogic.engine.mtp.loop import load_mtp_head
        head, spec = load_mtp_head(model, sidecar=found.path)
    except Exception as e:
        # A head that will not bind is a fact worth printing, not a crash:
        # the model serves perfectly well without one.
        _DRAFT.update(on=False, why=f"{type(e).__name__}: {e}")
        return None
    _DRAFT.update(head=head, spec=spec, on=True,
                  why=f"{found.path.name}, {found.gib:.2f} GiB")
    return head


def install_drafting(srv) -> bool:
    """Route the engine's own generation through the drafting loop.

    mlx-lm's server calls `stream_generate(...)` once per request and reads
    `.text`, `.token`, `.logprobs` and `.finish_reason` off what it yields.
    `mtp_stream_generate` yields all four, so this is a swap rather than a
    reimplementation -- knurlogic still does not have a second inference
    path, which is the rule this package is built on.

    THE SAMPLING PARAMETERS ARE THE AWKWARD PART. mlx-lm hands
    `stream_generate` a BUILT sampler, and the drafting loop needs the
    parameters themselves: verification is rejection sampling against the
    target distribution, so a callable that has already collapsed it is no
    use. They are available one frame up, in `_serve_single`, so that is
    wrapped to put them on a thread-local. Per request, per thread, and the
    server serves each request on its own thread.
    """
    if not _DRAFT.get("on"):
        return False

    import threading

    from knurlogic.engine.mtp.loop import mtp_stream_generate

    local = threading.local()
    real_single = srv.ResponseGenerator._serve_single
    real_stream = srv.stream_generate

    def _serve_single(self, request):
        local.args = request[2]
        try:
            return real_single(self, request)
        finally:
            local.args = None

    def _stream(model, tokenizer, prompt, **kw):
        args = getattr(local, "args", None)
        head = _DRAFT.get("head")
        # A draft model and a drafting head are two different mechanisms and
        # stacking them is not defined; the explicit one wins.
        if head is None or args is None or kw.get("draft_model") is not None:
            yield from real_stream(model=model, tokenizer=tokenizer,
                                   prompt=prompt, **kw)
            return
        s = args.sampling
        _DRAFT["requests"] += 1
        last = None
        for r in mtp_stream_generate(
                model, tokenizer, prompt, head,
                max_tokens=kw.get("max_tokens", 256),
                temp=s.temperature, top_p=s.top_p, top_k=s.top_k,
                min_p=s.min_p, xtc_probability=s.xtc_probability,
                xtc_threshold=s.xtc_threshold,
                logits_processors=kw.get("logits_processors"),
                prefill_step_size=kw.get("prefill_step_size", 2048),
                prompt_cache=kw.get("prompt_cache"),
                want_logprobs=True):
            last = r
            if r.tail:          # detokenizer flush, not a new token
                continue
            yield r
        if last is not None:
            _DRAFT["steps"] += last.steps
            _DRAFT["accepted"] += last.accepted

    srv.ResponseGenerator._serve_single = _serve_single
    srv.stream_generate = _stream
    _install_batch_drafting(srv)
    return True


def _install_batch_drafting(srv) -> None:
    """The batch half: hand the server a drafting BatchGenerator.

    The server constructs `BatchGenerator(model, ...)` by the name it
    imported, once per batch, so swapping that name is the whole
    installation. It decides per construction: with no head bound to THIS
    model (a switch to an artifact without one, or `--no-draft`) the server
    gets mlx-lm's own, untouched.
    """
    if getattr(srv.BatchGenerator, "_knurlogic", False):
        return
    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator, tag_samplers

    tag_samplers(srv)
    real = srv.BatchGenerator

    def _factory(model, *a, **kw):
        head = _DRAFT.get("head")
        prov = _SERVED.get("provider")
        ours = getattr(prov, "model", None) is model
        vision = _VISION.get("serve") if (
            ours and _VISION.get("model") is model) else None
        if head is None or not _DRAFT.get("on") or not ours:
            head = None
        if head is None and vision is None:
            return real(model, *a, **kw)
        if head is not None:
            try:
                return MTPBatchGenerator(model, head, stats=_DRAFT,
                                         vision=vision, *a, **kw)
            except Exception as e:
                _DRAFT.update(why=f"batch drafting refused: "
                                  f"{type(e).__name__}: {e}")
                if vision is None:
                    return real(model, *a, **kw)
        # Design D5: a vision model's batch is ALWAYS this engine, head or
        # not -- only its admit snaps prefill chunks to image blocks, and
        # only it reads a key. No fallback to mlx-lm's here: that one would
        # feed sentinels to mx.array.
        return MTPBatchGenerator(model, None, stats=_VISION_STATS,
                                 vision=vision, *a, **kw)

    _factory._knurlogic = True
    srv.BatchGenerator = _factory
    _DRAFT["batch_installed"] = True


# --- vision (design D3, D5, D6; docs/design/vision.md) ------------------------
# Images reach the model through three wraps of mlx-lm's server and nothing
# else:
#
#   APIHandler.do_POST          (HTTP thread) images sent to a model with no
#                               vision -> 400. Nothing else happens here.
#   ResponseGenerator.generate  (HTTP thread) a request with image parts is
#                               TAGGED (args._knurlogic_images); no decoding.
#   ResponseGenerator._is_batchable   a tagged request is always batchable,
#                               so it goes through the batch engine (D5).
#   ResponseGenerator._tokenize (GENERATOR thread) all image work, returns
#                               the cache key as the prompt (D3).
#
# plus the BatchGenerator factory above, which hands a vision model
# MTPBatchGenerator whether or not it has a head.
#
# Every wrapped method is resolved BY NAME with an assertion (design D2): a
# pinned mlx-lm that renames one fails at install, loudly, instead of a
# wrap that silently never runs. tests/test_pins.py pins server.py's digest.

#: What is served: the VisionServe (family + store + pins) and the model it
#: is bound to. Only bind_vision / clear_vision write here.
_VISION: dict = {"serve": None, "model": None, "error": ""}
#: Row counts for a headless vision batch (the drafting one uses _DRAFT).
_VISION_STATS: dict = {}

#: (owner attribute on srv, method name) for every mlx-lm method wrapped
#: for vision -- one list, so the install and the tests agree on it.
VISION_WRAPS = (("APIHandler", "do_POST"),
                ("ResponseGenerator", "generate"),
                ("ResponseGenerator", "_is_batchable"),
                ("ResponseGenerator", "_tokenize"))

#: Paths whose body is a chat request that may carry images.
_CHAT_PATHS = ("/v1/chat/completions", "/chat/completions")


def _method(srv, owner: str, name: str):
    cls = getattr(srv, owner, None)
    fn = getattr(cls, name, None) if cls is not None else None
    assert callable(fn), (
        f"mlx_lm.server.{owner}.{name} is gone: the vision serve path wraps "
        f"it by name. The pinned mlx-lm changed; re-read server.py before "
        f"re-pinning (tests/test_pins.py).")
    return cls, fn


def _set_spec(spec) -> None:
    from knurlogic.engine.vision import set_served_vision
    set_served_vision(spec)


def served_vision():
    """The served model's VisionSpec, or None. The same answer as
    engine.vision.served_vision(), which interfaces/ reads without
    importing this module (critique C4)."""
    from knurlogic.engine.vision import served_vision as _sv
    return _sv()


def vision_status() -> dict:
    """For /status.json: the spec, the store and the pins, or why not."""
    v = _VISION.get("serve")
    out = {"on": v is not None, "error": _VISION.get("error", "")}
    if v is not None:
        out.update(spec=v.spec.to_json(), store=v.store.stats(),
                   encodes=v.encodes, pinned=v.pinned_count())
    return out


def bind_vision(model_path: str, provider, *, store_bytes: int | None = None):
    """Build the loaded model's Family through the registry and serve it.

    Called after every load. model_type and config come from the artifact's
    config.json; `registry.build` answers None for a model without vision
    (unregistered type, no vision_config, family package absent), and that
    is the ordinary case, not an error. The store is keyed by mlx-lm's
    `model_key`, the same key its prompt cache uses."""
    import json
    from pathlib import Path

    from knurlogic.engine.vision import registry
    from knurlogic.engine.vision.request import VisionServe
    from knurlogic.engine.vision.store import DEFAULT_MAX_BYTES, ImageStore

    clear_vision()
    cfg_path = Path(str(model_path)) / "config.json"
    config = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
    fam = registry.build(config.get("model_type", ""), str(model_path),
                         provider.model, config)
    _VISION.update(model=provider.model, error="")
    if fam is None:
        return None
    n = fam.load_weights(str(model_path))
    serve = VisionServe(fam, ImageStore(store_bytes or DEFAULT_MAX_BYTES),
                        provider.model_key)
    serve.tensors = int(n)
    _VISION["serve"] = serve
    _set_spec(fam.spec)
    return serve


def clear_vision() -> None:
    """Drop the family and its store (features AND refs: the prompt cache
    they index dies with the model), and say no vision is served."""
    v = _VISION.get("serve")
    if v is not None:
        v.store.clear()
    _VISION.update(serve=None, model=None, error="")
    _set_spec(None)


def _refuse(handler, code: int, msg: str) -> None:
    import json

    body = json.dumps({"error": msg}).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def install_vision(srv) -> None:
    """Wrap mlx-lm's server for images. Idempotent; a server whose model
    has no vision behaves exactly as before -- every wrap passes straight
    through when no image part is present (G5)."""
    import io
    import json

    from knurlogic.engine.vision import VisionError
    from knurlogic.engine.vision import request as vreq

    Handler, real_post = _method(srv, "APIHandler", "do_POST")
    RG, real_generate = _method(srv, "ResponseGenerator", "generate")
    _, real_batchable = _method(srv, "ResponseGenerator", "_is_batchable")
    _, real_tokenize = _method(srv, "ResponseGenerator", "_tokenize")
    if getattr(real_tokenize, "_knurlogic_vision", False):
        return

    def do_POST(self):
        # Design D3: `_post` only refuses. The body is read to look for
        # image parts and put back for mlx-lm to read again.
        if self.path.split("?")[0] not in _CHAT_PATHS:
            return real_post(self)
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return real_post(self)           # mlx-lm answers that one
        raw = self.rfile.read(n) if n else b""
        self.rfile = io.BytesIO(raw)
        try:
            body = json.loads(raw.decode() or "null")
        except (ValueError, UnicodeDecodeError):
            body = None
        if (isinstance(body, dict)
                and vreq.has_images(body.get("messages"))
                and _VISION.get("serve") is None):
            why = _VISION.get("error") or "the served model has no vision"
            return _refuse(self, 400, f"this request has images but "
                                      f"{why}; send text only")
        return real_post(self)

    def generate(self, request, generation_args, *a, **kw):
        if vreq.has_images(getattr(request, "messages", None)):
            vreq.tag(generation_args)
        return real_generate(self, request, generation_args, *a, **kw)

    def _is_batchable(self, args):
        # D5: a request with an image is served by the batch engine even
        # when it carries a seed (which mlx-lm serves sequentially): the
        # sequential path chunks prefill with no hook and cannot read a key.
        if vreq.tagged(args):
            return True
        return real_batchable(self, args)

    def _tokenize(self, tokenizer, request, args):
        from knurlogic.engine.vision import cachehook
        from knurlogic.engine.vision import key as K
        # Release pins a request took and never handed to the batch engine
        # (it raised between tokenize and insert). Every request, text too.
        cachehook.sweep()
        if not vreq.has_images(getattr(request, "messages", None)):
            return real_tokenize(self, tokenizer, request, args)
        v = _VISION.get("serve")
        if v is None:
            # Raced a switch to a text-only model after _post let it in.
            raise VisionError("the served model has no vision")
        out = v.tokenize(real_tokenize, self, tokenizer, request, args)
        cachehook.pending(v, K.images_in(out[0]))
        return out

    for fn in (do_POST, generate, _is_batchable, _tokenize):
        fn._knurlogic_vision = True
    # The pin refcount, installed HERE with the _tokenize wrap that feeds it
    # and never separately: pending() without install_admit() lets a later
    # sweep release pins a queued row still owns. Images a cached
    # conversation references stay pinned (Flash-Next review point 1).
    from knurlogic.engine.vision import cachehook
    cachehook.install(srv, lambda: (_VISION["serve"].store
                                    if _VISION.get("serve") else None))
    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    cachehook.install_admit(MTPBatchGenerator)
    Handler.do_POST = do_POST
    RG.generate = generate
    RG._is_batchable = _is_batchable
    RG._tokenize = _tokenize
    _install_batch_drafting(srv)
