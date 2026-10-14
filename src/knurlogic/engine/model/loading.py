"""Loading and the engine's own knobs: what runs, how to load it, how much
memory it has, the server's argv, the cache limit, and what can change on a
running process.

A model is built by knurlogic's copy of mlx-lm's load_model
(load_model.py), which opens the weight files through weight_files.py and
always takes `trust_remote_code` (a VQ artifact runs its own model.py).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

from knurlogic.engine import backend
from knurlogic.machine.units import GIB

#: Packages that can host a model architecture, in lookup order.
#: Where an architecture module is looked up. Only mlx_lm's namespace: every
#: family registers there, GLM included (mlx-vlm is not needed).
HOST_PACKAGES = ("mlx_lm",)


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
    except (ImportError, OSError):
        return EngineInfo(host, "-", False)


def describe() -> str:
    return " | ".join(str(info(h)) for h in HOST_PACKAGES)


def memory() -> dict:
    try:
        import mlx.core as mx
    except ImportError:
        return {"available": False}
    try:
        info = backend.device_info()
        ws = int(info.get("max_recommended_working_set_size", 0))
        total = int(info.get("memory_size", ws))
        device = info.get("device_name", "unknown")
    except (AttributeError, TypeError, ValueError, RuntimeError):
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


def process_memory() -> dict:
    """This process's memory as /status.json reports it: the framework's
    accounting (memory()) plus the OS's resident size, scope "process"."""
    import os

    from knurlogic.machine import processes
    d = memory()
    d["process_rss_bytes"] = processes.rss(os.getpid())
    d["scope"] = "process"      # see `scope` in machine/status.aggregate
    return d


def models_module(host: str = "mlx_lm"):
    """The package a model architecture is looked up in."""
    return importlib.import_module(f"{host}.models")


def load(path: str, executes_artifact_code: bool = False):
    """Load a model and tokenizer, under the machine's load lock.

    `executes_artifact_code` says the artifact ships its own runtime, which
    WILL be executed. It is a separate argument rather than something inferred
    quietly, so a caller has to state it and can say so to the user.
    """
    from knurlogic.machine import loadlock

    # The box is shared: one model load at a time, across processes
    # (machine/loadlock.py; vision-contracts.md "Load lock" names this caller).
    with loadlock.model_load(str(path), "model.loading"):
        return load_unlocked(path, executes_artifact_code)


def load_unlocked(path: str, executes_artifact_code: bool = False,
                  lazy: bool = False, model_config: dict | None = None):
    """`load` for a caller already holding the load lock (the model host).
    A VQ model always runs its OWN bundled model.py, the runtime it ships;
    knurlogic carries no VQ runtime of its own. `state.SERVED["runtime"]`
    says "bundled"."""
    from pathlib import Path

    from mlx_lm.utils import load_tokenizer

    from knurlogic.engine import templates

    from . import state
    from .load_model import load_model
    p = Path(str(path))
    why = vq_without_runtime(p)
    if why:
        raise RuntimeError(why)
    overlay = long_context_overlay(p)
    state.SERVED["long_context"] = "yarn" if overlay else "off"
    text = text_only_overlay(p)
    if text:
        overlay = {**(overlay or {}), **text}
    state.SERVED["runtime"] = "bundled"
    # knurlogic's copy of mlx-lm's load_model: the weight files are opened
    # by engine/model/weight_files.py (a split rank may stream them)
    cfg = {**(overlay or {}), **(model_config or {})} or None
    model, config = load_model(p, lazy, model_config=cfg,
                               trust_remote_code=executes_artifact_code)
    tok = load_tokenizer(p, None,
                         eos_token_ids=config.get("eos_token_id", None))
    templates.install(tok)
    return model, tok


def text_only_overlay(p) -> dict:
    """The config overlay that builds a vision model's TEXT model when its
    weights carry none of the vision tensors its config.json describes
    (engine/vision/registry.vision_weights decides; e.g. DeepSeek-V4-Flash-
    Vision-Exp converted without its tower: vision_n_layers built as 0),
    or {}. Raises for a conversion with only part of them."""
    import json
    from pathlib import Path

    from knurlogic.engine.vision import registry
    try:
        cfg = json.loads((Path(str(p)) / "config.json").read_text())
    except (OSError, ValueError):
        return {}
    v = registry.vision_weights(cfg, p)
    if v["state"] == "partial":
        raise RuntimeError(v["why"])
    return v["text_config"] if v["state"] == "text_only" else {}


def vq_without_runtime(p) -> str:
    """Why a VQ artifact cannot load, or "": it declares VQ modules but
    ships no model.py to run them."""
    import json
    from pathlib import Path

    p = Path(str(p))
    try:
        cfg = json.loads((p / "config.json").read_text())
    except (OSError, ValueError):
        return ""
    if not any(cfg.get(k) for k in ("vq_modules", "vq_linear", "vq_embed")):
        return ""
    if (p / str(cfg.get("model_file") or "model.py")).is_file():
        return ""
    return ("this VQ model does not ship its runtime (model.py); "
            "re-download it")


def long_context_overlay(path, env=None) -> dict:
    """The config overlay KNURLOGIC_LONG_CONTEXT asks for (context_window.
    long_context_config): {} when off. Read from the environment the
    launch set (serve applies a model's launch settings there, on every
    rank of a split), and applied as mlx-lm's `model_config` -- the
    artifact's config.json is never written."""
    import json
    from pathlib import Path

    from knurlogic.tuning import context_window, knobs
    mode = context_window.long_context_of(
        knobs.env_value("KNURLOGIC_LONG_CONTEXT", env))
    if mode == "off":
        return {}
    cfg = json.loads((Path(str(path)) / "config.json").read_text())
    return context_window.long_context_config(cfg, mode)


def set_cache_limit(gib: float) -> str:
    """Bound mlx's freed-buffer cache for this process. mlx-lm never sets
    it, so without this the knob is a number on a page."""
    import mlx.core as mx
    setter = getattr(mx, "set_cache_limit", None) or \
        getattr(getattr(mx, "metal", None), "set_cache_limit", None)
    if setter is None:
        return "no cache-limit setter in this engine build"
    setter(int(float(gib) * GIB))
    return f"applied ({gib} GiB)"


def generate(model, tokenizer, prompt: str, max_tokens: int = 8) -> str:
    from mlx_lm.generate import generate as _generate

    return _generate(model, tokenizer, prompt=prompt,
                     max_tokens=max_tokens, verbose=False) or ""


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

    from knurlogic.tuning import knobs

    done = {}
    for k, v in env.items():
        logical = knobs.logical_of(k)
        if logical == "cache_limit_gb":
            try:
                said = set_cache_limit(v)
                if said.startswith("no "):
                    done[k] = said
                    continue
                os.environ[k] = str(v)
                done[k] = f"applied now ({v} GiB)"
            except (ValueError, TypeError, AttributeError, RuntimeError, OSError) as e:
                done[k] = f"failed: {e}"
        elif logical == "thinking_default":
            # engine/model/thinking reads it at every request
            try:
                v = knobs.thinking_default_of(v)
            except ValueError as e:
                done[k] = f"failed: {e}"
                continue
            os.environ[k] = v
            done[k] = ("applied: a request that names no level gets "
                       + ("the model's own default" if v == "model"
                          else f"its {v} level"))
        elif logical == "context_length":
            # the scheduler reads it at every admission
            try:
                v = int(v)
                if v <= 0:
                    raise ValueError(v)
            except (TypeError, ValueError):
                done[k] = f"failed: {v!r} is not a positive number of tokens"
                continue
            os.environ[k] = str(v)
            done[k] = f"applied: requests from now on are capped at {v} tokens"
        elif logical == "decode_chunk":
            try:
                v = int(v)
            except (TypeError, ValueError):
                done[k] = f"failed: {v!r} is not a whole number of tokens"
                continue
            mods = _artifact_runtime_modules()
            if not mods:
                # Before the first prefill the global does not exist yet, but
                # the environment is still what the runtime will read.
                os.environ[k] = str(v)
                done[k] = "set for the next prefill (runtime not resolved yet)"
                continue
            for _name, mod in mods:
                mod._DECODE_CHUNK = int(v)
            os.environ[k] = str(v)
            done[k] = (f"applied to {len(mods)} loaded runtime"
                       f"{'s' if len(mods) > 1 else ''}; takes effect on the "
                       f"next prefill")
        else:
            done[k] = "needs a restart: read at import and compiled into the "\
                      "kernel"
    return done


class _TemplateOnly:
    """What mlx-lm 0.32's `_infer_tool_parser(tokenizer)` reads, for a
    caller holding only the template (doctor reads the artifact, it loads
    no tokenizer): the template, and no vocab for its vocab fallback. 0.31
    took the template string itself."""

    def __init__(self, chat_template: str):
        self.chat_template = chat_template

    def get_vocab(self) -> dict:
        return {}


def tool_support(chat_template: str) -> dict:
    """Which tool-call dialect an artifact speaks, and whether knurlogic can read it.

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
    out: dict = {"has_template": bool(chat_template), "parser": None,
           "mentions_tools": "tool" in (chat_template or "").lower()}
    if not chat_template:
        return out
    try:
        from mlx_lm.tokenizer_utils import _infer_tool_parser
    except ImportError:
        out["parser"] = "unknown (this engine exposes no inference rule)"
        return out
    try:
        out["parser"] = _infer_tool_parser(_TemplateOnly(chat_template))
    except (ValueError, TypeError, KeyError, AttributeError, RuntimeError):
        out["parser"] = None
    from knurlogic.engine import templates
    fam = templates.served_family(chat_template)
    if fam in templates.PARSERS:
        # knurlogic's own template and parser (engine/templates)
        out["parser"] = f"{fam} (knurlogic)"
        out["mentions_tools"] = True
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
    except (ImportError, OSError, ValueError, AttributeError, TypeError):
        return None
    dropped = 'k.startswith(("mtp.", "model.mtp."))' in text or \
        ('"mtp."' in text and "continue" in text)
    return not dropped
