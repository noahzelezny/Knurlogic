"""Loading and the engine's own knobs: what runs, how to load it, how much
memory it has, the server's argv, the cache limit, and what can change on a
running process.

VERSION SKEW IS THIS FILE'S JOB. mlx-lm 0.31.3 executes `model_file`
unconditionally; 0.32.0 put it behind `trust_remote_code=` and raises without
it. Passing the kwarg blindly is a TypeError on one, omitting it a ValueError
on the other, so a VQ artifact cannot load on both unless something inspects
the signature. Nobody downstream should ever learn that.
"""

from __future__ import annotations

import importlib
import inspect
import subprocess
from dataclasses import dataclass

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
        info = (mx.device_info() if hasattr(mx, "device_info")
                else mx.metal.device_info())
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


def gpu_in_use() -> int | None:
    """Bytes of GPU memory in use on this Mac by EVERY process (the IOGPU
    driver's "In use system memory"), or None where it cannot be read.
    iogpu.wired_limit_mb caps this total, not one process's share: the
    27B on an M3 Ultra (96 GB) aborted Metal with its own peak under
    the working set while other processes held 2.9 GiB of it."""
    import re
    try:
        out = subprocess.run(["ioreg", "-r", "-c", "IOAccelerator", "-d1",
                              "-w0"], capture_output=True, text=True,
                             timeout=2).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r'"In use system memory"=(\d+)', out)
    return int(m.group(1)) if m else None


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
    with loadlock.model_load(str(path), "serve.load"):
        return load_unlocked(path, executes_artifact_code)


def load_unlocked(path: str, executes_artifact_code: bool = False,
                  lazy: bool = False):
    """`load` for a caller already holding the load lock (the model host).
    A VQ model always runs its OWN bundled model.py, the runtime it ships;
    knurlogic carries no VQ runtime of its own. `state.SERVED["runtime"]`
    says "bundled"."""
    from pathlib import Path

    from mlx_lm.utils import load as _load

    from knurlogic.engine import templates

    from . import state
    p = Path(str(path))
    why = vq_without_runtime(p)
    if why:
        raise RuntimeError(why)
    overlay = long_context_overlay(p)
    state.SERVED["long_context"] = "yarn" if overlay else "off"
    state.SERVED["runtime"] = "bundled"
    kw: dict = {"lazy": True} if lazy else {}
    if overlay:
        kw["model_config"] = overlay
    if executes_artifact_code and \
            "trust_remote_code" in inspect.signature(_load).parameters:
        kw["trust_remote_code"] = True
    model, tok = _load(path, **kw)
    templates.install(tok)
    return model, tok


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
    """The config overlay KNURLOGIC_LONG_CONTEXT asks for (settings.
    long_context_config): {} when off. Read from the environment the
    launch set (serve applies a model's launch settings there, on every
    rank of a split), and applied as mlx-lm's `model_config` -- the
    artifact's config.json is never written."""
    import json
    import os
    from pathlib import Path

    from knurlogic.tuning import settings as S
    mode = S.long_context_of((os.environ if env is None else env).get(
        "KNURLOGIC_LONG_CONTEXT"))
    if mode == "off":
        return {}
    cfg = json.loads((Path(str(path)) / "config.json").read_text())
    return S.long_context_config(cfg, mode)


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
#:   * VQ_CACHE_LIMIT_GB (and its old names) is applied through the framework's own live
#: API.
#:   * the eight GEMM/numerics flags are read into module globals AT IMPORT and
#:     baked into Metal kernel source that is compiled once. Those genuinely
#:     need a restart, or an override module that reads them per dispatch.
LIVE_KNOBS = ("VQ_DECODE_CHUNK", "VQ_CACHE_LIMIT_GB", "VQLAB_CACHE_LIMIT_GB",
              "KNURLOGIC_CACHE_LIMIT_GB", "KNURLOGIC_CONTEXT_LENGTH")


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
        if k in ("VQ_CACHE_LIMIT_GB", "VQLAB_CACHE_LIMIT_GB",
                 "KNURLOGIC_CACHE_LIMIT_GB"):
            try:
                said = set_cache_limit(v)
                if said.startswith("no "):
                    done[k] = said
                    continue
                os.environ[k] = str(v)
                done[k] = f"applied now ({v} GiB)"
            except (ValueError, TypeError, AttributeError, RuntimeError, OSError) as e:
                done[k] = f"failed: {e}"
        elif k == "KNURLOGIC_CONTEXT_LENGTH":
            # the scheduler reads it at every admission
            try:
                v = int(v)
                assert v > 0
            except (TypeError, ValueError, AssertionError):
                done[k] = f"failed: {v!r} is not a positive number of tokens"
                continue
            os.environ[k] = str(v)
            done[k] = f"applied: requests from now on are capped at {v} tokens"
        elif k == "VQ_DECODE_CHUNK":
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
        out["parser"] = _infer_tool_parser(chat_template)
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
