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


def generate(model, tokenizer, prompt: str, max_tokens: int = 8) -> str:
    from mlx_lm.generate import generate as _generate

    return _generate(model, tokenizer, prompt=prompt,
                     max_tokens=max_tokens, verbose=False) or ""
