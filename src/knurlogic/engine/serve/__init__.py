"""engine/serve/ -- the one place that knows what runs a model, and every
change knurlogic makes to mlx-lm's own server, one module each.

  server.py        serve / switch / unload: pin the served model, install
                   the changes below, add knurlogic's routes, hand over
  load.py          engine info, load, memory, the server's argv, the cache
                   limit, knobs a running process can change, tool dialects
  state.py         what is served, drafting, vision: the process's dicts
  vq_runtime.py    verified rungs load on knurlogic's VQ runtime (engine/vq)
  cache_report.py  usage.knurlogic.cache: what the prompt cache actually did
  cache_guard.py   an exact prompt-cache hit still leaves a token to process
                   (mlx-lm's batch path crashes on an empty remainder)
  thinking.py      reasoning_effort -> each chat template's own controls
  sampling.py      a request's seed reaches the sampler (mlx-lm's compiled
                   sampler ignores it off the main thread)
  drafting.py      an artifact's MTP head in mlx-lm's server (engine/mtp)
  vision.py        images through mlx-lm's server (engine/vision)

Callers import the package and use the names below; which module holds a
name is this package's business. Importing it imports no mlx -- only
calling into it does.

Today the engine is mlx-lm (and mlx-vlm for multimodal architectures). That
may not always be true, and the cost of keeping the option open is exactly
this package: every other module asks here instead of importing mlx directly.
Knurlogic's own code is ~1000 lines against ~3900 lines of vendored
architectures and a 4500-line runtime shipped inside each artifact -- so the
package itself is nearly uncoupled, and the job is to keep it that way rather
than to abstract anything clever.

WHAT A REPLACEMENT WOULD HAVE TO HONOUR, because these are not this package's
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

from .load import (HOST_PACKAGES, LIVE_KNOBS, EngineInfo, apply_live,
                   describe, generate, info, keeps_mtp_weights, load, memory,
                   models_module, server_argv, set_cache_limit, tool_support)
from .server import serve, switch, unload
from .state import served_path
from .drafting import drafting_status, install as install_drafting, \
    load_head as load_draft_head
from .thinking import status as thinking_status
from .vision import (VISION_WRAPS, bind as bind_vision, clear as clear_vision,
                     install as install_vision, served_vision, vision_status)

__all__ = [
    "HOST_PACKAGES", "LIVE_KNOBS", "EngineInfo", "VISION_WRAPS", "apply_live",
    "bind_vision", "clear_vision", "describe", "drafting_status", "generate",
    "info", "install_drafting", "install_vision", "keeps_mtp_weights", "load",
    "load_draft_head", "memory", "models_module", "serve", "served_path",
    "served_vision", "server_argv", "set_cache_limit", "switch",
    "thinking_status",
    "tool_support", "unload", "vision_status",
]
