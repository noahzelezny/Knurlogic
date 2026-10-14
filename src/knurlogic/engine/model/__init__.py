"""engine/model/ -- the model this process serves and what it can do: the
pieces knurlogic's own server (engine/runtime, interfaces/http) builds on.

  load.py      load it, its memory (and this process's, process_memory),
               the engine's knobs, tool dialects
  load_model.py  mlx-lm's load_model, opening weights through weight_files
  weight_files.py  the one place an artifact's tensor files are opened
  state.py     what is served and its vision (process state, by attribute)
  segments.py  the system prompt's own prompt-cache segment
  thinking.py  reasoning_effort, translated to each chat template's own
  vision.py    bind a vision family at load, clear it at unload

Drafting is engine/mtp/binding.py.

Callers outside the engine import the package and use the names below;
which module holds a name is this package's business. Importing it imports
no mlx -- only calling into it does. (Engine code itself imports mlx
where it computes; the boundary is engine/, see engine/__init__.py.)
`load` here is the function, which shadows the load module as a package
attribute: reach the module with `from knurlogic.engine.model.load import`.

Version skew is this package's job: mlx-lm 0.31.3 executes `model_file`
unconditionally, 0.32.0 requires `trust_remote_code=` and raises without
it; the signature is inspected here so nobody downstream learns that.
Design: docs/design/server.md (engine boundary).
"""

from .load import (
                   HOST_PACKAGES,
                   EngineInfo,
                   apply_live,
                   describe,
                   generate,
                   info,
                   keeps_mtp_weights,
                   load,
                   memory,
                   models_module,
                   process_memory,
                   set_cache_limit,
                   tool_support,
)
from .state import served_path
from .thinking import status as thinking_status
from .vision import bind as bind_vision
from .vision import clear as clear_vision
from .vision import served_vision, vision_status

__all__ = [
    "HOST_PACKAGES", "EngineInfo", "apply_live",
    "bind_vision", "clear_vision", "describe", "generate",
    "info", "keeps_mtp_weights", "load", "memory",
    "models_module", "process_memory", "served_path", "served_vision",
    "set_cache_limit",
    "thinking_status", "tool_support", "vision_status",
]
