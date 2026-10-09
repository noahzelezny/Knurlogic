"""engine/serve/ -- what the served model is and what it can do: the pieces
knurlogic's own server (engine/runtime, interfaces/http) builds on, one
module each: load.py (load, memory, knobs, tool dialects), state.py,
segments.py, thinking.py, drafting.py, vision.py.

Callers import the package and use the names below; which module holds a
name is this package's business. Importing it imports no mlx -- only
calling into it does. Every other module asks here instead of importing
mlx directly, which keeps the engine replaceable.

Version skew is this package's job: mlx-lm 0.31.3 executes `model_file`
unconditionally, 0.32.0 requires `trust_remote_code=` and raises without
it; the signature is inspected here so nobody downstream learns that.
Design: docs/design/server.md (engine boundary).
"""

from .drafting import drafting_status
from .drafting import load_head as load_draft_head
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
    "bind_vision", "clear_vision", "describe", "drafting_status", "generate",
    "info", "keeps_mtp_weights", "load", "load_draft_head", "memory",
    "models_module", "served_path", "served_vision", "set_cache_limit",
    "thinking_status", "tool_support", "vision_status",
]
