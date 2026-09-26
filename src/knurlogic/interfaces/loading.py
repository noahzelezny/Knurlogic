"""What a model must pass before it is loaded -- at startup and on every
switch (the page's load, POST /v1/ensure), the same checks:

  known     it is an artifact this machine's stores hold (machine/discover)
            or the one already served. A client names a model; it never
            names an arbitrary directory, so a request cannot make the
            server execute a `model.py` from wherever it likes.
  runnable  its architecture registered from knurlogic's vendored set, and
            every module it needs present.
  fits      its weights fit the memory it would have: the load budget
            (machine/wired.load_budget), plus what unloading the current
            model frees.

A refusal is a `NotLoadable` with an HTTP status and a message that names
the cause and what to do about it.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

GIB = 1 << 30
_KNOWN: dict = {"at": 0.0, "rows": None}


class NotLoadable(Exception):
    def __init__(self, status: int, message: str, code: str = ""):
        super().__init__(message)
        self.status, self.code = status, code


def known_artifacts(ttl: float = 60.0) -> list:
    """The servable artifacts in this machine's stores (cached: a scan
    walks every store and takes about a second)."""
    now = time.time()
    if _KNOWN["rows"] is None or now - _KNOWN["at"] > ttl:
        from knurlogic.machine import discover
        try:
            _KNOWN["rows"] = [f for f in discover.find()
                              if f.servable and f.format == "mlx"]
        except Exception:
            _KNOWN["rows"] = []
        _KNOWN["at"] = now
    return _KNOWN["rows"]


def resolve_name(model: str, served: Optional[str]) -> str:
    """A model id (directory name) or path -> the artifact directory, if it
    is the served one or a known one. NotLoadable(404) otherwise."""
    model = (model or "").strip()
    if served and model in ("", Path(served).name, served):
        return served
    try:
        want = str(Path(model).expanduser().resolve())
    except (OSError, RuntimeError):
        want = model
    for f in known_artifacts():
        p = str(Path(f.path).resolve())
        if model in (f.name, Path(f.path).name) or want == p:
            return str(f.path)
    raise NotLoadable(404, f"{model!r} is not an artifact in this machine's "
                           f"model stores. /models.json lists the ones that "
                           f"are; a model is named by its id there, not by "
                           f"an arbitrary path.", "model_not_found")


def register(artifact) -> list:
    """Register the artifact's vendored architecture modules; the names of
    any it still lacks (empty when it can load)."""
    from knurlogic.engine import arch, register as reg
    needed = arch.modules_for_artifact(artifact)
    if needed:
        reg.register(*needed)
    return [r.module for r in arch.check(artifact.model_type)
            if not r.present]


def prepare(model: str, *, served: Optional[str] = None,
            freed_bytes: int = 0):
    """-> the Artifact to load, or NotLoadable saying why not."""
    from knurlogic.machine import wired
    from knurlogic.machine.artifact import Artifact
    path = resolve_name(model, served)
    try:
        a = Artifact.load(path)
    except Exception as e:
        raise NotLoadable(422, f"{Path(path).name} could not be read as an "
                               f"artifact: {e}", "bad_artifact")
    missing = register(a)
    if missing:
        raise NotLoadable(422, f"{a.path.name} needs {missing}, which "
                               f"neither knurlogic's vendored set nor the "
                               f"installed engine provides.",
                          "unsupported_architecture")
    b = wired.load_budget()
    ws, avail = b["working_set_bytes"], b["available_bytes"]
    room = min(x for x in (ws, avail + int(freed_bytes)) if x > 0) \
        if (ws or avail) else 0
    if room and a.bytes_on_disk > room:
        raise NotLoadable(
            507, f"{a.path.name} needs {a.bytes_on_disk / GIB:.1f} GiB; "
                 f"{room / GIB:.1f} GiB is available to it (GPU working set "
                 f"{ws / GIB:.1f}, free now {avail / GIB:.1f}"
                 + (f" + {freed_bytes / GIB:.1f} from unloading the current "
                    f"model" if freed_bytes else "") + "). /loaded.json "
                 f"shows what else is holding memory.", "insufficient_memory")
    return a
