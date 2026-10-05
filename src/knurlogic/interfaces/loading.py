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

GIB = 1 << 30
_KNOWN: dict = {"at": 0.0, "rows": None, "error": ""}


class NotLoadable(Exception):
    def __init__(self, status: int, message: str, code: str = ""):
        super().__init__(message)
        self.status, self.code = status, code


def known_artifacts(rescan: bool = False) -> list:
    """The servable artifacts in this machine's stores. Read once and kept:
    the stores are read again only on `rescan` (a load asked for a model
    the last read did not have) -- never on a timer."""
    if _KNOWN["rows"] is None or rescan:
        from knurlogic.machine import discover
        try:
            _KNOWN["rows"] = [f for f in discover.find()
                              if f.servable and f.format == "mlx"]
            _KNOWN["at"], _KNOWN["error"] = time.time(), ""
        except (OSError, ValueError, KeyError, AttributeError) as e:
            # not cached: a store on a volume that was briefly away is
            # scanned again on the next request, and until then the
            # refusal names the scan, not the model
            _KNOWN["rows"] = None
            _KNOWN["error"] = f"{type(e).__name__}: {e}"
            return []
    return _KNOWN["rows"]


def resolve_name(model: str, served: str | None) -> str:
    """A model id (directory name) or path -> the artifact directory, if it
    is the served one or a known one. NotLoadable(404) otherwise."""
    model = (model or "").strip()
    if served and model in ("", Path(served).name, served):
        return served
    try:
        want = str(Path(model).expanduser().resolve())
    except (OSError, RuntimeError):
        want = model
    from knurlogic.machine import discover
    try:
        direct = [f for f in discover.find_named(model)
                  if f.servable and f.format == "mlx"]
    except (OSError, ValueError, KeyError, AttributeError):
        direct = []
    if direct:
        return str(direct[0].path)
    fresh = _KNOWN["rows"] is None
    for again in (False, True):
        if again and fresh:
            break       # just read: reading again finds nothing new
        for f in known_artifacts(rescan=again):
            p = str(Path(f.path).resolve())
            if model in (f.name, Path(f.path).name) or want == p:
                return str(f.path)
    if _KNOWN.get("error"):
        raise NotLoadable(503, f"the model stores could not be scanned "
                               f"({_KNOWN['error']}); is a volume holding "
                               f"them unmounted?", "store_unavailable")
    raise NotLoadable(404, f"{model!r} is not an artifact in this machine's "
                           f"model stores. /models.json lists the ones that "
                           f"are; a model is named by its id there, not by "
                           f"an arbitrary path.", "model_not_found")


def register(artifact) -> list:
    """Register the artifact's vendored architecture modules; the names of
    any it still lacks (empty when it can load)."""
    from knurlogic.engine import arch
    from knurlogic.engine import register as reg
    needed = arch.modules_for_artifact(artifact)
    if needed:
        reg.register(*needed)
    return [r.module for r in arch.check(artifact.model_type)
            if not r.present]


def prepare(model: str, *, served: str | None = None,
            freed_bytes: int = 0):
    """-> the Artifact to load, or NotLoadable saying why not."""
    from knurlogic.machine import wired
    from knurlogic.machine.artifact import Artifact
    path = resolve_name(model, served)
    try:
        a = Artifact.load(path)
    except (OSError, ValueError, AttributeError) as e:
        raise NotLoadable(422, f"{Path(path).name} could not be read as an "
                               f"artifact: {e}", "bad_artifact") from e
    missing = register(a)
    if missing:
        raise NotLoadable(422, f"{a.path.name} needs {missing}, which "
                               f"neither knurlogic's vendored set nor the "
                               f"installed engine provides.",
                          "unsupported_architecture")
    b = wired.load_budget()
    ws, avail = b["working_set_bytes"], b["available_bytes"]
    allow = b.get("allowance_bytes") or 0
    room = min((x for x in (ws, avail + int(freed_bytes), allow) if x > 0),
               default=0)
    if room and a.bytes_on_disk > room:
        raise NotLoadable(
            507, f"{a.path.name} needs {a.bytes_on_disk / GIB:.1f} GiB; "
                 f"{room / GIB:.1f} GiB is available to it (GPU working set "
                 f"{ws / GIB:.1f}, free now {avail / GIB:.1f}"
                 + (f", knurlogic allowance {allow / GIB:.1f}" if allow
                    else "")
                 + (f" + {freed_bytes / GIB:.1f} from unloading the current "
                    f"model" if freed_bytes else "") + "). /loaded.json "
                 "shows what else is holding memory.", "insufficient_memory")
    return a
