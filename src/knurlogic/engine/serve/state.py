"""The serve package's process state: what is served, vision. (Drafting's
is engine/mtp/binding.DRAFT.)

One module so every file reaches the same dicts by attribute at call time
(`state.SERVED[...]`), never a copy taken at import -- which is also what
lets a test swap one out.
"""

#: The one model this process is serving, and the provider holding it. Only
#: `server.serve()` and `server.switch()` write the path.
SERVED: dict = {"path": None, "provider": None}

#: What is served for images: the VisionServe (family + store + pins) and
#: the model it is bound to. Only vision.bind / vision.clear write here.
VISION: dict = {"serve": None, "model": None, "error": ""}

#: Row counts for a headless vision batch (the drafting one uses
#: engine/mtp/binding.DRAFT).
VISION_STATS: dict = {}


def served_path() -> str:
    return SERVED["path"] or ""
