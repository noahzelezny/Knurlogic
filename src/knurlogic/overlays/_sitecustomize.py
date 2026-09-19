"""Installed as `sitecustomize.py` on PYTHONPATH. Runs in EVERY process.

This file is deliberately standalone -- stdlib only, no knurlogic import.
exo runs in its OWN environment (python 3.13, its own rust bindings) and
knurlogic is not installed there; a sitecustomize that imported knurlogic
would work on the box it was developed on and fail on the one that matters.
The manifest carries absolute paths, so nothing here needs a package.

WHY THIS FILE AND NOT `sys.modules`. exo's runner -- where MLX inference and
the VQ kernels actually live -- is an `mp.Process` under start method
"spawn" (exo/utils/async_process.py, exo/main.py). A spawned child is a
fresh interpreter and inherits NOTHING from the parent's sys.modules, so
registering an overlay in the process that launches exo does not reach the
process that runs the model. Measured 2026-09-18:

    without overlay:  child saw parent's sys.modules edit: False | overlay: False
    with overlay:     child saw parent's sys.modules edit: False | overlay: True

The environment is what crosses a spawn boundary, so PYTHONPATH is, so this
is. It runs at interpreter startup -- before exo or mlx import anything --
in the master, the API and every spawned runner alike.

A FINDER, NOT A PRELOAD. Importing the overlaid modules here would drag mlx
into every python process on the box and invite circular imports at
interpreter startup. A meta-path finder fires only if something actually
imports the target, which is also what makes "did it fire" a real signal.
"""

import json
import os
import sys

_MANIFEST = os.environ.get("KNURLOGIC_OVERLAY_MANIFEST", "")
_LOG = os.environ.get("KNURLOGIC_OVERLAY_LOG", "")


def _note(**fields):
    """Append one line of evidence that this actually happened, in THIS pid.

    A spawned runner cannot be asked what it loaded, and an overlay that
    silently did not fire looks exactly like one that did. The log is the
    channel that separates them -- the same reason a probe counts syncs as
    well as seconds.
    """
    if not _LOG:
        return
    try:
        with open(_LOG, "a") as f:
            f.write(json.dumps(dict(pid=os.getpid(), **fields)) + "\n")
    except Exception:
        pass


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class _OverlayFinder:
    """Serve the named modules from knurlogic's tree instead of the install."""

    def __init__(self, entries):
        self.entries = entries

    def find_spec(self, fullname, path=None, target=None):
        e = self.entries.get(fullname)
        if e is None:
            return None
        import importlib.util

        src = e["path"]
        # A digest that does not match is NOT a warning. The whole claim of an
        # overlay is "this exact arithmetic"; serving a file that is not the
        # pinned one would make every measurement taken afterwards unciteable.
        want = e.get("sha256")
        if want:
            got = _sha256(src)
            if got != want:
                _note(module=fullname, event="digest-mismatch",
                      expected=want, actual=got)
                raise ImportError(
                    f"knurlogic overlay for {fullname} is {got[:12]} but the "
                    f"manifest pins {want[:12]} -- refusing to import a file "
                    f"that is not the one that was measured")
        spec = importlib.util.spec_from_file_location(
            fullname, src,
            submodule_search_locations=([os.path.dirname(src)]
                                        if e.get("package") else None))
        _note(module=fullname, event="applied", path=src,
              against=e.get("against"))
        return spec


def _chain():
    """Run any OTHER sitecustomize we are shadowing.

    Ours is first on PYTHONPATH, and python imports exactly one module by
    that name. Conda ships one; silently disabling somebody's environment
    setup would be a rude way to install a memory knob.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for entry in sys.path:
        try:
            if not entry or os.path.abspath(entry) == here:
                continue
            cand = os.path.join(entry, "sitecustomize.py")
            if not os.path.isfile(cand):
                continue
        except Exception:
            continue
        try:
            with open(cand) as f:
                code = f.read()
            exec(compile(code, cand, "exec"), {"__name__": "sitecustomize",
                                               "__file__": cand})
            _note(event="chained", path=cand)
        except Exception as exc:
            _note(event="chain-failed", path=cand, error=repr(exc))
        return


def _install():
    if not _MANIFEST or not os.path.isfile(_MANIFEST):
        return
    try:
        with open(_MANIFEST) as f:
            entries = json.load(f).get("overlays", {})
    except Exception as exc:
        _note(event="manifest-unreadable", path=_MANIFEST, error=repr(exc))
        return
    live = {k: v for k, v in entries.items() if os.path.exists(v.get("path", ""))}
    missing = sorted(set(entries) - set(live))
    if missing:
        _note(event="missing-files", modules=missing)
    if live:
        sys.meta_path.insert(0, _OverlayFinder(live))
        _note(event="installed", modules=sorted(live))


_install()
_chain()
