"""The model folders this Mac remembers (`knurlogic models add <folder>`).

Kept in ~/.knurlogic/model_folders.json (KNURLOGIC_HOME moves it), beside
peers.json, so starting knurlogic needs no environment variable. Discovery
(machine/discover._roots) reads these beside every tool's own store.

A folder on an external volume that is not mounted is skipped quietly: it is
still remembered, and is found again once the drive is back.

Adoption, once: while nothing has been saved, a folder named by
EXO_MODELS_DIRS, EXO_MODELS_READ_ONLY_DIRS or KNURLOGIC_MODELS that exists is
saved, so a Mac started with the old variable keeps its folder without it.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from knurlogic.machine import files, paths

#: the variables a folder is adopted from, once
ADOPT_FROM = ("EXO_MODELS_DIRS", "EXO_MODELS_READ_ONLY_DIRS",
              "KNURLOGIC_MODELS")


def store_path() -> Path:
    return paths.state_dir() / "model_folders.json"


def _clean(d: dict) -> dict:
    folders = [f for f in d.get("folders") or [] if isinstance(f, str) and f]
    return {"folders": folders, "adopted": bool(d.get("adopted"))}


def _read() -> dict:
    return _clean(files.read_json(store_path(), {}))


def _update(change: Callable[[dict], None]) -> dict:
    """The store (`_clean`), changed in place by `change`, saved as one
    step against every other writer -- the page and the MCP both add and
    remove folders (files.update_json). -> the store saved."""
    return files.update_json(store_path(), change, {}, clean=_clean,
                             indent=1)


def _norm(folder: str) -> str:
    return str(Path(folder).expanduser().absolute())


def saved() -> list:
    """Every remembered folder, mounted or not."""
    return _read()["folders"]


def _is_dir(p: Path) -> bool:
    try:
        return p.is_dir()
    except OSError:
        return False


def roots() -> list:
    """The remembered folders that are there right now (an unmounted
    volume's folder is left out, without an error)."""
    return [Path(f) for f in saved() if _is_dir(Path(f))]


def add(folder: str) -> dict:
    """Remember `folder`. It has to exist when it is added."""
    f = _norm(folder)
    if not _is_dir(Path(f)):
        return {"error": f"not a folder: {f}", "folders": saved()}

    def put(d: dict) -> None:
        if f not in d["folders"]:
            d["folders"].append(f)
        d["adopted"] = True
    return {"added": f, "folders": _update(put)["folders"]}


def remove(folder: str) -> dict:
    """Forget `folder` (the models in it are not touched)."""
    f = _norm(folder)
    was: list = []

    def drop(d: dict) -> None:
        was.extend(d["folders"])
        if f in d["folders"]:
            d["folders"].remove(f)
            d["adopted"] = True
    d = _update(drop)
    if f not in was:
        return {"error": f"not a saved folder: {f}", "folders": d["folders"]}
    return {"removed": f, "folders": d["folders"]}


def adopt_from_env(env=None) -> list:
    """Save the folders the old variables name, once: only while nothing was
    ever saved here. Returns what was saved."""
    env = os.environ if env is None else env
    found: list = []
    for var in ADOPT_FROM:
        for v in (env.get(var) or "").split(":"):
            if v and _is_dir(Path(v).expanduser()):
                f = _norm(v)
                if f not in found:
                    found.append(f)
    if not found:
        return []
    saved_now: list = []

    def adopt(d: dict) -> None:
        if not (d["adopted"] or d["folders"]):
            d.update(folders=list(found), adopted=True)
            saved_now.extend(found)
    _update(adopt)
    return saved_now


def listing() -> list:
    """The saved folders, one line each, an unmounted one marked."""
    return [f + ("" if _is_dir(Path(f)) else "   (not mounted)")
            for f in saved()]
