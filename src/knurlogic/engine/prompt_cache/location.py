"""Where the prompt cache lives (docs/design/prompt-cache-shards.md, "Where
it lives").

Beside the models, visible: `<models folder>/Prompt cache/`, with a
README.txt saying it is regenerable and safe to delete. The models folder
is the folder that holds the artifact's directory; for an artifact inside
a Hugging Face cache (`.../models--org--name/snapshots/<rev>/`) it is the
folder holding `models--org--name` (the hub folder), never a snapshot.
On shared model storage every machine resolves the same folder, so every
rank reads and writes one cache.

When that folder cannot be written (a read-only volume, a share mounted
read-only, a missing path), the cache falls back to the old location,
`~/.cache/knurlogic/prompt-cache/` (XDG_CACHE_HOME honoured).

Every root ever used is remembered in the fallback root's
`locations.json`, so a drop by session, a pin and the registry reach every
model's cache, wherever it lives.
"""
from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path

from knurlogic.machine import paths
from knurlogic.machine.files import read_json, update_json, write_atomic

logger = logging.getLogger(__name__)

FOLDER = "Prompt cache"
README = "README.txt"
LOCATIONS = "locations.json"
_README_TEXT = """\
This folder is knurlogic's prompt cache: the saved attention state of
prompts a client asked to keep, so a model loaded again does not have to
re-read them.

Everything here is regenerable. It is safe to delete this folder (or any
part of it) at any time while no model is saving to it; the next request
on a deleted session simply reads its prompt again.

Settings: KNURLOGIC_PROMPT_CACHE_DISK (on/off), KNURLOGIC_PROMPT_CACHE_DISK_GB
(budget), KNURLOGIC_PROMPT_CACHE_TTL_H (hours an unused entry is kept).
"""
#: roots already resolved (and announced) in this process: {str: kind}
_SEEN: dict = {}


def fallback() -> Path:
    """The old location, used when the models folder cannot be written."""
    return paths.cache_dir() / "prompt-cache"


def models_folder(artifact_path) -> Path | None:
    """The folder that holds the artifact's directory (see the module
    docstring), or None for no path."""
    if not artifact_path:
        return None
    p = Path(artifact_path).expanduser().absolute()
    for parent in [p, *p.parents]:
        if parent.name.startswith("models--"):
            return parent.parent
    return p.parent


def _writable(d: Path) -> bool:
    """`d` (made if missing, its parent must exist) can take a file."""
    try:
        if not d.parent.is_dir():
            return False
        d.mkdir(exist_ok=True)
        probe = d / f".probe-{os.getpid()}"
        probe.write_bytes(b"")
        probe.unlink()
        return True
    except OSError:
        return False


def _readme(d: Path) -> None:
    f = d / README
    if f.exists():
        return
    try:
        write_atomic(f, _README_TEXT)
    except OSError:
        pass


def root_for(artifact_path) -> tuple[Path, str]:
    """(root, kind) of the cache for the artifact at `artifact_path`: kind
    "models" (beside the models) or "fallback" (the old hidden location,
    the models folder not writable). Logged once per root and process."""
    folder = models_folder(artifact_path)
    d = folder / FOLDER if folder is not None else None
    if d is not None and _writable(d):
        _readme(d)
        kind = "models"
    else:
        d, kind = fallback(), "fallback"
    if str(d) not in _SEEN:
        _SEEN[str(d)] = kind
        if kind == "models":
            logger.info("prompt cache: saved beside the models, in %s", d)
        else:
            logger.info("prompt cache: the models folder %s cannot be "
                        "written; saved in %s", folder, d)
    remember(d)
    return d, kind


def remember(d: Path) -> None:
    """Add root `d` to the fallback root's locations.json."""
    fb = fallback()
    if Path(d) == fb:
        return
    if str(d) in map(str, known()):
        return
    # re-read under the lock (every server sharing the fallback root
    # remembers its own); an unmounted volume's root stays listed --
    # known() leaves it out while it is gone
    with contextlib.suppress(OSError):
        update_json(fb / LOCATIONS, lambda names: names.append(str(d)), [],
                    indent=1, clean=lambda names: [
                        str(n) for n in names if str(n) != str(d)])


def known() -> list[Path]:
    """Every root the cache has used on this machine that is there now
    (an unmounted volume's is left out), the fallback first."""
    fb = fallback()
    out = [fb]
    for n in read_json(fb / LOCATIONS, []):
        p = Path(str(n))
        try:
            if p not in out and p.is_dir():
                out.append(p)
        except OSError:
            continue
    return out
