"""Is a model from Hugging Face, or knurlogic itself, out of date? Asked
once per page start.

knurlogic: PyPI's JSON for the package names its latest release; when it is
newer than the one running, the page says so (`release_doc`).

A model in the Hugging Face cache sits at models--org--repo/snapshots/<sha>:
the directory name IS the commit it was downloaded at. Once, in a background
thread, each such repo is asked for its current commit sha (one short
request, no retries). The answers are kept for the page's lifetime; nothing
polls. A model not in the HF cache is never asked about. HF_HUB_OFFLINE=1,
or `knurlogic ui --offline`, skips the check entirely.

Downloading the repo again (the picker's Hugging Face dialog) puts the new
revision beside the old snapshot; the model stops being flagged because a
local snapshot now matches the Hub's sha.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

TIMEOUT = 5.0
_SHA = re.compile(r"[0-9a-f]{40}")

_LOCK = threading.Lock()
#: repo id -> the Hub's current sha; only repos the Hub answered for
_REMOTE: dict = {}
_STATE = {"started": False}


def local_ref(path):
    """(repo id, revision) of a model inside the HF cache, else None."""
    p = Path(str(path))
    if not _SHA.fullmatch(p.name) or p.parent.name != "snapshots":
        return None
    holder = p.parent.parent.name
    if not holder.startswith("models--"):
        return None
    return holder[len("models--"):].replace("--", "/"), p.name


def offline(flag: bool = False) -> bool:
    return bool(flag) or os.environ.get("HF_HUB_OFFLINE", "").strip().lower() \
        in ("1", "true", "yes", "on")


def remote_sha(repo: str) -> str:
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo, expand=["sha"], timeout=TIMEOUT)
    return str(info.sha or "")


def check(repos, ask=remote_sha) -> None:
    """Ask the Hub about each repo once; a failure leaves the repo unknown."""
    for repo in repos:
        try:
            sha = ask(repo)
        except (OSError, ValueError, ImportError, TypeError):
            continue
        if sha:
            with _LOCK:
                _REMOTE[repo] = sha


def start(paths_fn, offline_flag: bool = False, ask=remote_sha):
    """Begin the one background check, unless offline or already begun.
    `paths_fn()` returns the local model paths. Returns the thread, or None."""
    if offline(offline_flag):
        return None
    with _LOCK:
        if _STATE["started"]:
            return None
        _STATE["started"] = True

    def run():
        try:
            repos = sorted({r[0] for r in map(local_ref, paths_fn()) if r})
        # the update check runs once on a daemon thread; any failure means no check
        except Exception:
            logger.debug("update check: could not list local repos",
                         exc_info=True)
            return
        check(repos, ask)
    t = threading.Thread(target=run, daemon=True, name="hf-update-check")
    t.start()
    return t


def start_for_page(offline_flag: bool = False):
    """`start` over every model `discover` finds, and the release check:
    what a page start calls."""
    from knurlogic.machine import discover
    if not offline(offline_flag):
        threading.Thread(target=check_release, daemon=True,
                         name="release-check").start()
    return start(lambda: [f.path for f in discover.find()], offline_flag)


# ------------------------------------------------------------ knurlogic

PYPI = "https://pypi.org/pypi/knurlogic/json"
_RELEASE: dict = {"latest": None}


def pypi_latest() -> str:
    """PyPI's latest release of knurlogic (it skips pre-releases)."""
    import json
    import urllib.request
    with urllib.request.urlopen(PYPI, timeout=TIMEOUT) as r:
        return str(json.load(r)["info"]["version"])


def check_release(ask=pypi_latest) -> None:
    """Ask PyPI once; a failure leaves the latest release unknown."""
    try:
        v = ask()
    except (OSError, ValueError, KeyError, TypeError):
        return
    with _LOCK:
        _RELEASE["latest"] = v


def _key(v: str) -> tuple:
    """A release's order: its leading numbers (0.1.10 after 0.1.9); a
    version with none sorts first."""
    nums = re.match(r"\d+(?:\.\d+)*", v.strip())
    return tuple(int(x) for x in nums.group(0).split(".")) if nums else ()


def release_doc() -> dict:
    """{"current", "latest", "update"}: `latest` None until PyPI answered,
    `update` True only when it is newer than the running knurlogic."""
    from knurlogic import __version__
    with _LOCK:
        latest = _RELEASE["latest"]
    return {"current": __version__, "latest": latest,
            "update": bool(latest) and _key(latest) > _key(__version__),
            "command": "pip install -U knurlogic"}


def flagged(paths) -> set:
    """The paths that have an update: their repo's current sha is known and
    no local snapshot of the repo is at it."""
    with _LOCK:
        remote = dict(_REMOTE)
    local: dict = {}
    refs = {}
    for p in paths:
        ref = local_ref(p)
        if ref:
            refs[str(p)] = ref
            local.setdefault(ref[0], set()).add(ref[1])
    return {p for p, (repo, _) in refs.items()
            if repo in remote and remote[repo] not in local[repo]}


def reset() -> None:
    """Forget everything (tests)."""
    with _LOCK:
        _REMOTE.clear()
        _STATE["started"] = False
        _RELEASE["latest"] = None
