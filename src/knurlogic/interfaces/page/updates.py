"""Is a model from Hugging Face out of date? Asked once per page start.

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
    """`start` over every model `discover` finds: what a page start calls."""
    from knurlogic.machine import discover
    return start(lambda: [f.path for f in discover.find()], offline_flag)


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
