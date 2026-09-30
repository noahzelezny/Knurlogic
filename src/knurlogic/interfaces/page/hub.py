"""Hugging Face models for the picker: search the Hub for MLX models, see
whether this machine can run one, and download it into the standard cache
where discovery already looks. Design: docs/design/huggingface.md.

huggingface_hub is imported when a call needs it, never at import. The token
is whatever `hf auth login` left on this machine; nothing here asks for one.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

LIMIT = 50
#: what a runnable repo needs; the rest (READMEs, images) is left behind
WANTED = ("*.json", "*.safetensors", "*.txt", "*.model", "*.jinja",
          "*.tiktoken")
GATED_HINT = "run `hf auth login` on this Mac with an account that has access"

_DOWNLOADS: dict = {}
_LOCK = threading.Lock()


def _api():
    from huggingface_hub import HfApi
    return HfApi()


def _cache_dir(repo: str) -> Path:
    from huggingface_hub import constants
    return Path(constants.HF_HUB_CACHE) / ("models--" + repo.replace("/", "--"))


def _bytes_held(repo: str) -> int:
    total = 0
    try:
        for f in (_cache_dir(repo) / "blobs").iterdir():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    except OSError:
        pass
    return total


def search(q: str) -> dict:
    """MLX-format models matching `q`, most downloaded first."""
    # the Hub matches one substring; further words narrow what it sent back
    words = [w for w in (q or "").lower().split() if w != "mlx"]
    try:
        rows = _api().list_models(search=words[0] if words else None,
                                  filter="mlx", sort="downloads",
                                  limit=LIMIT * 4 if words[1:] else LIMIT,
                                  expand=["downloads", "gated", "likes"])
        out = [{"id": m.id, "downloads": m.downloads or 0,
                "likes": m.likes or 0, "gated": bool(m.gated)}
               for m in rows if all(w in m.id.lower() for w in words[1:])]
        out = out[:LIMIT]
    except Exception as e:
        return {"error": f"Hugging Face did not answer: {_brief(e)}",
                "results": []}
    return {"results": out}


def _brief(e: Exception) -> str:
    return (str(e).strip().splitlines() or [type(e).__name__])[0][:200]


def repo(repo_id: str) -> dict:
    """One repo's size, architecture, whether it can run here, and whether
    this machine's login can read it."""
    from knurlogic.engine import arch
    from knurlogic.machine.discover import NON_CHAT_MODEL_TYPES
    try:
        info = _api().model_info(repo_id, files_metadata=True)
    except Exception as e:
        return {"id": repo_id, "error": _brief(e)}
    files = [s for s in info.siblings or []]
    size = _wanted_bytes(files)
    mtype = ((info.config or {}).get("model_type") or "")
    gated = bool(info.gated)
    doc = {"id": repo_id, "size_bytes": size, "model_type": mtype,
           "gated": gated, "access": True}
    if gated:
        try:
            _api().auth_check(repo_id)
        except Exception:
            doc["access"] = False
            doc["hint"] = GATED_HINT
    if not any(f.rfilename.endswith(".safetensors") for f in files):
        doc["supported"], doc["why"] = False, "no safetensors weights (not MLX format)"
    elif not mtype:
        doc["supported"], doc["why"] = False, "no model_type in its config.json"
    elif mtype in NON_CHAT_MODEL_TYPES:
        doc["supported"], doc["why"] = False, f"{mtype} is not a chat model"
    elif not arch.supported(mtype):
        doc["supported"], doc["why"] = False, f"{mtype} is not an architecture Knurlogic runs"
    else:
        doc["supported"], doc["why"] = True, ""
    return doc


def _wanted_bytes(files) -> int:
    import fnmatch
    return sum(f.size or 0 for f in files
               if any(fnmatch.fnmatch(f.rfilename, p) for p in WANTED))


class Cancelled(Exception):
    pass


def _run(repo_id: str, d: dict) -> None:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import tqdm as hf_tqdm

    class Bar(hf_tqdm):
        def update(self, n=1):
            if d["cancel"]:
                raise Cancelled()
            return super().update(n)
    try:
        info = _api().model_info(repo_id, files_metadata=True)
        d["total"] = _wanted_bytes(info.siblings or [])
        snapshot_download(repo_id, allow_patterns=list(WANTED),
                          tqdm_class=Bar)
        d["state"] = "done"
        from knurlogic.interfaces.page import documents
        documents._MODELS["at"] = 0.0
    except Exception as e:
        if d["cancel"] or isinstance(e, Cancelled):
            d["state"] = "cancelled"
        else:
            d["state"], d["why"] = "failed", _why(e)


def _why(e: Exception) -> str:
    name = type(e).__name__
    if name in ("GatedRepoError", "RepositoryNotFoundError"):
        return f"no access to this repo; {GATED_HINT}"
    if "No space" in str(e):
        return "the disk is full"
    return _brief(e)


def start(repo_id: str) -> dict:
    with _LOCK:
        d = _DOWNLOADS.get(repo_id)
        if d and d["state"] == "downloading":
            return {"id": repo_id, "state": "downloading"}
        d = _DOWNLOADS[repo_id] = {"id": repo_id, "state": "downloading",
                                   "total": 0, "cancel": False, "why": "",
                                   "started": time.time()}
    threading.Thread(target=_run, args=(repo_id, d), daemon=True).start()
    return {"id": repo_id, "state": "downloading"}


def cancel(repo_id: str) -> dict:
    d = _DOWNLOADS.get(repo_id)
    if d and d["state"] == "downloading":
        d["cancel"] = True
    return {"id": repo_id}


def dismiss(repo_id: str) -> dict:
    with _LOCK:
        d = _DOWNLOADS.get(repo_id)
        if d and d["state"] != "downloading":
            del _DOWNLOADS[repo_id]
    return {"id": repo_id}


def downloads() -> dict:
    """Every download of this run: bytes on disk against the total, and why
    a failed one failed. A finished one stays listed until dismissed."""
    return {"downloads": [
        {"id": d["id"], "state": d["state"], "total_bytes": d["total"],
         "bytes": _bytes_held(d["id"]) if d["state"] == "downloading"
         else d["total"] if d["state"] == "done" else 0,
         "why": d["why"]}
        for d in list(_DOWNLOADS.values()) if d["state"] != "cancelled"]}


def act(body) -> dict:
    """POST /hub/download.json {action: download | cancel | dismiss, id}."""
    import json
    try:
        req = json.loads(body or b"{}")
        repo_id, action = str(req["id"]), req.get("action", "download")
    except (ValueError, KeyError, TypeError):
        return {"error": "send {id, action}"}
    if "/" not in repo_id:
        return {"error": "a repo id is org/name"}
    fn = {"download": start, "cancel": cancel, "dismiss": dismiss}.get(action)
    return fn(repo_id) if fn else {"error": f"unknown action {action!r}"}
