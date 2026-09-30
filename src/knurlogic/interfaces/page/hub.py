"""Hugging Face models for the picker: search the Hub for MLX models, see
whether this machine can run one, and download it into the standard cache
where discovery already looks. Design: docs/design/huggingface.md.

huggingface_hub is imported when a call needs it, never at import. The token
is whatever `hf auth login` left on this machine; nothing here asks for one.
"""

from __future__ import annotations

import re
import subprocess
import threading
from pathlib import Path

LIMIT = 50
#: how many listing rows are read to find LIMIT runnable ones
SCAN = 200
#: what a runnable repo needs; the rest (READMEs, images) is left behind
WANTED = ("*.json", "*.safetensors", "*.txt", "*.model", "*.jinja",
          "*.tiktoken")
GATED_HINT = "run `hf auth login` on this Mac with an account that has access"

_DOWNLOADS: dict = {}
_LOCK = threading.RLock()


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
    """MLX-format models matching `q` that Knurlogic can run, most downloaded
    first. The listing carries each repo's config and safetensors info, so
    support is decided per result without a request per repo; a wider page
    is read and filtered so the list still comes back full."""
    from knurlogic.engine import arch
    from knurlogic.machine.discover import NON_CHAT_MODEL_TYPES
    # the Hub matches one substring; further words narrow what it sent back
    words = [w for w in (q or "").lower().split() if w != "mlx"]
    try:
        rows = _api().list_models(search=words[0] if words else None,
                                  filter="mlx", sort="downloads",
                                  limit=SCAN,
                                  expand=["downloads", "gated", "likes",
                                          "config", "safetensors"])
        out = []
        for m in rows:
            mtype = (m.config or {}).get("model_type") or ""
            if (m.safetensors is None or not mtype
                    or mtype in NON_CHAT_MODEL_TYPES
                    or not arch.supported(mtype)
                    or not all(w in m.id.lower() for w in words[1:])):
                continue
            out.append({"id": m.id, "downloads": m.downloads or 0,
                        "likes": m.likes or 0, "gated": bool(m.gated)})
            if len(out) >= LIMIT:
                break
    except (OSError, ValueError, ImportError) as e:
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
    except (OSError, ValueError, ImportError) as e:
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
        except (OSError, ValueError):
            doc["access"] = False
            doc["hint"] = GATED_HINT
    if not any(f.rfilename.endswith(".safetensors") for f in files):
        doc["supported"], doc["why"] = False, "no safetensors weights (not MLX format)"
    elif not mtype:
        doc["supported"], doc["why"] = False, "no model_type in its config.json"
    elif mtype in NON_CHAT_MODEL_TYPES:
        doc["supported"], doc["why"] = False, f"{mtype} is not a chat model"
    elif not arch.supported(mtype):
        doc["supported"] = False
        doc["why"] = f"{mtype} is not an architecture Knurlogic runs"
    else:
        doc["supported"], doc["why"] = True, ""
    return doc


def _wanted_bytes(files) -> int:
    import fnmatch
    return sum(f.size or 0 for f in files
               if any(fnmatch.fnmatch(f.rfilename, p) for p in WANTED))


_CHILD = ("import sys, json; from huggingface_hub import snapshot_download; "
          "snapshot_download(sys.argv[1], allow_patterns=json.loads(sys.argv[2]))")
_PROCS: dict = {}
_LOADED: list = []


def _store() -> Path:
    from knurlogic.machine.servers import cache_dir
    return cache_dir() / "downloads.json"


def _load() -> None:
    """The list a previous run left; what was running then is stopped now."""
    import json
    if _LOADED:
        return
    _LOADED.append(True)
    try:
        for d in json.loads(_store().read_text()):
            if d["state"] == "downloading":
                d["state"] = "stopped"
            d["stop"] = False
            _DOWNLOADS[d["id"]] = d
    except (OSError, ValueError, KeyError, TypeError):
        pass


def _save() -> None:
    import json
    rows = [{k: v for k, v in d.items() if k != "stop"}
            for d in _DOWNLOADS.values()]
    try:
        _store().write_text(json.dumps(rows))
    except OSError:
        pass


def _refresh_models() -> None:
    from knurlogic.interfaces.page import documents
    documents.forget_models()


def _run(repo_id: str, d: dict) -> None:
    import json
    import os
    import sys
    env = dict(os.environ, HF_HUB_DISABLE_PROGRESS_BARS="1")
    try:
        info = _api().model_info(repo_id, files_metadata=True)
        total = _wanted_bytes(info.siblings or [])
        with _LOCK:
            d["total"] = total
            _save()
    except (OSError, ValueError, ImportError):
        pass
    with _LOCK:
        if d["stop"]:
            d["state"] = "stopped"
            _save()
            return
        p = _PROCS[repo_id] = subprocess.Popen(
            [sys.executable, "-c", _CHILD, repo_id, json.dumps(list(WANTED))],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True)
    err = p.communicate()[1] or ""
    with _LOCK:
        if _PROCS.get(repo_id) is p:
            del _PROCS[repo_id]
        if d["stop"]:
            d["state"] = "stopped"
        elif p.returncode == 0:
            d["state"] = "done"
            _refresh_models()
        else:
            d["state"], d["why"] = "failed", _why(err)
        _save()


def _why(err: str) -> str:
    if "GatedRepoError" in err or "RepositoryNotFoundError" in err:
        return f"no access to this repo; {GATED_HINT}"
    if "No space" in err:
        return "the disk is full"
    return (err.strip().splitlines() or ["the download failed"])[-1][:200]


def start(repo_id: str) -> dict:
    """Start it, or resume it: files already partway on disk carry on."""
    with _LOCK:
        _load()
        d = _DOWNLOADS.get(repo_id)
        if d and d["state"] == "downloading":
            return {"id": repo_id, "state": "downloading"}
        _DOWNLOADS[repo_id] = d = {"id": repo_id, "state": "downloading",
                                   "total": d["total"] if d else 0,
                                   "stop": False, "why": ""}
        _save()
    threading.Thread(target=_run, args=(repo_id, d), daemon=True).start()
    return {"id": repo_id, "state": "downloading"}


def cancel(repo_id: str) -> dict:
    """Stop a running download; its bytes stay for a resume."""
    with _LOCK:
        _load()
        d = _DOWNLOADS.get(repo_id)
        if d and d["state"] == "downloading":
            d["stop"] = True
            p = _PROCS.get(repo_id)
            if p:
                p.terminate()
    return {"id": repo_id}


def dismiss(repo_id: str) -> dict:
    with _LOCK:
        _load()
        d = _DOWNLOADS.get(repo_id)
        if d and d["state"] != "downloading":
            del _DOWNLOADS[repo_id]
            _save()
    return {"id": repo_id}


def _in_use(repo_id: str) -> str:
    """The port of a running server whose model is in this repo's cache
    folder, or ""."""
    from knurlogic.machine.servers import is_our_server, registry
    root = _cache_dir(repo_id)
    for port, rec in registry().items():
        if not is_our_server(int(rec.get("pid") or 0)):
            continue
        try:
            if Path(str(rec.get("artifact") or "")).resolve().is_relative_to(
                    root.resolve()):
                return str(port)
        except (OSError, ValueError):
            continue
    return ""


def delete(repo_id: str) -> dict:
    """Remove this repo's files from the cache, and its row. Refused while
    a running server has the model loaded from there."""
    import shutil
    port = _in_use(repo_id)
    if port:
        return {"id": repo_id, "error": f"a server on port {port} is "
                f"running this model; unload it first"}
    cancel(repo_id)
    with _LOCK:
        p = _PROCS.get(repo_id)
    if p:
        try:
            p.wait(10)
        except (OSError, subprocess.SubprocessError):
            pass
    shutil.rmtree(_cache_dir(repo_id), ignore_errors=True)
    shutil.rmtree(_cache_dir(repo_id).parent / ".locks" /
                  _cache_dir(repo_id).name, ignore_errors=True)
    with _LOCK:
        _load()
        _DOWNLOADS.pop(repo_id, None)
        _save()
    _refresh_models()
    return {"id": repo_id}


def downloads() -> dict:
    """Every download this machine has made through the page: bytes on disk
    against the total, and why a failed one failed. Finished ones stay
    listed until cleared."""
    with _LOCK:
        _load()
        rows = list(_DOWNLOADS.values())
    return {"downloads": [
        {"id": d["id"], "state": d["state"], "total_bytes": d["total"],
         "bytes": d["total"] if d["state"] == "done" else _bytes_held(d["id"]),
         "why": d["why"]}
        for d in rows]}


def act(body) -> dict:
    """POST /hub/download.json {action: download | cancel | dismiss | delete, id}."""
    import json
    try:
        req = json.loads(body or b"{}")
        repo_id, action = str(req["id"]), req.get("action", "download")
    except (ValueError, KeyError, TypeError):
        return {"error": "send {id, action}"}
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo_id):
        return {"error": "a repo id is org/name"}
    fn = {"download": start, "cancel": cancel, "dismiss": dismiss,
          "delete": delete}.get(action)
    return fn(repo_id) if fn else {"error": f"unknown action {action!r}"}
