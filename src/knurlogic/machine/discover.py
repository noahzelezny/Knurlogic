"""Find the models already on this machine, wherever they were downloaded.

Four tools keep four stores and none of them looks at the others, so the same
7 GB of weights gets downloaded three times and a person with 200 GB of
models on disk still sees an empty list. This reads all of them.

FINDING A MODEL IS NOT THE SAME AS BEING ABLE TO RUN IT, and conflating those
would be a worse version of the problem. Ollama and most of LM Studio hold
GGUF; the engine here does not load GGUF. Checked rather than assumed:
`mlx_lm.gguf` exposes `convert_to_gguf` and no loader, and the server module
never mentions the format. So a GGUF model is reported as FOUND and NOT
SERVABLE, with the reason -- the alternative is a menu of entries that 500 on
click, which is the same "why did it fail" that `doctor` exists to end.

Nothing here downloads, moves, links or deletes anything. It reads.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

GIB = 1 << 30

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf", ".npz")

#: `model_type` values (config.json) this engine can find on disk but that
#: are not chat models: embedders, ASR and image-only towers a fit/picker
#: listed as servable chat models before this check existed (P5, per the
#: vision v2 design's `models/fit` note) -- an agent asked to load one got a
#: server that starts and then 400s every /v1/chat/completions call, which
#: is a worse failure than not listing it. Not exhaustive; extend as new
#: non-chat `model_type`s are found on disk.
NON_CHAT_MODEL_TYPES = frozenset({
    # text/embedding encoders
    "bert", "nomic_bert", "roberta", "distilbert", "xlm-roberta",
    "gte", "e5",
    # audio (ASR)
    "whisper",
    # vision-only towers (no LM head)
    "clip", "clip_vision_model", "siglip", "siglip_vision_model",
    "siglip2", "siglip2_vision_model",
    # background/segmentation and other image-to-image utility models
    "rmbg", "briarmbg", "segformer",
})

#: Where each tool keeps its models. Env var first, then the defaults it
#: ships with. Adding a store is a line, which is the point -- there will be
#: a fifth.
#: knurlogic's own store: a VISIBLE folder, deliberately -- models are the
#: biggest files on the disk, and a hidden one is where space goes missing.
#: KNURLOGIC_MODELS moves it (e.g. to an external SSD).
MODELS_DIR = "~/Knurlogic/Models"

STORES = (
    ("knurlogic", ("KNURLOGIC_MODELS",), (MODELS_DIR,)),
    # exo's own dirs come from `exo_model_dirs`, which mirrors exo's
    # resolution; these are only the names older launch scripts exported.
    ("exo", ("EXO_MODELS_DIR",), ("~/.cache/exo/models",)),
    ("huggingface", ("HF_HUB_CACHE",), ("~/.cache/huggingface/hub",)),
    ("ollama", ("OLLAMA_MODELS",), ("~/.ollama/models",)),
    ("lm studio", ("LMSTUDIO_MODELS",),
     ("~/.lmstudio/models", "~/.cache/lm-studio/models")),
)


@dataclass
class Found:
    name: str
    path: Path
    store: str
    format: str = "unknown"          # mlx | gguf | unknown
    bytes_on_disk: int = 0
    model_type: str = ""
    is_vq: bool = False
    model_file: str | None = None
    servable: bool = False
    why: str = ""                    # why not, when not
    extra: dict = field(default_factory=dict)

    @property
    def gib(self) -> float:
        return self.bytes_on_disk / GIB


#: Variables worth reading off a RUNNING tool, since a store location is
#: per-tool configuration and another process's environment is not ours.
_ENV_OF_INTEREST = ("EXO_MODELS_DIR", "EXO_DEFAULT_MODELS_DIR", "HF_HOME",
                    "HF_HUB_CACHE", "OLLAMA_MODELS", "EXO_MODELS_DIRS",
                    "EXO_MODELS_READ_ONLY_DIRS")


def _running_tool_roots() -> list:
    """Ask a running exo/ollama where it keeps its models.

    A store lives wherever that tool was configured to put it, and this
    machine is the case in point: 37 artifacts on an external volume, named
    only in the environment of a process that was started hours ago. Guessing
    at external volumes would be wrong on every other machine; asking the
    process that knows is right on all of them.

    Parsed with a boundary regex, not `split()`: the value here is
    "/Volumes/Thunderbay SSD/Exo Models", and splitting on spaces turns one
    real path into two paths that do not exist.
    """
    import re
    import subprocess

    out = []
    try:
        ps = subprocess.run(["ps", "-xo", "pid=,command="], capture_output=True,
                            text=True, timeout=5).stdout
    except Exception:
        return out
    pids = [ln.split(None, 1)[0] for ln in ps.splitlines()
            if re.search(r"/(exo|ollama)(\s|$)", ln.split(None, 1)[-1])]
    for pid in pids[:8]:
        try:
            env = subprocess.run(["ps", "eww", "-o", "command=", pid],
                                 capture_output=True, text=True,
                                 timeout=5).stdout
        except Exception:
            continue
        for var in _ENV_OF_INTEREST:
            m = re.search(rf"\b{var}=(.*?)(?=\s+[A-Z_]{{3,}}=|$)", env)
            if not m:
                continue
            val = m.group(1).strip()
            if not val:
                continue
            store = "ollama" if var == "OLLAMA_MODELS" else (
                "huggingface" if var.startswith("HF") else "exo")
            vals = val.split(":") if var.endswith("_DIRS") else [val]
            for v in filter(None, vals):
                p = Path(v).expanduser()
                for cand in ((p, p / "hub") if var.startswith("HF")
                             else (p,)):
                    if cand.is_dir():
                        out.append((store, cand))
    return out


def exo_model_dirs(env=None, platform=None, home=None) -> list:
    """Where exo itself looks for models, resolved the way exo resolves it
    (exo/shared/constants.py), so this answers without exo running.

    The default is `<data home>/models`, and on anything but Linux the data
    home is `~/.exo` -- not the XDG path. Missing that is how a machine whose
    `~/.exo/models` is a symlink to a 37-artifact external volume showed 13
    artifacts: the store was only found while exo happened to be started by
    a script that also exported a directory variable.
    """
    import sys
    env = os.environ if env is None else env
    platform = sys.platform if platform is None else platform
    home = Path.home() if home is None else Path(home)
    if env.get("EXO_HOME"):
        data = home / env["EXO_HOME"]
    elif platform != "linux":
        data = home / ".exo"
    else:
        xdg = env.get("XDG_DATA_HOME")
        data = (Path(xdg) if xdg else home / ".local" / "share") / "exo"
    default = (Path(env["EXO_DEFAULT_MODELS_DIR"]).expanduser()
               if env.get("EXO_DEFAULT_MODELS_DIR") else data / "models")
    out = [default]
    for var in ("EXO_MODELS_DIRS", "EXO_MODELS_READ_ONLY_DIRS"):
        out += [Path(x).expanduser() for x in env.get(var, "").split(":") if x]
    return out


def _roots(extra=(), include_defaults: bool = True):
    """(store, path) for every store directory that exists on this machine."""
    out = [("given", Path(p).expanduser()) for p in extra
           if Path(p).expanduser().is_dir()]
    if not include_defaults:
        return out
    out += _running_tool_roots()
    out += [("exo", d) for d in exo_model_dirs() if d.is_dir()]
    for store, envs, defaults in STORES:
        seen = []
        for e in envs:
            v = os.environ.get(e)
            if v:
                seen.append(Path(v).expanduser())
        # HF_HOME is the parent of the hub cache, not the cache itself.
        if store == "huggingface" and os.environ.get("HF_HOME"):
            seen.append(Path(os.environ["HF_HOME"]).expanduser() / "hub")
            seen.append(Path(os.environ["HF_HOME"]).expanduser())
        seen += [Path(d).expanduser() for d in defaults]
        for p in seen:
            try:
                if p.is_dir():
                    out.append((store, p))
            except OSError:
                continue
    return out


def _weights_bytes(d: Path, depth: int = 1) -> int:
    total = 0
    try:
        for f in d.iterdir():
            if f.is_file() and f.suffix in WEIGHT_SUFFIXES:
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
            elif f.is_dir() and depth > 0:
                total += _weights_bytes(f, depth - 1)
    except OSError:
        pass
    return total


def _from_config_dir(d: Path, store: str) -> Found | None:
    """A directory that carries a config.json is an artifact we can read."""
    from knurlogic.engine import mtp
    from knurlogic.machine.artifact import Artifact
    try:
        a = Artifact.load(d)
    except Exception:
        return None
    size = a.bytes_on_disk or _weights_bytes(d)
    name = d.name
    # An HF cache entry's real name lives two levels up: models--org--repo.
    for parent in d.parents:
        if parent.name.startswith("models--"):
            name = parent.name[len("models--"):].replace("--", "/")
            break
    # A config.json with no weights beside it is an interrupted or evicted
    # download, not a model. Two of these sat in the hub cache here and read
    # as 0 GiB artifacts -- which then dragged their whole model group into
    # "fits in memory" in the picker, because something reporting no size
    # fits anywhere.
    not_chat = a.model_type in NON_CHAT_MODEL_TYPES
    return Found(name=name, path=d, store=store, format="mlx",
                 bytes_on_disk=size, model_type=a.model_type, is_vq=a.is_vq,
                 model_file=a.model_file, servable=size > 0 and not not_chat,
                 why=("" if size and not not_chat else
                      f"model_type {a.model_type!r} is not a chat model "
                      "this engine serves as one (an embedder, ASR or "
                      "vision-only tower)" if not_chat and size else
                      "config.json but no weight files -- an "
                      "interrupted or evicted download"),
                 # A built drafting head is a property of what is ON DISK, and
                 # it sits outside the model glob -- so nothing else in a
                 # listing would ever mention it.
                 extra={"mtp_head": str(h.path)} if (h := mtp.find_head(d))
                        else {})


def _from_gguf(d: Path, store: str, files: list) -> Found:
    return Found(
        name=d.name, path=d, store=store, format="gguf",
        bytes_on_disk=sum(f.stat().st_size for f in files),
        servable=False,
        why="GGUF; this engine loads safetensors. mlx-lm can convert TO gguf "
            "and not from it, so knurlogic cannot serve this one")


def _scan_tree(root: Path, store: str, max_depth: int = 3) -> list:
    """Walk a store looking for either a config.json or GGUF files."""
    out, stack = [], [(root, 0)]
    while stack:
        d, depth = stack.pop()
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        if (d / "config.json").is_file():
            f = _from_config_dir(d, store)
            if f:
                out.append(f)
                continue                    # do not descend into an artifact
        ggufs = [e for e in entries if e.is_file() and e.suffix == ".gguf"]
        if ggufs:
            out.append(_from_gguf(d, store, ggufs))
            continue
        if depth < max_depth:
            for e in entries:
                if e.is_dir() and not e.name.startswith("."):
                    stack.append((e, depth + 1))
    return out


def _scan_ollama(root: Path) -> list:
    """Ollama keeps manifests separately from content-addressed blobs.

    Implemented from the on-disk layout; this machine's store is empty, so
    it is UNVERIFIED against a real pull and says so rather than pretending.
    """
    out = []
    man = root / "manifests"
    if not man.is_dir():
        return out
    for f in man.rglob("*"):
        if not f.is_file():
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        layers = d.get("layers") or []
        if not layers:
            continue
        rel = f.relative_to(man).parts
        name = f"{rel[-2]}:{rel[-1]}" if len(rel) >= 2 else f.name
        out.append(Found(
            name=name, path=f, store="ollama", format="gguf",
            bytes_on_disk=sum(int(l.get("size") or 0) for l in layers),
            servable=False,
            why="GGUF; this engine loads safetensors, so knurlogic cannot "
                "serve this one",
            extra={"manifest": str(f)}))
    return out


def find(stores=None, extra=(), include_defaults: bool = True) -> list:
    """Everything on this machine, deduped by real path, biggest first."""
    out, seen = [], set()
    roots, done = [], set()
    for store, root in _roots(extra, include_defaults):
        key = (store, str(root.resolve()))
        if key not in done:
            done.add(key)
            roots.append((store, root))
    for store, root in roots:
        if stores and store not in stores:
            continue
        found = (_scan_ollama(root) if store == "ollama"
                 else _scan_tree(root, store))
        for f in found:
            key = str(Path(f.path).resolve())
            if key in seen:
                continue
            seen.add(key)
            out.append(f)
    return sorted(out, key=lambda f: -f.bytes_on_disk)


def render(rows: list, working_set_bytes: int = 0) -> str:
    if not rows:
        looked = ", ".join(str(p) for _s, p in _roots()) or "nowhere"
        return f"no models found. Looked in: {looked}"
    L = [f"{'STORE':<12}{'NAME':<44}{'SIZE':>8}  STATE"]
    for f in rows:
        if not f.servable:
            state = f.format.upper() + " -- cannot be served here"
        elif working_set_bytes and f.bytes_on_disk > working_set_bytes:
            # Not "does not fit" flatly: knurlogic serves across nodes, so a
            # model bigger than one box is a clustering question, not a wall.
            state = f"needs more than this box ({working_set_bytes / GIB:.0f} GiB)"
        else:
            state = f.model_type or "ok"
        # Flags describe what is ON DISK, so they hold whatever the state
        # line says -- a model too big for one box still has its head.
        if f.servable:
            if f.is_vq:
                state += "  [VQ]"
            if f.extra.get("mtp_head"):
                state += "  [MTP]"
        L.append(f"{f.store:<12}{f.name[:43]:<44}{f.gib:>7.1f}G  {state}")
    # Three different counts, kept apart because they answer three different
    # questions: is it here, can this engine read it, will it fit.
    loadable = [f for f in rows if f.servable]
    fits = [f for f in loadable
            if not working_set_bytes or f.bytes_on_disk <= working_set_bytes]
    L.append("")
    L.append(f"{len(rows)} found, {sum(f.bytes_on_disk for f in rows) / GIB:.0f}"
             f" GiB on disk. {len(loadable)} in a format this engine loads"
             + (f", {len(fits)} that fit one box." if working_set_bytes
                else "."))
    if len(rows) != len(loadable):
        L.append(f"{len(rows) - len(loadable)} are GGUF -- found, but this "
                 f"engine loads safetensors.")
    return "\n".join(L)


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="knurlogic models",
        description="find the models already on this machine, in every "
                    "store, and say which ones can actually run here")
    p.add_argument("--store", action="append", default=[],
                   help="limit to one store (exo, huggingface, ollama, "
                        "'lm studio')")
    p.add_argument("--servable", action="store_true",
                   help="only the ones this engine can load")
    p.add_argument("--path", action="append", default=[],
                   help="also scan this directory")
    p.add_argument("--only-path", action="store_true",
                   help="scan ONLY --path, not the tools' own stores")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)

    from knurlogic.machine import wired
    rows = find(stores=a.store or None, extra=a.path,
                include_defaults=not a.only_path)
    if a.servable:
        rows = [f for f in rows if f.servable]
    if a.json:
        print(json.dumps([{**vars(f), "path": str(f.path)} for f in rows],
                         indent=1))
        return 0
    print(render(rows, wired.detected_working_set_bytes()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
