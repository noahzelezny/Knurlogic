"""Replace a module inside an installed package, without forking it.

`register.py` puts a vendored architecture in front of mlx-lm's by putting it
in `sys.modules` first. This is the same move generalized to any module in
any package -- `mlx_lm.*`, `mlx_vlm.*`, `exo.*` -- and made to survive the
one boundary that defeats the original.

THE BOUNDARY. exo's runner, which is where MLX inference and the VQ kernels
actually run, is an `mp.Process` under start method "spawn". A spawned child
is a fresh interpreter: it inherits the parent's ENVIRONMENT and nothing of
its `sys.modules`. Measured 2026-09-18, with a second channel so a child
that never ran could not read as a pass:

    without overlay:  child saw parent's sys.modules edit: False | overlay: False
    with overlay:     child saw parent's sys.modules edit: False | overlay: True

So the installer is a `sitecustomize.py` on PYTHONPATH. Python imports it at
interpreter startup in every process -- master, API, each spawned runner --
before exo or mlx import anything.

IT MUST BE STANDALONE, and this is not a style preference: exo runs in its
own environment and knurlogic is not installed there. A sitecustomize that
imported knurlogic would work on the machine it was written on and fail on
the one that matters -- the same shape as a version check run with bare
`python3` inside a loop over env paths, which answered for the system
interpreter every iteration.

WHAT THIS DOES NOT REACH. Only processes Knurlogic launches, because the
mechanism rides on the environment. `mlx.core` is a compiled extension and
is not overlayable this way; kernel-level changes still belong in the
artifact's own `model_file`, which is already a per-artifact runtime
boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

OVERLAY_DIR = Path(__file__).parent / "overlays"
SITECUSTOMIZE = OVERLAY_DIR / "_sitecustomize.py"
MANIFEST = OVERLAY_DIR / "MANIFEST.json"

#: Packages an overlay is allowed to target. Not a safety rail -- a signpost:
#: an overlay outside these is almost certainly a module that belongs in this
#: package instead of in front of someone else's.
TARGETS = ("mlx_lm", "mlx_vlm", "exo")


@dataclass
class Overlay:
    module: str                 # e.g. "exo.master.placement_utils"
    path: Path                  # the file that replaces it
    against: str = ""           # the upstream version it was taken from
    why: str = ""               # and the measurement that justifies it
    package: bool = False
    sha256: str = ""

    @property
    def digest(self) -> str:
        return sha256_of(self.path)

    @property
    def state(self) -> str:
        if not self.path.exists():
            return "MISSING"
        if not self.sha256:
            return "UNPINNED"
        return "OK" if self.digest == self.sha256 else "DRIFTED"


def sha256_of(path: Path) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    if p.is_dir():
        h = hashlib.sha256()
        for f in sorted(p.rglob("*.py")):
            h.update(str(f.relative_to(p)).encode())
            h.update(f.read_bytes())
        return h.hexdigest()
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(manifest: Path | None = None) -> list:
    """Every overlay declared in the manifest, in declaration order."""
    f = Path(manifest or MANIFEST)
    if not f.is_file():
        return []
    raw = json.loads(f.read_text())
    root = f.parent
    out = []
    for module, e in raw.get("overlays", {}).items():
        p = Path(e["path"])
        out.append(Overlay(module=module,
                           path=p if p.is_absolute() else (root / p),
                           against=e.get("against", ""), why=e.get("why", ""),
                           package=bool(e.get("package")),
                           sha256=e.get("sha256", "")))
    return out


def write_manifest(overlays: list, path: Path | None = None) -> Path:
    """Record the set, pinning each file by digest as it stands now."""
    f = Path(path or MANIFEST)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"overlays": {
        o.module: {"path": str(o.path), "against": o.against, "why": o.why,
                   "package": o.package, "sha256": o.sha256 or o.digest}
        for o in overlays}}, indent=1) + "\n")
    return f


def install(overlays: list, root: Path, log: Path | None = None) -> dict:
    """Stage the installer and hand back the environment that activates it.

    Returns env to merge into a child process. The staging directory holds
    `sitecustomize.py` AND NOTHING ELSE: it goes on PYTHONPATH, so any other
    file in it would shadow a real module for every process that inherits
    this environment.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    drifted = [o for o in overlays if o.state == "DRIFTED"]
    if drifted:
        raise ValueError(
            "refusing to install overlays whose digest does not match the "
            f"manifest: {[o.module for o in drifted]}. Re-pin them "
            "deliberately, or the measurement they carry is about a file "
            "that no longer exists.")
    shutil.copyfile(SITECUSTOMIZE, root / "sitecustomize.py")
    manifest = write_manifest(overlays, root / "MANIFEST.json")
    env = {"KNURLOGIC_OVERLAY_MANIFEST": str(manifest)}
    if log is not None:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        env["KNURLOGIC_OVERLAY_LOG"] = str(log)
    existing = os.environ.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (f"{root}{os.pathsep}{existing}" if existing
                         else str(root))
    return env


def activations(log: Path | None) -> list:
    """What actually fired, per process, from the installer's own log.

    A spawned runner cannot be asked what it imported, and an overlay that
    never fired looks exactly like one that did. This is the channel that
    tells them apart -- without it, `--cluster` would be claiming an effect
    it cannot see.
    """
    f = Path(log) if log else None
    if not f or not f.is_file():
        return []
    out = []
    for line in f.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def status(overlays: list | None = None, log: Path | None = None) -> dict:
    """The shape `/status.json` reports overlays in."""
    ovs = load() if overlays is None else overlays
    acts = activations(log)
    fired = {a["module"] for a in acts if a.get("event") == "applied"}
    return {
        "declared": [{"module": o.module, "state": o.state,
                      "against": o.against, "why": o.why,
                      "path": str(o.path), "sha256": o.sha256 or o.digest}
                     for o in ovs],
        "activations": acts,
        # Named for what it is. This process can only answer for itself; the
        # log answers for the ones it launched, and nothing answers for a
        # process started outside this environment.
        "applied_in_launched_processes": sorted(fired),
    }


def render(d: dict) -> str:
    if not d.get("declared"):
        return ""
    L = ["overlays"]
    fired = set(d.get("applied_in_launched_processes", []))
    for o in d["declared"]:
        mark = "applied" if o["module"] in fired else "not seen"
        L.append(f"  {o['state']:<9s} {o['module']}  [{mark}]")
        if o.get("against"):
            L.append(f"            against {o['against']}")
        if o.get("why"):
            L.append(f"            {o['why']}")
    if not fired:
        L.append("  nothing has imported an overlaid module yet -- declared "
                 "is not applied, and this says which")
    return "\n".join(L)


# --- the command ------------------------------------------------------------

def _paths():
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "knurlogic" / "overlay", root / "knurlogic" / "overlay.log"


def add(module: str, source: Path, against: str = "", why: str = "") -> Overlay:
    """Take a replacement module under version control, mirroring its path.

    The file is COPIED into this package rather than referenced where it sits:
    a file referenced in place inherits whatever edits happen to it, which is
    the same "which arithmetic am I running" hole that vendoring architectures
    closed.
    """
    if module.split(".")[0] not in TARGETS:
        raise ValueError(
            f"{module} is not inside {TARGETS}. An overlay outside those is "
            f"almost certainly a module that belongs in knurlogic itself.")
    dest = OVERLAY_DIR.joinpath(*module.split(".")).with_suffix(".py")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(source), dest)
    o = Overlay(module=module, path=dest, against=against, why=why)
    o.sha256 = o.digest
    existing = [x for x in load() if x.module != module]
    write_manifest(existing + [o])
    return o


def main(argv=None) -> int:
    import argparse
    import subprocess
    import sys

    p = argparse.ArgumentParser(
        prog="knurlogic overlay",
        description="replace a module inside mlx-lm, mlx-vlm or exo without "
                    "forking it")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("list", help="what is declared, and what has fired")
    a = sub.add_parser("add", help="take a replacement module under control")
    a.add_argument("module")
    a.add_argument("file")
    a.add_argument("--against", default="",
                   help="the exact upstream version this was taken from")
    a.add_argument("--why", default="",
                   help="the measurement that justifies it")
    sub.add_parser("env", help="print the environment that applies them")
    r = sub.add_parser("run", help="run a command with the overlays applied")
    r.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args(argv)

    root, log = _paths()
    overlays = load()

    if args.cmd == "add":
        o = add(args.module, Path(args.file), args.against, args.why)
        print(f"{o.module}  <- {args.file}")
        print(f"  pinned {o.sha256[:12]}")
        if not o.against or not o.why:
            print("  NOTE: no --against/--why recorded. An overlay without "
                  "the version it was taken from and the measurement that "
                  "justifies it is just a fork with extra steps.")
        return 0

    if args.cmd == "env":
        if not overlays:
            print("# no overlays declared", file=sys.stderr)
            return 1
        env = install(overlays, root=root, log=log)
        for k, v in env.items():
            print(f"export {k}={v!r}")
        return 0

    if args.cmd == "run":
        cmd = [c for c in (args.command or []) if c != "--"]
        if not cmd:
            print("knurlogic overlay run -- <command>", file=sys.stderr)
            return 2
        env = dict(os.environ)
        if overlays:
            env.update(install(overlays, root=root, log=log))
            print(f"# {len(overlays)} overlays applied to {cmd[0]} and every "
                  f"process it spawns", file=sys.stderr)
        return subprocess.call(cmd, env=env)

    d = status(overlays, log)
    print(render(d) or "no overlays declared.\n"
          f"  add one with: knurlogic overlay add <module> <file> "
          f"--against <version> --why <measurement>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
