"""`knurlogic vendor` -- take an architecture file under version control.

Copying a file is the easy half. The half that matters is recording WHERE it
came from and WHAT was true about the env when it was taken, because the
alternative -- trusting whatever happens to be installed -- is exactly how
three architecture files drifted on one machine without anyone editing them.

Deliberately NOT automatic. Vendoring says "this is the arithmetic we
measured," which is a claim a person makes, not a script.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

from .register import ARCH_DIR

PROVENANCE = ARCH_DIR / "PROVENANCE.md"


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _mlx_lm_version(python: str) -> str:
    try:
        out = subprocess.run(
            [python, "-c", "import mlx_lm;print(mlx_lm.__version__)"],
            capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _models_dir(python: str) -> Path:
    out = subprocess.run(
        [python, "-c",
         "import mlx_lm.models as m,pathlib;print(pathlib.Path(m.__file__).parent)"],
        capture_output=True, text=True, check=True)
    return Path(out.stdout.strip())


def vendor(module: str, python: str, note: str) -> int:
    src_dir = _models_dir(python)
    src = src_dir / f"{module}.py"
    if not src.is_file():
        print(f"no {module}.py in {src_dir}", file=sys.stderr)
        return 2

    ARCH_DIR.mkdir(parents=True, exist_ok=True)
    dst = ARCH_DIR / f"{module}.py"
    if dst.is_file() and _sha256(dst) == _sha256(src):
        print(f"{module}: already vendored, identical")
        return 0
    existed = dst.is_file()
    shutil.copy2(src, dst)
    digest, ver = _sha256(dst), _mlx_lm_version(python)

    header = ("# Vendored architecture modules\n\n"
              "Each entry is a claim that this file is the arithmetic the\n"
              "artifacts were validated against -- not merely that it imports.\n\n")
    if not PROVENANCE.is_file():
        PROVENANCE.write_text(header)
    with PROVENANCE.open("a") as f:
        f.write(
            f"\n## {module}.py\n\n"
            f"- taken: {_dt.date.today().isoformat()}\n"
            f"- from: `{src}`\n"
            f"- interpreter: `{python}`\n"
            f"- mlx-lm: {ver}\n"
            f"- sha256: `{digest}`\n"
            f"- note: {note or '(none)'}\n")
    print(f"{module}: {'re-' if existed else ''}vendored from {src}")
    print(f"  mlx-lm {ver}  sha256 {digest[:16]}...")
    print(f"  -> add to PINNED_SHA256 in arch.py once validated")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="knurlogic vendor")
    p.add_argument("module", help="e.g. qwen4_exp")
    p.add_argument("--python", required=True,
                   help="interpreter of the env to take it FROM")
    p.add_argument("--note", default="",
                   help="why this env is the authoritative one")
    a = p.parse_args(argv)
    return vendor(a.module, a.python, a.note)


if __name__ == "__main__":
    raise SystemExit(main())
