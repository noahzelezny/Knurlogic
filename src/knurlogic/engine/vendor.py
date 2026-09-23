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

from knurlogic.engine.register import ARCH_DIR

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


def _models_dir(python: str, host: str) -> Path:
    out = subprocess.run(
        [python, "-c",
         f"import {host}.models as m,pathlib;print(pathlib.Path(m.__file__).parent)"],
        capture_output=True, text=True, check=True)
    return Path(out.stdout.strip())


def _host_version(python: str, host: str) -> str:
    try:
        out = subprocess.run(
            [python, "-c", f"import {host};print({host}.__version__)"],
            capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _dir_digest(d: Path) -> str:
    """Digest of a package: every .py, path-and-content, sorted."""
    h = hashlib.sha256()
    for f in sorted(d.rglob("*.py")):
        h.update(str(f.relative_to(d)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def vendor(module: str, python: str, note: str, host: str) -> int:
    src_dir = _models_dir(python, host)
    flat, pkg = src_dir / f"{module}.py", src_dir / module
    is_pkg = pkg.is_dir() and (pkg / "__init__.py").is_file()
    src = pkg if is_pkg else flat
    if not (is_pkg or flat.is_file()):
        print(f"no {module} (file or package) in {src_dir}", file=sys.stderr)
        return 2

    ARCH_DIR.mkdir(parents=True, exist_ok=True)
    dst = ARCH_DIR / (module if is_pkg else f"{module}.py")
    existed = dst.exists()
    if is_pkg:
        if existed and _dir_digest(dst) == _dir_digest(src):
            print(f"{module}: already vendored, identical")
            return 0
        if existed:
            shutil.rmtree(dst)
        shutil.copytree(src, dst,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        digest = _dir_digest(dst)
    else:
        if existed and _sha256(dst) == _sha256(src):
            print(f"{module}: already vendored, identical")
            return 0
        shutil.copy2(src, dst)
        digest = _sha256(dst)
    ver = _host_version(python, host)

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
            f"- host package: {host} {ver}\n"
            f"- layout: {'package' if is_pkg else 'file'}\n"
            f"- sha256: `{digest}`\n"
            f"- note: {note or '(none)'}\n")
    print(f"{module}: {'re-' if existed else ''}vendored from {src}")
    print(f"  {host} {ver}  sha256 {digest[:16]}...")
    print(f"  -> add to PINNED_SHA256 in arch.py once validated")
    print(f"  -> RECORD THE LICENSE in architectures/THIRD-PARTY.md: copied "
          f"source carries its project's terms with it")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="knurlogic vendor")
    p.add_argument("module", help="e.g. qwen4_exp")
    p.add_argument("--python", required=True,
                   help="interpreter of the env to take it FROM")
    p.add_argument("--note", default="",
                   help="why this env is the authoritative one")
    p.add_argument("--host", default="mlx_lm", choices=("mlx_lm", "mlx_vlm"),
                   help="package to take it FROM and register it UNDER")
    a = p.parse_args(argv)
    return vendor(a.module, a.python, a.note, a.host)


if __name__ == "__main__":
    raise SystemExit(main())
