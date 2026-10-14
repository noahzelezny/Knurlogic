"""`knurlogic vendor` -- take an architecture file under version control.

Copying a file is the easy half. The half that matters is recording WHERE it
came from and WHAT was true about the env when it was taken, because the
alternative -- trusting whatever happens to be installed -- is exactly how
three architecture files drifted on one machine without anyone editing them.

Deliberately NOT automatic. Vendoring says "this is the arithmetic
that was measured," which is a claim a person makes, not a script.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import shutil
import subprocess
from pathlib import Path

from knurlogic.engine import families
from knurlogic.engine.arch import digest
from knurlogic.machine.processes import PROC_ERRORS


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
    except PROC_ERRORS:
        return "unknown"


def vendor(module: str, python: str, note: str, host: str,
           family: str, say) -> int:
    """Copy `module` into engine/families/<family>/architecture/. The family
    must be listed (engine/families/__init__.py) and its manifest must name
    the module -- vendoring is part of adding a family, not a way around it.
    What it did goes to `say(line, is_error)` (the CLI prints it); returns
    the exit code."""
    if family not in families.FAMILIES:
        say(f"no family {family!r}; listed: {families.FAMILIES}", True)
        return 2
    ARCH_DIR = families.architecture_dir(family)
    PROVENANCE = ARCH_DIR / "PROVENANCE.md"
    src_dir = _models_dir(python, host)
    flat, pkg = src_dir / f"{module}.py", src_dir / module
    is_pkg = pkg.is_dir() and (pkg / "__init__.py").is_file()
    src = pkg if is_pkg else flat
    if not (is_pkg or flat.is_file()):
        say(f"no {module} (file or package) in {src_dir}", True)
        return 2

    ARCH_DIR.mkdir(parents=True, exist_ok=True)
    dst = ARCH_DIR / (module if is_pkg else f"{module}.py")
    existed = dst.exists()
    if is_pkg:
        if existed and digest(dst) == digest(src):
            say(f"{module}: already vendored, identical")
            return 0
        if existed:
            shutil.rmtree(dst)
        shutil.copytree(src, dst,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        sha = digest(dst)
    else:
        if existed and digest(dst) == digest(src):
            say(f"{module}: already vendored, identical")
            return 0
        shutil.copy2(src, dst)
        sha = digest(dst)
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
            f"- sha256: `{sha}`\n"
            f"- note: {note or '(none)'}\n")
    say(f"{module}: {'re-' if existed else ''}vendored from {src}")
    say(f"  {host} {ver}  sha256 {sha[:16]}...")
    say(f"  -> `knurlogic smoke --pin` on a real artifact records the pin "
        f"in {family}/architecture/pins.json")
    say(f"  -> RECORD THE LICENSE in {family}/architecture/THIRD-PARTY.md: copied "
        f"source carries its project's terms with it")
    return 0


def main(argv, say) -> int:
    p = argparse.ArgumentParser(prog="knurlogic vendor")
    p.add_argument("module", help="e.g. qwen4_exp")
    p.add_argument("--python", required=True,
                   help="interpreter of the env to take it FROM")
    p.add_argument("--note", default="",
                   help="why this env is the authoritative one")
    p.add_argument("--host", default="mlx_lm", choices=("mlx_lm", "mlx_vlm"),
                   help="package to take it FROM and register it UNDER")
    p.add_argument("--family", required=True,
                   help="the family folder it belongs to (engine/families/)")
    a = p.parse_args(argv)
    return vendor(a.module, a.python, a.note, a.host, a.family, say)

