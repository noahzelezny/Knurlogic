"""Every package says what each of its modules owns.

A package's `__init__` docstring is its map: one line per module and
subpackage, `name.py` or `name/`, saying the one job it owns. A module the
map cannot name in a line is doing more than one job, or none. Every
module opens with a docstring too. Vendored code (NOTICE) is named by its
parent's map but not read further.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src" / "knurlogic"
#: upstream's code, kept as upstream wrote it (CLAUDE.md, NOTICE)
VENDORED = re.compile(r"(^|/)architecture(/|$)|^engine/families/[^/]+/vision(/|$)")


def _rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def _ours(p: Path) -> bool:
    return not VENDORED.search(_rel(p)) and "__pycache__" not in p.parts


def _docstring(p: Path) -> str:
    return ast.get_docstring(ast.parse(p.read_text())) or ""


def _children(pkg: Path) -> list[str]:
    mods = [f"{p.stem}.py" for p in pkg.glob("*.py") if p.stem != "__init__"]
    subs = [f"{p.name}/" for p in pkg.iterdir()
            if p.is_dir() and (p / "__init__.py").exists()]
    return sorted(mods + subs)


def test_every_module_has_a_docstring():
    bare = [_rel(p) for p in ROOT.rglob("*.py")
            if _ours(p) and not _docstring(p).strip()]
    assert not bare, f"modules with no docstring saying what they own: {bare}"


def test_every_package_map_names_each_of_its_modules():
    missing = {}
    for init in ROOT.rglob("__init__.py"):
        pkg = init.parent
        if pkg != ROOT and not _ours(pkg):
            continue
        doc = _docstring(init)
        absent = [c for c in _children(pkg)
                  if not re.search(rf"(?<![\w/]){re.escape(c)}", doc)]
        if absent:
            missing[_rel(pkg) or "."] = absent
    assert not missing, (
        "each package's __init__ docstring names every module in it "
        "(`name.py` / `name/` and the one job it owns); not named: "
        f"{missing}")
