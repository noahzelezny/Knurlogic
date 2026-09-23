"""Install Knurlogic's architecture modules into mlx-lm, without touching it.

mlx-lm resolves a model class with

    importlib.import_module(f"mlx_lm.models.{model_type}")

and `import_module` consults `sys.modules` first. So registering a module
object under that name BEFORE the first load makes mlx-lm use ours, and
`site-packages` is never written to.

That matters more than the convenience. Measured across one machine's two
envs, three of four grafted architecture files differed and a fourth was
absent from both -- and the envs turned out to be on DIFFERENT mlx-lm
versions (0.32.0 and 0.31.9), which explains most of the difference. That is
the point, not a weaker version of it: a file grafted into someone else's
install inherits that install's version, so "which arithmetic am I running"
has no answer. A file vendored inside a versioned package does.

Reversible by construction: `unregister()` drops the entries, and an env that
never imported knurlogic is byte-identical to one that did.

The vendored file must still be validated against a known mlx-lm -- pinning
the file does not pin the library it calls into. PROVENANCE.md records which
mlx-lm each file was taken from.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ARCH_DIR = Path(__file__).parent / "architectures"
_installed: list = []


def available() -> list:
    """Architecture modules vendored here -- flat files AND packages."""
    if not ARCH_DIR.is_dir():
        return []
    flat = {p.stem for p in ARCH_DIR.glob("*.py") if not p.stem.startswith("_")}
    pkgs = {d.name for d in ARCH_DIR.iterdir()
            if d.is_dir() and (d / "__init__.py").is_file()}
    return sorted(flat | pkgs)


def source_for(name: str):
    """(path, is_package) for a vendored architecture."""
    pkg = ARCH_DIR / name
    if (pkg / "__init__.py").is_file():
        return pkg / "__init__.py", True
    flat = ARCH_DIR / f"{name}.py"
    return (flat, False) if flat.is_file() else (None, False)


def _with_dependencies(names: list) -> list:
    """Expand each name to [its bases..., itself], in load order.

    THE BUG THIS EXISTS FOR (caught 2026-09-18): qwen3_5_moe subclasses
    qwen3_5. Registering the subclass alone produced a VENDORED subclass
    sitting on a SITE-PACKAGES base -- two versions of the arithmetic silently
    mixed, which is the precise failure this package is meant to end. It
    passed the first time only because `sorted()` happens to put qwen3_5
    before qwen3_5_moe.
    """
    from knurlogic.engine.arch import ARCH_DEPENDS_ON

    out: list = []

    def visit(n: str) -> None:
        if n in out:
            return
        for dep in ARCH_DEPENDS_ON.get(n, []):
            visit(dep)
        out.append(n)

    for n in names:
        visit(n)
    return out


def register(*names: str, override: bool = False) -> list:
    """Make the named vendored architectures resolvable as mlx_lm.models.<n>.

    Dependencies are registered FIRST: a subclass must never land on a base
    from a different source. With `override=False` (default) a module mlx-lm
    already imported is left alone and reported, rather than swapped
    underneath a loaded model.
    """
    wanted = _with_dependencies(list(names) or available())
    done = []
    from knurlogic.engine.arch import host_for

    for name in wanted:
        src, is_pkg = source_for(name)
        if src is None:
            raise FileNotFoundError(f"no vendored architecture {name!r}")
        # Registered under the host whose siblings its relative imports need.
        target = f"{host_for(name)}.models.{name}"
        if target in sys.modules and not override:
            continue
        # A package needs __path__ or its `from .language import ...` fails.
        spec = importlib.util.spec_from_file_location(
            target, src,
            submodule_search_locations=[str(src.parent)] if is_pkg else None)
        mod = importlib.util.module_from_spec(spec)
        # Register BEFORE exec so intra-package relative imports
        # (`from .base import ...`) resolve against real mlx_lm.models.
        sys.modules[target] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            sys.modules.pop(target, None)
            raise
        _installed.append(target)
        done.append(name)
    return done


def unregister() -> None:
    for target in _installed:
        sys.modules.pop(target, None)
    _installed.clear()
