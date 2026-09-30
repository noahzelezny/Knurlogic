"""Install Knurlogic's architecture modules into mlx-lm, without touching it.

mlx-lm resolves a model class with

    importlib.import_module(f"mlx_lm.models.{model_type}")

and `import_module` consults `sys.modules` first. So registering a module
object under that name BEFORE the first load makes mlx-lm use ours, and
`site-packages` is never written to. A file grafted into someone else's
install inherits that install's mlx-lm version; a file vendored inside a
versioned package does not.

Reversible: `unregister()` drops the entries. The vendored file must still
be validated against a known mlx-lm -- PROVENANCE.md records which mlx-lm
each file was taken from. Design: docs/design/engine.md (architecture layer).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_installed: list = []


def available() -> list:
    """Every architecture module a family manifest lists AND vendors --
    flat files and packages, under engine/families/<family>/architecture/."""
    from knurlogic.engine import families
    return sorted(m for mf in families.manifests()
                  for m in mf["architectures"] if source_for(m)[0] is not None)


def source_for(name: str):
    """(path, is_package) for a vendored architecture, found through the
    family that lists it."""
    from knurlogic.engine import families
    fam = families.family_of_module(name)
    if fam is None:
        return None, False
    d = families.architecture_dir(fam)
    pkg = d / name
    if (pkg / "__init__.py").is_file():
        return pkg / "__init__.py", True
    flat = d / f"{name}.py"
    return (flat, False) if flat.is_file() else (None, False)


def is_vendored_path(path) -> bool:
    """Does this file live in one of knurlogic's vendored architecture dirs?"""
    from knurlogic.engine import families
    rp = str(Path(path).resolve())
    return any(rp.startswith(str(d.resolve()) + "/")
               for d in families.architecture_dirs())


def _with_dependencies(names: list) -> list:
    """Expand each name to [its bases..., itself], in load order.

    Why: qwen3_5_moe subclasses qwen3_5. Registering the subclass alone
    leaves a VENDORED subclass sitting on a SITE-PACKAGES base -- two
    versions of the arithmetic silently mixed, the precise failure this
    package exists to end. `sorted()` happening to put qwen3_5 before
    qwen3_5_moe must not be what makes it work.
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
        # An arch hosted under mlx_vlm (for its relative imports) is also
        # what mlx-lm's loader must find: `import_module("mlx_lm.models.X")`
        # is how both knurlogic's VQ runtime and a published bundle resolve
        # their base. Same module object under both names, so there is one
        # arithmetic. A module mlx-lm already has is left alone.
        alias = f"mlx_lm.models.{name}"
        if alias != target and alias not in sys.modules:
            sys.modules[alias] = mod
            _installed.append(alias)
        done.append(name)
    return done


def unregister() -> None:
    for target in _installed:
        sys.modules.pop(target, None)
    _installed.clear()
