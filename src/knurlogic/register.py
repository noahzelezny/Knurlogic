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
_PREFIX = "mlx_lm.models."
_installed: list = []


def available() -> list:
    """Architecture modules vendored in this package."""
    if not ARCH_DIR.is_dir():
        return []
    return sorted(p.stem for p in ARCH_DIR.glob("*.py")
                  if not p.stem.startswith("_"))


def register(*names: str, override: bool = False) -> list:
    """Make the named vendored architectures resolvable as mlx_lm.models.<n>.

    With `override=False` (default) a module mlx-lm already imported is left
    alone and reported, rather than swapped underneath a loaded model.
    """
    wanted = list(names) or available()
    done = []
    for name in wanted:
        src = ARCH_DIR / f"{name}.py"
        if not src.is_file():
            raise FileNotFoundError(f"no vendored architecture {name!r}")
        target = _PREFIX + name
        if target in sys.modules and not override:
            continue
        spec = importlib.util.spec_from_file_location(target, src)
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
