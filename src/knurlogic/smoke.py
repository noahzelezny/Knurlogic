"""`knurlogic smoke` -- generate a token, and prove WHERE the code came from.

Every other check in this package reads bytes. None of them runs the model,
and an artifact can pass all of them and still be unable to emit a token.

The provenance half is not decoration, and the rule is borrowed from a
project that paid for it: three published artifacts shipped a bundle
importing a module that existed only in the build venvs. They passed their
smoke -- because it ran where that module happened to exist. The gate was
testing the artifact in the AUTHOR's environment, not a downloader's.

Their rule was "a downloader has exactly two things: the artifact directory,
and a released mlx-lm." Knurlogic changes that premise to THREE -- the
artifact, a released mlx-lm, and this package -- so the assertion is the same
shape with one more allowed origin. What stays a failure: an architecture
resolved from a file hand-grafted into site-packages, because a downloader
does not have that and never will.

    knurlogic smoke <artifact> [--max-tokens N] [--pin]
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from . import arch, register
from .artifact import Artifact
from .resolve import resolve

GIB = 1 << 30
PINS = register.ARCH_DIR / "PINS.json"


def _origin(path, artifact_dir: Path) -> str:
    """Which of the three legitimate places did this come from?"""
    if not path:
        return "unknown"
    rp = Path(path).resolve()
    if str(rp).startswith(str(register.ARCH_DIR.resolve())):
        return "knurlogic"
    if str(rp).startswith(str(artifact_dir.resolve())):
        return "artifact"
    if "site-packages" in str(rp) or "dist-packages" in str(rp):
        return "site-packages"
    return "checkout"


def run(path: str, max_tokens: int, pin: bool, strict: bool,
        working_set_gib: float) -> int:
    a = Artifact.load(path)
    print(f"artifact  {a.path.name}  ({a.model_type}, {a.gib:.1f} GiB)")

    # A smoke that thrashes swap produces no verdict, only a slow one.
    if working_set_gib and a.gib > working_set_gib:
        print(f"REFUSED: {a.gib:.1f} GiB will not fit a "
              f"{working_set_gib:.1f} GiB working set. A smoke that swaps "
              f"measures nothing.", file=sys.stderr)
        return 2

    needed = arch.required_modules(a.model_type)
    if needed:
        done = register.register(*needed)
        print(f"registered {done or '(already imported)'}")
    else:
        print(f"no architecture mapping for {a.model_type!r} -- "
              f"relying on whatever mlx-lm ships")

    r = resolve(a, int(working_set_gib * GIB))
    import os
    os.environ.update(r.env)

    from mlx_lm.utils import load

    # The artifact's own model.py is executed -- that IS the VQ runtime.
    #
    # VERSION GLUE, and a small exhibit of why this package exists: mlx-lm
    # 0.31.3 (PyPI) executes `model_file` UNCONDITIONALLY; 0.32.0 put it
    # behind trust_remote_code= and raises without it. Passing the kwarg
    # blindly is a TypeError on 0.31.3; omitting it is a ValueError on
    # 0.32.0. So ask the installed signature instead of guessing, and say out
    # loud that artifact code is being executed either way.
    import inspect

    kw = {}
    if a.model_file:
        if "trust_remote_code" in inspect.signature(load).parameters:
            kw["trust_remote_code"] = True
        print(f"executing {a.model_file} from the artifact "
              f"(its VQ kernels live there)")
    model, tokenizer = load(str(a.path), **kw)

    print("\nprovenance")
    problems = []
    for mod in needed:
        try:
            m = importlib.import_module(f"{arch.host_for(mod)}.models.{mod}")
            where = _origin(getattr(m, "__file__", None), a.path)
        except Exception:
            where, m = "unknown", None
        print(f"  {mod:14s} {where}")
        if where in ("site-packages", "checkout", "unknown"):
            problems.append(
                f"{mod} resolved from {where}; a downloader installing "
                f"knurlogic would not have that copy")

    from mlx_lm.generate import generate
    print("\ngenerating...")
    out = generate(model, tokenizer, prompt="The capital of France is",
                   max_tokens=max_tokens, verbose=False)
    text = (out or "").strip()
    print(f"  -> {text!r}")
    if not text:
        print("FAIL: loaded but produced no tokens", file=sys.stderr)
        return 1

    if problems:
        for p in problems:
            print(f"\n{'FAIL' if strict else 'WARNING'}: {p}", file=sys.stderr)
        if strict:
            print("\n(--no-strict to record the run anyway)", file=sys.stderr)
            return 1

    print("\nSMOKE PASS")

    if pin:
        if problems:
            print("refusing to pin: provenance is not clean", file=sys.stderr)
            return 1
        data = json.loads(PINS.read_text()) if PINS.is_file() else {}
        import mlx_lm
        for row in arch.check(a.model_type):
            if row.vendored and row.sha256:
                data[row.module] = {
                    "sha256": row.sha256,
                    "host": arch.host_for(row.module),
                    "validated_with_mlx_lm": mlx_lm.__version__,
                    "validated_on_artifact": a.path.name,
                }
                print(f"pinned {row.module} -> {row.sha256[:16]}...")
        PINS.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="knurlogic smoke")
    p.add_argument("artifact")
    p.add_argument("--max-tokens", type=int, default=4)
    p.add_argument("--working-set-gib", type=float, default=0.0)
    p.add_argument("--pin", action="store_true",
                   help="on a clean pass, record the vendored digests as "
                        "validated against this mlx-lm")
    p.add_argument("--no-strict", dest="strict", action="store_false",
                   default=True)
    a = p.parse_args(argv)
    return run(a.artifact, a.max_tokens, a.pin, a.strict, a.working_set_gib)


if __name__ == "__main__":
    raise SystemExit(main())
