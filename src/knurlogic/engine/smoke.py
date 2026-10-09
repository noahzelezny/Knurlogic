"""`knurlogic smoke` -- generate a token, and prove WHERE the code came from.

Every other check in this package reads bytes; none runs the model, and an
artifact can pass all of them and still be unable to emit a token.

The provenance half asserts that a downloader has exactly three things --
the artifact directory, a released mlx-lm, and this package -- and that
every architecture resolved from one of them. An architecture resolved
from a file hand-grafted into site-packages is a failure: a smoke that
passes only in the author's environment tests nothing a downloader has.

    knurlogic smoke <artifact> [--max-tokens N] [--pin]
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from knurlogic.engine import arch, register
from knurlogic.engine import model as engine
from knurlogic.machine.artifact import Artifact
from knurlogic.tuning.resolve import resolve

GIB = 1 << 30


def _origin(path, artifact_dir: Path) -> str:
    """Which of the three legitimate places did this come from?"""
    if not path:
        return "unknown"
    rp = Path(path).resolve()
    if register.is_vendored_path(rp):
        return "knurlogic"
    if rp.is_relative_to(artifact_dir.resolve()):
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

    needed = arch.modules_for_artifact(a)
    if needed:
        done = register.register(*needed)
        print(f"registered {done or '(already imported)'}")
    else:
        print(f"no architecture mapping for {a.model_type!r} -- "
              f"relying on whatever mlx-lm ships")

    r = resolve(a, int(working_set_gib * GIB))
    import os
    os.environ.update(r.env)

    print(f"engine    {engine.describe()}")
    if a.model_file:
        print(f"executing {a.model_file} from the artifact "
              f"(its VQ kernels live there)")
    model, tokenizer = engine.load(str(a.path),
                                   executes_artifact_code=bool(a.model_file))

    print("\nprovenance")
    problems = []
    for mod in needed:
        try:
            m = importlib.import_module(f"{arch.host_for(mod)}.models.{mod}")
            where = _origin(getattr(m, "__file__", None), a.path)
        except (ImportError, ValueError, OSError):
            where, m = "unknown", None
        print(f"  {mod:14s} {where}")
        if where in ("site-packages", "checkout", "unknown"):
            problems.append(
                f"{mod} resolved from {where}; a downloader installing "
                f"knurlogic would not have that copy")

    print("\ngenerating...")
    text = engine.generate(model, tokenizer,
                           "The capital of France is", max_tokens).strip()
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
        from knurlogic.engine import families
        for row in arch.check(a.model_type):
            if not (row.vendored and row.sha256):
                continue
            # Each pin goes into the pins.json of the family that owns it.
            pins = families.architecture_dir(
                families.family_of_module(row.module)) / "pins.json"
            data = json.loads(pins.read_text()) if pins.is_file() else {}
            # MERGE into the row: notes a person wrote beside a pin
            # (text_path_of, text_path_held_by) survive a re-pin. One schema
            # for every row: the mlx-lm the run used, always, plus the host.
            row_pin = data.setdefault(row.module, {})
            row_pin.update({
                "sha256": row.sha256,
                "host": arch.host_for(row.module),
                "validated_with_mlx_lm": engine.info("mlx_lm").version,
                "validated_on_artifact": a.path.name,
            })
            if row_pin["host"] != "mlx_lm":
                row_pin[f"validated_with_{row_pin['host']}"] = \
                    engine.info(row_pin["host"]).version
            pins.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
            print(f"pinned {row.module} -> {row.sha256[:16]}...")
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
