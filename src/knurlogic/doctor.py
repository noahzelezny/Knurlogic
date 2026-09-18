"""`knurlogic doctor` -- say whether this artifact will run, and why not.

The failure this exists to prevent is a stack trace. A downloader who cannot
tell "you are missing an architecture file" from "you are out of memory" from
"this model does not fit your machine" gives up, and every one of those looks
identical from inside mlx-lm.
"""

from __future__ import annotations

import argparse
import sys

from . import arch
from .artifact import Artifact
from .resolve import resolve

GIB = 1 << 30


def run(path: str, working_set_gib: float, profile: str,
        exports: bool) -> int:
    try:
        a = Artifact.load(path)
    except FileNotFoundError as e:
        print(f"NOT AN ARTIFACT: {e}", file=sys.stderr)
        return 2

    ws = int(working_set_gib * GIB)
    r = resolve(a, ws, profile=profile)

    if exports:
        print(r.as_exports())
        return 0

    print(f"artifact   {a.path.name}")
    print(f"  type     {a.model_type}")
    print(f"  size     {a.gib:.1f} GiB"
          + (f"   working set {working_set_gib:.1f} GiB" if ws else ""))
    if a.is_vq:
        geo = ", ".join(f"d{d}-K{K} x{n}"
                        for (d, K), n in sorted(a.geometries.items()))
        print(f"  kernels  {a.model_file or 'NONE'}   ({geo})")

    print("\narchitecture")
    rows = arch.check(a.model_type)
    if not rows:
        print(f"  ?  {a.model_type}: no mapping. Unknown architecture -- this "
              f"is the case Knurlogic exists to shorten.")
    for row in rows:
        mark = {"OK": "ok", "UNPINNED": "??", "DRIFTED": "!!",
                "MISSING": "XX"}[row.state]
        extra = ""
        if row.state == "UNPINNED":
            extra = "  present but not pinned -- 'it imports' is not 'it is " \
                    "the arithmetic we measured'"
        elif row.state == "MISSING":
            extra = "  NOT INSTALLED -- this artifact cannot load"
        elif row.state == "DRIFTED":
            extra = "  does not match the pinned digest"
        print(f"  {mark} {row.module} [{row.origin}]{extra}")

    print("\nsettings")
    for k, v in sorted(r.env.items()):
        print(f"  {k}={v}")

    for n in r.notes:
        print(f"\n  note: {n}")
    for w in r.warnings:
        print(f"\n  WARNING: {w}")

    blocked = any(row.state == "MISSING" for row in rows) or r.warnings
    print("\n" + ("WILL NOT RUN as configured" if blocked
                  else "no blockers found"))
    return 1 if blocked else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="knurlogic doctor")
    p.add_argument("artifact")
    p.add_argument("--working-set-gib", type=float, default=0.0,
                   help="usable GPU working set. 0 = unknown, which leaves "
                        "the memory knobs at their defaults.")
    p.add_argument("--profile", default="v1.5", choices=("v1.5", "v2"))
    p.add_argument("--exports", action="store_true",
                   help="print only `export K=V` lines, for eval")
    a = p.parse_args(argv)
    return run(a.artifact, a.working_set_gib, a.profile, a.exports)


if __name__ == "__main__":
    raise SystemExit(main())
