"""`knurlogic doctor` -- say whether this artifact will run, and why not.

The failure this exists to prevent is a stack trace. A downloader who cannot
tell "you are missing an architecture file" from "you are out of memory" from
"this model does not fit your machine" gives up, and every one of those looks
identical from inside mlx-lm.
"""

from __future__ import annotations

import argparse
import sys

from knurlogic.engine import arch
from knurlogic.machine.artifact import Artifact
from knurlogic.machine.memory import wired
from knurlogic.tuning import presets
from knurlogic.tuning.resolve import resolve

GIB = 1 << 30


def run(path: str, working_set_gib: float, profile: str | None,
        exports: bool, tune: str = "default") -> int:
    try:
        a = Artifact.load(path)
    except FileNotFoundError as e:
        print(f"NOT AN ARTIFACT: {e}", file=sys.stderr)
        return 2

    ws = int(working_set_gib * GIB)
    detected = ""
    if ws == 0:
        ws = wired.detected_working_set_bytes()
        if ws:
            detected = " (detected)"
            from knurlogic.machine.memory import allowance
            if allowance.get() and allowance.cap(ws) < ws:
                ws = allowance.cap(ws)
                detected = " (the knurlogic allowance)"
            working_set_gib = ws / GIB

    r = resolve(a, ws, profile=profile, tune=tune)

    if exports:
        print(r.as_exports())
        return 0

    print(f"artifact   {a.path.name}")
    print(f"  type     {a.model_type}")
    print(f"  size     {a.gib:.1f} GiB"
          + (f"   working set {working_set_gib:.1f} GiB{detected}"
             if ws else "   working set UNKNOWN"))
    if ws and a.bytes_on_disk < ws:
        from knurlogic.tuning.fit import room_for
        room = room_for(a.bytes_on_disk, a.raw_config, ws)
        print(f"  room     {room['text']}"
              + ("   SMALL: little to talk in" if room["small"] else ""))
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

    from knurlogic.engine import model as engine
    from knurlogic.engine import mtp
    m = mtp.status(a)
    if m.state != mtp.NONE:
        print("\nmtp        " + m.render())
    if m.state == mtp.GRAFTABLE and engine.keeps_mtp_weights(a.model_type) is False:
        print("           (`sanitize()` in that architecture drops them at "
              "load, which is why\n           building the head is a "
              "separate step and not a flag.)")

    ts = engine.tool_support(a.chat_template())
    if ts["mentions_tools"]:
        if ts["parser"]:
            print(f"\ntools      {ts['parser']} dialect, read from the chat "
                  f"template")
        else:
            print("\ntools      TEMPLATE ASKS FOR TOOL CALLS AND THE ENGINE "
                  "INFERRED NO PARSER.\n           Calls will come back as "
                  "prose, so a harness sees a model that describes the "
                  "function\n           it would call instead of calling "
                  "it.")
    elif ts["has_template"]:
        print("\ntools      this artifact's chat template does not mention "
              "tools")

    reads = a.knobs_read()
    if reads:
        print(f"\n  this artifact's runtime reads {len(reads)} settings; "
              f"knurlogic has a measured answer for {len(r.env)}. The rest "
              f"keep the runtime's own defaults.")

    adv = wired.advise(a.bytes_on_disk)
    if adv.get("known"):
        print("\n" + wired.render(adv))

    # A model that does not fit the CURRENT limit but fits under the ceiling
    # is not a model that does not fit. Saying "will not run" there would
    # send someone to buy a machine they already own.
    blocked = any(row.state == "MISSING" for row in rows) or r.warnings
    if blocked and adv.get("action") == "raise":
        print("\n  ^ the memory warning above is a SETTING, not a limit of "
              "this machine: the command above is the fix.")
    print("\n" + ("WILL NOT RUN as configured" if blocked
                  else "no blockers found"))
    return 1 if blocked else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="knurlogic doctor")
    p.add_argument("artifact", nargs="?",
                   help="the artifact to check (not needed with --cluster)")
    p.add_argument("--cluster", action="store_true",
                   help="check what stops this Mac and the others finding "
                        "each other: addresses, firewall, Bonjour, peers. "
                        "Run it on each Mac.")
    p.add_argument("--port", type=int, default=8899,
                   help="with --cluster: the port the page runs on")
    p.add_argument("--working-set-gib", type=float, default=0.0,
                   help="usable GPU working set. 0 = ask the framework what "
                        "it may use; pass a number to override it.")
    p.add_argument("--profile", default=None, choices=("v1.5", "v2"),
                   help="force a VQ numerics profile. Default: none -- "
                        "each model runs the numerics its own model.py "
                        "ships with. Forcing v1.5 on a v2 model changes "
                        "its outputs.")
    p.add_argument("--tune", default="default", type=presets.preset_arg,
                   metavar="{default,lean}",
                   help="the launch preset, the same two `serve --tune` "
                        "takes: default (the measured settings) or lean "
                        "(lowest peak memory, most context). Capped by what "
                        "has been measured.")
    p.add_argument("--exports", action="store_true",
                   help="print only `export K=V` lines, for eval")
    a = p.parse_args(argv)
    if a.cluster:
        from knurlogic.cluster.checks import report
        text, bad = report(a.port)
        print(text)
        return 1 if bad else 0
    if not a.artifact:
        from knurlogic.machine import discover
        print(discover.render(discover.find()))
        print("\nRun `knurlogic doctor <artifact>` to check one model "
              "against this machine, or `knurlogic doctor --cluster`.")
        return 0
    return run(a.artifact, a.working_set_gib, a.profile, a.exports, a.tune)


if __name__ == "__main__":
    raise SystemExit(main())
