"""The CLI over machine/: `knurlogic loaded`, `knurlogic models` (and its
folder subcommands) and `knurlogic deps`. machine/ returns documents and
text; printing them is here."""

from __future__ import annotations

import argparse
import json
import sys


def loaded(argv=None) -> int:
    from knurlogic.machine import loaded as L
    p = argparse.ArgumentParser(
        prog="knurlogic loaded",
        description="what is in memory right now, in every runtime on this "
                    "machine")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    doc = L.survey()
    print(json.dumps(doc, indent=1) if a.json else L.render(doc))
    return 0


def deps(argv=None) -> int:
    from knurlogic.machine import deps as D
    p = argparse.ArgumentParser(prog="knurlogic deps",
                                description="what this stack stands on, "
                                            "and what is stock")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    doc = D.survey()
    print(json.dumps(doc, indent=1) if a.json else D.render(doc))
    return 0


def models(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("add", "remove", "folders"):
        return folders(argv)
    from knurlogic.machine import discover
    from knurlogic.machine.memory import wired
    p = argparse.ArgumentParser(
        prog="knurlogic models",
        description="find the models already on this machine, in every "
                    "store, and say which ones can actually run here. "
                    "`knurlogic models add <folder>` remembers a folder "
                    "(e.g. on an external drive), `remove <folder>` forgets "
                    "it, `folders` lists them.")
    p.add_argument("--store", action="append", default=[],
                   help="limit to one store (exo, huggingface, ollama, "
                        "'lm studio')")
    p.add_argument("--servable", action="store_true",
                   help="only the ones this engine can load")
    p.add_argument("--path", action="append", default=[],
                   help="also scan this directory")
    p.add_argument("--only-path", action="store_true",
                   help="scan ONLY --path, not the tools' own stores")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    rows = discover.find(stores=a.store or None, extra=a.path,
                         include_defaults=not a.only_path)
    if a.servable:
        rows = [f for f in rows if f.servable]
    if a.json:
        print(json.dumps([{**vars(f), "path": str(f.path)} for f in rows],
                         indent=1))
        return 0
    print(discover.render(rows, wired.detected_working_set_bytes()))
    return 0


def folders(argv) -> int:
    """knurlogic models add|remove <folder>, knurlogic models folders."""
    from knurlogic.machine import folders as F
    cmd, rest = argv[0], argv[1:]
    if cmd == "folders":
        lines = F.listing()
        if not lines:
            print("no saved model folders (knurlogic models add <folder>)")
        for line in lines:
            print(line)
        return 0
    if len(rest) != 1:
        print(f"usage: knurlogic models {cmd} <folder>", file=sys.stderr)
        return 2
    out = F.add(rest[0]) if cmd == "add" else F.remove(rest[0])
    if "error" in out:
        print(f"knurlogic: {out['error']}", file=sys.stderr)
        return 1
    print(f"{'added' if cmd == 'add' else 'removed'} "
          f"{out.get('added') or out.get('removed')}")
    return 0
