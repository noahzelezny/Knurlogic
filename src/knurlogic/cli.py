"""knurlogic -- one entry point, subcommands underneath."""

from __future__ import annotations

import sys

from . import __version__

COMMANDS = {
    "ui": ("ui", "open the page without loading anything: every model, every "
                 "runtime, and where the memory went"),
    "serve": ("serve", "run an OpenAI-compatible endpoint for an artifact"),
    "doctor": ("doctor", "say whether an artifact will run, and why not"),
    "smoke": ("smoke", "generate a token and prove where the code came from"),
    "vendor": ("vendor", "take an architecture file under version control"),
    "override": ("override", "replace a module inside mlx-lm, mlx-vlm or exo "
                             "without forking it"),
    "connect": ("connect", "print how to point a client at a running server"),
    "mcp": ("mcp", "serve the agent-facing tool interface on stdio"),
    "loaded": ("loaded", "what is in memory right now, in every runtime on "
                         "this machine"),
    "mtp": ("mtp", "which artifacts have a drafting head, and which only "
                   "declare one"),
    "models": ("discover", "find the models already on this machine, in "
                           "every tool's store"),
}


def usage() -> int:
    print(f"knurlogic {__version__}\n")
    print("usage: knurlogic <command> [args]\n")
    for name, (_, desc) in COMMANDS.items():
        print(f"  {name:8s} {desc}")
    print("\nknurlogic <command> --help for a command's own options.")
    return 0


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help", "help"):
        return usage()
    if args[0] in ("-V", "--version", "version"):
        print(__version__)
        return 0
    cmd = args[0]
    if cmd not in COMMANDS:
        print(f"knurlogic: unknown command {cmd!r}", file=sys.stderr)
        usage()
        return 2
    import importlib

    mod = importlib.import_module(f".{COMMANDS[cmd][0]}", __package__)
    return mod.main(args[1:])


if __name__ == "__main__":
    raise SystemExit(main())
