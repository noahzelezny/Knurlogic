"""knurlogic -- one entry point, subcommands underneath."""

from __future__ import annotations

import sys

from . import __version__

COMMANDS = {
    "serve": ("serve", "run an OpenAI-compatible endpoint for an artifact"),
    "doctor": ("doctor", "say whether an artifact will run, and why not"),
    "smoke": ("smoke", "generate a token and prove where the code came from"),
    "vendor": ("vendor", "take an architecture file under version control"),
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
