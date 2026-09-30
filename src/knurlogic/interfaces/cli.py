"""knurlogic -- one entry point, subcommands underneath."""

from __future__ import annotations

import sys

from knurlogic import __version__

#: command -> (module under knurlogic, what it does). The module path IS the
#: routing: interfaces.* talk to people and agents, machine.* read this box,
#: engine.* touch what runs a model.
COMMANDS = {
    "ui": ("interfaces.page.server", "open the page without loading "
           "anything: every model, every runtime, and where the memory "
           "went"),
    "serve": ("interfaces.serve", "run an OpenAI-compatible endpoint for an artifact"),
    "doctor": ("interfaces.doctor", "say whether an artifact will run, and why not"),
    "smoke": ("engine.smoke", "generate a token and prove where the code came from"),
    "vendor": ("engine.vendor", "take an architecture file under version control"),
    "connect": ("interfaces.connect",
                "print how to point a client at a running server"),
    "mcp": ("interfaces.mcp", "serve the agent-facing tool interface on stdio"),
    "loaded": ("machine.loaded", "what is in memory right now, in every runtime on "
                         "this machine"),
    "mtp": ("interfaces.drafting", "which artifacts have a drafting head, "
            "and which only declare one"),
    "models": ("machine.discover", "find the models already on this machine, in "
                           "every tool's store"),
    "deps": ("machine.deps", "what this stack stands on, and which pieces are "
                     "stock and which are forks"),
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

    mod = importlib.import_module(f"knurlogic.{COMMANDS[cmd][0]}")
    return mod.main(args[1:])


if __name__ == "__main__":
    raise SystemExit(main())
