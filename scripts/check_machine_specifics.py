"""Find machine-specific strings: a home directory, a mounted volume other than
the generic examples, a private-network address. Stdlib only.

Knurlogic runs on other people's Macs: code, tests and docs read paths and
addresses from config or the environment at runtime, and examples use
placeholders -- the home and volume names below and the RFC 5737
documentation ranges (192.0.2.x, 198.51.100.x, 203.0.113.x). Link-local
169.254.x (the Thunderbolt Bridge) and loopback are generic.

One home for the patterns, used by:
  tests/integration/test_no_machine_specifics.py   every test run
  scripts/git-hooks/pre-push                       every push (--range)
  .github/workflows/publish.yml                    every release (--tree)
A per-user scanner (`no-machine-names`, pre-commit) also catches machine and
host names, which a generic pattern cannot know.

  python scripts/check_machine_specifics.py --tree         tracked files
  python scripts/check_machine_specifics.py --range A..B   lines B adds over A
Exit 1 and a list when anything is found.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: example volumes the docs and tests use; a new one is added here, on purpose
EXAMPLE_VOLUMES = {"My SSD", "External SSD", "Models", "S", "x"}
#: placeholder home directories
EXAMPLE_HOMES = {"x", "you", "name", "me", "user", "Shared"}

PRIVATE_IP = re.compile(
    r"(?<![\d.])(10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    r"|192\.168\.\d{1,3}\.\d{1,3}"
    r"|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(?![\d.])")
# split so this file holds no hit of its own
HOME = re.compile("/" + "Users" + r"/([A-Za-z0-9._-]+)/")
VOLUME = re.compile("/" + "Volumes" + r"/([^/\"'`)\n]+)")


def hits(line: str) -> list[str]:
    out = [m.group(0) for m in PRIVATE_IP.finditer(line)]
    out += [m.group(0) for m in HOME.finditer(line)
            if m.group(1) not in EXAMPLE_HOMES]
    out += [m.group(0) for m in VOLUME.finditer(line)
            if m.group(1).strip() not in EXAMPLE_VOLUMES]
    return out


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True,
                          capture_output=True).stdout.decode("utf-8", "replace")


def scan_tree() -> list[str]:
    bad = []
    for rel in _git("ls-files", "-z").split("\0"):
        if not rel:
            continue
        try:
            text = (ROOT / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue                        # binary files (images, weights)
        for n, line in enumerate(text.splitlines(), 1):
            if h := hits(line):
                bad.append(f"{rel}:{n}: {', '.join(h)}")
    return bad


def scan_range(rng: str) -> list[str]:
    """Lines added by the commits in `rng`, and their messages."""
    bad, path = [], "?"
    for line in _git("log", "-p", "--format=MSG %h %s%n%b", rng).splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
        elif line.startswith("+") and not line.startswith("+++"):
            if h := hits(line):
                bad.append(f"{path}: {', '.join(h)}")
        elif not line.startswith(("-", " ", "@@", "diff ", "index ")):
            if h := hits(line):                 # commit message lines
                bad.append(f"(message) {line[:60]}: {', '.join(h)}")
    return bad


def main(argv: list[str]) -> int:
    if argv[:1] == ["--tree"]:
        bad = scan_tree()
    elif argv[:1] == ["--range"] and len(argv) == 2:
        bad = scan_range(argv[1])
    else:
        print(__doc__)
        return 2
    if bad:
        print("machine-specific strings (use config, an env var or a "
              "placeholder):\n  " + "\n  ".join(bad), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
