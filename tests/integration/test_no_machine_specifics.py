"""No tracked file names a particular machine: no home directory, no mounted
volume other than the generic examples, no private-network address.

Knurlogic runs on other people's Macs, so code, tests and docs read paths and
addresses from config or the environment at runtime, and examples use
placeholders: the home and volume names below, and the RFC 5737
documentation ranges (192.0.2.x, 198.51.100.x, 203.0.113.x). Link-local
169.254.x (the Thunderbolt Bridge) and loopback are generic, not a machine's.

This check is generic on purpose, so it runs for every contributor. A
per-user scanner (scripts/git-hooks/pre-commit, `no-machine-names`) also
catches machine and host names, which a generic pattern cannot know. The
self-test builds its samples at runtime so this file holds none.
"""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: example volumes the docs and tests use; a new one is added here, on purpose
EXAMPLE_VOLUMES = {"My SSD", "External SSD", "Models", "S", "x"}
#: placeholder home directories
EXAMPLE_HOMES = {"x", "you", "name", "me", "user", "Shared"}

PRIVATE_IP = re.compile(
    r"(?<![\d.])(10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    r"|192\.168\.\d{1,3}\.\d{1,3}"
    r"|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(?![\d.])")
HOME = re.compile(r"/Users/([A-Za-z0-9._-]+)/")
VOLUME = re.compile("/" + "Volumes" + r"/([^/\"'`)\n]+)")  # split: no hit here


def _tracked() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True,
                         capture_output=True).stdout.decode()
    return [ROOT / p for p in out.split("\0") if p]


def test_no_tracked_file_names_a_machine():
    bad = []
    for f in _tracked():
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue                        # binary files (images, weights)
        for n, line in enumerate(text.splitlines(), 1):
            hits = [m.group(0) for m in PRIVATE_IP.finditer(line)]
            hits += [m.group(0) for m in HOME.finditer(line)
                     if m.group(1) not in EXAMPLE_HOMES]
            hits += [m.group(0) for m in VOLUME.finditer(line)
                     if m.group(1).strip() not in EXAMPLE_VOLUMES]
            if hits:
                bad.append(f"{f.relative_to(ROOT)}:{n}: {', '.join(hits)}")
    assert not bad, ("machine-specific strings in tracked files (use config, "
                     "an env var or a placeholder):\n" + "\n".join(bad))


def _ip(*parts) -> str:
    return ".".join(str(p) for p in parts)


def test_the_patterns_catch_what_they_are_for():
    assert PRIVATE_IP.search("ssh " + _ip(10, 1, 2, 3))
    assert PRIVATE_IP.search("inet " + _ip(192, 168, 7, 7) + " netmask")
    assert PRIVATE_IP.search(_ip(172, 20, 1, 4))
    assert not PRIVATE_IP.search(_ip(192, 0, 2, 2) + " " + _ip(169, 254, 1, 1)
                                 + " " + _ip(127, 0, 0, 1))
    assert not PRIVATE_IP.search("version " + _ip(110, 0, 0, 12))
    home = "/" + "Users" + "/someone/models/"
    assert HOME.search(home).group(1) == "someone"
    vol = '"/' + "Volumes" + '/Some Drive/Models"'
    assert VOLUME.search(vol).group(1) == "Some Drive"
