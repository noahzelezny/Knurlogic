"""This machine's Nvidia GPU as `nvidia-smi` reports it: the readouts a Mac
gets from ioreg and sysctl, on Linux and Windows.

The one home for asking. `nvidia-smi` ships with the driver, so nothing is
installed for it; a machine without one answers None.
"""
from __future__ import annotations

import subprocess


def query(*fields: str) -> list | None:
    """The first GPU's `fields` (nvidia-smi --query-gpu names), as strings
    in that order, or None where there is no nvidia-smi or it fails."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2)
    except (OSError, subprocess.SubprocessError):
        return None
    line = out.stdout.strip().splitlines()[:1]
    if out.returncode != 0 or not line:
        return None
    got = [v.strip() for v in line[0].split(",")]
    return got if len(got) == len(fields) else None


def number(v: str) -> int | None:
    """An nvidia-smi figure as an int; "[N/A]" (a unified-memory GPU has no
    memory.used of its own) reads None."""
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None
