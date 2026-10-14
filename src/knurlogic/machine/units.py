"""Byte units, and the one way a byte count is written for a person.

  GIB, MIB  1 << 30 and 1 << 20: binary units, as the OS and the GPU
            working set count memory
  gib(n)    "12.3 GiB"
"""
from __future__ import annotations

GIB = 1 << 30
MIB = 1 << 20


def gib(n: float, places: int = 1) -> str:
    """`n` bytes in GiB to `places` decimals: gib(13 << 30) -> "13.0 GiB"."""
    return f"{n / GIB:.{places}f} GiB"
