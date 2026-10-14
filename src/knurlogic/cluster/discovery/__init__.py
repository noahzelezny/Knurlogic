"""Every knurlogic page on the network, found without being named.

DNS-SD service type `_knurlogic._tcp`, through the service the OS already
runs: Bonjour on macOS (bonjour.py), Avahi on Linux (avahi.py). The page
(`ui`) registers one advertisement per machine (TXT: id, name, ver, schema,
role) and every page browses. `Discovery` is this OS's; each has
`register`, `browse`, `start`, `stop`, `snapshot` and `status`, and `found`
is {(instance, if_index): {name, if_index, host, port, txt}} for services
that resolved to an IPv4 address. A failure is "nothing found", never a
crash, and `status()` says which.

Design: docs/design/discovery.md (Bonjour).
"""
from __future__ import annotations

import socket
import sys

SERVICE = "_knurlogic._tcp"


def txt_encode(d: dict) -> bytes:
    out = b""
    for k, v in d.items():
        item = f"{k}={v}".encode()[:255]
        out += bytes([len(item)]) + item
    return out


def txt_decode(b: bytes) -> dict:
    out, i = {}, 0
    while i < len(b):
        n = b[i]
        item = b[i + 1:i + 1 + n].decode(errors="replace")
        k, _, v = item.partition("=")
        if k:
            out[k] = v
        i += 1 + n
    return out


def label(s: str) -> bytes:
    """A DNS label: at most 63 BYTES, never cut inside a UTF-8 character."""
    b = s.encode()
    return b[:63].decode(errors="ignore").encode() if len(b) > 63 else b


def interface_of(ip: str) -> int:
    """The index of the interface an address is configured on, or 0 (all)."""
    if ip in ("", "0.0.0.0", "::"):
        return 0
    import psutil
    for iface, addrs in psutil.net_if_addrs().items():
        if any(a.family == socket.AF_INET and a.address == ip for a in addrs):
            try:
                return socket.if_nametoindex(iface)
            except OSError:
                return 0
    return 0


if sys.platform == "darwin":
    from knurlogic.cluster.discovery.bonjour import Discovery
else:
    from knurlogic.cluster.discovery.avahi import Discovery

__all__ = ["SERVICE", "Discovery", "interface_of", "label", "txt_decode",
           "txt_encode"]
