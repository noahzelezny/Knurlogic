"""Which link a peer is reached over, and which links a page answers on.

Bonjour will happily hand back a peer's Wi-Fi address when a Thunderbolt
cable joins the two machines -- the owner's experience with exo: peers
preferred Wi-Fi and were "rediscovered" after connecting, which destabilised
the ring, until addresses were pinned by hand. So knurlogic is explicit:

  --host cluster   bind every address, ANSWER only on loopback and the
                   Thunderbolt links (checked per connection against the
                   address it arrived on, so a replugged bridge that
                   re-addresses keeps working), advertise only there.
  link_of(ip)      the local interface the OS routes to a peer through,
                   and what kind it is; a peer reachable two ways is kept
                   on Thunderbolt.
"""

from __future__ import annotations

import subprocess
import time

_CACHE: dict = {}


def _cached(key, ttl, fn):
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    val = fn()
    _CACHE[key] = (time.time(), val)
    return val


def local_interfaces() -> list[dict]:
    from knurlogic.cluster.checks import interfaces
    return _cached("ifs", 10.0, interfaces)


def thunderbolt() -> list[dict]:
    return [i for i in local_interfaces() if "Thunderbolt" in i["kind"]]


def kind_of_iface(iface: str) -> str:
    k = next((i["kind"] for i in local_interfaces() if i["iface"] == iface),
             "")
    if "Thunderbolt" in k:
        return "thunderbolt"
    if "Wi-Fi" in k:
        return "wifi"
    if iface == "lo0":
        return "loopback"
    return "ethernet" if k and k != "?" else "other"


def link_of(ip: str) -> str:
    """thunderbolt / wifi / ethernet / loopback / other, by the route."""
    def ask():
        try:
            out = subprocess.run(["route", "-n", "get", ip],
                                 capture_output=True, text=True,
                                 timeout=3).stdout
        except Exception:
            return "other"
        for line in out.splitlines():
            if line.strip().startswith("interface:"):
                return kind_of_iface(line.split(":", 1)[1].strip())
        return "other"
    return _cached(("link", ip), 60.0, ask)


class Gate:
    """For --host cluster: answer on loopback and Thunderbolt only."""

    LOOPBACK = {"127.0.0.1", "::1", "::ffff:127.0.0.1"}

    def allows(self, local_ip: str) -> bool:
        ip = (local_ip or "").removeprefix("::ffff:")
        if ip in self.LOOPBACK or ip.startswith("127."):
            return True
        return ip in {i["ip"] for i in thunderbolt()}

    def refusal(self, local_ip: str) -> bytes:
        tb = ", ".join(i["ip"] for i in thunderbolt()) or "none right now"
        return (f"this knurlogic answers on its Thunderbolt link only "
                f"(--host cluster); the request arrived on {local_ip}. "
                f"Thunderbolt addresses: {tb}.").encode()
