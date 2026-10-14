"""Which link a peer is reached over, and which links a page answers on.

Bonjour will happily hand back a peer's Wi-Fi address when a Thunderbolt
cable joins the machines, and a ring built over Wi-Fi is slow and
drops when a peer is "rediscovered". So knurlogic is explicit:

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

from knurlogic.machine.processes import PROC_ERRORS

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
    """This machine's Thunderbolt addresses, each with its cable's link
    speed: {"iface", "ip", "kind", "gbps", "generation"} (gbps/generation
    None for a bridge or when system_profiler does not say)."""
    try:
        speeds = _cached("tbspeed", 30.0, receptacle_speeds)
    except (*PROC_ERRORS, ValueError, KeyError):
        speeds = {}
    return [with_speed(i, speeds) for i in local_interfaces()
            if "Thunderbolt" in i["kind"]]


# ------------------------------------------------------- link speed
#
# Two Macs can be joined by two cables of different kinds; only a
# Thunderbolt 5 (80 Gb/s) one carries RDMA -- over the 40 Gb/s Thunderbolt 4
# cable, ibv_devinfo still says PORT_ACTIVE and jaccl
# fails RTR with errno 96. networksetup names each interface's hardware
# port "Thunderbolt N"; system_profiler reports receptacle N's current
# speed. The two numbers agree (e.g. Thunderbolt 6 = en7 = receptacle 6
# at 80 Gb/s).

TB5_GBPS = 80


def generation(gbps) -> int | None:
    """5 for an 80 Gb/s (or faster) link, 4 for 40, 3 for 20 or less."""
    if not gbps:
        return None
    return 5 if gbps >= TB5_GBPS else 4 if gbps >= 40 else 3


def parse_receptacle_speeds(doc: dict) -> dict:
    """{receptacle number: current Gb/s} for connected receptacles, from
    `system_profiler SPThunderboltDataType -json`. "Up to 120 Gb/s" is an
    empty port's capability, not a link, and is left out."""
    out = {}

    def walk(o):
        if isinstance(o, dict):
            rid, sp = o.get("receptacle_id_key"), o.get("current_speed_key")
            if rid and isinstance(sp, str) and \
                    o.get("receptacle_status_key") == "receptacle_connected" \
                    and not sp.lower().startswith("up to"):
                try:
                    out[int(rid)] = int(float(sp.split()[0]))
                except (ValueError, IndexError):
                    pass
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(doc if isinstance(doc, dict) else {})
    return out


def receptacle_speeds(run=None) -> dict:
    import json
    out = (run or _out)(["system_profiler", "SPThunderboltDataType",
                         "-json"])
    try:
        return parse_receptacle_speeds(json.loads(out or "{}"))
    except ValueError:
        return {}


def with_speed(i: dict, speeds: dict) -> dict:
    """An interface entry with its link speed, by its "Thunderbolt N"
    hardware port -> receptacle N."""
    k = str(i.get("kind") or "").split()
    n = int(k[1]) if len(k) == 2 and k[0] == "Thunderbolt" \
        and k[1].isdigit() else None
    g = speeds.get(n) if n is not None else None
    return {**i, "gbps": g, "generation": generation(g)}


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
        except PROC_ERRORS:
            return "other"
        for line in out.splitlines():
            if line.strip().startswith("interface:"):
                return kind_of_iface(line.split(":", 1)[1].strip())
        return "other"
    return _cached(("link", ip), 60.0, ask)


def iface_of(ip: str) -> str:
    """The local interface the OS routes to `ip` through ("" if none)."""
    def ask():
        out = _out(["route", "-n", "get", ip]) or ""
        for line in out.splitlines():
            if line.strip().startswith("interface:"):
                return line.split(":", 1)[1].strip()
        return ""
    return _cached(("iface", ip), 60.0, ask)


def gbps_of(ip: str):
    """The Thunderbolt link speed (Gb/s) to `ip`, None when the route is
    not over Thunderbolt or the speed is not known."""
    ifc = iface_of(ip)
    return next((i.get("gbps") for i in thunderbolt() if i["iface"] == ifc),
                None) if ifc else None


class Gate:
    """For --host cluster: answer on loopback and Thunderbolt only."""

    LOOPBACK = {"127.0.0.1", "::1", "::ffff:127.0.0.1"}

    def allows(self, local_ip: str) -> bool:
        ip = (local_ip or "").removeprefix("::ffff:")
        if ip in self.LOOPBACK or ip.startswith("127."):
            return True
        # per connection: the addresses only, never the speed probe
        return ip in {i["ip"] for i in local_interfaces()
                      if "Thunderbolt" in i["kind"]}

    def refusal(self, local_ip: str) -> bytes:
        tb = ", ".join(i["ip"] for i in thunderbolt()) or "none right now"
        return (f"this knurlogic answers on its Thunderbolt link only "
                f"(--host cluster); the request arrived on {local_ip}. "
                f"Thunderbolt addresses: {tb}.").encode()


# ----------------------------------------------------------------- RDMA

def _out(cmd) -> str | None:
    """stdout, or None when the tool is not there or fails to run."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    except PROC_ERRORS:
        return None
    return r.stdout if r.returncode == 0 else None


def rdma(run=_out) -> dict:
    """Whether jaccl can run here: {"available", "reason", "devices",
    "active"}. `rdma_ctl status` says whether RDMA over Thunderbolt is
    enabled (it is off until `rdma_ctl enable` from recovery), `ibv_devices`
    lists the rdma_en* devices, and `ibv_devinfo` says which have a cable
    in (PORT_ACTIVE). Nothing is changed."""
    st = run(["rdma_ctl", "status"])
    if st is None:
        return {"available": False, "devices": [], "active": [],
                "reason": "no rdma_ctl: this macOS has no RDMA over "
                          "Thunderbolt (macOS 26.2 or later)"}
    if "enabled" not in st.lower() or "disabled" in st.lower():
        return {"available": False, "devices": [], "active": [],
                "reason": "RDMA over Thunderbolt is disabled: run "
                          "`rdma_ctl enable` from recoveryOS, then reboot"}
    devs = [ln.split()[0] for ln in (run(["ibv_devices"]) or "").splitlines()
            if ln.strip().startswith("rdma_")]
    if not devs:
        return {"available": False, "devices": [], "active": [],
                "reason": "RDMA is enabled but ibv_devices lists no device"}
    active, cur = [], None
    for ln in (run(["ibv_devinfo"]) or "").splitlines():
        s = ln.strip()
        if s.startswith("hca_id:"):
            cur = s.split(":", 1)[1].strip()
        elif s.startswith("state:") and "PORT_ACTIVE" in s and cur:
            active.append(cur)
    if not active:
        return {"available": False, "devices": devs, "active": [],
                "reason": "no Thunderbolt port has an RDMA link up (no "
                          "cable, or the other Mac has RDMA off)"}
    return {"available": True, "devices": devs, "active": active,
            "reason": ""}


# ------------------------------------------------------- the cable a job runs on
#
# A job's ranks talk over one Thunderbolt /24 per pair of machines. Each
# function here reads only the machines' cluster blocks (`thunderbolt`,
# `rdma`) and BAD_CABLES, so every page picks the same cable from the same
# inputs.

#: frozenset of two machine ids -> {subnet: why} -- a cable a rank's link
#: init failed on while this page runs (e.g. one cable failing jaccl QP
#: RTR with errno 96 while the other works): tried last from then on
BAD_CABLES: dict = {}


def subnet(ip: str) -> str:
    """The /24 an IPv4 address sits on, as its first three octets."""
    return ".".join(str(ip).split(".")[:3])


def machine_pair(a: dict, b: dict) -> frozenset:
    """Two machines as BAD_CABLES' key: their ids (else names)."""
    return frozenset((str(a.get("id") or a.get("name") or ""),
                      str(b.get("id") or b.get("name") or "")))


def link_gbps(a: dict, b: dict, net: str):
    """The speed of the cable on subnet `net` between two machines: the
    slower end's Gb/s, None when either end does not say (an older
    knurlogic)."""
    sp = []
    for m in (a, b):
        g = next((t.get("gbps") for t in m.get("thunderbolt") or []
                  if t.get("ip") and subnet(t["ip"]) == net), None)
        if not isinstance(g, (int, float)) or isinstance(g, bool) or g <= 0:
            return None
        sp.append(g)
    return min(sp)


def cable_name(gbps) -> str:
    g = generation(gbps)
    return f"Thunderbolt {g} ({gbps:g} Gb/s)" if g else "speed unknown"


def shared_subnets(a: dict, b: dict, rdma: bool = False) -> list:
    """Every Thunderbolt /24 two machines both sit on, the order a launch
    tries them: the fastest cable first (Thunderbolt 5 over Thunderbolt 4;
    a link whose speed an older peer does not report counts as slowest),
    then the lowest subnet, so every page picks the same -- except that a
    cable whose link init failed between these two this session goes last.
    `rdma`: only a subnet with RDMA up at both ends AND a Thunderbolt 5
    link at both ends: over a 40 Gb/s Thunderbolt 4 cable ibv_devinfo
    still says PORT_ACTIVE, and jaccl fails RTR with errno 96."""
    def on(m):
        act = set((m.get("rdma") or {}).get("active") or [])
        return {subnet(t["ip"]) for t in m.get("thunderbolt") or []
                if t.get("ip") and (not rdma
                                    or f"rdma_{t.get('iface')}" in act)}
    bad = BAD_CABLES.get(machine_pair(a, b)) or {}
    both = on(a) & on(b)
    if rdma:
        both = {n for n in both
                if (link_gbps(a, b, n) or TB5_GBPS) >= TB5_GBPS}
    return sorted(both, key=lambda n: (n in bad, -(link_gbps(a, b, n) or 0),
                                       n))


def rdma_pair_reason(a: dict, b: dict) -> str:
    """Why jaccl cannot join these two machines ("" when it can): RDMA off
    at either end, or no Thunderbolt 5 cable between them."""
    for m in (a, b):
        rd = m.get("rdma") or {}
        if not rd.get("available"):
            return f"RDMA on {m.get('name')}: {rd.get('reason') or 'unknown'}"
    if shared_subnets(a, b, rdma=True):
        return ""
    slow = [n for n in shared_subnets(a, b)
            if (link_gbps(a, b, n) or TB5_GBPS) < TB5_GBPS]
    if slow:
        return ("RDMA needs a Thunderbolt 5 cable between these Macs; "
                + "; ".join(f"the {n} link is "
                            f"{cable_name(link_gbps(a, b, n))}"
                            for n in slow))
    return ""


def shared_subnet(a: dict, b: dict, rdma: bool = False) -> str:
    """The one Thunderbolt /24 two machines both sit on ("" if none) --
    lowest first, so every page picks the same. Two Macs joined by two
    cables share two subnets; BOTH ends of a link must be on the same one
    (en2 on one /24 and en4 on another are different cables, and a jaccl queue
    pair across them fails RTR with errno 60).
    `rdma`: only a subnet whose interface has RDMA up on both ends. A
    cable that failed link init between the two this session goes last
    (`shared_subnets`)."""
    both = shared_subnets(a, b, rdma)
    return both[0] if both else ""


def on_subnet(m: dict, net: str, key: str = "ip"):
    return next((t.get(key) for t in m.get("thunderbolt") or []
                 if t.get("ip") and subnet(t["ip"]) == net), None)


def ring_ips(infos: list, rdma: bool = False, net: str = "") -> list:
    """Each rank's address on the ring, in rank order. Two machines: both
    on the subnet they share (`shared_subnet`). More: a Thunderbolt
    address sharing a /24 with a neighbour's when there is one, else its
    first. Raises ValueError naming a machine with none."""
    ips = []
    n = len(infos)
    for _r, m in enumerate(infos):
        mine = [t["ip"] for t in m.get("thunderbolt") or [] if t.get("ip")]
        if not mine:
            raise ValueError(f"{m['name']} has no Thunderbolt address: the "
                             f"ring runs over the Thunderbolt bridge")
    if n == 2:
        net = net or shared_subnet(infos[0], infos[1], rdma) \
            or shared_subnet(infos[0], infos[1])
        if net:
            return [on_subnet(m, net) for m in infos]
    for r, m in enumerate(infos):
        mine = [t["ip"] for t in m.get("thunderbolt") or [] if t.get("ip")]
        near = {subnet(t["ip"]) for k in ((r - 1) % n, (r + 1) % n)
                for t in infos[k].get("thunderbolt") or []}
        ips.append(next((ip for ip in mine if subnet(ip) in near), mine[0]))
    return ips


def rdma_device(m: dict, peer: dict, net: str = ""):
    """The rdma_<iface> device on `m` that reaches `peer`: the one on the
    Thunderbolt subnet both share with RDMA up at both ends -- the same
    subnet from either side --,
    else None: a device on another subnet is another cable, never a
    fallback."""
    net = net or shared_subnet(m, peer, rdma=True)
    if net:
        return "rdma_" + str(on_subnet(m, net, "iface"))
    return None
