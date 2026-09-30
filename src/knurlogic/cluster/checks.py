"""`knurlogic doctor --cluster`: everything that stops Macs finding each
other, checked from the machine you run it on, each with the fix.

Read-only by rule: the firewall is READ (`socketfilterfw --get...`) to name
the exact binary it lists, never written; the Local Network privacy
setting is named, never touched. The machine that has the problem is the
one that can fix it, so this is run on EACH Mac.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

from knurlogic.cluster import PROC_ERRORS

FW = "/usr/libexec/ApplicationFirewall/socketfilterfw"


def _run(cmd, timeout=5) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout).stdout.strip()
    except PROC_ERRORS:
        return ""


def interfaces() -> list[dict]:
    """IPv4 addresses that are not loopback, with what the hardware is."""
    ports, cur = {}, None
    for line in _run(["networksetup", "-listallhardwareports"]).splitlines():
        if line.startswith("Hardware Port:"):
            cur = line.split(":", 1)[1].strip()
        elif line.startswith("Device:") and cur:
            ports[line.split(":", 1)[1].strip()] = cur
    out: list = []
    iface = ""
    for line in _run(["ifconfig"]).splitlines():
        if line and not line[0].isspace():
            iface = line.split(":", 1)[0]
        elif line.strip().startswith("inet ") and iface != "lo0":
            ip = line.split()[1]
            kind = ports.get(iface, "")
            if iface.startswith("bridge") or "Thunderbolt" in kind:
                kind = kind or "Thunderbolt Bridge"
            out.append({"iface": iface, "ip": ip, "kind": kind or "?"})
    return out


def python_binary() -> str:
    """What the firewall lists: the real interpreter behind the venv's
    symlink -- for a framework build, its Python.app."""
    real = os.path.realpath(sys.executable)
    marker = "/Resources/Python.app/Contents/MacOS/Python"
    base = real.split("/bin/")[0]
    app = base + marker
    return app if os.path.exists(app) else real


def firewall() -> dict:
    if not os.path.exists(FW):
        return {"known": False}
    on = "enabled" in _run([FW, "--getglobalstate"]).lower()
    binary = python_binary()
    blocked = _run([FW, "--getappblocked", binary]).lower()
    return {"known": True, "on": on,
            "stealth": "is on" in _run([FW, "--getstealthmode"]).lower(),
            "block_all": "enabled" in _run([FW, "--getblockall"]).lower(),
            "binary": binary,
            "binary_state": ("blocked" if "block" in blocked and
                             "permitted" not in blocked else
                             "allowed" if "permitted" in blocked or
                             "allow" in blocked else "not listed")}


def sleep_on_ac():
    """Minutes until system sleep on AC power (0 = never), or None.

    A satellite that sleeps drops off the network and every peer sees it
    go and come back: on this project's own M4, one minute on AC meant
    about nine disconnects a night and a ring re-election each time
    (fixed with `pmset -c sleep 0`)."""
    txt = _run(["pmset", "-g", "custom"])
    if "AC Power" not in txt:
        return None
    ac = txt.split("AC Power", 1)[1].split("Battery Power", 1)[0]
    for line in ac.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "sleep" and parts[1].isdigit():
            return int(parts[1])
    return None


def browse(seconds: float = 4.0, me_id: str = "") -> list[dict]:
    from knurlogic.cluster.discovery import Discovery
    d = Discovery()
    try:
        if not d.browse():
            return []
        d.start()
        time.sleep(seconds)
        return [s for s in d.snapshot()
                if (s.get("txt") or {}).get("id") != me_id]
    except (OSError, ImportError, ValueError, RuntimeError):
        return []
    finally:
        d.stop()


def report(port: int = 8899) -> tuple[str, int]:
    """(text, problems found)."""
    from knurlogic.cluster.peers import Peers
    from knurlogic.machine.identity import identity
    me = identity()
    L, bad = [], 0
    L.append(f"this machine   {me['name']}  (id {me['id']})")
    ifs = interfaces()
    L.append("")
    L.append("addresses")
    for i in ifs:
        L.append(f"  {i['ip']:<16s} {i['iface']:<8s} {i['kind']}")
    tb = [i for i in ifs if "Thunderbolt" in i["kind"]]
    if not ifs:
        L.append("  none but loopback: nothing else can reach this machine")
        bad += 1
    fw = firewall()
    L.append("")
    if fw.get("known"):
        L.append(f"firewall       {'on' if fw['on'] else 'off'}"
                 + (", stealth" if fw.get("stealth") else "")
                 + (", BLOCKING ALL incoming" if fw.get("block_all") else ""))
        if fw["on"]:
            L.append(f"  knurlogic's Python: {fw['binary_state']}")
            L.append(f"    {fw['binary']}")
            if fw["binary_state"] != "allowed" or fw.get("block_all"):
                bad += 1
                L.append("  -> other machines' requests to this page will "
                         "hang until it is allowed: the first `knurlogic ui "
                         "--host ...` shows an \"accept incoming "
                         "connections?\" prompt ON THIS SCREEN. If it was "
                         "missed or denied: System Settings -> Network -> "
                         "Firewall -> Options -> the Python above -> Allow.")
    zz = sleep_on_ac()
    if zz:
        bad += 1
        L.append("")
        L.append(f"sleep          after {zz} min on AC power")
        L.append("  -> a machine that sleeps leaves the cluster and comes "
                 "back, and every peer sees it happen. For a machine that "
                 "serves: `sudo pmset -c sleep 0` (never sleep on AC).")
    found = browse(me_id=me["id"])
    L.append("")
    L.append(f"bonjour        {len(found)} other knurlogic page(s) found")
    for s in found:
        t = s.get("txt") or {}
        L.append(f"  {t.get('name', s['name'])}  {s['host']}:{s['port']}"
                 f"  (knurlogic {t.get('ver', '?')})")
    ps = Peers(me, port, persist=False, reachable=False)
    for s in found:
        ps.add(s["host"], s["port"], "bonjour")
    if ps.all():                 # found now, or remembered from before
        ps.refresh()
        L.append("")
        L.append("peers")
        for p in ps.all():
            L.append(f"  {p.name or p.key:<22s} {p.state:<16s} "
                     f"via {', '.join(sorted(p.found_by))}")
            if p.problem:
                bad += 1
                L.append(f"    -> {p.problem}")
    elif not found:
        L.append("  -> none seen. Start `knurlogic ui --host <this "
                 "machine's address>` on each Mac. If one is running and "
                 "still not seen: System Settings -> Privacy & Security -> "
                 "Local Network -> allow the app that started knurlogic "
                 "(the terminal; or, under launchd or a script, the Python "
                 f"binary itself: {python_binary()}); or name it once "
                 "with `knurlogic ui --peer HOST:PORT` (it is remembered).")
    L.append("")
    L.append("to be found  " + (
        f"knurlogic ui --host {tb[0]['ip']}   (the Thunderbolt link)"
        if tb else "knurlogic ui --host <an address above>"))
    return "\n".join(L), bad
