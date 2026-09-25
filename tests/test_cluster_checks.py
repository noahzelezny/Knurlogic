"""cluster/checks: parsing what macOS prints, with canned output -- the
real commands are read-only, but a test should not depend on this box's
network or firewall."""
from knurlogic.cluster import checks

PORTS = """Hardware Port: Thunderbolt 3
Device: en4
Ethernet Address: aa

Hardware Port: Wi-Fi
Device: en1
Ethernet Address: bb
"""
IFCONFIG = """lo0: flags=8049<UP,LOOPBACK,RUNNING,MULTICAST> mtu 16384
\tinet 127.0.0.1 netmask 0xff000000
en1: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\tinet 203.0.113.168 netmask 0xffffff00 broadcast 203.0.113.255
en4: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\tinet 192.0.2.1 netmask 0xffffff00 broadcast 192.0.2.255
bridge0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\tinet 169.254.3.4 netmask 0xffff0000
"""


def test_interfaces_names_thunderbolt_and_skips_loopback(monkeypatch):
    monkeypatch.setattr(checks, "_run", lambda cmd, timeout=5:
                        PORTS if cmd[0] == "networksetup" else IFCONFIG)
    got = {i["ip"]: i["kind"] for i in checks.interfaces()}
    assert got == {"203.0.113.168": "Wi-Fi", "192.0.2.1": "Thunderbolt 3",
                   "169.254.3.4": "Thunderbolt Bridge"}


def test_firewall_reads_the_binary_state(monkeypatch):
    out = {"--getglobalstate": "Firewall is enabled. (State = 1)",
           "--getstealthmode": "Firewall stealth mode is off",
           "--getblockall": "Firewall has block all state set to disabled.",
           "--getappblocked": "Incoming connection to /x/Python is blocked."}
    monkeypatch.setattr(checks.os.path, "exists", lambda p: True)
    monkeypatch.setattr(checks, "_run", lambda cmd, timeout=5: out[cmd[1]])
    fw = checks.firewall()
    assert fw["on"] and not fw["stealth"] and not fw["block_all"]
    assert fw["binary_state"] == "blocked"
    out["--getappblocked"] = "Incoming connection to /x/Python is permitted."
    assert checks.firewall()["binary_state"] == "allowed"
