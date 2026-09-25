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
\tinet 192.168.1.68 netmask 0xffffff00 broadcast 192.168.1.255
en4: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\tinet 10.0.0.1 netmask 0xffffff00 broadcast 10.0.0.255
bridge0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\tinet 169.254.3.4 netmask 0xffff0000
"""


def test_interfaces_names_thunderbolt_and_skips_loopback(monkeypatch):
    monkeypatch.setattr(checks, "_run", lambda cmd, timeout=5:
                        PORTS if cmd[0] == "networksetup" else IFCONFIG)
    got = {i["ip"]: i["kind"] for i in checks.interfaces()}
    assert got == {"192.168.1.68": "Wi-Fi", "10.0.0.1": "Thunderbolt 3",
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


def test_sleep_on_ac_is_read_from_pmset(monkeypatch):
    out = ("Battery Power:\n sleep                1\nAC Power:\n"
           " displaysleep         10\n sleep                1\n")
    monkeypatch.setattr(checks, "_run", lambda cmd, timeout=5: out)
    assert checks.sleep_on_ac() == 1
    monkeypatch.setattr(checks, "_run", lambda cmd, timeout=5:
                        "AC Power:\n sleep                0\n")
    assert checks.sleep_on_ac() == 0
