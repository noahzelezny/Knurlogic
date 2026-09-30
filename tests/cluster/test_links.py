"""cluster/links and the peer preference: answer on Thunderbolt only in
cluster mode; keep a peer found two ways on the cable."""
from knurlogic.cluster import links
from knurlogic.cluster.peers import Peer, Peers

IFS = [{"iface": "en4", "ip": "192.0.2.1", "kind": "Thunderbolt 3"},
       {"iface": "en1", "ip": "203.0.113.168", "kind": "Wi-Fi"},
       {"iface": "en0", "ip": "203.0.113.173", "kind": "Ethernet"}]


def test_the_gate_answers_loopback_and_thunderbolt_only(monkeypatch):
    monkeypatch.setattr(links, "local_interfaces", lambda: IFS)
    g = links.Gate()
    assert g.allows("127.0.0.1") and g.allows("::ffff:127.0.0.1")
    assert g.allows("192.0.2.1")
    assert not g.allows("203.0.113.168") and not g.allows("203.0.113.173")
    assert b"192.0.2.1" in g.refusal("203.0.113.168")


def test_one_machine_on_wifi_and_the_cable_is_kept_on_the_cable():
    wifi = Peer("203.0.113.172", 8899, {"bonjour"}, id="m4", state="answering",
                link="wifi", last_seen=200.0)
    cable = Peer("192.0.2.2", 8899, {"manual"}, id="m4", state="answering",
                 link="thunderbolt", last_seen=100.0)
    keep, drop = sorted((wifi, cable), key=Peers._preference)
    assert keep is cable
    # but a silent cable does not win over a Wi-Fi address that answers
    cable.state = "gone"
    keep, _ = sorted((wifi, cable), key=Peers._preference)
    assert keep is wifi


# --- Thunderbolt link speed (fixtures: the real M3/M4 outputs) --------------

import json

from fixtures_thunderbolt import (
    PORTS_M3,
    PORTS_M4,
    SP_THUNDERBOLT_M3,
    SP_THUNDERBOLT_M4,
)

from knurlogic.cluster import launch as C


def ports_of(text):
    out, cur = {}, None
    for ln in text.splitlines():
        if ln.startswith("Hardware Port:"):
            cur = ln.split(":", 1)[1].strip()
        elif ln.startswith("Device:") and cur:
            out[ln.split(":", 1)[1].strip()] = cur
    return out


def test_receptacle_speeds_read_the_real_system_profiler_output():
    assert links.parse_receptacle_speeds(SP_THUNDERBOLT_M3) == {
        6: 80, 3: 40, 2: 40}               # "Up to 120 Gb/s" = empty port
    assert links.parse_receptacle_speeds(SP_THUNDERBOLT_M4) == {2: 80, 3: 40}
    run = lambda cmd: json.dumps(SP_THUNDERBOLT_M4)
    assert links.receptacle_speeds(run) == {2: 80, 3: 40}
    assert links.receptacle_speeds(lambda cmd: None) == {}


def test_thunderbolt_n_is_receptacle_n_on_both_macs():
    m3, m4 = ports_of(PORTS_M3), ports_of(PORTS_M4)
    s3 = links.parse_receptacle_speeds(SP_THUNDERBOLT_M3)
    s4 = links.parse_receptacle_speeds(SP_THUNDERBOLT_M4)

    def sp(ports, speeds, iface):
        return links.with_speed({"iface": iface, "kind": ports[iface]},
                                speeds)
    assert m3["en7"] == "Thunderbolt 6" and m3["en4"] == "Thunderbolt 3"
    assert sp(m3, s3, "en7")["gbps"] == 80
    assert sp(m3, s3, "en7")["generation"] == 5
    assert sp(m3, s3, "en4")["gbps"] == 40
    assert sp(m3, s3, "en4")["generation"] == 4
    assert m4["en2"] == "Thunderbolt 2" and m4["en3"] == "Thunderbolt 3"
    assert sp(m4, s4, "en2")["gbps"] == 80
    assert sp(m4, s4, "en3")["gbps"] == 40
    # a bridge has no receptacle
    b = links.with_speed({"iface": "bridge0", "kind": "Thunderbolt Bridge"},
                         s3)
    assert b["gbps"] is None and b["generation"] is None


def test_the_thunderbolt_list_carries_each_links_speed(monkeypatch):
    monkeypatch.setattr(links, "local_interfaces", lambda: [
        {"iface": "en7", "ip": "198.51.100.1", "kind": "Thunderbolt 6"},
        {"iface": "en4", "ip": "192.0.2.1", "kind": "Thunderbolt 3"},
        {"iface": "en1", "ip": "203.0.113.168", "kind": "Wi-Fi"}])
    monkeypatch.setattr(links, "receptacle_speeds",
                        lambda: links.parse_receptacle_speeds(
                            SP_THUNDERBOLT_M3))
    links._CACHE.pop("tbspeed", None)
    tb = {i["iface"]: i for i in links.thunderbolt()}
    links._CACHE.pop("tbspeed", None)
    assert set(tb) == {"en7", "en4"}
    assert (tb["en7"]["gbps"], tb["en4"]["gbps"]) == (80, 40)


RD = {"available": True, "reason": "", "devices": [],
      "active": ["rdma_en4", "rdma_en7", "rdma_en2", "rdma_en3"]}


def rig(m4_tb5=True):
    """The M3/M4 rig as each page's status reports it: TB5 on 198.51.100,
    TB4 on 192.0.2; RDMA ports "active" on both cables."""
    m3 = {"id": "m3", "name": "studio", "rdma": RD, "thunderbolt": [
        {"iface": "en4", "ip": "192.0.2.1", "gbps": 40, "generation": 4},
        {"iface": "en7", "ip": "198.51.100.1", "gbps": 80, "generation": 5}]}
    m4 = {"id": "m4", "name": "laptop", "rdma": RD, "thunderbolt": [
        {"iface": "en3", "ip": "192.0.2.2", "gbps": 40, "generation": 4},
        {"iface": "en2", "ip": "198.51.100.2", "gbps": 80 if m4_tb5 else 40,
         "generation": 5 if m4_tb5 else 4}]}
    return m3, m4


def test_jaccl_takes_only_the_thunderbolt_5_cable(monkeypatch):
    monkeypatch.setattr(C, "BAD_CABLES", {})
    m3, m4 = rig()
    assert C._shared_subnets(m3, m4, rdma=True) == ["198.51.100"]
    assert C._rdma_device(m3, m4) == "rdma_en7"
    assert C._rdma_device(m4, m3) == "rdma_en2"
    assert C._ring_ips([m3, m4], rdma=True) == ["198.51.100.1", "198.51.100.2"]
    assert C.rdma_pair_reason(m3, m4) == ""


def test_the_ring_prefers_the_fastest_cable_then_the_lowest(monkeypatch):
    monkeypatch.setattr(C, "BAD_CABLES", {})
    m3, m4 = rig()
    assert C._shared_subnets(m3, m4) == ["198.51.100", "192.0.2"]
    assert C._ring_ips([m3, m4]) == ["198.51.100.1", "198.51.100.2"]
    # two cables of one speed: the lowest subnet, as before
    slow3, slow4 = rig(m4_tb5=False)
    assert C._shared_subnets(slow3, slow4) == ["192.0.2", "198.51.100"]
    # an older peer that does not report speeds: its links count slowest
    old = {**m4, "thunderbolt": [{"iface": t["iface"], "ip": t["ip"]}
                                 for t in m4["thunderbolt"]]}
    assert C._shared_subnets(m3, old) == ["192.0.2", "198.51.100"]
    assert C._shared_subnets(m3, old, rdma=True) == ["192.0.2", "198.51.100"]


def test_no_thunderbolt_5_cable_greys_rdma_with_the_reason(monkeypatch):
    monkeypatch.setattr(C, "BAD_CABLES", {})
    m3, m4 = rig(m4_tb5=False)
    assert C._shared_subnets(m3, m4, rdma=True) == []
    why = C.rdma_pair_reason(m3, m4)
    assert why.startswith("RDMA needs a Thunderbolt 5 cable between these "
                          "Macs; the 192.0.2 link is Thunderbolt 4")
    # the page greys the button with the same words
    from knurlogic.interfaces.page.documents import ASSETS
    page = "".join(p.read_text() for p in ASSETS.rglob("*.js"))
    assert "RDMA needs a Thunderbolt 5 cable between these Macs; " in page
    assert "tb5Why(" in page


def test_the_cable_note_says_which_cable_and_why(monkeypatch):
    monkeypatch.setattr(C, "BAD_CABLES", {})
    m3, m4 = rig()
    monkeypatch.setattr(C, "_resolve", lambda i, name="": "/m/x")
    monkeypatch.setattr(C, "shape_of", lambda p, w, s: {
        "layers": 8, "layer_bytes": [1 << 30] * 8, "other_bytes": 0,
        "kv_bytes_per_token": 0})
    monkeypatch.setattr(C, "placement", lambda infos, shape, split, order: {
        "order": ["studio", "laptop"], "layers": [4, 4],
        "leader": "studio", "shares": [], "reason": ""})
    from types import SimpleNamespace
    peer = SimpleNamespace(id="m4", name="laptop", host="198.51.100.2",
                           key="198.51.100.2:8899", state="answering",
                           link="thunderbolt", node={"cluster": m4})
    got = []

    def post(page, kind, doc, **kw):
        got.append((kind, doc))
        return {"refused": "stop here"}
    monkeypatch.setattr(C, "prepare", lambda spec: (200, {"ok": True}))
    for link in ("jaccl", "ring"):
        got.clear()
        C.launch({"action": "load", "identity": "abc",
                  "nodes": ["m3", "m4"], "split": "pipeline", "link": link},
                 me={"id": "m3", "name": "studio"}, peers=[peer],
                 local_info=m3, ui_port=1, serve_port=2, post=post,
                 follow=lambda j, c: None)
        spec = next(d for k, d in got if k == "Prepare")
        assert spec["cable"] == "198.51.100", spec["cable_note"]
        assert "Thunderbolt 5 (80 Gb/s)" in spec["cable_note"]
        assert "fastest" in spec["cable_note"]
        if link == "jaccl":
            assert spec["ibv_devices"] == [[None, "rdma_en7"],
                                           ["rdma_en2", None]]
            assert spec["coordinator"].startswith("198.51.100.1:")
            assert "192.0.2 (Thunderbolt 4 (40 Gb/s): RDMA needs " \
                   "Thunderbolt 5)" in spec["cable_note"]
            assert "next if link init fails" not in spec["cable_note"]
        else:
            assert "192.0.2 next if link init fails" in spec["cable_note"]
    # no TB5 cable: jaccl is refused with the reason, never tried
    s3, s4 = rig(m4_tb5=False)
    peer.node = {"cluster": s4}
    got.clear()
    out = C.launch({"action": "load", "identity": "abc",
                    "nodes": ["m3", "m4"], "split": "pipeline",
                    "link": "jaccl"},
                   me={"id": "m3", "name": "studio"}, peers=[peer],
                   local_info=s3, ui_port=1, serve_port=2, post=post,
                   follow=lambda j, c: None)
    assert "Thunderbolt 5" in out.get("refused", ""), out
    assert not got
