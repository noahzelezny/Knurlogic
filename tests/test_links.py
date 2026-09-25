"""cluster/links and the peer preference: answer on Thunderbolt only in
cluster mode; keep a peer found two ways on the cable."""
from knurlogic.cluster import links
from knurlogic.cluster.peers import Peer, Peers

IFS = [{"iface": "en4", "ip": "10.0.0.1", "kind": "Thunderbolt 3"},
       {"iface": "en1", "ip": "192.168.1.68", "kind": "Wi-Fi"},
       {"iface": "en0", "ip": "192.168.1.73", "kind": "Ethernet"}]


def test_the_gate_answers_loopback_and_thunderbolt_only(monkeypatch):
    monkeypatch.setattr(links, "local_interfaces", lambda: IFS)
    g = links.Gate()
    assert g.allows("127.0.0.1") and g.allows("::ffff:127.0.0.1")
    assert g.allows("10.0.0.1")
    assert not g.allows("192.168.1.68") and not g.allows("192.168.1.73")
    assert b"10.0.0.1" in g.refusal("192.168.1.68")


def test_one_machine_on_wifi_and_the_cable_is_kept_on_the_cable():
    wifi = Peer("192.168.1.72", 8899, {"bonjour"}, id="m4", state="answering",
                link="wifi", last_seen=200.0)
    cable = Peer("10.0.0.2", 8899, {"manual"}, id="m4", state="answering",
                 link="thunderbolt", last_seen=100.0)
    keep, drop = sorted((wifi, cable), key=Peers._preference)
    assert keep is cable
    # but a silent cable does not win over a Wi-Fi address that answers
    cable.state = "not_answering"
    keep, _ = sorted((wifi, cable), key=Peers._preference)
    assert keep is wifi
