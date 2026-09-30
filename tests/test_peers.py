"""cluster/peers.py: states, introductions, the cross-check, and memory.

No network: every fetch is a stub returning a status document or raising
what a real failure raises.
"""

import json
import socket

from knurlogic.cluster.peers import HEADER, Peers
from knurlogic.machine.status import SCHEMA

ME = {"id": "aaaaaaaaaaaa", "name": "Studio"}


def doc(pid, name, peers=(), schema=SCHEMA):
    return {"schema": schema, "peers": list(peers),
            "nodes": [{"node": name, "role": "local", "id": pid,
                       "reachable": True}]}


def make(tmp_path, fetch, manual=(("10.0.0.2", 8899),), **kw):
    return Peers(ME, 8899, manual=manual, store=tmp_path / "peers.json",
                 fetch=fetch, **kw)


def test_an_answering_peer_is_named_by_its_own_status(tmp_path):
    ps = make(tmp_path, lambda url: doc("bbbbbbbbbbbb", "M4"))
    ps.refresh()
    [p] = ps.all()
    assert (p.state, p.id, p.name) == ("answering", "bbbbbbbbbbbb", "M4")
    assert p.found_by == {"manual"}


def test_a_timeout_names_the_firewall_on_the_other_machine(tmp_path):
    def fetch(url):
        raise socket.timeout("timed out")
    ps = make(tmp_path, fetch)
    ps.refresh()
    [p] = ps.all()
    assert p.state == "not_answering"
    assert "firewall" in p.problem and "10.0.0.2" in p.problem
    assert p.failing_since


def test_refused_says_nothing_is_listening(tmp_path):
    def fetch(url):
        raise ConnectionRefusedError("Connection refused")
    ps = make(tmp_path, fetch)
    ps.refresh()
    assert "nothing is listening" in ps.all()[0].problem


def test_a_different_schema_is_named_not_drawn_wrong(tmp_path):
    ps = make(tmp_path, lambda url: doc("bbbbbbbbbbbb", "M4", schema=1))
    ps.refresh()
    assert ps.all()[0].state == "version_mismatch"


def test_the_blocked_machine_learns_it_from_its_peers(tmp_path):
    # The M4 lists THIS machine as not answering: that is a measurement of
    # our firewall, and the only one this machine can get.
    seen = [{"id": ME["id"], "state": "not_answering",
             "address": "10.0.0.1:8899"}]
    ps = make(tmp_path, lambda url: doc("bbbbbbbbbbbb", "M4", peers=seen))
    ps.refresh()
    assert "firewall on THIS machine" in ps.self_problem()
    assert "M4" in ps.self_problem()


def test_no_problem_claimed_when_peers_reach_us(tmp_path):
    seen = [{"id": ME["id"], "state": "answering"}]
    ps = make(tmp_path, lambda url: doc("bbbbbbbbbbbb", "M4", peers=seen))
    ps.refresh()
    assert ps.self_problem() == ""


def test_an_introduction_adds_the_caller_and_ignores_ourselves(tmp_path):
    ps = make(tmp_path, lambda url: doc("x", "x"), manual=())
    ps.introduce("10.0.0.2", "bbbbbbbbbbbb 8899")
    ps.introduce("10.0.0.1", f"{ME['id']} 8899")
    ps.introduce("10.0.0.3", "garbage")
    [p] = ps.all()
    assert (p.key, p.found_by) == ("10.0.0.2:8899", {"introduced"})


def test_a_loopback_page_does_not_introduce_itself(tmp_path):
    sent = []
    ps = make(tmp_path, None, reachable=False)
    import urllib.request

    class R:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return json.dumps(doc("b", "M4")).encode()

    def fake(req, timeout):
        sent.append(dict(req.header_items()))
        return R()
    orig = urllib.request.urlopen
    urllib.request.urlopen = fake
    try:
        ps.refresh()
    finally:
        urllib.request.urlopen = orig
    assert sent and not any(HEADER.lower() in {k.lower() for k in h}
                            for h in sent)


def test_one_machine_at_two_addresses_is_one_peer(tmp_path):
    ps = make(tmp_path, lambda url: doc("bbbbbbbbbbbb", "M4"),
              manual=(("10.0.0.2", 8899), ("192.168.1.72", 8899)))
    ps.refresh()
    [p] = ps.all()
    assert p.found_by == {"manual"}


def test_remembered_after_answering_and_only_then(tmp_path):
    bad = make(tmp_path, lambda url: (_ for _ in ()).throw(OSError("x")),
               manual=(("10.9.9.9", 1),))
    bad.refresh()
    assert not (tmp_path / "peers.json").exists()   # a typo is not kept

    make(tmp_path, lambda url: doc("bbbbbbbbbbbb", "M4")).refresh()
    again = Peers(ME, 8899, store=tmp_path / "peers.json",
                  fetch=lambda url: doc("bbbbbbbbbbbb", "M4"))
    [p] = again.all()
    assert p.found_by == {"remembered"} and p.name == "M4"
    stored = json.loads((tmp_path / "peers.json").read_text())
    assert stored["schema"] == 1 and "bbbbbbbbbbbb" in stored["peers"]


def test_a_peer_that_lies_about_its_shape_cannot_break_status():
    from knurlogic.cluster.peers import clean_node
    from knurlogic.machine import status
    n = clean_node({"id": 7, "node": ["x"], "memory": {"active_bytes": "x",
                   "total_bytes": -5, "peak_bytes": 3}, "uptime_seconds": "a",
                   "deep": {"a": {"b": {"c": {"d": {"e": {"f": 1}}}}}}})
    assert n["memory"] == {"active_bytes": 0, "total_bytes": 0,
                           "peak_bytes": 3}
    assert n["id"] == "7" and isinstance(n["node"], str)
    agg = status.aggregate([n, {"memory": {"active_bytes": "junk"}}])
    assert agg["cluster"]["memory"]["active_bytes"] == 0


def test_introductions_are_capped():
    from knurlogic.cluster import peers as P
    ps = P.Peers({"id": "me"}, 8899, store="/nonexistent/peers.json",
                 persist=False)
    for i in range(P.MAX_INTRODUCED + 20):
        ps.introduce(f"10.1.{i // 250}.{i % 250}", f"id{i} 8899")
    assert len(ps.all()) == P.MAX_INTRODUCED
    ps.introduce("10.9.9.9", "x" * 100 + " 8899")          # junk id
    ps.introduce("10.9.9.9", "idz 70000")                   # junk port
    assert len(ps.all()) == P.MAX_INTRODUCED


def test_two_machines_claiming_one_id_are_both_kept_and_flagged():
    from knurlogic.cluster import peers as P
    ps = P.Peers({"id": "me"}, 8899, store="/nonexistent/peers.json",
                 persist=False)
    a, b = ps.add("10.0.0.2", 8899, "bonjour"), ps.add("10.0.0.9", 8899,
                                                         "introduced")
    a.id = b.id = "same"
    a.name, b.name = "Studio", "Impostor"
    ps._dedupe()
    assert len(ps.all()) == 2
    assert "both claim" in a.problem and "both claim" in b.problem


# --- one machine on two cables (after a replug) -----------------------------

M4 = "002779f847e1"
SPEED = {"10.0.1.2": ("thunderbolt", 80.0), "10.0.0.2": ("thunderbolt", 40.0)}


def rig_doc():
    d = doc(M4, "laptop")
    d["nodes"][0]["cluster"] = {"thunderbolt": [
        {"iface": "en3", "ip": "10.0.0.2", "gbps": 40},
        {"iface": "en2", "ip": "10.0.1.2", "gbps": 80}]}
    return d


def rig(tmp_path, monkeypatch, answers=("10.0.0.2", "10.0.1.2"), **kw):
    monkeypatch.setattr(Peers, "_speed", staticmethod(
        lambda h: SPEED.get(h, ("other", 0.0))))
    asked = []

    def fetch(url):
        host = url.split("//")[1].split(":")[0]
        asked.append(host)
        if host in answers:
            return rig_doc()
        raise socket.timeout("timed out")
    return make(tmp_path, fetch, **kw), asked


def test_an_unanswering_address_of_a_known_machine_is_not_a_second_one(
        tmp_path, monkeypatch):
    # the peer answers on the TB4 cable; Bonjour then offers its TB5 address
    # (no TXT id yet) and that address does not answer: still one machine
    ps, _ = rig(tmp_path, monkeypatch, answers=("10.0.0.2",))
    ps.refresh()
    ps.add("10.0.1.2", 8899, "bonjour")
    ps.refresh()
    [p] = ps.all()
    assert (p.id, p.state, p.key) == (M4, "answering", "10.0.0.2:8899")
    assert "bonjour" in p.found_by
    assert p.addresses == {"10.0.0.2:8899", "10.0.1.2:8899"}
    assert p.public()["other_addresses"] == ["10.0.1.2:8899"]


def test_an_id_less_silent_address_learned_first_folds_in(tmp_path,
                                                          monkeypatch):
    # the silent address was already a peer of its own before the machine
    # answered on the other cable: the refresh folds it in
    ps, _ = rig(tmp_path, monkeypatch, answers=("10.0.0.2",),
                manual=(("10.0.1.2", 8899), ("10.0.0.2", 8899)))
    ps.refresh()
    [p] = ps.all()
    assert p.key == "10.0.0.2:8899" and p.found_by == {"manual"}
    assert "10.0.1.2:8899" in p.addresses


def test_the_same_machine_at_both_cables_keeps_the_faster(tmp_path,
                                                          monkeypatch):
    ps, _ = rig(tmp_path, monkeypatch,
                manual=(("10.0.0.2", 8899), ("10.0.1.2", 8899)))
    ps.refresh()
    [p] = ps.all()
    assert p.key == "10.0.1.2:8899" and p.gbps == 80.0
    assert p.addresses == {"10.0.0.2:8899", "10.0.1.2:8899"}


def test_a_machine_is_asked_at_its_fastest_address_and_moves_there(
        tmp_path, monkeypatch):
    ps, asked = rig(tmp_path, monkeypatch)
    ps.refresh()                        # known only at 10.0.0.2 (TB4)
    ps.add("10.0.1.2", 8899, "bonjour", id=M4)   # its TB5 address
    assert len(ps.all()) == 1
    asked.clear()
    ps.refresh()
    [p] = ps.all()
    assert asked == ["10.0.1.2"]
    assert p.key == "10.0.1.2:8899" and p.state == "answering"
    # the TB5 cable pulled: it answers on the other one, still one machine
    ps._fetch = lambda url: (rig_doc() if "10.0.0.2" in url
                             else (_ for _ in ()).throw(
                                 socket.timeout("timed out")))
    ps.refresh()
    [p] = ps.all()
    assert p.key == "10.0.0.2:8899" and p.state == "answering"


def test_introductions_and_bonjour_instances_fold_into_the_machine(
        tmp_path, monkeypatch):
    ps, _ = rig(tmp_path, monkeypatch, answers=("10.0.0.2",))
    ps.refresh()
    ps.introduce("10.0.1.2", f"{M4} 8899")
    assert len(ps.all()) == 1 and "introduced" in ps.all()[0].found_by
    assert ps.id_of_instance(f"laptop {M4[:6]}") == M4
    assert ps.id_of_instance("Someone Else 123456") == ""
    # a stranger at an unknown address is still its own peer
    ps.add("10.0.9.9", 8899, "bonjour")
    assert len(ps.all()) == 2
