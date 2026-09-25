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
