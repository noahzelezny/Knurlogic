"""interfaces/page/server.py: what peers are serving, gathered for the page.

No network: peers are stand-ins and every fetch is a stub.
"""

import time
from types import SimpleNamespace

from knurlogic.interfaces.page import loads as page_loads
from knurlogic.interfaces.page import peers as page_peers
from knurlogic.interfaces.page import router as page_router


def peer(name, host, state="answering", port=8899):
    return SimpleNamespace(name=name, host=host, port=port, state=state,
                           key=f"{host}:{port}")


class Peers:
    def __init__(self, *ps):
        self.ps = ps

    def all(self):
        return list(self.ps)


ROW = {"runtime": "knurlogic", "name": "Qwen", "where": "http://127.0.0.1:8097",
       "state": "ready"}


def test_rows_are_labelled_by_machine_and_addressed_at_the_peer():
    got = page_peers.peer_residency(Peers(peer("M4", "192.0.2.2")),
                            fetch=lambda url, t: {"resident": [ROW]})
    [m] = got
    assert m["machine"] == "M4" and "error" not in m
    [r] = m["resident"]
    assert r["machine"] == "M4"
    assert r["where"] == "http://192.0.2.2:8097"
    assert "http://192.0.2.2:8097" in page_router.chat_targets()


def test_a_dead_peer_is_reported_and_does_not_take_the_others_down():
    page_peers._PEER_LAST.clear()
    def fetch(url, t):
        if "192.0.2.3" in url:
            raise ConnectionRefusedError("refused")
        return {"resident": [ROW]}
    got = page_peers.peer_residency(Peers(peer("M4", "192.0.2.2"),
                                  peer("Air", "192.0.2.3")), fetch=fetch)
    by = {m["machine"]: m for m in got}
    assert by["M4"]["resident"] and "error" not in by["M4"]
    assert by["Air"]["resident"] == [] and "refused" in by["Air"]["error"]


def test_a_slow_peer_costs_at_most_the_deadline():
    page_peers._PEER_LAST.clear()
    def fetch(url, t):
        time.sleep(2.0)
        return {"resident": [ROW]}
    t0 = time.time()
    [m] = page_peers.peer_residency(Peers(peer("M4", "192.0.2.2")), timeout=0.2,
                            fetch=fetch)
    assert time.time() - t0 < 1.0
    # late is not a failure: nothing heard yet, no "not answering" error
    assert m["resident"] == [] and m["late"] and "error" not in m


def test_a_late_reply_shows_what_the_peer_last_said_and_its_age():
    page_peers._PEER_LAST.clear()
    p = peer("M4", "192.0.2.2")
    page_peers.peer_residency(Peers(p), fetch=lambda url, t: {
        "resident": [ROW]})

    def slow(url, t):
        time.sleep(2.0)
        return {"resident": []}
    [m] = page_peers.peer_residency(Peers(p), timeout=0.2, fetch=slow)
    assert m["resident"] and "error" not in m and m["heard_ago"] >= 0


def test_a_peer_whose_survey_fails_mid_load_keeps_its_load_row():
    # a busy peer's Survey timed out while it loaded its half of a cluster
    # job: dropping its loads made the job's % fall to this machine's share
    page_peers._PEER_LAST.clear()
    p = peer("M4", "192.0.2.2")
    load = {"job": "j", "rank": 0, "bytes": 15 << 30, "total_bytes": 101 << 30}
    page_peers.peer_residency(Peers(p), fetch=lambda url, t: {
        "resident": [], "loads": [load]})

    def fails(url, t):
        raise TimeoutError("timed out")
    [m] = page_peers.peer_residency(Peers(p), fetch=fails)
    assert m["loads"] == [load] and "error" not in m and m["heard_ago"] >= 0


def test_peers_not_answering_are_not_asked():
    asked = []
    page_peers.peer_residency(Peers(peer("M4", "192.0.2.2", state="gone")),
                      fetch=lambda url, t: asked.append(url) or {})
    assert asked == []


def test_peers_are_only_gathered_when_asked_for(monkeypatch):
    monkeypatch.setattr(page_loads.documents, "loaded_document",
                        lambda: (lambda q: {"resident": []}))
    monkeypatch.setattr(page_peers, "peer_residency", lambda ps: [{"machine": "M4"}])
    h = page_loads.loaded_fn()
    assert "peers" not in h({})
    assert h({"peers": ["1"]})["peers"] == [{"machine": "M4"}]


def test_a_serve_started_by_hand_is_found_by_its_listening_port(monkeypatch):
    from knurlogic.machine import servers

    def run(cmd, **k):
        out = ("  101 /venv/bin/python /venv/bin/knurlogic serve /m --port 8097\n"
               "  102 python -m knurlogic ui --port 8899\n"
               if cmd[0] == "ps" else "p101\nn*:8097\nn127.0.0.1:8097\n")
        assert cmd[0] == "ps" or "101" in cmd       # ui is not a serve
        return SimpleNamespace(stdout=out)
    monkeypatch.setattr(servers.subprocess, "run", run)
    assert servers.listening_serves() == {8097: 101}


def test_a_serve_started_by_hand_is_shown_but_not_offered_for_unload(
        monkeypatch):
    from knurlogic.machine import loaded, servers
    from knurlogic.machine.memory import footprint
    monkeypatch.setattr(servers, "registry", lambda: {})
    monkeypatch.setattr(servers, "listening_serves", lambda: {8097: 101})
    monkeypatch.setattr(loaded, "_exo", lambda b: [])
    monkeypatch.setattr(loaded, "_ollama", lambda b: [])
    monkeypatch.setattr(loaded, "_openai_port", lambda b: [])
    monkeypatch.setattr(footprint, "memory_map", lambda: {})
    monkeypatch.setattr(loaded, "_knurlogic", lambda b: [loaded.Resident(
        runtime="knurlogic", name="Qwen", where=b, can_unload=True)]
        if b.endswith(":8097") else [])
    [r] = loaded.survey(ports={"openai": []})["resident"]
    assert r["where"] == "http://127.0.0.1:8097" and r["can_unload"] is False


def test_a_peer_may_only_offer_endpoints_on_its_own_address():
    """What a peer reports becomes a chat proxy target: an endpoint on
    another host (a rogue Bonjour advertiser pointing at the router, say)
    is dropped, not proxied to."""
    from knurlogic.interfaces.page.peers import _peer_where
    assert _peer_where("http://127.0.0.1:8097", "192.0.2.2") == \
        "http://192.0.2.2:8097"
    assert _peer_where("http://192.0.2.2:8097", "192.0.2.2") == \
        "http://192.0.2.2:8097"
    assert _peer_where("http://203.0.113.101:80", "192.0.2.2") == ""
    assert _peer_where("file:///etc/passwd", "192.0.2.2") == ""
    assert _peer_where("", "192.0.2.2") == ""
