"""interfaces/ui.py: what peers are serving, gathered for the page.

No network: peers are stand-ins and every fetch is a stub.
"""

import time
from types import SimpleNamespace

from knurlogic.interfaces import ui


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
    got = ui.peer_residency(Peers(peer("M4", "10.0.0.2")),
                            fetch=lambda url, t: {"resident": [ROW]})
    [m] = got
    assert m["machine"] == "M4" and "error" not in m
    [r] = m["resident"]
    assert r["machine"] == "M4"
    assert r["where"] == "http://10.0.0.2:8097"
    assert "http://10.0.0.2:8097" in ui.chat_targets()


def test_a_dead_peer_is_reported_and_does_not_take_the_others_down():
    def fetch(url, t):
        if "10.0.0.3" in url:
            raise ConnectionRefusedError("refused")
        return {"resident": [ROW]}
    got = ui.peer_residency(Peers(peer("M4", "10.0.0.2"),
                                  peer("Air", "10.0.0.3")), fetch=fetch)
    by = {m["machine"]: m for m in got}
    assert by["M4"]["resident"] and "error" not in by["M4"]
    assert by["Air"]["resident"] == [] and "refused" in by["Air"]["error"]


def test_a_slow_peer_costs_at_most_the_deadline():
    def fetch(url, t):
        time.sleep(2.0)
        return {"resident": [ROW]}
    t0 = time.time()
    [m] = ui.peer_residency(Peers(peer("M4", "10.0.0.2")), timeout=0.2,
                            fetch=fetch)
    assert time.time() - t0 < 1.0
    assert m["resident"] == [] and "did not answer" in m["error"]


def test_peers_not_answering_are_not_asked():
    asked = []
    ui.peer_residency(Peers(peer("M4", "10.0.0.2", state="not_answering")),
                      fetch=lambda url, t: asked.append(url) or {})
    assert asked == []


def test_peers_are_only_gathered_when_asked_for(monkeypatch):
    monkeypatch.setattr(ui.web, "loaded_document",
                        lambda: (lambda q: {"resident": []}))
    monkeypatch.setattr(ui, "peer_residency", lambda ps: [{"machine": "M4"}])
    h = ui._loaded_fn()
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
    monkeypatch.setattr(servers, "registry", lambda: {})
    monkeypatch.setattr(servers, "listening_serves", lambda: {8097: 101})
    monkeypatch.setattr(loaded, "_exo", lambda b: [])
    monkeypatch.setattr(loaded, "_ollama", lambda b: [])
    monkeypatch.setattr(loaded, "_openai_port", lambda b: [])
    monkeypatch.setattr(loaded, "memory_map", lambda: {})
    monkeypatch.setattr(loaded, "_knurlogic", lambda b: [loaded.Resident(
        runtime="knurlogic", name="Qwen", where=b, can_unload=True)]
        if b.endswith(":8097") else [])
    [r] = loaded.survey(ports={"openai": []})["resident"]
    assert r["where"] == "http://127.0.0.1:8097" and r["can_unload"] is False
