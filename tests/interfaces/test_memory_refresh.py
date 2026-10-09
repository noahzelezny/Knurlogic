"""The liveness document's memory map: 30 s old while the machine is quiet,
3 s around anything that moves memory, so a card does not trail a load."""

from knurlogic.interfaces import spawn
from knurlogic.interfaces.page import documents
from knurlogic.interfaces.page import nodes as page_nodes


def test_quiet_machine_reuses_the_map_long(monkeypatch):
    monkeypatch.setitem(page_nodes._HOT, "until", 0.0)
    monkeypatch.setitem(documents._LOADED, "doc", {"resident": []})
    assert page_nodes._map_age_limit(1000.0) == page_nodes.LIGHT_MAP_S


def test_an_event_makes_it_hot_for_a_while(monkeypatch):
    monkeypatch.setitem(documents._LOADED, "doc", None)
    monkeypatch.setattr(spawn.time, "time", lambda: 1000.0)
    page_nodes.hot()
    assert page_nodes._map_age_limit(1000.0 + 5) == page_nodes.HOT_MAP_S
    later = 1000.0 + page_nodes.HOT_S + 1
    assert page_nodes._map_age_limit(later) == page_nodes.LIGHT_MAP_S


def test_a_model_answering_keeps_it_hot(monkeypatch):
    monkeypatch.setitem(page_nodes._HOT, "until", 0.0)
    monkeypatch.setitem(documents._LOADED, "doc", {"resident": [
        {"name": "m", "requests": {"in_flight": 1, "pending": 0}}]})
    assert page_nodes._map_age_limit(1000.0) == page_nodes.HOT_MAP_S
