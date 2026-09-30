"""Knurlogic-wide settings (machine/preferences): compaction and identical
results across chips are set once for every model, not per model -- read
per request (compaction) and at launch (cross-chip)."""
import json

import pytest

from knurlogic.context_management import compaction as C
from knurlogic.interfaces.page import documents
from knurlogic.interfaces.page import server as page_server
from knurlogic.machine import preferences
from knurlogic.tuning import settings as S


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    for k in preferences.NAMES:
        monkeypatch.delenv(k, raising=False)
    return tmp_path


def test_round_trip_clear_and_refusals(home):
    assert preferences.get() == {}
    preferences.set({"KNURLOGIC_COMPACT_KEEP_TURNS": "4",
                     "KNURLOGIC_CROSS_CHIP": "on"})
    assert preferences.get() == {"KNURLOGIC_COMPACT_KEEP_TURNS": "4",
                                 "KNURLOGIC_CROSS_CHIP": "on"}
    preferences.set({"KNURLOGIC_CROSS_CHIP": ""})
    assert preferences.get() == {"KNURLOGIC_COMPACT_KEEP_TURNS": "4"}
    with pytest.raises(ValueError):
        preferences.set({"KNURLOGIC_KV_BITS": "8"})      # a model's
    with pytest.raises(ValueError):
        preferences.set({"KNURLOGIC_COMPACT_TOOL_RESULTS": "burn"})
    assert preferences.get() == {"KNURLOGIC_COMPACT_KEEP_TURNS": "4"}
    preferences.set({"KNURLOGIC_COMPACT_KEEP_TURNS": None})
    assert not preferences.path().exists()
    preferences.path().write_text("{not json")
    assert preferences.get() == {}


def test_compaction_is_read_per_request_and_beats_a_servers_env(
        home, monkeypatch):
    # a per-model value a server was launched with (the old way) ...
    monkeypatch.setenv("KNURLOGIC_COMPACT_KEEP_TURNS", "2")
    assert C.settings()["keep"] == 2
    # ... is ignored once knurlogic-wide is set, with no restart
    preferences.set({"KNURLOGIC_COMPACT_KEEP_TURNS": "12",
                     "KNURLOGIC_COMPACT_AUTO": "on"})
    cfg = C.settings()
    assert cfg["keep"] == 12 and cfg["auto"] is True
    # an explicit env (a caller's own) is still read as given
    assert C.settings({"KNURLOGIC_COMPACT_KEEP_TURNS": "2"})["keep"] == 2


def test_compaction_document_is_one_live_set(home):
    preferences.set({"KNURLOGIC_COMPACT_TRIGGER": "0.6"})
    doc = documents.compaction_document()
    by = {k["name"]: k for k in doc["knobs"]}
    assert set(by) == set(S.COMPACT_KNOBS)
    assert by["KNURLOGIC_COMPACT_TRIGGER"]["value"] == "0.6"
    assert all(k["reach"] == "live" for k in doc["knobs"])
    assert doc["effective"]["trigger"] == 0.6


def test_peer_machine_saves_knurlogic_wide_settings(home):
    code, doc = page_server.peer_machine(json.dumps(
        {"strategy": "stable",
         "settings": {"KNURLOGIC_CROSS_CHIP": "auto",
                      "KNURLOGIC_COMPACT_AUTO": "on"}}).encode())
    assert code == 200
    assert doc["knurlogic"]["saved"] == {"KNURLOGIC_CROSS_CHIP": "auto",
                                         "KNURLOGIC_COMPACT_AUTO": "on"}
    assert doc["knurlogic"]["cross_chip"]["value"] == "auto"
    code, doc = page_server.peer_machine(
        b'{"settings": {"KNURLOGIC_PREFILL_CHUNK": "512"}}')
    assert code == 400
    assert preferences.get()["KNURLOGIC_CROSS_CHIP"] == "auto"


def test_set_knurlogic_clears_and_reports(home):
    out = documents.set_knurlogic(b'{"KNURLOGIC_CROSS_CHIP": "on"}')
    assert out["applied"] == {"KNURLOGIC_CROSS_CHIP": "on"}
    out = documents.set_knurlogic(b'{"KNURLOGIC_CROSS_CHIP": ""}')
    assert out["applied"] == {"KNURLOGIC_CROSS_CHIP": "cleared"}
    assert "error" in documents.set_knurlogic(b'{"KNURLOGIC_CROSS_CHIP": "x"}')


def test_every_setting_says_what_it_costs():
    """Each explanation names a trade-off, not only what the knob does."""
    words = ("cost", "slower", "faster", "memory", "loses", "lost",
             "no trade", "changes the output", "speed")
    for name, (what, why) in S.KNOB_DOC.items():
        assert any(w in why.lower() for w in words), name
    for name, spec in S.COMPACT_KNOBS.items():
        assert any(w in spec[4].lower() for w in words), name
    # a preset is the settings it sets: balanced all of them, the rest
    # only what differs from balanced
    assert S.PRESET_GUIDE["balanced"]["settings"].count(" · ") >= 4
    assert S.PRESET_GUIDE["fast"]["settings"] == "cache 8 GiB"
    assert S.PRESET_GUIDE["lean"]["settings"] == \
        "prompt chunk 512 · MTP off · KV 8-bit"
    assert "+2-6%" in S.KNOB_DOC["KNURLOGIC_CROSS_CHIP"][1]


def test_a_launch_reads_cross_chip_from_knurlogic_wide(home):
    assert preferences.launch_sets({"KNURLOGIC_KV_BITS": "8"}) == \
        {"KNURLOGIC_KV_BITS": "8"}         # unset: the preset decides
    preferences.set({"KNURLOGIC_CROSS_CHIP": "on"})
    assert preferences.launch_sets({})["KNURLOGIC_CROSS_CHIP"] == "on"
    # an explicit --set at that launch still wins
    assert preferences.launch_sets({"KNURLOGIC_CROSS_CHIP": "off"}) == \
        {"KNURLOGIC_CROSS_CHIP": "off"}
    # and it beats the preset's value, as any launch set does
    from knurlogic.tuning.resolve import preset_env  # noqa: F401
    launch = S.engine_settings({"KNURLOGIC_CROSS_CHIP": "off",
                                **preferences.launch_sets({})})
    assert launch["cross_chip"] == "on"


def test_serve_takes_knurlogic_wide_before_the_preset():
    import inspect
    from knurlogic.interfaces import serve
    src = inspect.getsource(serve)
    assert src.index("preferences.launch_sets(overrides)") < \
        src.index("S.preset_of(overrides.pop(\"KNURLOGIC_PRESET\"")
