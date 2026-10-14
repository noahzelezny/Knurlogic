"""Knurlogic-wide settings (tuning/preferences): compaction and per-chip
rounding are set once for every model, not per model -- read per request
(compaction) and at launch (per-chip rounding)."""

import pytest

from knurlogic.context_management import compaction as C
from knurlogic.interfaces import documents
from knurlogic.interfaces.page import peers as page_peers
from knurlogic.tuning import groups, knobs, preferences, presets


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    for k in preferences.NAMES:
        monkeypatch.delenv(k, raising=False)
    return tmp_path


def test_round_trip_clear_and_refusals(home):
    assert preferences.get() == {}
    preferences.set({"KNURLOGIC_COMPACT_KEEP_TURNS": "4",
                     "KNURLOGIC_PER_CHIP_ROUNDING": "off"})
    assert preferences.get() == {"KNURLOGIC_COMPACT_KEEP_TURNS": "4",
                                 "KNURLOGIC_PER_CHIP_ROUNDING": "off"}
    preferences.set({"KNURLOGIC_PER_CHIP_ROUNDING": ""})
    assert preferences.get() == {"KNURLOGIC_COMPACT_KEEP_TURNS": "4"}
    with pytest.raises(ValueError):
        preferences.set({"KNURLOGIC_KV_KERNEL": "off"})  # a model's
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
    assert set(by) == set(groups.COMPACT_KNOBS)
    assert by["KNURLOGIC_COMPACT_TRIGGER"]["value"] == "0.6"
    assert all(k["reach"] == "live" for k in doc["knobs"])
    assert doc["effective"]["trigger"] == 0.6


def test_peer_machine_saves_knurlogic_wide_settings(home):
    code, doc = page_peers.peer_machine(
        {"strategy": "lean",
         "settings": {"KNURLOGIC_PER_CHIP_ROUNDING": "off",
                      "KNURLOGIC_COMPACT_AUTO": "on"}})
    assert code == 200
    assert doc["knurlogic"]["saved"] == {"KNURLOGIC_PER_CHIP_ROUNDING": "off",
                                         "KNURLOGIC_COMPACT_AUTO": "on"}
    assert doc["knurlogic"]["per_chip_rounding"]["value"] == "off"
    code, doc = page_peers.peer_machine(
        {"settings": {"KNURLOGIC_CONTEXT_LENGTH": "512"}})
    assert code == 400
    assert preferences.get()["KNURLOGIC_PER_CHIP_ROUNDING"] == "off"


def test_set_knurlogic_clears_and_reports(home):
    out = documents.set_knurlogic(b'{"KNURLOGIC_PER_CHIP_ROUNDING": "off"}')
    assert out["applied"] == {"KNURLOGIC_PER_CHIP_ROUNDING": "off"}
    assert out["per_chip_rounding"]["value"] == "off"
    out = documents.set_knurlogic(b'{"KNURLOGIC_PER_CHIP_ROUNDING": ""}')
    assert out["applied"] == {"KNURLOGIC_PER_CHIP_ROUNDING": "cleared"}
    assert "error" in documents.set_knurlogic(
        b'{"KNURLOGIC_PER_CHIP_ROUNDING": "x"}')


def test_every_setting_says_what_it_costs():
    """Each explanation names a trade-off, not only what the knob does."""
    words = ("cost", "slower", "faster", "memory", "loses", "lost",
             "no trade", "changes the output", "speed")
    for name, (_what, why) in knobs.KNOB_DOC.items():
        assert any(w in why.lower() for w in words), name
    for name, spec in groups.COMPACT_KNOBS.items():
        assert any(w in spec[4].lower() for w in words), name
    assert "+2-6%" in knobs.KNOB_DOC["KNURLOGIC_PER_CHIP_ROUNDING"][1]


def test_a_launch_reads_per_chip_rounding_from_knurlogic_wide(home):
    assert preferences.launch_sets({"KNURLOGIC_KV_BITS": "8"}) == \
        {"KNURLOGIC_KV_BITS": "8"}         # unset: the preset decides
    preferences.set({"KNURLOGIC_PER_CHIP_ROUNDING": "off"})
    assert preferences.launch_sets({})["KNURLOGIC_PER_CHIP_ROUNDING"] == "off"
    # an explicit --set at that launch still wins
    assert preferences.launch_sets({"KNURLOGIC_PER_CHIP_ROUNDING": "on"}) == \
        {"KNURLOGIC_PER_CHIP_ROUNDING": "on"}
    # and it beats the preset's value, as any launch set does
    launch = knobs.engine_settings({"KNURLOGIC_PER_CHIP_ROUNDING": "on",
                                    **preferences.launch_sets({})})
    assert launch["per_chip_rounding"] == "off"


def test_a_custom_preset_is_saved_rows_the_launch_reads(home):
    assert presets.preset_row_values("lean") == {
        "KNURLOGIC_PREFILL_CHUNK": "512", "KNURLOGIC_CACHE_LIMIT_GB": "1",
        presets.MTP_MODE: "off", "KNURLOGIC_KV_BITS": "8"}
    assert presets.preset_row_values("default")[presets.MTP_MODE] == "dynamic"
    preferences.set({presets.MTP_MODE: "every", "KNURLOGIC_KV_BITS": "8",
                     "KNURLOGIC_CACHE_LIMIT_GB": "2"})
    assert preferences.launch_sets({}) == {
        "KNURLOGIC_MTP": "on", "KNURLOGIC_MTP_DYNAMIC": "off",
        "KNURLOGIC_KV_BITS": "8", "KNURLOGIC_CACHE_LIMIT_GB": "2"}
    # a model's own knob, or its own preset, wins
    assert preferences.launch_sets({"KNURLOGIC_KV_BITS": "bf16"})[
        "KNURLOGIC_KV_BITS"] == "bf16"
    assert preferences.launch_sets({"KNURLOGIC_PRESET": "lean"}) == \
        {"KNURLOGIC_PRESET": "lean"}
    assert "error" in documents.set_knurlogic(b'{"KNURLOGIC_MTP_MODE": "x"}')
    preferences.set({presets.MTP_MODE: ""})
    assert presets.MTP_MODE not in preferences.get()


def test_serve_takes_knurlogic_wide_before_the_preset():
    import inspect

    from knurlogic.interfaces import serve
    src = inspect.getsource(serve)
    assert src.index("preferences.launch_sets(overrides)") < \
        src.index("presets.preset_of(overrides.pop(\"KNURLOGIC_PRESET\"")
