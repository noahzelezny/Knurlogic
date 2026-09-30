"""A deterministic refusal -- bad settings, a model that cannot take them
-- is caught BEFORE a process starts (serve.launch_refusal, asked by the
cluster prepare, the coordinator and mcp.load) and handed back with serve's
own words; a server that refuses anyway exits REFUSED_EXIT and says why."""
import json

import pytest

from knurlogic.interfaces import serve
from knurlogic.cluster import launch as C

from test_cluster_jobs import SHAPE, info, spec  # noqa: F401


def _model(tmp_path, **cfg):
    d = tmp_path / "m"
    d.mkdir(exist_ok=True)
    base = {"model_type": "qwen3_5_moe", "max_position_embeddings": 262144,
            "num_hidden_layers": 4}
    base.update(cfg)
    (d / "config.json").write_text(json.dumps(base))
    return d


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return tmp_path


def test_launch_refusal_says_what_serve_would(cache):
    from knurlogic.machine.artifact import Artifact
    a = Artifact.load(_model(cache))
    assert serve.launch_refusal(a, {}) is None
    why = serve.launch_refusal(a, {"KNURLOGIC_PRESET": "bogus"})
    assert why and "bogus" in why
    why = serve.launch_refusal(a, {"KNURLOGIC_KV_BITS": "3"})
    assert why and "KNURLOGIC_KV_BITS" in why


def test_prepare_refuses_bad_settings_before_a_rank_starts(cache):
    d = _model(cache)
    code, doc = C.prepare(spec(sets={"KNURLOGIC_PRESET": "bogus"}),
                          resolve=lambda i: str(d),
                          info=info("Apple M3 Ultra", "10.0.0.2"),
                          shape=lambda p, w, sp: SHAPE,
                          registry=lambda: {})
    assert code == 200 and not doc["ok"]
    assert "launch settings are refused" in doc["refused"]
    assert "bogus" in doc["refused"]
    assert "ab12cd34ef567890" not in C.PREPARED


def test_mcp_load_refuses_before_spawning(cache, monkeypatch):
    from knurlogic.interfaces import loading, mcp
    from knurlogic.interfaces.page import server as page_server
    d = _model(cache)
    monkeypatch.setattr(loading, "resolve_name", lambda a, _: str(d))
    spawned = []
    monkeypatch.setattr(page_server, "_spawn",
                        lambda *a, **k: spawned.append(a) or {"pid": 1})
    out = mcp.load(artifact="m", sets={"KNURLOGIC_PRESET": "bogus"})
    assert out["loaded"] is False and "bogus" in out["refused"]
    assert spawned == []


def test_a_serve_refusal_exits_with_its_own_code():
    assert serve.REFUSED_EXIT not in (0, 1, 2)


# --- a saved context past the model's window never bricks a launch --------

from knurlogic.tuning import settings as S  # noqa: E402

QWEN = {"max_position_embeddings": 262144}


def test_past_native_turns_long_context_on_for_a_yarn_family():
    sets, notes = S.settle_context("qwen3_5_moe", QWEN,
                                   {"KNURLOGIC_CONTEXT_LENGTH": "1048576"})
    assert sets["KNURLOGIC_LONG_CONTEXT"] == "yarn"
    assert sets["KNURLOGIC_CONTEXT_LENGTH"] == "1048576"
    assert len(notes) == 1 and "YaRN" in notes[0]


def test_past_even_yarn_is_lowered_to_what_yarn_reaches():
    sets, notes = S.settle_context("qwen3_5_moe", QWEN,
                                   {"KNURLOGIC_CONTEXT_LENGTH": "4000000"})
    assert sets["KNURLOGIC_CONTEXT_LENGTH"] == "1048576"
    assert sets["KNURLOGIC_LONG_CONTEXT"] == "yarn"
    assert len(notes) == 2


def test_a_family_without_yarn_is_clamped_with_a_note():
    sets, notes = S.settle_context("glm5_next", QWEN,
                                   {"KNURLOGIC_CONTEXT_LENGTH": "1048576"})
    assert sets["KNURLOGIC_CONTEXT_LENGTH"] == "262144"
    assert "KNURLOGIC_LONG_CONTEXT" not in sets
    assert notes and "lowered to 262,144" in notes[0]


def test_within_the_window_nothing_changes():
    for v in ("32768", "262144", "junk"):
        sets, notes = S.settle_context("qwen3_5_moe", QWEN,
                                       {"KNURLOGIC_CONTEXT_LENGTH": v})
        assert sets == {"KNURLOGIC_CONTEXT_LENGTH": v} and notes == []


def test_the_maintainers_saved_context_launches(cache):
    """KNURLOGIC_CONTEXT_LENGTH=1048576 saved without long context: every
    rank used to print REFUSING and exit."""
    from knurlogic.machine.artifact import Artifact
    a = Artifact.load(_model(cache))
    assert serve.launch_refusal(
        a, {"KNURLOGIC_CONTEXT_LENGTH": "1048576"}) is None
    b = Artifact.load(_model(cache, model_type="glm5_next"))
    assert serve.launch_refusal(
        b, {"KNURLOGIC_CONTEXT_LENGTH": "1048576"}) is None


def test_the_page_offers_up_to_the_yarn_window():
    from knurlogic.interfaces.page.documents import knob_limit
    from knurlogic.machine.artifact import Artifact
    from pathlib import Path
    a = Artifact(path=Path("/x"), model_type="qwen3_5_moe", model_file=None,
                 bytes_on_disk=0, hidden_size=None,
                 moe_intermediate_size=None, raw_config=dict(QWEN))
    lim = knob_limit(a, "KNURLOGIC_CONTEXT_LENGTH")
    assert lim["max"] == 1048576 and "YaRN" in lim["max_why"]
    a.model_type = "glm5_next"
    assert knob_limit(a, "KNURLOGIC_CONTEXT_LENGTH")["max"] == 262144
