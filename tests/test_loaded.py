"""What is resident, across runtimes -- and the distinctions that matter.

The failure these pin is a page that says a model is loaded when it is not.
Every runtime here reports something slightly different, and flattening them
into "has a model" loses the only fact anybody opened the page for.
"""
import json

import pytest

from knurlogic import loaded


class _Fake:
    """Stands in for the HTTP layer. Keyed by URL so one fixture can serve
    four runtimes at once, which is the case that actually gets rendered."""
    def __init__(self, table):
        self.table = table
        self.asked = []

    def __call__(self, url, timeout=1.5):
        self.asked.append(url)
        for k, v in self.table.items():
            if url.endswith(k):
                return v
        return None


@pytest.fixture
def fake(monkeypatch):
    def install(table):
        f = _Fake(table)
        monkeypatch.setattr(loaded, "_get", f)
        return f
    return install


def test_an_exo_instance_with_no_live_runner_is_not_loaded(fake):
    """Measured against the real daemon: it held 33 RunnerShuttingDown, and
    an instance map that still listed the model. Reading `instances` alone
    would have reported a loaded model that is on its way out."""
    fake({"/state": {
        "instances": {"i1": {"shard_assignments": {"model_id": "some/model"}}},
        "runners": {"r1": {"RunnerShuttingDown": {}},
                    "r2": {"RunnerShuttingDown": {}}}}})
    rows = loaded._exo("http://x")
    assert len(rows) == 1
    assert rows[0].state == "offered"
    assert rows[0].name == "some/model"
    assert "no runner up" in rows[0].detail


def test_exo_runners_up_without_an_instance_still_report(fake):
    """Also measured live: 2 WarmingUp + 1 Loading against an EMPTY instance
    map. Something is taking memory; saying "nothing loaded" is the wrong
    answer in the direction that matters."""
    fake({"/state": {"instances": {},
                     "runners": {"a": {"RunnerWarmingUp": {}},
                                 "b": {"RunnerWarmingUp": {}},
                                 "c": {"RunnerLoading": {}}}}})
    rows = loaded._exo("http://x")
    assert len(rows) == 1
    assert rows[0].state == "loading"
    assert "3 runners starting" in rows[0].name


def test_exo_reports_loaded_when_a_runner_is_actually_up(fake):
    fake({"/state": {
        "instances": {"i1": {"shardAssignments": {"modelId": "m/x"}}},
        "runners": {"r1": {"RunnerReady": {}}}}})
    r = loaded._exo("http://x")[0]
    assert r.state == "loaded" and r.can_unload and r.ident == "i1"


def test_ollama_reads_ps_not_tags(fake):
    """`/api/tags` is the disk and `/api/ps` is the memory. Reading tags here
    would list every model ever pulled as though it were resident."""
    f = fake({"/api/ps": {"models": [
        {"name": "qwen3:8b", "model": "qwen3:8b", "size_vram": 5 << 30,
         "details": {"parameter_size": "8B"}}]}})
    rows = loaded._ollama("http://x")
    assert [u for u in f.asked] == ["http://x/api/ps"]
    assert rows[0].bytes_resident == 5 << 30
    assert rows[0].runtime == "ollama" and rows[0].can_unload


def test_an_openai_port_is_offered_not_loaded(fake):
    """mlx-lm exposes no "what is loaded" anywhere: /v1/models is what it
    COULD serve. Calling that residency invents the fact being asked for."""
    fake({"/v1/models": {"data": [{"id": "some-model"}]}})
    rows = loaded._openai_port("http://x")
    assert rows[0].state == "offered"
    assert rows[0].bytes_resident == 0


def test_our_own_port_is_read_as_ours_not_as_an_anonymous_mlx_server(fake):
    fake({"/status.json": {"schema": 2,
                           "artifact": {"name": "A", "model_type": "qwen3_5",
                                        "path": "/p/A"},
                           "memory": {"active_bytes": 3 << 30}}})
    rows = loaded._knurlogic("http://x")
    assert rows[0].runtime == "knurlogic"
    assert rows[0].bytes_resident == 3 << 30
    assert rows[0].ident == "/p/A"


def test_render_never_prints_an_unreported_size_as_zero(fake):
    doc = {"resident": [{"runtime": "mlx", "state": "offered", "name": "m",
                         "bytes_resident": 0, "detail": ""}],
           "bytes_resident": 0}
    out = loaded.render(doc)
    assert "--" in out and "0.0G" not in out


def test_nothing_running_is_an_ordinary_answer(monkeypatch):
    monkeypatch.setattr(loaded, "_get", lambda *a, **k: None)
    doc = loaded.survey()
    assert doc["resident"] == []
    assert "nothing is loaded" in loaded.render(doc)


def test_exo_load_uses_exo_s_own_placement_object(monkeypatch):
    """The shard assignment is exo's decision, not one assembled here: the
    object returned by /instance/placement is posted back verbatim."""
    placement = {"MlxRingInstance": {"instanceId": "abc",
                                     "shardAssignments": {"modelId": "m/x"}}}
    monkeypatch.setattr(loaded, "_get", lambda url, timeout=1.5: placement)
    sent = {}

    def _post(url, payload=None, method="POST", timeout=30.0):
        sent.update(url=url, payload=payload, method=method)
        return {"message": "Command received."}
    monkeypatch.setattr(loaded, "_post", _post)

    loaded.exo_load("http://x", "m/x")
    assert sent["url"] == "http://x/instance"
    assert sent["payload"] == {"instance": placement}


def test_exo_load_refuses_rather_than_posting_a_guess(monkeypatch):
    monkeypatch.setattr(loaded, "_get", lambda url, timeout=1.5: None)
    monkeypatch.setattr(loaded, "_post",
                        lambda *a, **k: pytest.fail("posted without placement"))
    with pytest.raises(RuntimeError, match="would not place"):
        loaded.exo_load("http://x", "m/x")
