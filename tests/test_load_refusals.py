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
