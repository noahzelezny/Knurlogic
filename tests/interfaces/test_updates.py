"""The once-per-start Hugging Face update check (page/updates.py), and the
refusal of a VQ artifact that ships no runtime."""
import json

import pytest

from knurlogic.interfaces.page import updates

OLD = "a" * 40
NEW = "b" * 40


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    updates.reset()
    yield
    updates.reset()


def snap(tmp_path, repo, sha):
    d = tmp_path / ("models--" + repo.replace("/", "--")) / "snapshots" / sha
    d.mkdir(parents=True)
    return d


def test_local_ref_reads_repo_and_revision_from_the_cache_path(tmp_path):
    d = snap(tmp_path, "org/model-8bit", OLD)
    assert updates.local_ref(d) == ("org/model-8bit", OLD)
    assert updates.local_ref(tmp_path / "some-folder") is None


def test_a_newer_hub_sha_flags_the_model(tmp_path):
    d = snap(tmp_path, "org/m", OLD)
    other = tmp_path / "plain"
    other.mkdir()
    updates.check(["org/m"], ask=lambda r: NEW)
    assert updates.flagged([d, other]) == {str(d)}


def test_the_same_sha_is_not_an_update(tmp_path):
    d = snap(tmp_path, "org/m", OLD)
    updates.check(["org/m"], ask=lambda r: OLD)
    assert updates.flagged([d]) == set()


def test_a_redownload_clears_the_flag_for_both_snapshots(tmp_path):
    old, new = snap(tmp_path, "org/m", OLD), snap(tmp_path, "org/m", NEW)
    updates.check(["org/m"], ask=lambda r: NEW)
    assert updates.flagged([old, new]) == set()


def test_a_failed_ask_leaves_the_model_unflagged(tmp_path):
    d = snap(tmp_path, "org/m", OLD)

    def boom(repo):
        raise TimeoutError
    updates.check(["org/m"], ask=boom)
    assert updates.flagged([d]) == set()


def test_start_asks_once_in_the_background(tmp_path):
    d = snap(tmp_path, "org/m", OLD)
    asked = []
    t = updates.start(lambda: [d, tmp_path], ask=lambda r: asked.append(r)
                      or NEW)
    t.join(5)
    assert asked == ["org/m"] and updates.flagged([d]) == {str(d)}
    assert updates.start(lambda: [d], ask=lambda r: NEW) is None


def test_offline_skips_the_check(tmp_path, monkeypatch):
    assert updates.start(lambda: [], offline_flag=True) is None
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert updates.start(lambda: []) is None


def test_models_json_carries_update(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from knurlogic.interfaces.page import documents
    d = snap(tmp_path, "org/m", OLD)
    f = SimpleNamespace(name="org/m", path=d, store="huggingface",
                        bytes_on_disk=10, model_type="qwen3_5", is_vq=False,
                        servable=True, why="", extra={})
    monkeypatch.setitem(documents._MODELS, "rows", [f])
    monkeypatch.setitem(documents._MODELS, "at", 1e18)
    monkeypatch.setattr(documents, "_room", lambda *a: None)
    monkeypatch.setattr(documents, "_splits", lambda *a: None)
    updates.check(["org/m"], ask=lambda r: NEW)
    rows = documents.models_document()({})["models"]
    assert rows[0]["update"] is True


def test_a_vq_model_without_model_py_is_refused_plainly(tmp_path):
    from knurlogic.engine.serve.load import vq_without_runtime
    (tmp_path / "config.json").write_text(json.dumps(
        {"model_type": "qwen3_5_moe", "vq_modules": {"a": {}}}))
    assert "does not ship its runtime (model.py)" in vq_without_runtime(
        tmp_path)
    (tmp_path / "model.py").write_text("")
    assert vq_without_runtime(tmp_path) == ""
