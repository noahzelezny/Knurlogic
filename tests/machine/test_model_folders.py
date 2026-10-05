"""Remembered model folders: `knurlogic models add|remove|folders`, read
by discovery; an unmounted folder skipped; adoption from the old env once."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest

from knurlogic.machine import discover, folders


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    h = tmp_path / "kh"
    monkeypatch.setenv("KNURLOGIC_HOME", str(h))
    return h


def _model(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    (d / "model.safetensors").write_bytes(b"\0" * 4096)
    return d


def test_add_list_remove(tmp_path, capsys):
    f = tmp_path / "Models"
    f.mkdir()
    assert folders.main(["add", str(f)]) == 0
    assert folders.saved() == [str(f)]
    assert folders.main(["folders"]) == 0
    assert str(f) in capsys.readouterr().out
    assert folders.main(["remove", str(f)]) == 0
    assert folders.saved() == []
    assert folders.main(["add", str(tmp_path / "nope")]) == 1


def test_models_command_routes_folder_subcommands(tmp_path):
    f = tmp_path / "m"
    f.mkdir()
    assert discover.main(["add", str(f)]) == 0
    assert folders.saved() == [str(f)]


def test_discovery_reads_saved_folders(tmp_path, monkeypatch):
    f = tmp_path / "external drive"
    _model(f / "qwen-x")
    folders.add(str(f))
    monkeypatch.setattr(discover, "_running_tool_roots", lambda: [])
    assert f in [p for _s, p in discover._roots()]
    assert any(r.path == f / "qwen-x" for r in discover.find())


def test_unmounted_folder_skipped_quietly(tmp_path, monkeypatch):
    f = tmp_path / "Volumes" / "Gone"
    f.mkdir(parents=True)
    folders.add(str(f))
    f.rmdir()
    monkeypatch.setattr(discover, "_running_tool_roots", lambda: [])
    assert f not in [p for _s, p in discover._roots()]
    assert folders.saved() == [str(f)]          # still remembered


def test_adopt_from_env_once(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b b"
    a.mkdir()
    b.mkdir()
    env = {"EXO_MODELS_DIRS": f"{a}:{tmp_path / 'missing'}",
           "KNURLOGIC_MODELS": str(b)}
    assert folders.adopt_from_env(env) == [str(a), str(b)]
    assert folders.saved() == [str(a), str(b)]
    folders.remove(str(a))
    folders.remove(str(b))
    assert folders.adopt_from_env(env) == []    # once only
    assert folders.saved() == []


def test_mcp_model_folders_tool(tmp_path):
    from knurlogic.interfaces import mcp
    f = tmp_path / "x"
    f.mkdir()
    out = mcp._call("model_folders", {"add": str(f)})
    assert out["folders"] == [{"path": str(f), "mounted": True}]
    out = mcp._call("model_folders", {"remove": str(f)})
    assert out["folders"] == []
