"""machine/files.py and machine/paths.py: whole-or-nothing writes, tolerant
reads, and the three state directories."""
from pathlib import Path

import pytest

from knurlogic.machine import files, paths, servers


def test_write_json_then_read_json(tmp_path):
    f = tmp_path / "a" / "b.json"
    files.write_json(f, {"x": 1}, end="\n", indent=1)
    assert f.read_text() == '{\n "x": 1\n}\n'
    assert files.read_json(f, {}) == {"x": 1}
    assert [p.name for p in f.parent.iterdir()] == ["b.json"]


def test_read_json_falls_back(tmp_path):
    f = tmp_path / "x.json"
    assert files.read_json(f, {}) == {}
    f.write_text("[1, 2")
    assert files.read_json(f, {}) == {}
    f.write_text("[1, 2]")
    assert files.read_json(f, {}) == {}          # not of default's type
    assert files.read_json(f) == [1, 2]


def test_atomic_replaces_on_success_and_cleans_up_on_failure(tmp_path):
    f = tmp_path / "d" / "x.bin"
    with files.atomic(f) as tmp:
        assert tmp.name.startswith(".tmp-") and tmp.name.endswith("x.bin")
        tmp.write_bytes(b"new")
    assert f.read_bytes() == b"new"
    with pytest.raises(RuntimeError):
        with files.atomic(f) as tmp:
            tmp.write_bytes(b"half")
            raise RuntimeError
    assert f.read_bytes() == b"new"
    assert [p.name for p in f.parent.iterdir()] == ["x.bin"]


def test_a_failed_write_keeps_the_old_file(tmp_path, monkeypatch):
    f = tmp_path / "x.json"
    f.write_text("old")

    def half(self, data, *a, **k):
        Path.write_bytes(self, data[:2].encode())
        raise OSError("disk full")
    with monkeypatch.context() as m:
        m.setattr(Path, "write_text", half)
        with pytest.raises(OSError):
            files.write_atomic(f, "new contents")
    assert f.read_text() == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


def test_save_registry_is_atomic(tmp_path, monkeypatch):
    """servers.json is read by the page, the MCP and every server: a write
    that fails halfway leaves the previous registry, never half a file."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    servers.save_registry({8080: {"pid": 1, "model": "/m"}})
    before = servers.registry_path().read_text()

    def half(self, data, *a, **k):
        Path.write_bytes(self, data[:5].encode())
        raise OSError("disk full")
    with monkeypatch.context() as m:
        m.setattr(Path, "write_text", half)
        with pytest.raises(OSError):
            servers.save_registry({8081: {"pid": 2, "model": "/n"}})
    assert servers.registry_path().read_text() == before
    assert list(servers.registry()) == [8080]


def test_the_state_directories(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "c"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "g"))
    monkeypatch.setenv("KNURLOGIC_HOME", str(tmp_path / "h"))
    assert paths.cache_dir() == tmp_path / "c" / "knurlogic"
    assert paths.config_dir() == tmp_path / "g" / "knurlogic"
    assert paths.state_dir() == tmp_path / "h"
    monkeypatch.delenv("KNURLOGIC_HOME")
    assert paths.state_dir() == Path.home() / ".knurlogic"
