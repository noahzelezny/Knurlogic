"""machine/files.py and machine/paths.py: whole-or-nothing writes, tolerant
reads, read-change-write as one step, and the three state directories."""
import subprocess
import sys
import threading
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
    with pytest.raises(RuntimeError), files.atomic(f) as tmp:
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


_COUNT = """
import sys
from knurlogic.machine import files
for _ in range(int(sys.argv[2])):
    files.update_json(sys.argv[1], lambda d: d.update(n=d.get("n", 0) + 1),
                      {})
"""


def test_update_json_loses_no_update_across_threads_and_processes(tmp_path):
    """A writer that read before another one saved must not write its
    stale copy over that save: two processes and four threads each add
    one, 200 times, and every addition is there."""
    f = tmp_path / "count.json"
    procs = [subprocess.Popen([sys.executable, "-c", _COUNT, str(f), "200"])
             for _ in range(2)]
    threads = [threading.Thread(target=lambda: [
        files.update_json(f, lambda d: d.update(n=d.get("n", 0) + 1), {})
        for _ in range(200)]) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(p.wait(60) == 0 for p in procs)
    assert files.read_json(f) == {"n": 1200}


def test_update_json_writes_the_edit_never_what_change_returns(tmp_path):
    """`change` edits in place; its return is ignored -- a pop returns the
    popped value, which once replaced a whole registry in a test."""
    f = tmp_path / "doc.json"
    assert files.update_json(f, lambda d: d.update(a=1, b=2), {}) == \
        {"a": 1, "b": 2}
    assert files.update_json(f, lambda d: d.pop("a"), {}) == {"b": 2}
    assert files.read_json(f) == {"b": 2}
    # `clean` normalises what was read before `change` sees it
    assert files.update_json(f, lambda d: d.append("z"), [],
                             clean=sorted) == ["z"]


def test_a_filesystem_without_locks_still_writes(tmp_path, monkeypatch):
    """flock on an SMB or NFS mount can fail ENOTSUP: the write goes on
    unlocked (atomic, as before) -- a pin on a models folder over SMB must
    not fail. Any other flock error is raised."""
    import errno

    def no_locks(fh, op):
        raise OSError(errno.ENOTSUP, "Operation not supported")
    monkeypatch.setattr(files.fcntl, "flock", no_locks)
    f = tmp_path / "pins.json"
    assert files.update_json(f, lambda d: d.update(s=True), {}) == {"s": True}

    def broken(fh, op):
        raise OSError(errno.EBADF, "Bad file descriptor")
    monkeypatch.setattr(files.fcntl, "flock", broken)
    with pytest.raises(OSError):
        files.update_json(f, lambda d: d.clear(), {})


def test_a_registry_update_keeps_what_another_writer_saved(tmp_path,
                                                          monkeypatch):
    """servers.update_registry re-reads under the lock: a stop that read
    the registry before a start saved its server drops only its own port
    (interfaces/spawn waits up to STOP_SAVE_S between the two)."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    servers.save_registry({8080: {"pid": 1, "artifact": "/m"}})
    stale = servers.registry()                     # read before the wait
    servers.update_registry(lambda r: r.update({8081: {"pid": 2,
                                                       "artifact": "/n"}}))
    servers.update_registry(lambda r: r.pop(8080))
    assert stale == {8080: {"pid": 1, "artifact": "/m"}}
    assert servers.registry() == {8081: {"pid": 2, "artifact": "/n"}}


def test_the_state_directories(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "c"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "g"))
    monkeypatch.setenv("KNURLOGIC_HOME", str(tmp_path / "h"))
    assert paths.cache_dir() == tmp_path / "c" / "knurlogic"
    assert paths.config_dir() == tmp_path / "g" / "knurlogic"
    assert paths.state_dir() == tmp_path / "h"
    monkeypatch.delenv("KNURLOGIC_HOME")
    assert paths.state_dir() == Path.home() / ".knurlogic"
