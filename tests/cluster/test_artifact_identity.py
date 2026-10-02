"""Artifact identity: two artifacts with byte-identical config.json and the
same total size, differing only in shard contents (or in the model.py they
ship), must not collapse to one identity; two paths WITH one identity are
the same weights, and one of them is chosen, never refused."""
import json
import os
import struct

from knurlogic.cluster import launch
from knurlogic.machine import artifact as A


def _shard(path, fill: bytes, size=300_000):
    hdr = json.dumps({"w": {"dtype": "U8", "shape": [size],
                            "data_offsets": [0, size]}}).encode()
    path.write_bytes(struct.pack("<Q", len(hdr)) + hdr + fill * size)


def _art(root, name, fill, real=None):
    d = (real or root) / name
    d.mkdir(parents=True)
    (d / "config.json").write_text('{"model_type": "x"}')
    (d / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    _shard(d / "model-00001-of-00001.safetensors", fill)
    if real is not None:        # a store entry that is a symlink to a pin
        os.symlink(d, root / name)
        return root / name
    return d


def test_same_config_different_weights_differ(tmp_path):
    a = _art(tmp_path, "e112-A", b"a")
    b = _art(tmp_path, "e112-B", b"b")
    assert A.identity(a) and A.identity(b)
    assert A.identity(a) != A.identity(b)


def test_same_weights_agree_and_ignore_mtime(tmp_path):
    a = _art(tmp_path / "m3", "m", b"a")
    b = _art(tmp_path / "m4", "m", b"a")
    os.utime(b / "model-00001-of-00001.safetensors", (1, 1))
    assert A.identity(a) == A.identity(b)


def test_symlinked_pins_read_through(tmp_path):
    store, pins = tmp_path / "store", tmp_path / "pins"
    store.mkdir()
    a = _art(store, "pin_A", b"a", real=pins)
    b = _art(store, "pin_B", b"b", real=pins)
    assert A.identity(a) != A.identity(b)
    assert A.resolve_identity(A.identity(a), [b, a]) == str(a)


def test_same_identity_is_the_same_weights_one_is_chosen(tmp_path,
                                                          monkeypatch):
    monkeypatch.setattr(A, "_network_mounts", lambda: [])
    a = _art(tmp_path / "s1", "one", b"a")
    b = _art(tmp_path / "s2", "two", b"a")
    ident = A.identity(a)
    assert A.identity(b) == ident
    assert A.resolve_identity(ident, [b, a]) == str(a)      # first by path
    assert A.resolve_identity(ident, [a, b], name="two") == str(b)
    assert A.resolve_identity(ident, [a, b], name="/x/y/one") == str(a)
    # a name that matches none of them: still the same weights
    assert A.resolve_identity(ident, [a, b], name="three") == str(a)


def test_a_local_copy_beats_the_network_one(tmp_path, monkeypatch):
    smb = tmp_path / "Volumes" / "shared"
    a = _art(smb, "m", b"a")
    b = _art(tmp_path / "local", "m", b"a")
    monkeypatch.setattr(A, "_network_mounts", lambda: [str(smb.resolve())])
    assert A.on_network(a) and not A.on_network(b)
    assert A.resolve_identity(A.identity(a), [a, b]) == str(b)
    assert A.resolve_identity(A.identity(a), [a, b], name="m") == str(b)


def test_same_real_dir_twice_is_not_ambiguous(tmp_path):
    a = _art(tmp_path, "real", b"a")
    os.symlink(a, tmp_path / "alias")
    assert A.resolve_identity(A.identity(a), [a, tmp_path / "alias"])


def test_rank_prepare_takes_either_copy_of_the_same_weights(tmp_path,
                                                            monkeypatch):
    a = _art(tmp_path / "s1", "one", b"a")
    b = _art(tmp_path / "s2", "two", b"a")
    monkeypatch.setattr(A, "resolve_identity",
                        lambda i, paths=None, name="", _r=A.resolve_identity:
                        _r(i, [a, b], name=name))
    assert launch._resolve(A.identity(a), "two") == str(b)
    assert launch._resolve(A.identity(a), "") == str(a)


def test_model_py_is_part_of_the_identity(tmp_path):
    a = _art(tmp_path / "p1", "m", b"a")
    b = _art(tmp_path / "p2", "m", b"a")
    assert A.identity(a) == A.identity(b)
    (a / "model.py").write_text("X = 1\n")
    (b / "model.py").write_text("X = 2\n")
    assert A.identity(a) != A.identity(b)
    (b / "model.py").write_text("X = 1\n")
    assert A.identity(a) == A.identity(b)          # content, never mtime
    (b / "kernels.py").write_text("")
    assert A.identity(a) != A.identity(b)


def test_a_differing_local_copy_yields_to_the_shared_one(tmp_path,
                                                         monkeypatch):
    smb = tmp_path / "smb"
    shared = _art(smb, "m", b"s")
    local = _art(tmp_path / "local", "m", b"l")
    monkeypatch.setattr(A, "_network_mounts", lambda: [str(smb.resolve())])
    paths = [shared, local]
    got, alert = A.prefer_shared(A.identity(local), "m", paths)
    assert got == A.identity(shared)
    assert "differs from the shared copy" in alert and str(local) in alert
    # identical copies, or the shared one asked for: nothing to say
    assert A.prefer_shared(A.identity(shared), "m", paths) == \
        (A.identity(shared), "")
    same = _art(tmp_path / "local2", "m", b"s")
    assert A.prefer_shared(A.identity(same), "m", [shared, same]) == \
        (A.identity(same), "")
    # no shared copy at all
    monkeypatch.setattr(A, "_network_mounts", lambda: [])
    assert A.prefer_shared(A.identity(local), "m", paths) == \
        (A.identity(local), "")


def test_network_mounts_are_read_from_mount(monkeypatch):
    import subprocess

    class R:
        stdout = ("/dev/disk3s1 on / (apfs, local, journaled)\n"
                  "//user@studio-a._smb._tcp.local/Models on /Volumes/Models "
                  "(smbfs, nodev, nosuid, mounted by user)\n")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: R)
    monkeypatch.setattr(A, "_MOUNTS", [])
    assert A._network_mounts() == ["/Volumes/Models"]


def test_spec_name_is_never_a_path():
    assert "name" in launch.check_spec({"name": "/etc/x", "job": "ab12cd34ef56ab12",
                                        "rank": 0, "world": 2, "prefill_chunk": 0,
                                        "split": "pipeline", "link": "ring",
                                        "hosts": []})


def test_the_plain_name_beats_a_prefixed_copy(tmp_path, monkeypatch):
    # sp190--X sorts before X on some paths; without a picked name every
    # machine should still land on X, so the ranks name the same folder
    monkeypatch.setattr(A, "_network_mounts", lambda: [])
    a = _art(tmp_path / "a", "sp190--Qwen--M", b"a")
    b = _art(tmp_path / "b", "Qwen--M", b"a")
    ident = A.identity(a)
    assert A.resolve_identity(ident, [a, b]) == str(b)
    assert A.resolve_identity(ident, [a, b], name="sp190--Qwen--M") == str(a)
    assert A.resolve_identity(ident, [a, b], name="Qwen--M") == str(b)


def _fresh_process():
    A._DISK.flush()
    A._IDENT.clear()
    A._DISK.file = None
    A._DISK.data = {}


def _boom(*a, **k):
    raise AssertionError("rehashed")


def test_disk_cache_survives_a_new_process(tmp_path, monkeypatch):
    a = _art(tmp_path, "m", b"a")
    first = A.identity(a)
    _fresh_process()
    monkeypatch.setattr(A, "_shard_digest", _boom)
    assert A.identity(a) == first


def test_disk_cache_recomputes_on_changed_shard(tmp_path, monkeypatch):
    a = _art(tmp_path, "m", b"a")
    first = A.identity(a)
    _fresh_process()
    shard = a / "model-00001-of-00001.safetensors"
    st = shard.stat()
    os.utime(shard, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    calls = []
    real = A._shard_digest
    monkeypatch.setattr(A, "_shard_digest",
                        lambda f, n: calls.append(f) or real(f, n))
    assert A.identity(a) == first and calls
    _fresh_process()
    _shard(shard, b"b", size=300_001)       # size changes too
    calls.clear()
    assert A.identity(a) != first and calls


def test_corrupt_disk_cache_is_ignored(tmp_path):
    a = _art(tmp_path, "m", b"a")
    first = A.identity(a)
    _fresh_process()
    f = A._DISK.path()
    f.write_text("{not json")
    assert A.identity(a) == first
    A._DISK.flush()
    assert json.loads(f.read_text())


def test_flush_prunes_gone_paths(tmp_path):
    a = _art(tmp_path, "m", b"a")
    A.identity(a)
    A._DISK.data["/no/such/path"] = [[], "x"]
    A._DISK.flush()
    assert list(json.loads(A._DISK.path().read_text())) == [str(a)]


def _counting(monkeypatch):
    calls = []
    real = A._shard_digest
    monkeypatch.setattr(A, "_shard_digest",
                        lambda f, n: calls.append(f) or real(f, n))
    return calls


def test_same_size_same_mtime_rewrite_recomputes(tmp_path, monkeypatch):
    """cp -p / rsync -a keep mtime and a rebuilt shard keeps its size: the
    ctime still moves, so the stamp must see it (memory and disk)."""
    import time
    a = _art(tmp_path, "m", b"a")
    first = A.identity(a)
    shard = a / "model-00001-of-00001.safetensors"
    st = shard.stat()
    time.sleep(0.01)
    _shard(shard, b"b")
    os.utime(shard, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert shard.stat().st_mtime_ns == st.st_mtime_ns
    calls = _counting(monkeypatch)
    assert A.identity(a) != first and calls
    second = A.identity(a)
    _fresh_process()
    calls.clear()
    assert A.identity(a) == second and not calls


def test_index_change_recomputes(tmp_path, monkeypatch):
    a = _art(tmp_path, "m", b"a")
    first = A.identity(a)
    _fresh_process()
    (a / "model.safetensors.index.json").write_text('{"weight_map": {"x": 1}}')
    calls = _counting(monkeypatch)
    assert A.identity(a) != first and calls


def test_flush_races_writers_without_losing_entries(tmp_path, monkeypatch):
    import threading
    a = _art(tmp_path, "m", b"a")
    A.identity(a)
    keys = [str(tmp_path / f"k{i}") for i in range(4000)]
    for k in keys:
        os.mkdir(k)
    errors = []

    def writer(part):
        try:
            for k in part:
                with A._DISK.lock:
                    A._DISK._live()[k] = [[], "v"]
                    A._DISK._schedule()
        except Exception as e:      # noqa: BLE001
            errors.append(e)

    def flusher():
        try:
            for _ in range(50):
                A._DISK.flush()
        except Exception as e:      # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=writer, args=(keys[i::4],))
          for i in range(4)] + [threading.Thread(target=flusher)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors
    A._DISK.flush()
    on_disk = json.loads(A._DISK.path().read_text())
    assert set(keys) <= set(on_disk) and str(a) in on_disk
    assert not A._DISK.dirty


def test_the_pickers_splits_are_kept_on_disk_under_the_identity(
        tmp_path, monkeypatch):
    # after a page restart the picker read every model's shard headers
    # again (~8 s over the library); the answer is kept under the identity
    from types import SimpleNamespace
    from knurlogic.interfaces.page import documents as D
    a = _art(tmp_path, "m", b"a")
    calls = []
    monkeypatch.setattr(D, "splits_of",
                        lambda p, n=2: calls.append(p) or ["pipeline"])
    f = SimpleNamespace(path=a, servable=True)
    assert D._splits(f) == ["pipeline"]
    D._SPLITS.flush()
    D._SPLITS.file = None                    # a fresh process
    D._SPLITS.data = {}
    assert D._splits(f) == ["pipeline"] and len(calls) == 1
    (a / "config.json").write_text('{"model_type": "other"}')
    assert D._splits(f) == ["pipeline"] and len(calls) == 2
    # a build whose split rules differ asks again (a reverted rule kept
    # offering tensor from the cache)
    monkeypatch.setattr(D, "_RULES", ["other-build"])
    assert D._splits(f) == ["pipeline"] and len(calls) == 3
