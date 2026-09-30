"""Artifact identity: two artifacts with byte-identical config.json and the
same total size, differing only in shard contents (or in the model.py they
ship), must not collapse to one identity; two paths WITH one identity are
the same weights, and one of them is chosen, never refused."""
import json
import os
import struct

import pytest

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
                  "//noah@m3._smb._tcp.local/Models on /Volumes/Models "
                  "(smbfs, nodev, nosuid, mounted by noah)\n")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: R)
    monkeypatch.setattr(A, "_MOUNTS", [])
    assert A._network_mounts() == ["/Volumes/Models"]


def test_spec_name_is_never_a_path():
    assert "name" in launch.check_spec({"name": "/etc/x", "job": "ab12cd34ef56ab12",
                                        "rank": 0, "world": 2, "prefill_chunk": 0,
                                        "split": "pipeline", "link": "ring",
                                        "hosts": []})
