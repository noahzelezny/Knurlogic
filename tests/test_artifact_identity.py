"""Artifact identity: two artifacts with byte-identical config.json and the
same total size, differing only in shard contents, must not collapse to one
identity, and resolution must never silently pick the other one."""
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


def test_ambiguous_refused_unless_named(tmp_path):
    a = _art(tmp_path / "s1", "one", b"a")
    b = _art(tmp_path / "s2", "two", b"a")
    ident = A.identity(a)
    assert A.identity(b) == ident
    with pytest.raises(A.AmbiguousIdentity) as e:
        A.resolve_identity(ident, [a, b])
    assert str(a) in str(e.value) and str(b) in str(e.value)
    assert A.resolve_identity(ident, [a, b], name="two") == str(b)
    assert A.resolve_identity(ident, [a, b], name="/x/y/one") == str(a)


def test_same_real_dir_twice_is_not_ambiguous(tmp_path):
    a = _art(tmp_path, "real", b"a")
    os.symlink(a, tmp_path / "alias")
    assert A.resolve_identity(A.identity(a), [a, tmp_path / "alias"])


def test_rank_prepare_refuses_ambiguity(tmp_path, monkeypatch):
    a = _art(tmp_path / "s1", "one", b"a")
    b = _art(tmp_path / "s2", "two", b"a")
    monkeypatch.setattr(A, "resolve_identity",
                        lambda i, paths=None, name="", _r=A.resolve_identity:
                        _r(i, [a, b], name=name))
    spec = {"job": "ab12cd34ef567890", "rank": 1, "world": 2,
            "split": "tensor", "link": "ring", "identity": A.identity(a),
            "hosts": ["10.0.0.1:47200", "10.0.0.2:47201"],
            "ibv_devices": None, "coordinator": "", "layers": [],
            "prefill_chunk": 512, "tune": "balanced", "port": 0,
            "working_set_gib": 60.0, "bandwidth_gbs": 0, "sets": {},
            "versions": {}, "jaccl_timeout_ms": 0,
            "nodes": [{"rank": 0, "id": "a", "name": "A", "page": "x:1"},
                      {"rank": 1, "id": "b", "name": "B", "page": "y:1"}]}
    assert launch.check_spec(spec) == ""
    code, doc = launch.prepare(spec)
    assert code == 200 and doc["ok"] is False
    assert "one" in doc["refused"] and "two" in doc["refused"]
    assert launch._resolve(A.identity(a), "two") == str(b)


def test_spec_name_is_never_a_path():
    assert "name" in launch.check_spec({"name": "/etc/x", "job": "ab12cd34ef56ab12",
                                        "rank": 0, "world": 2, "prefill_chunk": 0,
                                        "split": "pipeline", "link": "ring",
                                        "hosts": []})
