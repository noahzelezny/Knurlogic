"""The load-time warm-up: one tiny generation before the host says ready."""
import importlib

import mlx.nn as nn
import pytest

from knurlogic.engine.runtime import host as H


@pytest.fixture
def host(tmp_path, monkeypatch):
    L = importlib.import_module("knurlogic.engine.serve.load")
    monkeypatch.setenv("KNURLOGIC_LOADLOCK", str(tmp_path / "load.lock"))
    monkeypatch.setattr(L, "load_unlocked",
                        lambda path, code, lazy=False: (nn.Linear(2, 2),
                                                        object()))
    monkeypatch.setattr(H.ModelHost, "_bind_head", lambda self, path: None)
    monkeypatch.delenv("KNURLOGIC_WARMUP", raising=False)
    return H.ModelHost(vision=False), tmp_path


def test_warm_up_runs_once_while_warming_then_ready(host):
    h, path = host
    seen = []
    h.warm = lambda: seen.append(h.state)
    h.load(str(path))
    assert seen == ["warming"]
    assert h.state == "ready"


def test_warm_up_skipped_when_off(host, monkeypatch):
    h, path = host
    monkeypatch.setenv("KNURLOGIC_WARMUP", "off")
    seen = []
    h.warm = lambda: seen.append(1)
    h.load(str(path))
    assert seen == [] and h.state == "ready"


def test_a_failed_warm_up_still_serves(host):
    h, path = host

    def boom():
        raise RuntimeError("no kernel")
    h.warm = boom
    h.load(str(path))
    assert h.state == "ready"
