"""engine/backend.py: which MLX back end computes, and the Metal-only
kernels staying off elsewhere."""
import pytest

pytest.importorskip("mlx.core")

from knurlogic.engine import backend, kvattn


def test_the_back_end_is_named():
    assert backend.name() in ("metal", "cuda", "cpu")
    assert backend.metal() == (backend.name() == "metal")


def test_the_kv_kernel_is_off_without_metal(monkeypatch):
    monkeypatch.delenv(kvattn.ENV, raising=False)
    monkeypatch.setattr(backend, "metal", lambda: False)
    assert not kvattn.enabled()
    monkeypatch.setattr(backend, "metal", lambda: True)
    assert kvattn.enabled()
