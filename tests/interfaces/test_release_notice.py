"""The page says when a newer knurlogic is on PyPI: one request at page
start, nothing when offline or when PyPI does not answer."""
from __future__ import annotations

import pytest

from knurlogic.interfaces.page import updates


@pytest.fixture(autouse=True)
def _clean():
    updates.reset()
    yield
    updates.reset()


def test_a_newer_release_is_an_update(monkeypatch):
    monkeypatch.setattr("knurlogic.__version__", "0.1.2")
    updates.check_release(lambda: "0.1.10")
    d = updates.release_doc()
    assert d == {"current": "0.1.2", "latest": "0.1.10", "update": True,
                 "command": "pip install -U knurlogic"}


@pytest.mark.parametrize("latest", ["0.1.2", "0.1.1", "0.0.9"])
def test_the_same_or_an_older_release_is_not(monkeypatch, latest):
    monkeypatch.setattr("knurlogic.__version__", "0.1.2")
    updates.check_release(lambda: latest)
    assert updates.release_doc()["update"] is False


def test_pypi_not_answering_leaves_it_unknown(monkeypatch):
    def down():
        raise OSError("no network")
    updates.check_release(down)
    d = updates.release_doc()
    assert d["latest"] is None and d["update"] is False


def test_offline_asks_nothing(monkeypatch):
    asked = []
    monkeypatch.setattr(updates, "check_release", lambda *a: asked.append(1))
    monkeypatch.setattr(updates, "start", lambda *a, **k: None)
    updates.start_for_page(offline_flag=True)
    assert asked == []
