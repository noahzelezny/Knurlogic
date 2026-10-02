"""Placement re-reads memory fresh before it refuses: a peer's last status is
stale right after an unload."""

from types import SimpleNamespace

import pytest
from test_cluster_nmachines import GIB, SHAPE, mesh_infos

from knurlogic.cluster import launch as C


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "A"))
    return tmp_path


def _stale_launch(monkeypatch, fresh, exiting=()):
    """Two Macs; the peer's last status says 1 GiB available. `fresh`: what
    its Survey answers now (GiB, one per call); `exiting`: its exiting count
    per call."""
    monkeypatch.setattr(C, "_resolve", lambda i, name="": "/m/x")
    monkeypatch.setattr(C, "shape_of", lambda p, w, s, vision=True, mtp=True: SHAPE)
    monkeypatch.setattr(C, "BAD_CABLES", {})
    monkeypatch.setattr(C, "prepare", lambda spec: (200, {"ok": True}))
    monkeypatch.setattr(C, "start", lambda job, **k: (200, {"started": job}))
    monkeypatch.setattr(C, "available_now", lambda: 64 * GIB)
    monkeypatch.setattr(C.time, "sleep", lambda s: None)
    infos = mesh_infos(2)
    stale = dict(infos[1], available_bytes=1 * GIB)
    peers = [SimpleNamespace(id="m1", name="M1", host="127.0.0.1",
                             key="127.0.0.1:8765", state="answering",
                             link="thunderbolt", node={"cluster": stale})]
    surveys = []
    fresh, exiting = list(fresh), list(exiting)

    def post(page, kind, d, **kw):
        if kind == "Survey":
            surveys.append(1)
            i = len(surveys) - 1
            return {"available_bytes": fresh[min(i, len(fresh) - 1)] * GIB,
                    "exiting": exiting[i] if i < len(exiting) else 0}
        return {"ok": True, "started": d.get("job")}
    out = C.launch({"action": "load", "identity": "abc",
                    "nodes": ["m0", "m1"], "split": "pipeline",
                    "link": "tcp"},
                   me={"id": "m0", "name": "M0"}, peers=peers,
                   local_info=infos[0], ui_port=1, serve_port=8080,
                   post=post, follow=lambda j, c: None)
    return out, surveys


def test_placement_retries_with_fresh_memory_and_succeeds(cache, monkeypatch):
    out, surveys = _stale_launch(monkeypatch, [60])
    assert "refused" not in out, out
    assert len(surveys) == 1


def test_placement_waits_for_a_rank_still_exiting(cache, monkeypatch):
    out, surveys = _stale_launch(monkeypatch, [2, 60], exiting=[1, 0])
    assert "refused" not in out, out
    assert len(surveys) == 2


def test_placement_still_refuses_when_fresh_memory_is_short(cache,
                                                            monkeypatch):
    out, surveys = _stale_launch(monkeypatch, [2])
    assert out["refused"].startswith("cannot place it")
    assert surveys
