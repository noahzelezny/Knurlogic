"""Placement re-reads memory fresh before it refuses: a peer's last status is
stale right after an unload."""

from types import SimpleNamespace

import pytest
from test_cluster_nmachines import GIB, SHAPE, mesh_infos

from knurlogic.cluster import launch as C
from knurlogic.cluster import links


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
    monkeypatch.setattr(links, "BAD_CABLES", {})
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


def _kept():
    return {"order": ["A", "B"], "leader": "A", "split": "pipeline",
            "layers": [30, 30], "reason": "",
            "shares": [{"rank": 0, "machine": "A", "bytes": 40 * GIB},
                       {"rank": 1, "machine": "B", "bytes": 60 * GIB}]}


def test_a_relaunch_keeps_its_split_while_it_still_fits(monkeypatch):
    monkeypatch.setattr(C, "refresh_memory", lambda infos, post: False)
    infos = [{"name": "A", "working_set_bytes": 84 * GIB},
             {"name": "B", "working_set_bytes": 120 * GIB,
              "available_bytes": 110 * GIB}]
    assert C._kept_split(_kept(), infos, post=None) == _kept()


def test_a_relaunch_onto_a_machine_still_holding_memory_is_refused(
        monkeypatch):
    """B's hung GPU still holds the old job's memory: the relaunch is
    refused saying how much, never re-placed with a smaller share on B."""
    monkeypatch.setattr(C, "refresh_memory", lambda infos, post: False)
    infos = [{"name": "A", "working_set_bytes": 84 * GIB},
             {"name": "B", "working_set_bytes": 120 * GIB,
              "available_bytes": 30 * GIB}]
    with pytest.raises(C._Refused) as e:
        C._kept_split(_kept(), infos, post=None)
    why = e.value.doc["refused"]
    assert why.startswith("B still holds 90") and "check B" in why
    assert e.value.doc["placement"] == _kept()
    from knurlogic.cluster import recovery as R
    assert R.kind(f"relaunch 1 refused: {why}") == "memory"   # not retried


def test_a_relaunch_whose_machine_is_absent_is_refused(monkeypatch):
    monkeypatch.setattr(C, "refresh_memory", lambda infos, post: False)
    infos = [{"name": "A", "working_set_bytes": 84 * GIB}]
    with pytest.raises(C._Refused) as e:
        C._kept_split(_kept(), infos, post=None)
    assert e.value.doc["refused"] == ("B is not answering; the split this "
                                      "job started with needs it")
