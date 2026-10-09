"""`load`'s role words (interfaces/mcp/placement.py) against a fake page
/status.json: no page, no load."""

import pytest

from knurlogic.interfaces.mcp import inspection, lifecycle, page_client, placement

GIB = 1024 ** 3


def _status(peers):
    nodes = [{"role": "local", "node": "m3",
              "cluster": {"working_set_bytes": 100 * GIB}}]
    for i, (_, _, ws) in enumerate(peers):
        nodes.append({"role": "remote", "address": f"192.0.2.{i}:8899",
                      "cluster": {"working_set_bytes": ws * GIB}})
    return {"me": {"id": "id-me", "name": "m3"},
            "nodes": nodes,
            "peers": [{"id": f"id-{n}", "name": n, "state": s,
                       "address": f"192.0.2.{i}:8899"}
                      for i, (n, s, _) in enumerate(peers)]}


@pytest.fixture
def page(monkeypatch):
    doc = {}

    def get(path):
        assert path == "/status.json"
        return doc["st"]
    monkeypatch.setattr(page_client, "page_get", get)
    return doc


def test_roles_order_and_dedupe(page):
    page["st"] = _status([("m4", "answering", 64), ("old", "unreachable", 8),
                          ("m5", "answering", 32)])
    ids, split, no = placement.resolve_machines(["peers", "here", "m4"])
    assert no is None and ids == ["id-me", "id-m4", "id-m5"]
    assert split == "pipeline"
    ids, split, no = placement.resolve_machines(["all"], split="tensor")
    assert ids == ["id-me", "id-m4", "id-m5"] and split == "tensor"
    ids, split, no = placement.resolve_machines(["here"])
    assert ids == ["id-me"] and split == ""


def test_peers_none_answering(page):
    page["st"] = _status([("old", "unreachable", 8)])
    ids, _, no = placement.resolve_machines(["peers"])
    assert ids == [] and "no peer is answering" in no["refused"]
    assert no["machines"] == ["m3", "old (unreachable)"]


def test_name_not_answering_refused(page):
    page["st"] = _status([("old", "unreachable", 8)])
    _, _, no = placement.resolve_machines(["here", "old"])
    assert "not answering" in no["refused"]


def test_fit_alone_only(page):
    _, _, no = placement.resolve_machines(["fit", "here"])
    assert "fit stands alone" in no["refused"]


def test_fit_here(monkeypatch, page):
    monkeypatch.setattr(inspection, "fit", lambda **k: {"fits": True})
    assert placement.resolve_machines(["fit"], "x") == ([], "pipeline", None)


def test_fit_smallest_set(monkeypatch, page):
    page["st"] = _status([("m4", "answering", 40), ("m5", "answering", 90)])
    monkeypatch.setattr(inspection, "fit",
                        lambda **k: {"fits": False, "verdict": "will not fit"})
    monkeypatch.setattr(placement, "_local_path", lambda a: "/m/x")
    from knurlogic.cluster import launch
    monkeypatch.setattr(launch, "shape_of",
                        lambda *a, **k: {"need": 150 * GIB})

    def place(infos, shape, split):
        if sum(m["working_set_bytes"] for m in infos) < shape["need"]:
            raise ValueError("does not fit")
        return {}
    monkeypatch.setattr(launch, "placement", place)
    ids, split, no = placement.resolve_machines(["fit"], "x")
    assert no is None and ids == ["id-me", "id-m5"] and split == "pipeline"


def test_fit_nothing_fits(monkeypatch, page):
    page["st"] = _status([("m4", "answering", 10)])
    monkeypatch.setattr(inspection, "fit",
                        lambda **k: {"fits": False, "verdict": "will not fit",
                                     "budget_gib": 100.0})
    monkeypatch.setattr(placement, "_local_path", lambda a: "/m/x")
    from knurlogic.cluster import launch
    monkeypatch.setattr(launch, "shape_of", lambda *a, **k: {})

    def place(*a):
        raise ValueError("m4: share does not fit")
    monkeypatch.setattr(launch, "placement", place)
    _, _, no = placement.resolve_machines(["fit"], "x", split="tensor")
    assert no["loaded"] is False and "fits on no set" in no["refused"]
    assert "tensor" in no["refused"] and "m4: share" in no["refused"]
    assert [m["machine"] for m in no["machines"]] == ["m3", "m4"]
    assert no["here"]["budget_gib"] == 100.0


def test_load_passes_roles_through(monkeypatch, page):
    page["st"] = _status([("m4", "answering", 64)])
    seen = {}

    def on(names, *a):
        seen["names"], seen["split"] = names, a[6]
        return {"ok": True}
    monkeypatch.setattr(lifecycle, "_load_on", on)
    assert lifecycle.load(artifact="x", machines=["all"]) == {"ok": True}
    assert seen == {"names": ["id-me", "id-m4"], "split": "pipeline"}
