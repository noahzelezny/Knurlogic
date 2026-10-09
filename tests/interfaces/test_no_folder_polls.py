"""The page reads model folders only when someone acts (the picker opens, a
launch, a download finishing, an explicit call) -- never on a poll. A folder
on a share that stalls held the page, and a stalled page got a healthy job
stopped."""
import time

import pytest

from knurlogic.cluster import jobs as J
from knurlogic.interfaces import spawn
from knurlogic.interfaces.page import documents
from knurlogic.interfaces.page import loads as page_loads
from knurlogic.interfaces.page import messages as page_messages
from knurlogic.interfaces.page import nodes as page_nodes
from knurlogic.interfaces.page import peek as page_peek
from knurlogic.interfaces.page import router as page_router
from knurlogic.machine import discover, servers
from knurlogic.machine.artifact import Artifact


@pytest.fixture
def no_folders(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a poll read a model folder")
    monkeypatch.setattr(discover, "find", boom)
    monkeypatch.setattr(Artifact, "load", staticmethod(boom))


@pytest.mark.parametrize("answers", [False, True])   # loading, then loaded
def test_polls_never_read_a_model_folder(monkeypatch, tmp_path, no_folders,
                                         answers):
    log = tmp_path / "r.log"
    log.write_text("loading\n")
    now = time.time()
    rec = {"pid": 100, "artifact": str(tmp_path / "M"), "log": str(log),
           "t": now, "job": "j1", "bytes": 1000}
    monkeypatch.setattr(spawn, "registry", lambda: {8000: rec})
    monkeypatch.setattr(page_loads, "registry", lambda: {8000: rec})
    monkeypatch.setattr(J, "registry", lambda: {
        "j1/0": dict(rec, rank=0, port=8000)})
    monkeypatch.setattr(J, "read_marker", lambda job, rank: None)
    monkeypatch.setattr(servers, "is_our_server", lambda pid: True)
    monkeypatch.setattr(spawn, "is_our_server", lambda pid: True)
    monkeypatch.setattr(spawn, "answers", lambda port: answers)
    mm = {"processes": [{"pid": 100, "bytes": 950}]}
    monkeypatch.setattr(spawn.footprint, "memory_map", lambda: mm)
    monkeypatch.setattr(page_loads.loaded, "survey", lambda: {
        "resident": [], "runtimes": [], "bytes_resident": 0, "memory": mm})
    documents.LOADED.update(doc=None, at=0.0)
    page_nodes._LIGHT.update(doc=None, at=0.0)

    (load,) = page_loads.load_progress({"memory": mm})
    assert load["total_bytes"] == 1000          # from the launch record
    (c,) = spawn.children()
    assert c["phase"] == ("serving" if answers else "loading")
    page_loads.loaded_fn()({})
    page_nodes.status_light()


def _rows(monkeypatch):
    calls = []

    def find(*a, **k):
        calls.append(1)
        return []
    monkeypatch.setattr(discover, "find", find)
    documents.forget_models()
    return calls


def test_models_json_reads_folders_only_when_asked(monkeypatch):
    calls = _rows(monkeypatch)
    h = documents.models_document()
    h({})                                       # the first read
    h({})
    h({})
    assert len(calls) == 1
    h({"rescan": ["1"]})                        # the picker opened
    assert len(calls) == 2
    documents.forget_models()                   # a download finished
    h({})
    assert len(calls) == 3


def test_peek_passes_rescan_for_models_json_only(monkeypatch):
    from types import SimpleNamespace
    p = SimpleNamespace(name="M4", host="192.0.2.2", port=8899,
                        state="answering", key="192.0.2.2:8899")
    monkeypatch.setattr(page_nodes, "PEERS",
                        SimpleNamespace(all=lambda: [p]))
    monkeypatch.setattr(page_router, "chat_targets", lambda: set())
    seen = []

    def fetch(url, t):
        seen.append(url)
        return b"{}"
    for path in ("/models.json", "/settings.json"):
        page_peek.peek({"where": ["http://192.0.2.2:8899"], "path": [path],
                     "rescan": ["1"]}, fetch=fetch)
    assert seen == ["http://192.0.2.2:8899/models.json?rescan=1",
                    "http://192.0.2.2:8899/settings.json"]


def test_a_peers_read_passes_rescan_to_its_models_json():
    got = []

    def models(q, _n=0):
        got.append(q)
        return b"{}", "application/json"
    page_messages.read_here({"/models.json": models},
                     {"path": "/models.json", "query": {"rescan": "1"}})
    assert got == [{"rescan": ["1"]}]
