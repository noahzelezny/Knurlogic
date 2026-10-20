"""interfaces/http/embeddings.py over real sockets with a fake embedder:
/v1/embeddings' request and response shapes, its input forms and
refusals, /v1/models and /status.json; the page's router reaching it by
`model`; and an embedding artifact told apart from a chat one."""

import base64
import json
import struct
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from knurlogic.engine.embeddings import Embedded
from knurlogic.interfaces.http import embeddings as H
from knurlogic.interfaces.page import nodes as page_nodes
from knurlogic.interfaces.page import router as page_router
from knurlogic.interfaces.page import server as page_server
from knurlogic.machine.artifact import Artifact


class FakeEmbedder:
    state, loaded_at = "empty", 0.0

    def __init__(self):
        self.seen = []

    def status(self):
        return {"state": self.state, "error": ""}

    def embed(self, texts):
        self.seen.append(list(texts))
        self.state = "ready"
        return Embedded([[float(len(t)), 0.5] for t in texts],
                        tokens=sum(len(t) for t in texts))


@pytest.fixture
def server():
    emb = FakeEmbedder()
    app = H.EmbeddingApp(emb, lambda: {"id": "Qwen3-Embedding-test",
                                       "capabilities": ["embeddings"],
                                       "size_bytes": 7})
    srv = H.make_server(app, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", emb
    srv.shutdown()


def _post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())


def test_a_list_of_texts_is_one_embedding_each_in_order(server):
    url, emb = server
    code, doc = _post(url + "/v1/embeddings",
                      {"input": ["ab", "cdef"], "model": "x"})
    assert code == 200
    assert doc == {"object": "list", "model": "Qwen3-Embedding-test",
                   "data": [{"object": "embedding", "index": 0,
                             "embedding": [2.0, 0.5]},
                            {"object": "embedding", "index": 1,
                             "embedding": [4.0, 0.5]}],
                   "usage": {"prompt_tokens": 6, "total_tokens": 6}}
    assert emb.seen == [["ab", "cdef"]]


def test_a_string_is_one_text(server):
    url, emb = server
    code, doc = _post(url + "/v1/embeddings", {"input": "hello"})
    assert code == 200 and len(doc["data"]) == 1
    assert emb.seen == [["hello"]]


def test_base64_is_little_endian_float32(server):
    url, _ = server
    _, doc = _post(url + "/v1/embeddings",
                   {"input": "abc", "encoding_format": "base64"})
    raw = base64.b64decode(doc["data"][0]["embedding"])
    assert struct.unpack("<2f", raw) == (3.0, 0.5)


@pytest.mark.parametrize("body, param", [
    ({"input": []}, "input"),
    ({}, "input"),
    ({"input": [[1, 2, 3]]}, "input"),
    ({"input": ["a", 3]}, "input"),
    ({"input": "a", "encoding_format": "int8"}, "encoding_format"),
    ({"input": "a", "dimensions": 64}, "dimensions"),
])
def test_what_it_cannot_embed_is_a_400_naming_the_field(server, body, param):
    url, emb = server
    code, doc = _post(url + "/v1/embeddings", body)
    assert code == 400 and doc["error"]["param"] == param
    assert emb.seen == []


def test_a_chat_request_is_refused_saying_what_it_serves(server):
    url, _ = server
    code, doc = _post(url + "/v1/chat/completions", {"messages": []})
    assert code == 404 and "/v1/embeddings" in doc["error"]["message"]


def test_models_lists_it_ready_before_it_has_loaded(server):
    url, emb = server
    (m,) = _get(url + "/v1/models")["data"]
    assert emb.state == "empty"
    assert m["id"] == "Qwen3-Embedding-test" and m["status"] == "ready"
    assert m["capabilities"] == ["embeddings"]


def test_status_json_carries_the_artifact_and_its_load_state(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "qwen3"}')
    art = Artifact.load(tmp_path)
    emb = FakeEmbedder()
    snap, _text = H.status_fn(art, emb)(0)
    assert snap["artifact"]["name"] == tmp_path.name
    assert snap["load"]["state"] == "empty"


def test_the_page_router_reaches_it_by_model(server, monkeypatch):
    url, _ = server
    monkeypatch.setattr(page_router, "chat_targets", lambda: {url})
    monkeypatch.setattr(page_nodes, "PEERS", None)
    page_router._ROUTES.update(at=0.0, all={})
    page = ThreadingHTTPServer(
        ("127.0.0.1", 0), page_server.make_handler({}))
    threading.Thread(target=page.serve_forever, daemon=True).start()
    try:
        purl = f"http://127.0.0.1:{page.server_address[1]}"
        code, doc = _post(purl + "/v1/embeddings",
                          {"input": ["xy"], "model": "Qwen3-Embedding-test"})
        assert code == 200 and doc["data"][0]["embedding"] == [2.0, 0.5]
        assert "Qwen3-Embedding-test" in {
            m["id"] for m in _get(purl + "/v1/models")["data"]}
    finally:
        page.shutdown()


@pytest.mark.parametrize("parts, embeds", [
    (("Qwen3-Embedding-0.6B-4bit-DWQ",), True),
    (("hub", "models--mlx-community--Qwen3-Embedding-0.6B-4bit-DWQ",
      "snapshots", "0123abcd"), True),
    (("Qwen3-0.6B-4bit",), False),
])
def test_an_embedding_artifact_is_told_by_its_name(tmp_path, parts, embeds):
    p = tmp_path.joinpath(*parts)
    p.mkdir(parents=True)
    (p / "config.json").write_text('{"model_type": "qwen3"}')
    assert Artifact.load(p).embeds is embeds


def test_sentence_transformers_modules_mark_an_embedder(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "qwen3"}')
    (tmp_path / "modules.json").write_text("[]")
    assert Artifact.load(tmp_path).embeds


@pytest.mark.parametrize("model_type, ring", [
    ("bert", None), ("qwen3", {"world": 2})])
def test_serve_refuses_an_unknown_recipe_or_a_split(tmp_path, model_type,
                                                     ring):
    from knurlogic.cluster import REFUSED_EXIT
    from knurlogic.interfaces import serve
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": model_type}))
    art = Artifact.load(tmp_path)
    assert serve.serve_embeddings(art, "127.0.0.1", 0, None, ring) \
        == REFUSED_EXIT


def test_a_hugging_face_snapshot_is_served_by_its_repo_id(tmp_path):
    """models--org--repo/snapshots/<sha>: the id a client asks for is
    org/repo, as /models.json names it, never the hash -- on a chat
    model's /v1/models and residency as on an embedder's."""
    from knurlogic.interfaces.http import residency
    from knurlogic.machine.artifact import name_of
    p = tmp_path.joinpath("hub", "models--mlx-community--Qwen3-Embedding-"
                          "0.6B-4bit-DWQ", "snapshots", "6c3ae708")
    p.mkdir(parents=True)
    (p / "config.json").write_text('{"model_type": "qwen3"}')
    art = Artifact.load(p)
    want = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
    assert name_of(p) == art.name == want
    assert name_of(tmp_path / "Qwen3-8B") == "Qwen3-8B"
    assert H.served(art, FakeEmbedder())()["id"] == want

    class Host:
        path, loaded_at = str(p), 0.0
    assert residency.served(art, Host())()["id"] == want
    snap, _ = H.status_fn(art, FakeEmbedder())(0)
    assert snap["artifact"]["name"] == want


@pytest.mark.parametrize("embeds, phase", [(True, "serving"),
                                           (False, "warming")])
def test_an_idle_dropped_embedder_does_not_block_a_load(monkeypatch,
                                                        tmp_path, embeds,
                                                        phase):
    """Its weights dropped on idle, an embedder answers holding a fraction
    of its size: that is not memory about to move, so the MCP's ready and
    the page's load check are not held up by it (a chat model is)."""
    from knurlogic.interfaces import spawn
    from knurlogic.interfaces.mcp import inspection
    from knurlogic.machine import loadlock
    log = tmp_path / "e.log"
    log.write_text("dropped\n")
    snap = tmp_path.joinpath("models--org--Qwen3-Embedding-x", "snapshots",
                             "6c3ae708")
    rec = {"pid": 100, "artifact": str(snap), "log": str(log),
           "t": 0.0, "bytes": 1000, "embeds": embeds}
    monkeypatch.setattr(spawn, "registry", lambda: {8080: rec})
    monkeypatch.setattr(spawn, "is_our_server", lambda pid: True)
    monkeypatch.setattr(spawn, "answers", lambda port: True)
    monkeypatch.setattr(spawn.footprint, "memory_map", lambda: {
        "processes": [{"pid": 100, "bytes": 100}]})
    monkeypatch.setattr(loadlock, "holder", lambda: None)
    (c,) = spawn.children()
    assert c["phase"] == phase
    got = inspection.ready()
    assert got["ready"] is embeds
    if not embeds:      # the blocker names the model, not the hash
        assert got["blockers"][0]["detail"]["artifact"] == \
            "org/Qwen3-Embedding-x"


def test_a_launch_records_that_it_embeds(tmp_path):
    from knurlogic.machine.artifact import embeds
    p = tmp_path / "Qwen3-Embedding-0.6B"
    p.mkdir()
    (p / "config.json").write_text('{"model_type": "qwen3"}')
    assert embeds(p) and not embeds(tmp_path / "missing")
