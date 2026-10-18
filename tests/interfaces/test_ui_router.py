"""interfaces/page/server.py: the page's router (POST /v1/messages and
/v1/chat/completions forwarded by `model`, GET /v1/models) and the live
settings proxy (POST /apply). Every upstream is a fake on loopback."""

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from knurlogic.interfaces.page import nodes as page_nodes
from knurlogic.interfaces.page import peek as page_peek
from knurlogic.interfaces.page import peers as page_peers
from knurlogic.interfaces.page import router as page_router
from knurlogic.interfaces.page import server as page_server


def _serve(handler_cls):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _fake_model(model_id, seen):
    """A knurlogic model server as far as the router can tell: /v1/models,
    a streamed chat, and POST /settings.json with a per-knob report."""
    class F(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_GET(self):
            out = json.dumps({"data": [{
                "id": model_id, "context_length": 4096,
                "sampling_defaults": {"temp": 0.6},
                "thinking": {"dialect": "qwen_toggle", "default": "on"}}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n)
            seen.append((model_id, self.path, body,
                         self.headers.get("Origin")))
            if self.path == "/settings.json":
                want = json.loads(body)
                out = json.dumps({"applied": {k: "applied now"
                                              for k in want}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            for part in (b": keepalive\n\n", b"data: {\"a\": 1}\n\n",
                         b"data: [DONE]\n\n"):
                self.wfile.write(part)
                self.wfile.flush()
    return F


@pytest.fixture
def cluster(monkeypatch):
    """Two running models -- one here, one a peer reported -- and the page
    in front of them."""
    seen = []
    a, base_a = _serve(_fake_model("qwen-local", seen))
    b, base_b = _serve(_fake_model("glm-peer", seen))
    monkeypatch.setattr(page_router, "chat_targets", lambda: {base_a, base_b})
    monkeypatch.setattr(page_nodes, "PEERS", None)
    page_router._ROUTES.update(at=0.0, all={})
    page, page_url = _serve(page_server.make_handler({}))
    yield page_url, base_a, base_b, seen
    for s in (a, b, page):
        s.shutdown()


def _post(url, body, headers=None):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers=dict({"Content-Type": "application/json"}, **(headers or {})))
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.headers.get("Content-Type"), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type"), e.read()


def test_routes_by_model_and_streams_through(cluster):
    page, _, _, seen = cluster
    code, ctype, raw = _post(page + "/v1/messages",
                             {"model": "glm-peer", "stream": True})
    assert code == 200 and ctype == "text/event-stream"
    # byte for byte, keepalive included
    assert raw == (b": keepalive\n\ndata: {\"a\": 1}\n\n"
                   b"data: [DONE]\n\n")
    code, _, _ = _post(page + "/v1/chat/completions", {"model": "qwen-local"})
    assert code == 200
    assert [(m, p) for m, p, _, _ in seen] == [
        ("glm-peer", "/v1/messages"), ("qwen-local", "/v1/chat/completions")]
    # the body goes as the client sent it
    assert json.loads(seen[0][2]) == {"model": "glm-peer", "stream": True}


def test_an_unknown_model_is_a_404_listing_what_runs(cluster):
    page, _, _, seen = cluster
    code, _, raw = _post(page + "/v1/messages", {"model": "claude-opus"})
    doc = json.loads(raw)
    assert code == 404 and doc["models"] == ["glm-peer", "qwen-local"]
    assert "glm-peer" in doc["error"]["message"]
    assert not seen


def test_models_lists_every_routable_model(cluster):
    page, base_a, base_b, _ = cluster
    with urllib.request.urlopen(page + "/v1/models", timeout=5) as r:
        doc = json.loads(r.read())
    assert {m["id"]: m["server"] for m in doc["data"]} == {
        "glm-peer": base_b, "qwen-local": base_a}


def test_models_name_the_machine_each_runs_on(cluster, monkeypatch):
    """A client tells which Mac serves a model from /v1/models
    alone, without the page's loaded.json."""
    from knurlogic.machine import identity as I
    page, base_a, base_b, _ = cluster
    monkeypatch.setattr(I, "identity", lambda: {"name": "Studio"})
    monkeypatch.setitem(page_peers.PEER_TARGETS, base_b,
                        {"machine": "Laptop B"})
    # the fake peer answers directly, not through a peer page's relay
    monkeypatch.setattr(page_peers, "upstream", lambda b, p: b + p)
    with urllib.request.urlopen(page + "/v1/models", timeout=5) as r:
        doc = json.loads(r.read())
    assert {m["id"]: m["machine"] for m in doc["data"]} == {
        "glm-peer": "Laptop B", "qwen-local": "Studio"}


def test_models_carry_each_servers_own_entry(cluster):
    """The page's chat reads a model's recommended sampling and thinking
    levels from /v1/models; the router listed only ids, so a cluster job
    reached through another machine's page had neither."""
    page, _, _, _ = cluster
    with urllib.request.urlopen(page + "/v1/models", timeout=5) as r:
        doc = json.loads(r.read())
    for m in doc["data"]:
        assert m["sampling_defaults"] == {"temp": 0.6}
        assert m["thinking"]["dialect"] == "qwen_toggle"
        assert m["context_length"] == 4096
        assert m["object"] == "model" and m["owned_by"] == "knurlogic"


def test_the_browser_guard_applies_to_the_router(cluster):
    page, _, _, seen = cluster
    code, _, raw = _post(page + "/v1/messages", {"model": "glm-peer"},
                         {"Origin": "http://evil.example"})
    assert code == 403 and b"web page" in raw
    code, _, _ = _post(page + "/apply?where=x", {"a": 1},
                       {"Origin": "http://evil.example"})
    assert code == 403
    assert not seen


def test_apply_forwards_to_the_model_servers_settings_only(cluster):
    page, base_a, _, seen = cluster
    code, _, raw = _post(f"{page}/apply?where={base_a}/",
                         {"VQ_DECODE_CHUNK": "256"})
    assert code == 200
    assert json.loads(raw) == {"applied": {"VQ_DECODE_CHUNK": "applied now"}}
    assert [(m, p) for m, p, _, _ in seen] == [("qwen-local",
                                                "/settings.json")]


def test_apply_refuses_what_it_does_not_know(cluster):
    page, base_a, _, seen = cluster
    # not a running model server (a peer's page, anything else)
    code, _, _ = _post(page + "/apply?where=http://192.0.2.2:8899",
                       {"VQ_DECODE_CHUNK": "1"})
    assert code == 403
    # JSON objects only, and small
    code, _, _ = _post(f"{page}/apply?where={base_a}", [1, 2])
    assert code == 400
    code, _, _ = _post(f"{page}/apply?where={base_a}",
                       {"x": "y" * (page_peek.APPLY_MAX + 1)})
    assert code == 413
    assert not seen


def test_apply_passes_the_servers_report_back(monkeypatch):
    monkeypatch.setattr(page_router, "chat_targets", lambda: {"http://h:1"})
    calls = []

    def post(url, data, t):
        calls.append((url, data, t))
        return 200, b'{"applied": {"K": "failed: no"}}'
    code, doc = page_peek.apply_settings("http://h:1", b'{"K": "2"}', post=post)
    assert code == 200 and doc == {"applied": {"K": "failed: no"}}
    assert calls == [("http://h:1/settings.json", b'{"K": "2"}',
                      page_peek.APPLY_S)]
    code, doc = page_peek.apply_settings("http://h:1", b"{}",
                                  post=lambda u, d, t: (200, b"<html>"))
    assert code == 502


def test_count_tokens_goes_to_the_model_it_names(cluster):
    """Claude Code asks the token count per model, like its messages."""
    page, _, _, seen = cluster
    code, _, _ = _post(page + "/v1/messages/count_tokens",
                       {"model": "qwen-local", "messages": []})
    assert code == 200
    assert [(m, p) for m, p, _, _ in seen] == [
        ("qwen-local", "/v1/messages/count_tokens")]


@pytest.fixture
def twin(monkeypatch):
    """One model id on two Macs: this one's server and a peer's (reached
    through its page relay), and the page in front of them."""
    from types import SimpleNamespace
    seen = []
    a, here = _serve(_fake_model("qwen", seen))
    b, there = _serve(_fake_model("qwen", seen))
    monkeypatch.setattr(page_router, "chat_targets", lambda: {here, there})
    monkeypatch.setitem(page_peers.PEER_TARGETS, there,
                        {"machine": "Peer Mac", "relay": there, "job": ""})
    monkeypatch.setattr(page_nodes, "PEERS", SimpleNamespace(
        all=lambda: [SimpleNamespace(name="Peer Mac", id="abc123")]))
    monkeypatch.setattr(page_peers, "PEER_AT", [1e18])
    monkeypatch.setattr(page_router, "_AFFINITY", {})
    page_router._ROUTES.update(at=0.0, all={})
    page, page_url = _serve(page_server.make_handler({}))
    yield page_url, seen
    for s in (a, b, page):
        s.shutdown()


def _went(seen):
    """Which server each forwarded request reached: the peer's comes in
    through its relay (/peer/...)."""
    return ["peer" if p.startswith("/peer") else "here"
            for _, p, _, _ in seen]


def test_one_model_on_two_macs_goes_to_the_same_instance_every_time(twin):
    """The table kept whichever server answered first, rebuilt every 5 s:
    a model on both Macs flipped between them, and a session's prompt
    cache with it (a fork test served 0 of 4523 cached tokens)."""
    page, seen = twin
    for _ in range(6):
        page_router._ROUTES["at"] = 0.0              # a new table each time
        assert _post(page + "/v1/messages", {"model": "qwen"})[0] == 200
    assert _went(seen) == ["here"] * 6               # this Mac's, first


def test_an_instance_is_named_by_machine_in_the_model_or_a_header(twin):
    page, seen = twin
    for body, headers in (({"model": "qwen@abc123"}, None),
                          ({"model": "qwen@peer mac"}, None),
                          ({"model": "qwen"},
                           {page_router.MACHINE_HEADER: "abc123"})):
        assert _post(page + "/v1/messages", body, headers)[0] == 200
    assert _went(seen) == ["peer"] * 3
    assert {json.loads(b)["model"] for _, _, b, _ in seen} == {"qwen"}
    code, _, raw = _post(page + "/v1/messages", {"model": "qwen@nowhere"})
    assert code == 404 and "'nowhere'" in json.loads(raw)["error"]["message"]


def test_a_session_stays_on_the_instance_that_served_it(twin):
    page, seen = twin
    s1 = {"X-Client-Session": "s1"}
    _post(page + "/v1/messages", {"model": "qwen@abc123"}, s1)
    _post(page + "/v1/messages", {"model": "qwen"}, s1)    # names none now
    _post(page + "/v1/messages", {"model": "qwen"}, {"X-Client-Session": "s2"})
    assert _went(seen) == ["peer", "peer", "here"]


def test_models_lists_every_instance_of_a_model(twin):
    page, _ = twin
    with urllib.request.urlopen(page + "/v1/models", timeout=5) as r:
        doc = json.loads(r.read())
    (m,) = doc["data"]
    assert m["id"] == "qwen" and len(m["instances"]) == 2
    first, second = m["instances"]
    assert first["machine_id"] != "abc123"           # this Mac's first
    assert second == dict(second, model="qwen@abc123", machine="Peer Mac",
                          machine_id="abc123")
