"""interfaces/page/server.py: the page's router (POST /v1/messages and
/v1/chat/completions forwarded by `model`, GET /v1/models) and the live
settings proxy (POST /apply). Every upstream is a fake on loopback."""

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

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
    monkeypatch.setattr(page_server, "chat_targets", lambda: {base_a, base_b})
    monkeypatch.setattr(page_server, "PEERS", None)
    page_server._ROUTES.update(at=0.0, map={})
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


def test_models_carry_each_servers_own_entry(cluster):
    """The page's chat reads a model's recommended sampling and thinking
    levels from /v1/models; the router listed only ids, so a cluster job
    reached through the M3 page had neither (2026-09-27)."""
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
                       {"x": "y" * (page_server.APPLY_MAX + 1)})
    assert code == 413
    assert not seen


def test_apply_passes_the_servers_report_back(monkeypatch):
    monkeypatch.setattr(page_server, "chat_targets", lambda: {"http://h:1"})
    calls = []

    def post(url, data, t):
        calls.append((url, data, t))
        return 200, b'{"applied": {"K": "failed: no"}}'
    code, doc = page_server.apply_settings("http://h:1", b'{"K": "2"}', post=post)
    assert code == 200 and doc == {"applied": {"K": "failed: no"}}
    assert calls == [("http://h:1/settings.json", b'{"K": "2"}',
                      page_server.APPLY_S)]
    code, doc = page_server.apply_settings("http://h:1", b"{}",
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
