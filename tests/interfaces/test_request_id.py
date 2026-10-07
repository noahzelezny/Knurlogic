"""X-Request-Id: echoed by the model server and passed both ways by the
page's router (interfaces/http/request_id.py). No model is loaded: the
model server is answered on its error path, the router in front of fakes."""
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace as NS

import pytest

from knurlogic.interfaces.http import request_id as RID
from knurlogic.interfaces.page import server as page_server


def test_what_is_echoed_and_what_is_not():
    assert RID.valid("client-7f3a 2026") == "client-7f3a 2026"
    assert RID.valid("x" * 128) == "x" * 128
    assert RID.valid("x" * 129) is None
    assert RID.valid("") is None and RID.valid(None) is None
    assert RID.valid("a\r\nSet-Cookie: x") is None      # no header injection
    assert RID.valid("é") is None
    assert RID.of({"X-Request-Id": "abc"}) == "abc"
    assert RID.of(None) is None


def _post(url, body: bytes, rid=None):
    headers = {"Content-Type": "application/json"}
    if rid is not None:
        headers["X-Request-Id"] = rid
    req = urllib.request.Request(url, data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.headers.get("X-Request-Id"), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("X-Request-Id"), e.read()


def _serve(handler_cls):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


@pytest.fixture
def model_server():
    from knurlogic.interfaces.http.server import App, make_server
    sched = NS(host=NS(state="ready", path=None, tokenizer=None))
    app = App(sched, served=lambda: {"id": "tiny"})
    srv = make_server(app, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_the_model_server_echoes_it_even_on_an_error(model_server):
    code, echo, _ = _post(model_server + "/v1/chat/completions", b"{nope",
                          rid="client-1")
    assert code == 400 and echo == "client-1"


def test_no_id_in_no_id_out_and_a_bad_one_is_not_repeated(model_server):
    _, echo, _ = _post(model_server + "/v1/chat/completions", b"{nope")
    assert echo is None
    _, echo, _ = _post(model_server + "/v1/chat/completions", b"{nope",
                       rid="x" * 200)
    assert echo is None


def _fake_model(model_id, seen, echo=True):
    """A model server as the router sees it: /v1/models and a stream that
    ends with a usage chunk carrying the id it was sent."""
    class F(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_GET(self):
            out = json.dumps({"data": [{"id": model_id}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            rid = self.headers.get("X-Request-Id")
            seen.append(rid)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            if echo and rid:
                self.send_header("X-Request-Id", rid)
            self.end_headers()
            usage = {"usage": {"knurlogic": {"request_id": rid}}}
            self.wfile.write(b"data: " + json.dumps(usage).encode() +
                             b"\n\ndata: [DONE]\n\n")
            self.wfile.flush()
    return F


def _page(monkeypatch, *bases):
    monkeypatch.setattr(page_server, "chat_targets", lambda: set(bases))
    monkeypatch.setattr(page_server, "PEERS", None)
    page_server._ROUTES.update(at=0.0, map={})
    _, url = _serve(page_server.make_handler({}))
    return url


def test_the_router_passes_it_up_and_the_echo_back(monkeypatch):
    seen = []
    _, base = _serve(_fake_model("qwen", seen))
    page = _page(monkeypatch, base)
    for path in ("/v1/chat/completions", "/v1/messages"):
        code, echo, raw = _post(page + path,
                                json.dumps({"model": "qwen"}).encode(),
                                rid="client-42")
        assert code == 200 and echo == "client-42"
        assert b'"request_id": "client-42"' in raw
    assert seen == ["client-42", "client-42"]


def test_the_router_echoes_it_when_the_upstream_does_not(monkeypatch):
    seen = []
    _, base = _serve(_fake_model("qwen", seen, echo=False))
    page = _page(monkeypatch, base)
    _, echo, _ = _post(page + "/v1/chat/completions",
                       json.dumps({"model": "qwen"}).encode(), rid="c-9")
    assert echo == "c-9"


def test_a_refusal_from_the_router_carries_it_too(monkeypatch):
    page = _page(monkeypatch)
    code, echo, _ = _post(page + "/v1/messages",
                          json.dumps({"model": "nobody"}).encode(), rid="c-3")
    assert code == 404 and echo == "c-3"
