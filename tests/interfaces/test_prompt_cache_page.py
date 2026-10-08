"""The page forwards /v1/prompt-cache* to the model server serving the
named model (interfaces/page/server.prompt_cache_forward)."""
import io
import json
from types import SimpleNamespace as NS

from knurlogic.interfaces.page import server as page_server


class _Handler:
    def __init__(self, ip="127.0.0.1"):
        self.client_address = (ip, 5)
        self.headers = {}
        self.wfile = io.BytesIO()
        self.code = None

    def send_response(self, code):
        self.code = code

    def send_header(self, *a):
        pass

    def end_headers(self):
        pass

    def doc(self):
        return json.loads(self.wfile.getvalue())


def _run(monkeypatch, table, method, path, query=None, body=b"", ip=None):
    monkeypatch.setattr(page_server, "local_models",
                        lambda fetch=None: dict(table))
    sent = []
    h = _Handler(ip or "127.0.0.1")
    page_server.prompt_cache_forward(
        h, method, path, query or {}, body,
        send=lambda url, m, b: sent.append((url, m, b)) or (200, {"ok": 1}))
    return h, sent


def test_one_model_needs_no_name(monkeypatch):
    h, sent = _run(monkeypatch, {"glm": "http://127.0.0.1:8081"}, "POST",
                   "/v1/prompt-cache/save", body=b"{}")
    assert h.code == 200 and sent == [
        ("http://127.0.0.1:8081/v1/prompt-cache/save", "POST", b"{}")]


def test_several_models_go_by_name_from_query_or_body(monkeypatch):
    t = {"glm": "http://127.0.0.1:8081", "qwen": "http://127.0.0.1:8082"}
    h, sent = _run(monkeypatch, t, "GET", "/v1/prompt-cache",
                   query={"model": ["qwen"]})
    assert sent[0][:2] == ("http://127.0.0.1:8082/v1/prompt-cache", "GET")
    h, sent = _run(monkeypatch, t, "POST", "/v1/prompt-cache/drop",
                   body=json.dumps({"model": "glm", "session": "a"}).encode())
    assert sent[0][0] == "http://127.0.0.1:8081/v1/prompt-cache/drop"


def test_several_unnamed_is_a_400_listing_them(monkeypatch):
    t = {"glm": "http://127.0.0.1:8081", "qwen": "http://127.0.0.1:8082"}
    h, sent = _run(monkeypatch, t, "POST", "/v1/prompt-cache/save")
    assert h.code == 400 and h.doc()["models"] == ["glm", "qwen"]
    assert not sent
    h, sent = _run(monkeypatch, t, "GET", "/v1/prompt-cache",
                   query={"model": ["nope"]})
    assert h.code == 404 and not sent


def test_only_loopback(monkeypatch):
    h, sent = _run(monkeypatch, {"glm": "http://127.0.0.1:8081"}, "GET",
                   "/v1/prompt-cache", ip="192.0.2.5")
    assert h.code == 403 and not sent


def test_the_client_headers_go_up_with_a_chat():
    h = NS(headers={"X-Client-Session": "c1", "X-Cache-Retain": "pin",
                    "Authorization": "no"})
    assert page_server._client_headers(h) == {"X-Client-Session": "c1",
                                              "X-Cache-Retain": "pin"}
