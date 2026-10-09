"""The page forwards /v1/prompt-cache* to the model server serving the
named model (interfaces/page/server.prompt_cache_forward)."""
import io
import json
from types import SimpleNamespace as NS

from knurlogic.interfaces.page import peers as page_peers
from knurlogic.interfaces.page import prompt_cache as page_prompt_cache
from knurlogic.interfaces.page import relay as page_relay
from knurlogic.interfaces.page import router as page_router


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


def _run(monkeypatch, table, method, path, query=None, body=b"", ip=None,
         routed=None):
    monkeypatch.setattr(page_router, "local_models",
                        lambda fetch=None: dict(table))
    monkeypatch.setattr(page_router, "routable",
                        lambda fetch=None: dict(routed or table))
    sent = []
    h = _Handler(ip or "127.0.0.1")
    page_prompt_cache.prompt_cache_forward(
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


def test_a_peers_model_goes_through_its_pages_relay(monkeypatch):
    """a coordinator session runs on the M4: the M3's page sends its cache calls to
    the M4 page's relay, like a chat, which resolves the name there."""
    far = "http://192.0.2.2:8081"
    monkeypatch.setitem(page_peers.PEER_TARGETS, far,
                        {"relay": "http://192.0.2.2:8899",
                         "machine": "Laptop B"})
    routed = {"qwen": "http://127.0.0.1:8080", "flash": far}
    h, sent = _run(monkeypatch, {"qwen": "http://127.0.0.1:8080"}, "GET",
                   "/v1/prompt-cache", query={"model": ["flash"]},
                   routed=routed)
    assert h.code == 200 and sent == [
        ("http://192.0.2.2:8899/peer/v1/prompt-cache?model=flash", "GET",
         None)]
    body = json.dumps({"model": "flash", "session": "pm"}).encode()
    h, sent = _run(monkeypatch, {"qwen": "http://127.0.0.1:8080"}, "POST",
                   "/v1/prompt-cache/pin", body=body, routed=routed)
    assert sent == [("http://192.0.2.2:8899/peer/v1/prompt-cache/pin",
                     "POST", body)]


def _relay(monkeypatch, table, method, path, url_path, body=b""):
    monkeypatch.setattr(page_router, "local_models",
                        lambda fetch=None, docs=None: dict(table))
    sent = []
    monkeypatch.setattr(page_router, "send_up",
                        lambda u, m, b: sent.append((u, m, b)) or (200, {}))
    h = _Handler()
    h.path = url_path
    page_relay.peer_relay(h, method, path, body)
    return h, sent


def test_the_relay_serves_its_own_models_cache(monkeypatch):
    t = {"flash": "http://127.0.0.1:8081"}
    h, sent = _relay(monkeypatch, t, "GET", "/v1/prompt-cache",
                     "/peer/v1/prompt-cache?model=flash")
    assert h.code == 200 and sent == [
        ("http://127.0.0.1:8081/v1/prompt-cache", "GET", None)]
    b = json.dumps({"model": "flash", "session": "pm"}).encode()
    h, sent = _relay(monkeypatch, t, "POST", "/v1/prompt-cache/drop",
                     "/peer/v1/prompt-cache/drop", b)
    assert sent == [("http://127.0.0.1:8081/v1/prompt-cache/drop", "POST", b)]
    # nothing else under the path, and no model it does not serve
    h, sent = _relay(monkeypatch, t, "POST", "/v1/prompt-cache/other",
                     "/peer/v1/prompt-cache/other", b)
    assert h.code == 404 and not sent
    h, sent = _relay(monkeypatch, t, "GET", "/v1/prompt-cache",
                     "/peer/v1/prompt-cache?model=glm")
    assert h.code == 404 and not sent


def test_only_loopback(monkeypatch):
    h, sent = _run(monkeypatch, {"glm": "http://127.0.0.1:8081"}, "GET",
                   "/v1/prompt-cache", ip="192.0.2.5")
    assert h.code == 403 and not sent


def test_the_client_headers_go_up_with_a_chat():
    h = NS(headers={"X-Client-Session": "c1", "X-Cache-Retain": "pin",
                    "Authorization": "no"})
    assert page_router._client_headers(h) == {"X-Client-Session": "c1",
                                              "X-Cache-Retain": "pin"}
