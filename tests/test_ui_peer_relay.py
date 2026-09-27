"""interfaces/ui.py: a peer's model reached through the peer's PAGE.

A peer's model server listens on the peer's loopback; this page reaches it
through the peer page's /peer/v1/... relay, by model name. Everything here
is a fake on loopback: one model server, one "peer" page, one local page.
"""

import http.client
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from knurlogic.interfaces import ui

SSE = b": keepalive\n\ndata: {\"a\": 1}\n\ndata: [DONE]\n\n"
PLAIN = {"id": "x", "choices": [{"message": {"content": "hi"}}]}


def _serve(cls):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), cls)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _fake_model(model_id, seen):
    class F(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, doc):
            out = json.dumps(doc).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def do_GET(self):
            self._json({"data": [{"id": model_id}]})

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length")
                                       or 0))
            seen.append((self.path, json.loads(body)))
            if not json.loads(body).get("stream"):
                self._json(PLAIN)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            for part in SSE.split(b"\n\n")[:-1]:
                self.wfile.write(part + b"\n\n")
                self.wfile.flush()
    return F


class Peers:
    def __init__(self, *ps):
        self.ps = ps

    def all(self):
        return list(self.ps)


@pytest.fixture
def two(monkeypatch):
    """A model on the "peer" (its loopback), the peer's page, this page."""
    seen = []
    model, mport = _serve(_fake_model("glm-peer", seen))
    mbase = f"http://127.0.0.1:{mport}"
    # both pages share this process: the peer page's own servers are the
    # fake model; this page has none of its own
    monkeypatch.setattr(ui, "local_models",
                        lambda fetch=None: {"glm-peer": mbase})
    monkeypatch.setattr(ui, "registry", lambda: {})
    peer_page, pport = _serve(ui.make_handler({}))
    here, hport = _serve(ui.make_handler({}))
    p = SimpleNamespace(name="M4", host="127.0.0.1", port=pport,
                        state="answering", key=f"127.0.0.1:{pport}",
                        id="m4", found_by={"bonjour"})
    monkeypatch.setattr(ui, "PEERS", Peers(p))
    ui._ROUTES.update(at=0.0, map={})
    # the peer reports its model where it runs: on ITS loopback
    row = {"runtime": "knurlogic", "name": "glm-peer", "state": "ready",
           "where": "http://127.0.0.1:8080"}
    ui.peer_residency(ui.PEERS, fetch=lambda url, t: {"resident": [row]})
    yield f"http://127.0.0.1:{hport}", f"http://127.0.0.1:{pport}", seen
    for s in (model, peer_page, here):
        s.shutdown()
    ui._PEER_TARGETS.clear()
    ui._ROUTES.update(at=0.0, map={})


def _post(url, body, headers=None):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers=dict({"Content-Type": "application/json"}, **(headers or {})))
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.headers.get("Content-Type"), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type"), e.read()


def test_peer_targets_go_through_the_peer_page(two):
    _, peer, _ = two
    key = "http://127.0.0.1:8080"
    assert key in ui.chat_targets()
    assert ui.upstream(key, "/v1/messages") == peer + "/peer/v1/messages"


def test_router_lists_and_reaches_the_peer_model_streaming(two):
    here, _, seen = two
    with urllib.request.urlopen(here + "/v1/models", timeout=5) as r:
        ids = [m["id"] for m in json.loads(r.read())["data"]]
    assert ids == ["glm-peer"]
    code, ctype, raw = _post(here + "/v1/chat/completions",
                             {"model": "glm-peer", "stream": True})
    assert code == 200 and ctype == "text/event-stream" and raw == SSE
    code, ctype, raw = _post(here + "/v1/messages",
                             {"model": "glm-peer"})
    assert code == 200 and json.loads(raw) == PLAIN
    assert [p for p, _ in seen] == ["/v1/chat/completions", "/v1/messages"]


def test_page_chat_by_where_reaches_the_peer_model(two):
    here, _, seen = two
    code, _, raw = _post(here + "/chat?where=http://127.0.0.1:8080",
                         {"model": "glm-peer", "stream": True})
    assert code == 200 and raw == SSE
    assert seen[0][0] == "/v1/chat/completions"


def test_relay_count_tokens_and_models(two):
    _, peer, seen = two
    code, _, raw = _post(peer + "/peer/v1/messages/count_tokens",
                         {"model": "glm-peer", "messages": []})
    assert code == 200
    with urllib.request.urlopen(peer + "/peer/v1/models", timeout=5) as r:
        assert [m["id"] for m in json.loads(r.read())["data"]] == \
            ["glm-peer"]


def test_relay_refuses_any_origin(two):
    _, peer, seen = two
    code, _, raw = _post(peer + "/peer/v1/chat/completions",
                         {"model": "glm-peer"},
                         {"Origin": "http://127.0.0.1:8899"})
    assert code == 403 and b"web page" in raw
    req = urllib.request.Request(peer + "/peer/v1/models",
                                 headers={"Origin": "null"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=5)
    assert e.value.code == 403
    assert not seen


def test_relay_unknown_model_is_a_404(two):
    _, peer, seen = two
    code, _, raw = _post(peer + "/peer/v1/chat/completions",
                         {"model": "claude-opus"})
    doc = json.loads(raw)
    assert code == 404 and doc["models"] == ["glm-peer"]
    assert not seen


def test_relay_refuses_transfer_encoding(two):
    _, peer, seen = two
    host, port = peer.removeprefix("http://").split(":")
    c = http.client.HTTPConnection(host, int(port), timeout=5)
    c.putrequest("POST", "/peer/v1/chat/completions")
    c.putheader("Content-Type", "application/json")
    c.putheader("Transfer-Encoding", "chunked")
    c.endheaders()
    c.send(b"5\r\n{\"a\":\r\n0\r\n\r\n")
    assert c.getresponse().status == 411
    assert not seen


def test_relay_gate_refuses_other_networks():
    class No:
        def allows(self, ip):
            return False
    code, doc = ui.peer_refusal({}, "192.168.1.9", "192.168.1.2", No())
    assert code == 403 and "Thunderbolt" in doc["error"]
    assert ui.peer_refusal({}, "10.0.0.1", "192.168.1.2", No(),
                           manual_hosts=["10.0.0.1"]) is None


def test_relay_resolves_by_folder_name_or_the_only_model():
    t = {"org--Qwen": "http://127.0.0.1:1"}
    assert ui._resolve(t, "/models/org--Qwen/") == "http://127.0.0.1:1"
    assert ui._resolve(t, None) == "http://127.0.0.1:1"
    assert ui._resolve(t, "other") is None


def _counting_survey(monkeypatch, rows):
    """peer_residency, answering `rows` for the peer; -> the call count."""
    real, calls = ui.peer_residency, []

    def survey(peers, timeout=ui.PEER_LOADED_S, fetch=None):
        calls.append(1)
        return real(peers, timeout,
                    fetch=lambda url, t: {"resident": list(rows)})
    monkeypatch.setattr(ui, "peer_residency", survey)
    return calls


def test_a_chat_to_a_model_the_page_has_not_surveyed_yet_resurveys(
        two, monkeypatch):
    here, _, seen = two
    ui._PEER_TARGETS.clear()            # launched since the last survey
    calls = _counting_survey(monkeypatch, [
        {"runtime": "knurlogic", "name": "glm-peer",
         "where": "http://127.0.0.1:8080"}])
    code, _, raw = _post(here + "/chat?where=http://127.0.0.1:8080",
                         {"model": "glm-peer", "stream": True})
    assert code == 200 and raw == SSE and calls == [1]
    # and a model nobody serves is still refused, after one survey
    code, _, raw = _post(here + "/chat?where=http://127.0.0.1:9",
                         {"model": "x"})
    assert code == 403 and calls == [1, 1]


def test_the_router_resurveys_once_on_a_miss(two, monkeypatch):
    here, _, seen = two
    ui._PEER_TARGETS.clear()
    ui._ROUTES.update(at=1e18, map={})      # a fresh, empty cached table
    calls = _counting_survey(monkeypatch, [
        {"runtime": "knurlogic", "name": "glm-peer",
         "where": "http://127.0.0.1:8080"}])
    code, _, raw = _post(here + "/v1/messages", {"model": "glm-peer"})
    assert code == 200 and json.loads(raw) == PLAIN and len(calls) >= 1


@pytest.mark.parametrize("req,which,answer,refresh", [
    ({"action": "load", "identity": "abc", "nodes": ["m4", "m3"],
      "split": "tensor", "link": "ring"}, "cluster_launch",
     {"job": "ab", "port": 8080}, True),
    ({"action": "load", "identity": "abc", "nodes": ["m4", "m3"],
      "split": "tensor", "link": "ring"}, "cluster_launch",
     {"refused": "nothing started"}, False),
    ({"action": "load", "identity": "abc", "node": "m4"}, "forward_launch",
     {"starting": "/m", "port": 8080, "machine": "M4"}, True),
    ({"action": "load", "identity": "abc", "node": "m4"}, "forward_launch",
     {"error": "M4 did not answer"}, False),
])
def test_a_launch_refreshes_what_the_page_can_reach(monkeypatch, req, which,
                                                    answer, refresh):
    calls = []
    monkeypatch.setattr(ui, which, lambda *a, **k: answer)
    monkeypatch.setattr(ui, "refresh_targets", lambda: calls.append(1))
    from knurlogic.machine import identity
    monkeypatch.setitem(identity._ID, "id", "m3")
    out = ui._load_fn(8080)({}, json.dumps(req).encode())
    assert out == answer and bool(calls) is refresh
