"""Load/unload on ONE peer: the coordinator forwards by identity; the
peer refuses anything off its gate, with an Origin, or naming a path. No model loads: the
peer's loader and resolver are stubs, and the "peer" in the happy path is a
local http server running the real handler."""

import json
import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from knurlogic.interfaces import ui
from knurlogic.machine import artifact


class Open:
    def allows(self, _ip):
        return True


class Shut:
    def allows(self, _ip):
        return False


def hdr(**kw):
    return {k.replace("_", "-"): v for k, v in kw.items()}


def call(body, headers=None, gate=Open(), **kw):
    loads = []
    code, doc = ui.peer_launch(
        {} if headers is None else headers, "10.0.0.1", "10.0.0.2",
        json.dumps(body).encode(), gate=gate,
        load=lambda **a: loads.append(a) or {"starting": a["artifact"]},
        resolve=kw.get("resolve", lambda i: "/models/X" if i == "abc"
                       else None),
        stop=lambda port: {"stopped": port})
    return code, doc, loads


LOAD = {"action": "load", "identity": "abc", "tune": "fast",
        "sets": {"VQ_DECODE_CHUNK": "16"}}


# --- the peer's side ---------------------------------------------------------

def test_refused_with_any_origin_header():
    code, doc, loads = call(LOAD, headers={"Origin": "http://10.0.0.2:8899"})
    assert code == 403 and not loads


def test_refused_off_the_gate_unless_a_named_peer():
    code, _, loads = call(LOAD, gate=Shut())
    assert code == 403 and not loads
    code, _ = ui.peer_launch({}, "10.0.0.1",
                                "192.168.1.5", json.dumps(LOAD).encode(),
                                gate=Shut(), manual_hosts=["10.0.0.1"],
                                load=lambda **a: {}, resolve=lambda i: "/m")
    assert code == 200


@pytest.mark.parametrize("key", ["path", "target", "artifact", "where"])
def test_refused_when_a_path_is_in_the_payload(key):
    code, doc, loads = call({**LOAD, key: "/etc"})
    assert code == 400 and "identity" in doc["error"] and not loads


def test_a_body_that_is_not_json_says_why():
    code, doc = ui.peer_launch({}, "10.0.0.1", "10.0.0.2", b'{"action": ',
                               gate=Open(), load=lambda **a: {},
                               resolve=lambda i: "/m")
    assert code == 400 and "JSON" in doc["error"]
    assert "line 1" in doc["error"]          # the parser's own message


def test_unknown_identity_is_a_plain_refusal():
    code, doc, loads = call({**LOAD, "identity": "zzz"})
    assert code == 404 and doc["refused"].startswith("not on") and not loads


def test_knobs_outside_the_allow_list_are_refused():
    code, doc, loads = call({**LOAD, "sets": {"DYLD_INSERT_LIBRARIES": "x"}})
    assert code == 400 and not loads
    code, doc, loads = call({**LOAD, "sets": {"VQ_DECODE_CHUNK": "1;rm"}})
    assert code == 400 and not loads


def test_accepted_load_resolves_identity_here():
    code, doc, [a] = call(LOAD)
    assert code == 200 and a["artifact"] == "/models/X"
    assert a["tune"] == "fast" and a["sets"] == {"VQ_DECODE_CHUNK": "16"}


# --- identity ----------------------------------------------------------------

def make(tmp_path, name, cfg=b'{"model_type": "qwen3"}', size=10):
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_bytes(cfg)
    (d / "model.safetensors").write_bytes(b"\0" * size)
    return d


def test_identity_is_content_not_place(tmp_path):
    a, b = make(tmp_path, "a"), make(tmp_path, "b")
    c = make(tmp_path, "c", size=11)
    d = make(tmp_path, "d", cfg=b'{"model_type": "glm"}')
    assert artifact.identity(a) == artifact.identity(b) != ""
    assert artifact.identity(c) != artifact.identity(a)
    assert artifact.identity(d) != artifact.identity(a)
    assert artifact.identity(tmp_path / "missing") == ""
    ident = artifact.identity(c)
    assert artifact.resolve_identity(ident, [a, c]) == str(c)
    assert artifact.resolve_identity("nope", [a, c]) is None
    assert artifact.resolve_identity(str(a), [a, c]) is None   # not a path


# --- the coordinator's side --------------------------------------------------

def peers_with(*ps, monkeypatch):
    monkeypatch.setattr(ui, "PEERS", SimpleNamespace(all=lambda: list(ps)))


def peer(state="answering"):
    return SimpleNamespace(id="m4id", name="M4", host="10.0.0.2", port=8899,
                           state=state, key="10.0.0.2:8899",
                           found_by={"bonjour"})


def test_forward_refuses_unknown_or_silent_peer(monkeypatch):
    peers_with(peer("not_answering"), monkeypatch=monkeypatch)
    sent = []
    doc = ui.forward_launch({"action": "load", "node": "m4id",
                             "identity": "abc"},
                            post=lambda *a: sent.append(a))
    assert "not a machine that is answering" in doc["error"] and not sent
    doc = ui.forward_launch({"action": "load", "node": "other",
                             "identity": "abc"},
                            post=lambda *a: sent.append(a))
    assert "error" in doc and not sent


def test_forward_never_sends_a_path(monkeypatch):
    peers_with(peer(), monkeypatch=monkeypatch)
    for extra in ({"target": "/x"}, {"path": "/x"}, {"artifact": "/x"}):
        doc = ui.forward_launch({"action": "load", "node": "m4id",
                                 "identity": "abc", **extra},
                                post=lambda *a: 1 / 0)
        assert "identity" in doc["error"]


def test_forward_to_a_real_peer_page(monkeypatch):
    """The coordinator's forward_launch against the real handler on a local
    port: identity resolution, refusal text passed back."""
    loads = []
    monkeypatch.setattr(ui, "peer_launch", _stubbed(ui.peer_launch, loads))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), ui.make_handler({}))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        port = srv.server_address[1]
        p = SimpleNamespace(id="m4id", name="M4", host="127.0.0.1",
                            port=port, state="answering",
                            key=f"127.0.0.1:{port}", found_by={"manual"})
        peers_with(p, monkeypatch=monkeypatch)
        doc = ui.forward_launch({"action": "load", "node": "m4id",
                                 "identity": "abc", "tune": "safe",
                                 "sets": {"VQ_DECODE_CHUNK": "8"}})
        assert doc.get("starting") == "/models/X", doc
        assert doc["machine"] == "M4" and loads[0]["port"] == 8080
        doc = ui.forward_launch({"action": "load", "node": "m4id",
                                 "identity": "zzz"})
        assert doc["refused"].startswith("not on")
        doc = ui.forward_launch({"action": "unload", "node": "m4id",
                                 "port": 8123})
        assert doc["stopped"] == 8123
        # a browser page cannot reach the peer route
        import urllib.error
        import urllib.request
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{ui.PEER_LOAD_PATH}", method="POST",
            data=json.dumps({"action": "load", "identity": "abc"}).encode(),
            headers={"Origin": "http://evil.example"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req, timeout=5)
        assert e.value.code == 403
    finally:
        srv.shutdown()
    assert len(loads) == 1


def _stubbed(real, loads):
    def f(headers, client_ip, local_ip, body, **kw):
        return real(headers, client_ip, local_ip, body,
                    load=lambda **a: loads.append(a) or
                    {"starting": a["artifact"]},
                    resolve=lambda i: "/models/X" if i == "abc" else None,
                    stop=lambda port: {"stopped": port}, **kw)
    return f
