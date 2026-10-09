"""A machine's own settings -- the knurlogic allowance and the knurlogic
strategy (the default launch preset) -- set from ANY page: this machine's
through /machine.json, a peer's through that peer's own page
(/peer/machine.json behind the peer gate), which applies it to itself."""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from knurlogic.interfaces.page import documents
from knurlogic.interfaces.page import server as page_server
from knurlogic.machine.memory import allowance, wired
from knurlogic.tuning import strategy

GIB = 1 << 30


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(wired, "detected_working_set_bytes", lambda: 100 * GIB)
    monkeypatch.setattr(wired, "advise", lambda b: {"total_bytes": 128 * GIB})
    return tmp_path


def test_strategy_round_trips_and_defaults(home):
    assert strategy.get() == "default"
    assert strategy.set("lean") == "lean" and strategy.get() == "lean"
    assert json.loads(strategy.path().read_text()) == {"preset": "lean"}
    strategy.set("default")
    assert not strategy.path().exists() and strategy.get() == "default"
    # the names presets once had: fast and stable are default, safe is lean
    assert strategy.set("fast") == "default" and not strategy.path().exists()
    assert strategy.set("safe") == "lean" and strategy.get() == "lean"
    with pytest.raises(ValueError):
        strategy.set("reckless")
    strategy.path().write_text("{not json")
    assert strategy.get() == "default"


def test_every_preset_is_explained(home):
    from knurlogic.tuning.presets import PRESETS
    doc = documents.strategy_doc()
    assert [p["name"] for p in doc["presets"]] and \
        {p["name"] for p in doc["presets"]} == set(PRESETS)
    for p in doc["presets"]:
        assert p["title"] and set(p["values"]) == {
            r["name"] for r in doc["rows"]}
    assert all(r["help"] and r["options"] for r in doc["rows"])
    out = documents.set_strategy(b'{"preset": "lean"}')
    assert out["preset"] == "lean" and out["applied"]
    assert "error" in documents.set_strategy(b'{"preset": "x"}')
    assert strategy.get() == "lean"


def test_a_launch_without_a_tune_takes_the_strategy(home):
    assert page_server._default_tune() == "default"
    strategy.set("lean")
    assert page_server._default_tune() == "lean"


def test_peer_machine_applies_to_this_machine(home):
    code, doc = page_server.peer_machine({"allowance_gib": 64, "strategy": "lean"})
    assert code == 200
    assert allowance.get() == 64 * GIB and strategy.get() == "lean"
    assert doc["allowance"]["allowance_gib"] == 64
    assert doc["strategy"]["preset"] == "lean"
    assert set(doc["applied"]) == {"knurlogic allowance",
                                   "knurlogic strategy"}
    assert page_server.peer_machine({"allowance_gib": 500})[0] == 400
    assert page_server.peer_machine({"wired_mb": 1})[0] == 400
    assert page_server.peer_machine([])[0] == 400
    assert allowance.get() == 64 * GIB      # refusals change nothing


def _peers(monkeypatch, *keys):
    ps = [SimpleNamespace(key=k, state="answering", found_by=("bonjour",),
                          host=k.split(":")[0]) for k in keys]
    monkeypatch.setattr(page_server, "PEERS", SimpleNamespace(all=lambda: ps))


def test_machine_apply_forwards_only_to_an_answering_peer(home, monkeypatch):
    _peers(monkeypatch, "192.0.2.2:8899")
    sent = []

    def post(page, kind, doc):
        sent.append((page, kind, doc))
        return {"applied": {"knurlogic allowance": "80 GiB"}}
    code, doc = page_server.machine_apply("http://192.0.2.2:8899",
                                 b'{"allowance_gib": 80}', post=post)
    assert code == 200 and doc["applied"]
    assert sent == [("192.0.2.2:8899", "MachineSet", {"allowance_gib": 80})]
    # nothing of this machine's changed: the peer sets its own
    assert allowance.get() == 0
    assert page_server.machine_apply("http://203.0.113.9:8899", b"{}", post=post)[0] \
        == 403
    from knurlogic.cluster.protocol import VersionMismatch

    def old(*a):
        raise VersionMismatch("M4 does not speak this protocol: update "
                              "knurlogic on M4")
    code, doc = page_server.machine_apply(
        "http://192.0.2.2:8899", b'{"strategy": "lean"}', post=old)
    assert code == 502 and "update knurlogic on M4" in doc["error"]


def test_machine_apply_with_no_peer_is_this_machine(home, monkeypatch):
    _peers(monkeypatch)
    code, doc = page_server.machine_apply("", b'{"strategy": "lean"}')
    assert code == 200 and strategy.get() == "lean"


def _serve(handler):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _post(url, doc, headers=None):
    req = urllib.request.Request(url, data=json.dumps(doc).encode(),
                                 method="POST", headers={
                                     "Content-Type": "application/json",
                                     **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def test_one_page_sets_a_peers_allowance_through_the_peers_page(
        home, monkeypatch):
    """Two pages on loopback: the local page relays to the 'peer' page,
    which applies the setting (a MachineSet message) to itself."""
    peer, pport = _serve(page_server.make_handler({}))
    here, hport = _serve(page_server.make_handler({}))
    try:
        _peers(monkeypatch, f"127.0.0.1:{pport}")
        code, doc = _post(f"http://127.0.0.1:{hport}/machine.json?where="
                          f"http://127.0.0.1:{pport}",
                          {"allowance_gib": 48, "strategy": "lean"})
        assert code == 200, doc
        assert doc["allowance"]["allowance_gib"] == 48
        assert allowance.get() == 48 * GIB and strategy.get() == "lean"
        # a browser never reaches the peer route itself
        from knurlogic.cluster import transport
        env = json.loads(transport.encode("MachineSet",
                                          {"strategy": "default"})[0])
        code, doc = _post(f"http://127.0.0.1:{pport}{transport.MSG_PATH}",
                          env, {"Origin": "http://evil.example"})
        assert code == 403 and strategy.get() == "lean"
    finally:
        peer.shutdown()
        here.shutdown()
