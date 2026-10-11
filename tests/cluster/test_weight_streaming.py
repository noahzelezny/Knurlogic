"""cluster/weights.py: the read plan (who fetches from whom, who waits), a
page serving a shard's byte range over the peer route, and the rank's
client for it -- end to end against a real page on loopback."""

import threading
import time
from http.server import ThreadingHTTPServer

import pytest

from knurlogic.cluster import launch
from knurlogic.cluster import transport as TR
from knurlogic.cluster import weights as CW
from knurlogic.machine import artifact as A

PAGES = ["192.0.2.1:8899", "192.0.2.2:8899", "192.0.2.3:8899"]


# ------------------------------------------------------------ the plan

def test_the_mount_rank_fetches_first_and_the_holder_reads_after():
    plan = CW.read_plan(["leader", "holder"], [False, True], PAGES[:2])
    assert plan == [
        {"from": PAGES[1], "source": "holder", "turn": 0, "ordered": True},
        {"from": "", "source": "", "turn": 1, "ordered": True}]


def test_three_machines_fetch_from_the_first_holder():
    plan = CW.read_plan(["a", "b", "c"], [True, False, True], PAGES)
    assert [(e["from"], e["turn"]) for e in plan] == [
        ("", 1), (PAGES[0], 0), ("", 1)]


@pytest.mark.parametrize("local", [[True, True], [False, False]])
def test_no_order_unless_both_kinds_of_rank(local):
    plan = CW.read_plan(["a", "b"], local, PAGES[:2])
    assert not any(e["ordered"] for e in plan)
    assert not any(e["from"] for e in plan)        # nobody local: the mount


def test_the_plan_reaches_the_rank_environment():
    spec = {"rank": 1, "link": "ring", "hosts": [], "read": {
        "from": PAGES[0], "source": "holder", "turn": 0, "ordered": True}}
    env = launch.rank_env(spec, {"hostfile": "h"}, False)
    from knurlogic.engine.split import weights as W
    plan, page, source = W.plan_from_env(env)
    assert (page, source, plan.turn, plan.ordered) == (
        PAGES[0], "holder", 0, True)
    assert "KNURLOGIC_WEIGHTS" not in launch.rank_env(
        dict(spec, read=None), {"hostfile": "h"}, False)


def test_a_start_carries_this_ranks_entry(monkeypatch):
    seen = {}
    monkeypatch.setattr(launch, "_start", lambda prep, spawn, wait_s:
                        seen.update(prep["spec"]) or (200, {}))
    launch.PREPARED["ab" * 8] = {"spec": {"job": "ab" * 8, "rank": 1,
                                          "world": 2}, "path": "/x", "t": 9e9}
    reads = CW.read_plan(["a", "b"], [True, False], PAGES[:2])
    launch.start("ab" * 8, ring={"job": "ab" * 8, "reads": reads})
    assert seen["read"] == reads[1]


# ------------------------------------------------------ the serving page

@pytest.fixture
def art(tmp_path):
    d = tmp_path / "Model-8bit"
    d.mkdir()
    (d / "model-00001-of-00002.safetensors").write_bytes(bytes(range(256)) * 64)
    CW._RESOLVED.clear()
    yield d
    CW._RESOLVED.clear()


def _resolve(d):
    return lambda ident, name="": str(d) if ident == "0123456789abcdef" else None


GOOD = {"identity": "0123456789abcdef", "name": "Model-8bit",
        "file": "model-00001-of-00002.safetensors", "offset": 10, "length": 300}


def test_a_range_of_a_shard_is_served(art, monkeypatch):
    monkeypatch.setattr(A, "on_network", lambda p: False)
    code, b = CW.serve(GOOD, resolve=_resolve(art))
    assert code == 200 and isinstance(b, TR.Bytes)
    assert (b.offset, b.length) == (10, 300) and b.path.endswith(GOOD["file"])


@pytest.mark.parametrize("change,code", [
    ({"file": "../secret.safetensors"}, 400),
    ({"file": "/etc/passwd"}, 400),
    ({"file": "config.json"}, 400),
    ({"file": "model-00002-of-00002.safetensors"}, 404),
    ({"identity": "ffffffffffffffff"}, 404),
    ({"offset": 16000, "length": 1000}, 416),
    ({"length": -1}, 400),
])
def test_anything_but_a_shards_range_is_refused(art, monkeypatch, change, code):
    monkeypatch.setattr(A, "on_network", lambda p: False)
    assert CW.serve(dict(GOOD, **change), resolve=_resolve(art))[0] == code


def test_a_copy_on_a_mount_is_not_served(art, monkeypatch):
    monkeypatch.setattr(A, "on_network", lambda p: True)
    code, doc = CW.serve(GOOD, resolve=_resolve(art))
    assert code == 404 and "own disk" in doc["error"]


@pytest.fixture
def page(art, monkeypatch):
    from knurlogic.interfaces.page import server as page_server
    monkeypatch.setattr(A, "on_network", lambda p: False)
    monkeypatch.setattr(A, "resolve_identity", _resolve(art))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), page_server.make_handler({}))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_the_rank_reads_ranges_over_the_peer_route(page, art):
    read = CW.reader(page, GOOD["identity"], GOOD["name"])
    data = (art / GOOD["file"]).read_bytes()
    for off, n in ((0, 8), (10, 300), (16000, 384)):
        buf = bytearray(n)
        read(GOOD["file"], off, n, memoryview(buf))
        assert bytes(buf) == data[off:off + n]


def test_a_refusal_raises_with_the_peers_reason(page):
    read = CW.reader(page, "ffffffffffffffff", "")
    with pytest.raises(TR.PeerRefused, match="own disk"):
        read(GOOD["file"], 0, 8, memoryview(bytearray(8)))


def test_a_rank_fills_its_share_from_a_real_page(tmp_path, monkeypatch):
    """The whole path: a lazily loaded model, its arrays fetched from a
    page serving the shard, equal to reading the file."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from knurlogic.engine.model import weight_files as WF
    from knurlogic.engine.split import weights as W
    from knurlogic.interfaces.page import server as page_server
    d = tmp_path / "Tiny"
    d.mkdir()
    m = nn.Linear(64, 32)
    f = d / "model.safetensors"
    mx.save_safetensors(str(f), dict(tree_flatten(m.parameters())))
    monkeypatch.setattr(A, "on_network", lambda p: False)
    monkeypatch.setattr(A, "resolve_identity",
                        lambda ident, name="": str(d))
    CW._RESOLVED.clear()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), page_server.make_handler({}))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        read = CW.reader(f"127.0.0.1:{srv.server_address[1]}", "id", "Tiny")
        WF.use_peer(WF.Peer(read=read, source="holder", root=d))
        lazy = nn.Linear(64, 32)
        lazy.load_weights(list(WF.load(f, deferred=True).items()))
        out = W.fill(lazy)
        assert out["left"] == 0 and out["fetched"] == (64 * 32 + 32) * 4
        for k, v in tree_flatten(lazy.parameters()):
            assert mx.array_equal(v, dict(tree_flatten(m.parameters()))[k])
    finally:
        WF.use_peer(None)
        srv.shutdown()
        CW._RESOLVED.clear()


# ------------------------------------------------------ the page's labels

def _rank(monkeypatch, tmp_path, marker, held=0):
    import os

    from knurlogic.cluster import jobs as J
    from knurlogic.interfaces import spawn
    from knurlogic.interfaces.page import loads as page_loads
    from knurlogic.machine import servers
    art = tmp_path / "397B"
    art.mkdir(exist_ok=True)
    log = tmp_path / "r1.log"
    log.write_text("joined\n")
    now = time.time()
    os.utime(log, (now - 600, now - 600))          # quiet for 10 minutes
    jobs = {"j1/1": {"job": "j1", "rank": 1, "pid": 101, "artifact": str(art),
                     "log": str(log), "t": now - 700}}
    monkeypatch.setattr(J, "read_marker", lambda job, rank: dict(
        {"share_bytes": 500, "phase": "loading"}, **marker))
    monkeypatch.setattr(page_loads, "registry", lambda: {})
    monkeypatch.setattr(J, "registry", lambda: jobs)
    monkeypatch.setattr(servers, "is_our_server", lambda pid: True)
    monkeypatch.setattr(spawn, "_GROWTH", {})
    [e] = page_loads.load_progress({
        "memory": {"processes": [{"pid": 101, "bytes": held}]},
        "jobs": [{"job": "j1", "phase": "loading"}]})
    return e


def test_a_rank_waiting_its_turn_is_not_stalled(monkeypatch, tmp_path):
    e = _rank(monkeypatch, tmp_path, {"turn": "waiting"})
    assert (e["phase"], e["waiting"]) == ("waiting", "turn")


def test_a_rank_with_its_share_in_waits_for_the_others(monkeypatch, tmp_path):
    e = _rank(monkeypatch, tmp_path, {"turn": "reading"}, held=500)
    assert (e["phase"], e["waiting"]) == ("waiting", "others")


def test_a_fetching_rank_says_where_from(monkeypatch, tmp_path):
    e = _rank(monkeypatch, tmp_path, {"reading_from": "holder"}, held=100)
    assert e["reading_from"] == "holder"


def test_the_marker_phase_stays_one_the_watcher_knows():
    # jobs.phase_of reads an unknown phase as "joining", and a rank joining
    # for JOIN_S is stopped: the wait is the `turn` field, never a phase
    from knurlogic.cluster import jobs as J
    assert J.phase_of("j", [{"rank": 0}], read=lambda j, r: {
        "phase": "loading", "turn": "waiting"}) == "loading"


def test_a_turn_change_is_written_at_once(tmp_path, monkeypatch):
    from knurlogic.cluster import jobs as J
    monkeypatch.setattr(J, "marker_path", lambda job, rank: tmp_path / "m.json")
    m = J.Marker("j", 1)
    m.set(phase="loading")
    m.set(turn="waiting")
    m.set(turn="reading")
    import json
    assert json.loads((tmp_path / "m.json").read_text())["turn"] == "reading"
