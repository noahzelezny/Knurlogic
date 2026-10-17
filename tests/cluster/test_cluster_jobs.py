"""Cluster jobs: the page-to-page two-phase launch and the failure path.

No model loads anywhere. The pure parts (placement, the RDMA probe, the
prepare checks, the stall verdict) are tested directly. The failure path
runs for real across processes: this test process is page A (the
coordinator, with its page served on a thread), tests/support/cluster_fake_page.py
is page B in its own process with its own cache dir, and each page spawns
tests/support/cluster_fake_rank.py as its rank -- which writes real markers and,
as rank 0, answers through the real scheduler abort and SIGTERM path."""

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from procs import reap_registries

from knurlogic.cluster import jobs as J
from knurlogic.cluster import launch as C
from knurlogic.cluster import links
from knurlogic.engine.split import marker
from knurlogic.interfaces import spawn
from knurlogic.interfaces.page import loads as page_loads
from knurlogic.interfaces.page import nodes as page_nodes
from knurlogic.interfaces.page import peers as page_peers
from knurlogic.interfaces.page import router as page_router
from knurlogic.interfaces.page import server as page_server
from knurlogic.machine import identity
from knurlogic.machine.files import write_json
from knurlogic.tuning import checks


def _started() -> threading.Event:
    """The watcher's started flag, set: no thread; the test watches by hand."""
    ev = threading.Event()
    ev.set()
    return ev

# this test process plays the page: wire launch and recovery to it
page_server.wire()

GIB = 1 << 30
HERE = Path(__file__).resolve().parents[1] / "support"
SRC = str(HERE.parents[1] / "src")
VERSIONS = {"knurlogic": "0.1.0.dev0", "mlx": "0.31.2",
            "build": "0123456789ab+mlx0.31.2"}


def info(chip, tb, ws=64 * GIB, rdma=None):
    return {"chip": chip, "bandwidth_gbs": None,
            "thunderbolt": [{"iface": "en4", "ip": tb}],
            "rdma": rdma or {"available": False, "reason": "off",
                             "devices": [], "active": []},
            "versions": dict(VERSIONS), "jaccl_selfheal": False,
            "working_set_bytes": ws}


SHAPE = {"layer_bytes": [GIB] * 8, "other_bytes": GIB,
         "tensor_per_rank_bytes": 4 * GIB, "refusals": []}


# --- placement ---------------------------------------------------------------

def test_placement_leader_is_the_newest_chip_and_it_is_deterministic():
    ms = [{"name": "M3", **info("Apple M3 Ultra", "192.0.2.1")},
          {"name": "M4", **info("Apple M4 Max", "192.0.2.2")}]
    a = C.placement(ms, SHAPE, "tensor")
    assert a["order"] == ["M4", "M3"] and a["leader"] == "M4"
    assert [s["bytes"] for s in a["shares"]] == [4 * GIB] * 2
    assert C.placement(list(reversed(ms)), SHAPE, "tensor") == a


def test_pipeline_placement_gives_rank_0_the_last_layers():
    ms = [{"name": "A", **info("Apple M4 Max", "192.0.2.1", ws=20 * GIB)},
          {"name": "B", **info("Apple M3 Ultra", "192.0.2.2", ws=9 * GIB)}]
    p = C.placement(ms, SHAPE, "pipeline")
    assert p["order"] == ["A", "B"] and sum(p["layers"]) == 8
    assert p["shares"][0]["bounds"][1] == 8          # rank 0 ends the model
    assert p["layers"][0] > p["layers"][1]           # more room, more layers
    assert "rank 0 A leads" in p["reason"]


def test_placement_refuses_a_share_that_does_not_fit():
    ms = [{"name": "A", **info("Apple M4 Max", "192.0.2.1", ws=2 * GIB)},
          {"name": "B", **info("Apple M3 Ultra", "192.0.2.2")}]
    with pytest.raises(ValueError, match="A: its tensor share"):
        C.placement(ms, SHAPE, "tensor")


def test_placement_leaves_the_step_margin():
    """A share that fits the working set but not its step margin (5%, at
    least 4 GiB) is refused; the reason says what each rank leaves."""
    ms = [{"name": "A", **info("Apple M4 Max", "192.0.2.1", ws=7 * GIB)},
          {"name": "B", **info("Apple M3 Ultra", "192.0.2.2")}]
    with pytest.raises(ValueError, match=r"4\.0 GiB step margin"):
        C.placement(ms, SHAPE, "tensor")              # 4 of 7 GiB: 3 left
    ms[0]["working_set_bytes"] = 8 * GIB
    t = C.placement(ms, SHAPE, "tensor")
    assert "A leaves 4.0 GiB" in t["reason"]
    # pipeline: 8 x 1 GiB layers + 1 GiB each; after the 4 GiB margins
    # 12 and 6 GiB hold 7 and 1 layers (the bare working sets would 11 + 5)
    ms = [{"name": "A", **info("Apple M4 Max", "192.0.2.1", ws=12 * GIB)},
          {"name": "B", **info("Apple M3 Ultra", "192.0.2.2", ws=6 * GIB)}]
    p = C.placement(ms, SHAPE, "pipeline")
    assert p["layers"] == [7, 1]
    assert all(s["bytes"] <= m - 4 * GIB for s, m in
               zip(p["shares"], [12 * GIB, 6 * GIB]))
    assert "leaves 4.0 GiB" in p["reason"]


def test_ring_ips_pick_the_shared_thunderbolt_subnet():
    a = {"name": "A", "thunderbolt": [{"ip": "198.51.100.1"}, {"ip": "192.0.2.1"}]}
    b = {"name": "B", "thunderbolt": [{"ip": "192.0.2.2"}]}
    assert links.ring_ips([a, b]) == ["192.0.2.1", "192.0.2.2"]
    with pytest.raises(ValueError, match="no Thunderbolt address"):
        links.ring_ips([a, {"name": "C", "thunderbolt": []}])


def test_rdma_device_is_the_one_on_the_peers_subnet():
    a = {"thunderbolt": [{"iface": "en7", "ip": "198.51.100.1"},
                         {"iface": "en4", "ip": "192.0.2.1"}],
         "rdma": {"active": ["rdma_en4", "rdma_en7"]}}
    b = {"thunderbolt": [{"iface": "en2", "ip": "192.0.2.2"}],
         "rdma": {"active": ["rdma_en2"]}}
    assert links.rdma_device(a, b) == "rdma_en4"
    assert links.rdma_device(b, a) == "rdma_en2"


def test_two_cables_both_ends_on_one_subnet():
    # two cables: 192.0.2.x (M3 en4 - M4 en3) and 198.51.100.x (M3 en7 -
    # M4 en2). Each side picking "a device on the peer's subnet" would take
    # M4 en2 and M3 en4 -- two different cables; jaccl fails RTR.
    m3 = {"name": "M3", "thunderbolt": [{"iface": "en4", "ip": "192.0.2.1"},
                                        {"iface": "en7", "ip": "198.51.100.1"}],
          "rdma": {"active": ["rdma_en4", "rdma_en7"]}}
    m4 = {"name": "M4", "thunderbolt": [{"iface": "en2", "ip": "198.51.100.2"},
                                        {"iface": "en3", "ip": "192.0.2.2"}],
          "rdma": {"active": ["rdma_en2", "rdma_en3"]}}
    assert links.rdma_device(m4, m3) == "rdma_en3"
    assert links.rdma_device(m3, m4) == "rdma_en4"
    assert links.ring_ips([m4, m3]) == ["192.0.2.2", "192.0.2.1"]
    assert links.ring_ips([m4, m3], rdma=True) == ["192.0.2.2", "192.0.2.1"]
    # RDMA down on 192.0.2.x at one end: both move to 198.51.100.x
    m4d = {**m4, "rdma": {"active": ["rdma_en2"]}}
    assert links.rdma_device(m4d, m3) == "rdma_en2"
    assert links.rdma_device(m3, m4d) == "rdma_en7"
    assert links.ring_ips([m4d, m3], rdma=True) == ["198.51.100.2", "198.51.100.1"]


# --- RDMA probe --------------------------------------------------------------

DEVINFO = """hca_id:\trdma_en2
\t\tport:\t1
\t\t\tstate:\t\t\tPORT_DOWN (1)
hca_id:\trdma_en4
\t\tport:\t1
\t\t\tstate:\t\t\tPORT_ACTIVE (4)
"""


def fake_run(status="enabled", devices=True, devinfo=DEVINFO):
    def run(cmd):
        if cmd[0] == "rdma_ctl":
            return status
        if cmd[0] == "ibv_devices":
            return ("    device   node GUID\n    ------   ----\n"
                    "    rdma_en2  aa\n    rdma_en4  bb\n") if devices else ""
        return devinfo
    return run


def test_rdma_probe_reads_active_ports():
    r = links.rdma(fake_run())
    assert r["available"] and r["active"] == ["rdma_en4"]
    assert r["devices"] == ["rdma_en2", "rdma_en4"]


@pytest.mark.parametrize("run,why", [
    (lambda cmd: None, "no rdma_ctl"),
    (fake_run(status="disabled"), "disabled"),
    (fake_run(devices=False), "lists no device"),
    (fake_run(devinfo=DEVINFO.replace("PORT_ACTIVE", "PORT_DOWN")),
     "no Thunderbolt port"),
])
def test_rdma_probe_says_why_not(run, why):
    r = links.rdma(run)
    assert not r["available"] and why in r["reason"]


# --- prepare -----------------------------------------------------------------

def spec(**kw):
    s = {"job": "ab12cd34ef567890", "rank": 1, "world": 2, "split": "tensor",
         "link": "ring", "identity": "abc",
         "hosts": ["192.0.2.1:47200", "192.0.2.2:47201"], "ibv_devices": None,
         "coordinator": "", "layers": [], "prefill_chunk": 512,
         "tune": "default", "port": 0, "working_set_gib": 60.0,
         "bandwidth_gbs": 0, "sets": {}, "versions": dict(VERSIONS),
         "jaccl_timeout_ms": 0,
         "nodes": [{"rank": 0, "id": "a", "name": "A", "page": "x:1"},
                   {"rank": 1, "id": "b", "name": "B", "page": "y:1"}]}
    s.update(kw)
    return s


def prep(s, **kw):
    return C.prepare(s, resolve=kw.get("resolve", lambda i: "/m/x"),
                     info=kw.get("info", info("Apple M3 Ultra", "192.0.2.2")),
                     shape=lambda p, w, sp, vision=True, mtp=True:
                     kw.get("shape", SHAPE),
                     registry=dict)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "A"))
    return tmp_path


def test_prepare_accepts_and_remembers(cache):
    code, doc = prep(spec())
    assert code == 200 and doc["ok"]
    assert "ab12cd34ef567890" in C.PREPARED


@pytest.mark.parametrize("change,why", [
    ({"versions": {"knurlogic": "0.0.9", "mlx": "0.31.2"}}, "knurlogic 0.1.0"),
    ({"hosts": ["192.0.2.1:1", "203.0.113.9:2"]}, "not one of"),
    ({"link": "jaccl", "ibv_devices": [[None, "rdma_en4"], ["rdma_en2", None]],
      "coordinator": "192.0.2.1:1"}, "RDMA on"),
])
def test_prepare_refuses_with_its_reason(cache, change, why):
    code, doc = prep(spec(**change))
    assert code == 200 and not doc["ok"] and why in doc["refused"]


def test_an_unreadable_spec_from_another_build_names_the_build(cache):
    """A newer coordinator's spec carries a key this page does not know:
    the refusal says the builds differ, the cause, not only the key."""
    other = dict(VERSIONS, build="ba9876543210+mlx0.31.2")
    code, doc = prep(spec(versions=other, some_new_key=1))
    assert code == 400 and "same build" in doc["error"]
    assert "ba9876543210" in doc["error"]


def test_prepare_refuses_another_build_of_the_same_version(cache):
    other = dict(VERSIONS, build="ba9876543210+mlx0.31.2")
    code, doc = prep(spec(versions=other))
    assert code == 200 and not doc["ok"]
    # both fingerprints, so the person sees which side is stale
    assert VERSIONS["build"] in doc["refused"]
    assert other["build"] in doc["refused"]
    code, doc = prep(spec(versions={k: v for k, v in VERSIONS.items()
                                    if k != "build"}))
    assert not doc["ok"] and "build" in doc["refused"]


def test_build_fingerprint_hashes_the_source_and_names_mlx(tmp_path,
                                                          monkeypatch):
    pkg = tmp_path / "pkg"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "a.py").write_text("x = 1\n")
    (pkg / "sub" / "b.py").write_text("y = 2\n")
    monkeypatch.setattr(C, "_mlx_version", lambda: "0.31.2")
    one = C.build_fingerprint(root=pkg, cache={})
    assert one == C.build_fingerprint(root=pkg, cache={})
    assert one.endswith("+mlx0.31.2")
    (pkg / "sub" / "b.py").write_text("y = 3\n")
    assert C.build_fingerprint(root=pkg, cache={}) != one
    # cached per process: a second call does not re-read
    memo = {}
    first = C.build_fingerprint(root=pkg, cache=memo)
    (pkg / "a.py").write_text("x = 9\n")
    assert C.build_fingerprint(root=pkg, cache=memo) == first


def test_the_cluster_block_carries_the_build(monkeypatch):
    monkeypatch.setattr(C, "_INFO", {"doc": None, "at": 0.0})
    monkeypatch.setattr(C.identity, "chip", lambda: "Apple M4 Max")
    monkeypatch.setattr(C, "_selfheal", lambda: False)
    monkeypatch.setattr(links, "thunderbolt", list)
    monkeypatch.setattr(links, "rdma", lambda: {"available": False})
    v = C.node_info(0)["versions"]
    assert v["build"] == C.build_fingerprint()
    assert v["build"].endswith("+mlx" + (v["mlx"] or "none"))


def test_prepare_reserve_short_but_step_margin_kept_fits(cache):
    """Short of the full reserve but clear of the step margin: it fits (the
    old tight band); past the step margin it is refused."""
    shape = dict(SHAPE, reserve={"transient_bytes": 6 * GIB,
                                 "kv_bytes": GIB})        # 8.5 GiB margin
    i = info("Apple M3 Ultra", "192.0.2.2", ws=10 * GIB)  # 4 + 8.5 > 10 > 4 + 4
    code, doc = prep(spec(), info=i, shape=shape)
    assert doc["ok"], doc
    code, doc = prep(spec(), shape=shape,
                     info=info("Apple M3 Ultra", "192.0.2.2", ws=7 * GIB))
    assert not doc["ok"] and "more space required" in doc["refused"]


def test_prepare_refuses_a_share_that_does_not_fit_here(cache):
    code, doc = prep(spec(), info=info("Apple M3 Ultra", "192.0.2.2",
                                       ws=2 * GIB))
    assert not doc["ok"] and "working set" in doc["refused"]
    s = spec(split="pipeline", layers=[2, 6])        # rank 1: layers 0..5
    code, doc = prep(s, info=info("Apple M3 Ultra", "192.0.2.2", ws=6 * GIB))
    assert not doc["ok"] and "7.0 GiB" in doc["refused"]
    # fits the working set, not the step margin: 4 GiB in 7 leaves 3
    code, doc = prep(spec(), info=info("Apple M3 Ultra", "192.0.2.2",
                                       ws=7 * GIB))
    assert not doc["ok"] and "4.0 GiB step margin" in doc["refused"]
    code, doc = prep(spec(), info=info("Apple M3 Ultra", "192.0.2.2",
                                       ws=8 * GIB))
    assert doc["ok"], doc


def test_prepare_says_not_on_and_rejects_bad_shapes(cache):
    code, doc = prep(spec(), resolve=lambda i: None)
    assert not doc["ok"] and doc["refused"].startswith("not on")
    for bad in ({"job": "../../etc"}, {"world": 1}, {"split": "data"},
                {"hosts": ["a:1"]}, {"path": "/etc"}):
        code, doc = prep(spec(**bad))
        assert code == 400


@pytest.mark.parametrize("bad", [
    {"port": "8080"}, {"port": -1}, {"port": True},
    {"working_set_gib": "60"}, {"bandwidth_gbs": [1]}, {"tune": "turbo"},
    {"nodes": [1, 2]},
    {"nodes": [{"rank": 0, "id": "a", "page": "x:1"},
               {"rank": 1, "id": "b", "name": "B", "page": "y:1"}]},
    {"nodes": [{"rank": 0, "id": "a", "name": "A", "page": 5},
               {"rank": 1, "id": "b", "name": "B", "page": "y:1"}]},
    {"sets": ["a=b"]},
])
def test_prepare_type_checks_the_spec(cache, bad):
    code, doc = prep(spec(**bad))
    assert code == 400, doc


def test_prepare_refuses_unknown_sets_and_stores_only_clean_ones(
        cache, monkeypatch):
    monkeypatch.setattr(checks, "launch_knobs",
                        lambda: frozenset({"kv_bits"}))
    code, doc = prep(spec(sets={"kv_bits": "8", "evil": "x"}))
    assert not doc["ok"] and "evil" in doc["refused"]
    code, doc = prep(spec(sets={"kv_bits": "8; rm"}))
    assert not doc["ok"] and "kv_bits" in doc["refused"]
    code, doc = prep(spec(sets={"kv_bits": "8"}))
    assert doc["ok"]
    assert C.PREPARED["ab12cd34ef567890"]["spec"]["sets"] == {"kv_bits": "8"}


def test_stop_tells_only_pages_this_page_knows(cache, monkeypatch):
    """The spec's page addresses are never posted to: a node is told at
    the address this page's PEERS store has for its id, or not at all."""
    from knurlogic.cluster.peers import Peer
    s = spec(nodes=[{"rank": 0, "id": "me", "name": "A", "page": "x:1"},
                    {"rank": 1, "id": "b", "name": "B",
                     "page": "evil.example:80"},
                    {"rank": 2, "id": "c", "name": "C", "page": "z:1"}])
    monkeypatch.setattr(identity, "identity", lambda: {"id": "me"})
    peers = [Peer(host="192.0.2.2", port=8765, id="b", name="B",
                  state="answering")]
    monkeypatch.setattr(page_nodes, "PEERS", SimpleNamespace(all=lambda: peers))
    C.SPECS["ab12cd34ef567890"] = s
    posted = []
    out = C.stop("ab12cd34ef567890",
                 post=lambda page, kind, d: posted.append((page, kind)),
                 grace=0)
    assert posted == [("192.0.2.2:8765", "Stop")]
    assert out["told"] == ["B"]
    C.ENDED.pop("ab12cd34ef567890", None)


def test_stop_tells_the_other_pages_before_it_stops_its_own_ranks(
        cache, monkeypatch):
    """A rank stopped here before its peers hear of it fails their ranks'
    ring ("[ring] connection to a peer was lost"): their pages would record
    a crash, and recovery relaunch, a job that was asked to stop."""
    from knurlogic.cluster.peers import Peer
    job = "ab12cd34ef567890"
    monkeypatch.setattr(identity, "identity", lambda: {"id": "me"})
    monkeypatch.setattr(page_nodes, "PEERS", SimpleNamespace(all=lambda: [
        Peer(host="192.0.2.2", port=8765, id="b", name="B",
             state="answering")]))
    C.SPECS[job] = spec(nodes=[{"rank": 0, "id": "me", "name": "A"},
                               {"rank": 1, "id": "b", "name": "B"}])
    J.save_registry({f"{job}/0": {"job": job, "rank": 0, "pid": 4242}})
    monkeypatch.setattr(C, "_alive", lambda j, pid: True)
    order = []
    monkeypatch.setattr(J, "terminate", lambda pids, **kw: order.append(
        "own ranks") or [])
    C.stop(job, post=lambda page, kind, d: order.append(kind), grace=0)
    assert order == ["Stop", "own ranks"]
    C.ENDED.pop(job, None)
    J.save_registry({})


def test_another_pages_stop_answers_once_its_ranks_are_marked(
        cache, monkeypatch):
    """A follower leaves only on rank 0's stop, and rank 0's page stops
    rank 0 once every other page has answered: a page answering only after
    its follower's grace held an idle job's unload for that grace (25 s,
    measured) and SIGKILLed the follower. It answers once its ranks are
    marked stopping, and ends them on a thread."""
    import threading
    job = "ab12cd34ef567890"
    J.save_registry({f"{job}/1": {"job": job, "rank": 1, "pid": 4242}})
    ended = threading.Event()
    monkeypatch.setattr(C, "_alive", lambda j, pid: not ended.is_set())
    release = threading.Event()

    calls = []

    def terminate(pids, **kw):
        calls.append(pids)
        release.wait(5)
        ended.set()
        return []
    monkeypatch.setattr(J, "terminate", terminate)
    code, out = C.peer_step("Stop", {"job": job, "reason": "unloaded"})
    assert code == 200 and out["stopped"] == job and out["exiting"] == [4242]
    assert not ended.is_set()                  # answered before the end
    assert J.registry()[f"{job}/1"]["stopping"]
    # a second Stop (a retry, another page's) answers the same, and does
    # not end the ranks a second time
    code, again = C.peer_step("Stop", {"job": job, "reason": "unloaded"})
    assert code == 200 and again["exiting"] == [4242]
    release.set()
    assert ended.wait(5)
    for t in threading.enumerate():        # the stop's own end, then clean
        if t.name == f"knurlogic-stop-{job}":
            t.join(5)
    assert f"{job}/1" not in J.registry() and calls == [[4242]]
    assert job not in C._STOPPING
    C.ENDED.pop(job, None)
    J.save_registry({})


# --- the stall verdict -------------------------------------------------------

def test_verdict_idle_is_not_stalled_but_busy_and_still_is(monkeypatch):
    w = J.Watch()
    ranks = [{"rank": 0, "pid": 1, "machine": "A"}]
    idle = {"phase": "ready", "step": 5, "busy": False}
    busy = {"phase": "ready", "step": 5, "busy": True}
    alive = lambda p: True
    assert w.check("j", ranks, alive, now=0, read=lambda j, r: idle)[0] == ""
    assert w.check("j", ranks, alive, now=1e6, read=lambda j, r: idle)[0] == ""
    assert w.check("j", ranks, alive, now=1e6 + J.STALL_S - 1,
                   read=lambda j, r: busy)[0] == ""
    why = w.check("j", ranks, alive, now=1e6 + J.STALL_S + 1,
                  read=lambda j, r: busy)[0]
    assert "stalled" in why
    moved = {**busy, "step": 6}
    assert w.check("j", ranks, alive, now=2e6, read=lambda j, r: moved)[0] == ""


def test_verdict_a_rank_whose_peer_machine_went_silent(tmp_path, monkeypatch):
    """The rank's bell watcher (engine/split/liveness) marks peer_gone; the
    page stops the job on it -- the way out for a rank blocked in a transfer
    nothing else can unblock (jaccl) -- and the mark is written at once."""
    w = J.Watch()
    ranks = [{"rank": 0, "pid": 7, "machine": "A"}]
    gone = {"phase": "ready", "busy": True, "pid": 7,
            "peer_gone": "the machine holding rank 1 stopped answering"}
    why, kind = w.check("j", ranks, lambda p: True, now=0,
                        read=lambda j, r: gone)
    assert why == ("rank 0 on A: the machine holding rank 1 stopped "
                   "answering") and kind == "machine"
    # an earlier process's mark (a reused marker file) is not this rank's
    assert w.check("j", ranks, lambda p: True, now=0,
                   read=lambda j, r: {**gone, "pid": 8})[0] == ""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    m = J.Marker("ab12cd34ef567890", 0)
    m._last = time.time()                 # just written: a plain set waits
    m.set(peer_gone="the machine holding rank 1 stopped answering")
    assert (J.read_marker("ab12cd34ef567890", 0) or {}).get("peer_gone")


def test_verdict_a_long_prefill_is_one_step_of_many_chunks_not_a_stall():
    """397B over two Macs: a 30k-token prompt missing the prompt cache is
    one step of ~2 minutes, and must not read as stalled at 122 s while
    both ranks are computing. Chunks are progress."""
    w = J.Watch()
    ranks = [{"rank": 0, "pid": 1, "machine": "A"}]
    alive = lambda p: True
    doc = lambda c: (lambda j, r: {"phase": "ready", "step": 9, "busy": True,
                                   "chunk": c})
    assert w.check("j", ranks, alive, now=0, read=doc(0))[0] == ""
    for k in range(1, 6):          # a chunk every 60 s, no step for 300 s
        assert w.check("j", ranks, alive, now=60 * k, read=doc(k))[0] == ""
    assert "stalled" in w.check("j", ranks, alive,
                                now=300 + J.STALL_S + 1, read=doc(5))[0]


def test_verdict_a_request_waiting_on_a_slow_load_is_not_a_stall():
    """Qwen3.8-Flash-Next-6bit read over SMB: a request arriving while
    the ranks are still loading (busy, step 0) must not read as stalled
    at 122 s, mid-load. Only a loaded ring can stall; loading has its
    own clock (the pid, the join deadline)."""
    w = J.Watch()
    ranks = [{"rank": 0, "pid": 1, "machine": "A"}]
    alive = lambda p: True
    loading = lambda j, r: {"phase": "loading", "step": 0, "busy": True}
    assert w.check("j", ranks, alive, now=0, read=loading)[0] == ""
    assert w.check("j", ranks, alive, now=10 * J.STALL_S,
                   read=loading)[0] == ""
    # once ready, the clock starts from there
    ready = lambda j, r: {"phase": "ready", "step": 0, "busy": True}
    t = 10 * J.STALL_S + 1
    assert w.check("j", ranks, alive, now=t, read=ready)[0] == ""
    assert "stalled" in w.check("j", ranks, alive, now=t + J.STALL_S + 1,
                                read=ready)[0]


def test_a_prefill_chunk_bumps_the_marker(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    m = J.Marker("ab12cd34ef567890", 0)
    monkeypatch.setitem(marker.CURRENT, "marker", m)
    marker.chunk_done()
    marker.chunk_done()
    m.beat()
    assert J.read_marker("ab12cd34ef567890", 0)["chunk"] == 2


def test_verdict_dead_pid_and_never_joined():
    w = J.Watch()
    ranks = [{"rank": 1, "pid": 7, "machine": "B", "t": 0}]
    assert "exited" in w.check("ab12cd34ef567890", ranks, lambda p: False,
                               now=1)[0]
    why = w.check("j", ranks, lambda p: True, now=J.JOIN_S + 5,
                  read=lambda j, r: {"phase": "joining", "since": 0})[0]
    assert "not joined" in why


def test_marker_writes_phase_changes_at_once(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    m = J.Marker("ab12cd34ef567890", 1)
    m.set(phase="loading")
    assert J.read_marker("ab12cd34ef567890", 1)["phase"] == "loading"
    m.set(step=3)                    # throttled: not yet on disk
    m.beat()
    assert J.read_marker("ab12cd34ef567890", 1)["step"] == 3


def test_after_load_arms_the_jaccl_deadline(monkeypatch):
    monkeypatch.setenv("KNURLOGIC_JACCL_TIMEOUT_MS", "60000")
    monkeypatch.setenv("JACCL_COLLECTIVE_TIMEOUT_MS", "0")
    marker.after_load()
    assert os.environ["JACCL_COLLECTIVE_TIMEOUT_MS"] == "60000"


def test_rank_env_sets_the_timeout_to_zero_for_the_load_only_with_the_fork():
    s = spec(link="jaccl", jaccl_timeout_ms=60000, coordinator="192.0.2.1:9")
    e = C.rank_env(s, {"ibv": "/j/ibv.json", "hostfile": ""}, selfheal=True)
    assert e["JACCL_COLLECTIVE_TIMEOUT_MS"] == "0"
    assert e["KNURLOGIC_JACCL_TIMEOUT_MS"] == "60000"
    assert e["MLX_IBV_DEVICES"] == "/j/ibv.json" and e["MLX_RANK"] == "1"
    e = C.rank_env(s, {"ibv": "/j/ibv.json", "hostfile": ""}, selfheal=False)
    assert "JACCL_COLLECTIVE_TIMEOUT_MS" not in e
    e = C.rank_env(spec(), {"hostfile": "/j/h.json"}, selfheal=True)
    assert e["MLX_HOSTFILE"] == "/j/h.json"


def test_a_ring_failure_is_a_503():
    from knurlogic.engine.runtime.scheduler import RingFailed
    from knurlogic.interfaces.http.openai import status_of
    e = status_of(RingFailed("a rank exited"))
    assert e.status == 503 and e.body()["error"]["code"] == "cluster_failed"


# --- two pages, real processes -----------------------------------------------

def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait(pred, t=20.0, every=0.1):
    end = time.time() + t
    while time.time() < end:
        v = pred()
        if v:
            return v
        time.sleep(every)
    return pred()


def alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                             capture_output=True, text=True).stdout
    except Exception:
        return True
    return bool(out.strip()) and not out.strip().startswith("Z")


@pytest.fixture
def two_pages(tmp_path, monkeypatch, owned_procs):
    """Page A (here) and page B (a process), each with its own cache."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "A"))
    monkeypatch.setenv("PYTHONPATH", SRC + os.pathsep
                       + os.environ.get("PYTHONPATH", ""))
    monkeypatch.setitem(identity._ID, "id", "aaaa")
    monkeypatch.setitem(identity._ID, "name", "A")
    monkeypatch.setattr(C, "_resolve",
                        lambda i, name="": "/fake/artifact" if i == "abc" else None)
    monkeypatch.setattr(C, "shape_of", lambda p, w, s, vision=True, mtp=True: SHAPE)
    info_a = info("Apple M4 Max", "127.0.0.1")
    info_b = info("Apple M3 Ultra", "127.0.0.1")
    monkeypatch.setattr(C, "_local_info", lambda: info_a)
    sys.path.insert(0, str(HERE))
    import cluster_fake_page
    monkeypatch.setattr(C, "RANK_ARGV", [cluster_fake_page.fake_argv])
    monkeypatch.setattr(C, "_WATCHER", _started())        # A is watched by hand
    ui_a, ui_b = free_port(), free_port()
    srv = ThreadingHTTPServer(("127.0.0.1", ui_a), page_server.make_handler({}))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    # page B, both ranks and the server are reaped however setup or the
    # test ends: a failed assert before the yield used to skip the
    # teardown and leave rank 0 (its own session) running
    try:
        env = {**os.environ, "XDG_CACHE_HOME": str(tmp_path / "B")}
        page_b = owned_procs(subprocess.Popen(
            [sys.executable, str(HERE / "cluster_fake_page.py"), str(ui_b),
             "bbbb", "B", json.dumps(info_b), "aaaa", f"127.0.0.1:{ui_a}"],
            env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
        assert "up" in page_b.stdout.readline()
        peer = SimpleNamespace(id="bbbb", name="B", host="127.0.0.1",
                               key=f"127.0.0.1:{ui_b}", state="answering",
                               link="thunderbolt", node={"cluster": info_b})
        from knurlogic.cluster.peers import Peer
        known = Peer(host="127.0.0.1", port=ui_b, id="bbbb", name="B",
                     state="answering")
        monkeypatch.setattr(page_nodes, "PEERS", SimpleNamespace(
            all=lambda: [known], introduce=lambda *a, **k: None))
        serve_port = free_port()
        out = C.launch({"action": "load", "identity": "abc",
                        "nodes": ["aaaa", "bbbb"], "split": "tensor",
                        "link": "ring"},
                       me={"id": "aaaa", "name": "A"}, peers=[peer],
                       local_info=info_a, ui_port=ui_a, serve_port=serve_port)
        assert out.get("job"), out
        job = out["job"]
        # rank 0 is A (the newer chip) and answers on serve_port
        assert out["leader"] == "A" and out["port"] == serve_port
        b_dir = tmp_path / "B" / "knurlogic" / "jobs"
        rank1 = wait(lambda: (json.loads((b_dir / "jobs.json").read_text())
                              .get(f"{job}/1") if (b_dir / "jobs.json").exists()
                              else None))
        rank0 = J.registry()[f"{job}/0"]
        wait(lambda: (J.read_marker(job, 0) or {}).get("phase") == "ready")
        yield SimpleNamespace(job=job, out=out, rank0=rank0, rank1=rank1,
                              port=serve_port, page_b=page_b, b_dir=b_dir,
                              ui_b=ui_b)
    finally:
        for job in list(C.SPECS):
            C.stop(job, propagate=False, grace=1)
        reap_registries(tmp_path / "A", tmp_path / "B")
        srv.shutdown()
        C.ENDED.clear()


def in_flight(port):
    """POST a chat to rank 0 on a thread; -> (thread, result dict)."""
    res = {}

    def go():
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions", data=b"{}",
            method="POST", headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                res.update(code=r.status, body=json.loads(r.read()))
        except urllib.error.HTTPError as e:
            res.update(code=e.code, body=json.loads(e.read()))
        except Exception as e:
            res.update(code=None, error=repr(e))
    t = threading.Thread(target=go, daemon=True)
    t.start()
    # held in flight: rank 0's marker says busy
    assert wait(lambda: any((J.read_marker(j, 0) or {}).get("busy")
                            for j in J.by_job()))
    return t, res


def test_launch_shows_placement_and_lists_the_job_once(two_pages):
    p = two_pages
    assert p.out["placement"]["order"] == ["A", "B"]
    assert "tensor split 2 ways" in p.out["placement"]["reason"]
    docs = C.jobs_document()
    assert [d["job"] for d in docs] == [p.job]
    assert docs[0]["machines"] == ["A", "B"] and docs[0]["port"] == p.port
    # the page merges its launch card with the job card by job id, else by
    # identity -- never by folder name, which may differ per machine
    assert docs[0]["identity"] and docs[0]["identity"] == p.out["starting"]
    doc = page_loads.with_jobs({"resident": [
        {"runtime": "knurlogic", "where": f"http://127.0.0.1:{p.port}"}]})
    assert doc["resident"][0]["cluster"]["machines"] == ["A", "B"]


def test_kill_rank_1_stops_the_job_and_rank_0_answers_503(two_pages):
    p = two_pages
    t, res = in_flight(p.port)
    os.kill(p.rank1["pid"], 9)          # B's page sees it and stops the job
    t.join(40)
    assert res.get("code") == 503, res
    assert res["body"]["error"]["code"] == "cluster_failed"
    assert wait(lambda: not alive(p.rank0["pid"]), 20)
    assert wait(lambda: p.job not in {d["job"] for d in C.jobs_document()
                                      if d.get("phase") != "stopped"}, 10)


def test_kill_rank_0_and_its_page_has_rank_1_killed(two_pages):
    p = two_pages
    os.kill(p.rank0["pid"], 9)
    assert wait(lambda: not alive(p.rank0["pid"]), 5)
    stopped = C.watch_once()
    assert stopped and stopped[0][0] == p.job and "exited" in stopped[0][1]
    # the reason names the machine rank 0 ran on, not "this machine":
    # stop() sends these same words to B's page
    assert p.rank0["machine"] == "A" and p.rank1["machine"] == "B"
    assert "rank 0 on A " in stopped[0][1], stopped
    assert wait(lambda: not alive(p.rank1["pid"]), 20)
    ended = [d for d in C.jobs_document() if d["job"] == p.job]
    assert ended and ended[0]["phase"] == "stopped"
    # an ended job still says how it was split (a card must never read "null")
    assert ended[0]["split"] == "tensor" and ended[0]["link"]


def test_the_verdict_names_a_ranks_machine_from_the_job_when_unrecorded():
    w = J.Watch()
    why = w.check("ab12cd34ef567890",
                  [{"rank": 1, "pid": 5, "machines": ["A", "B"]}],
                  lambda p: False, now=1)[0]
    assert "rank 1 on B " in why


def test_a_stalled_ring_is_torn_down(two_pages, monkeypatch):
    p = two_pages
    monkeypatch.setattr(J, "STALL_S", 1.0)
    t, res = in_flight(p.port)          # busy, and the step never moves
    assert C.watch_once() == []         # first sight of the step
    time.sleep(1.5)
    stopped = C.watch_once()
    assert stopped and "stalled" in stopped[0][1]
    t.join(20)
    assert res.get("code") == 503
    assert wait(lambda: not alive(p.rank1["pid"]), 20)


def test_unload_from_the_page_stops_every_rank(two_pages):
    p = two_pages
    from knurlogic.machine import servers
    assert servers.registry()[p.port]["job"] == p.job
    out = spawn.stop(p.port)              # the Unload button on rank 0's row
    assert out["stopped"] and out["told"] == ["B"]
    assert not alive(p.rank0["pid"])
    assert wait(lambda: not alive(p.rank1["pid"]), 20)
    assert p.port not in servers.registry()


def test_peer_cluster_routes_are_gated_like_peer_loaded(two_pages):
    p = two_pages
    from knurlogic.cluster import transport
    body, _ = transport.encode("Stop", {"job": p.job})
    req = urllib.request.Request(
        f"http://127.0.0.1:{p.ui_b}{transport.MSG_PATH}",
        data=body, method="POST",
        headers={"Content-Type": "application/json",
                 "Origin": f"http://127.0.0.1:{p.ui_b}"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=10)
    assert e.value.code == 403
    assert alive(p.rank1["pid"])
    code, doc = C.peer_step("Prepare", {**spec(), "path": "/etc"})
    assert code == 400 and "identity" in doc["error"]


def test_a_ranks_working_set_is_the_wired_limit_not_the_ram():
    from knurlogic.machine.memory import wired
    gib = 1 << 30
    assert wired.gpu_working_set(96 * gib, 84 * gib) == 84 * gib
    assert wired.gpu_working_set(96 * gib, 0) == 96 * gib
    assert wired.gpu_working_set(0, 84 * gib) == 84 * gib
    assert wired.gpu_working_set(0, 0) == 0


def test_every_rank_syncs_the_gpu_fast():
    for link, files in (("ring", {"hostfile": "/j/h.json"}),
                        ("jaccl", {"ibv": "/j/ibv.json", "hostfile": ""})):
        s = {**spec(), "link": link}
        assert C.rank_env(s, files, selfheal=False)[
            "MLX_METAL_FAST_SYNCH"] == "1"


class _Capture:
    """Just enough of a request handler for page_router.send_json."""

    def __init__(self):
        import io
        self.code, self.wfile = None, io.BytesIO()

    def send_response(self, code):
        self.code = code

    def send_header(self, *_a):
        pass

    def end_headers(self):
        pass

    def doc(self):
        return json.loads(self.wfile.getvalue())


def test_a_dead_rank_0_is_a_503_cluster_failed_with_the_reason(two_pages):
    p = two_pages
    os.kill(p.rank0["pid"], 9)
    assert wait(lambda: not alive(p.rank0["pid"]), 5)
    base = f"http://127.0.0.1:{p.port}"
    h = _Capture()
    page_router.stream(h, base + "/v1/chat/completions", b"{}", base=base)
    assert h.code == 503, h.doc()
    err = h.doc()["error"]
    assert err["code"] == "cluster_failed" and err["type"] == "server_error"
    assert "rank 0 on A " in err["message"] and "exited" in err["message"]
    # and once the job is stopped, still that reason (not a 502)
    h = _Capture()
    page_router.stream(h, base + "/v1/chat/completions", b"{}", base=base)
    assert h.code == 503 and "rank 0 on A " in h.doc()["error"]["message"]


def test_an_unreachable_server_that_is_no_job_is_still_a_502(cache):
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    h = _Capture()
    page_router.stream(h, base + "/v1/chat/completions", b"{}", base=base)
    assert h.code == 502


def test_a_peers_dead_rank_0_is_a_503_with_the_peers_stop_reason(
        monkeypatch):
    base = "http://192.0.2.2:8080"
    dead = f"http://127.0.0.1:{free_port()}"
    monkeypatch.setitem(page_peers.PEER_TARGETS, base, {
        "machine": "M4", "relay": dead, "job": "ab12cd34ef567890"})

    def survey():
        page_peers.PEER_JOBS["ab12cd34ef567890"] = {
            "job": "ab12cd34ef567890", "phase": "stopped",
            "reason": "rank 0 on M4 (pid 7) exited"}
    monkeypatch.setattr(page_router, "refresh_targets", survey)
    h = _Capture()
    page_router.stream(h, page_peers.upstream(base, "/v1/chat/completions"), b"{}",
               base=base)
    assert h.code == 503
    assert h.doc()["error"]["code"] == "cluster_failed"
    assert "rank 0 on M4 (pid 7) exited" in h.doc()["error"]["message"]
    page_peers.PEER_JOBS.clear()


def _sse_upstream(events, done):
    """A one-shot upstream that answers an event stream -- `events`, then
    [DONE] when `done` -- and drops the socket (a rank 0 dying mid-way
    when not `done`)."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)

    def run():
        c, _ = srv.accept()
        c.recv(65536)
        c.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                  b"Connection: close\r\n\r\n")
        for e in events:
            c.sendall(b"data: " + e + b"\n\n")
        if done:
            c.sendall(b"data: [DONE]\n\n")
        # a FIN, not a reset: closing with request bytes still unread
        # would reset the socket and lose what was sent before it
        c.shutdown(socket.SHUT_WR)
        c.settimeout(2)
        try:
            while c.recv(65536):
                pass
        except OSError:
            pass
        c.close()
        srv.close()
    threading.Thread(target=run, daemon=True).start()
    return f"http://127.0.0.1:{srv.getsockname()[1]}"


@pytest.mark.parametrize("done", [False, True])
def test_a_rank_0_dying_mid_stream_ends_the_stream_with_cluster_failed(
        monkeypatch, done):
    asked = []
    monkeypatch.setattr(page_router, "cluster_failure",
                        lambda b: asked.append(b) or "rank 1 on B exited")
    base = _sse_upstream([b'{"x": 1}'], done)
    h = _Capture()
    page_router.stream(h, base + "/v1/chat/completions", b"{}", base=base)
    out = h.wfile.getvalue().decode()
    assert h.code == 200 and out.startswith('data: {"x": 1}')
    if done:
        assert "cluster_failed" not in out and not asked
        return
    last = out.strip().split("\n\n")[-1]
    assert last.startswith("data: ")
    err = json.loads(last[6:])["error"]
    assert err["code"] == "cluster_failed"
    assert "rank 1 on B exited" in err["message"]


def test_a_relayed_cluster_failed_is_not_said_twice(monkeypatch):
    """A peer's rank 0 dies: the peer page's relay already ends the stream
    with its cluster_failed event; this page passes that on and adds none
    of its own (which would name whichever rank it saw exit last)."""
    monkeypatch.setattr(page_router, "cluster_failure",
                        lambda b: "rank 1 on B exited")
    relayed = json.dumps(page_router.cluster_failed("rank 0 on A exited")).encode()
    base = _sse_upstream([b'{"x": 1}', relayed], False)
    h = _Capture()
    page_router.stream(h, base + "/v1/chat/completions", b"{}", base=base)
    out = h.wfile.getvalue().decode()
    assert out.count("cluster_failed") == 1
    assert "rank 0 on A exited" in out and "rank 1 on B" not in out


def test_a_cut_stream_that_is_no_cluster_job_just_ends(monkeypatch):
    monkeypatch.setattr(page_router, "cluster_failure", lambda b: "")
    base = _sse_upstream([b'{"x": 1}'], False)
    h = _Capture()
    page_router.stream(h, base + "/v1/chat/completions", b"{}", base=base)
    assert h.wfile.getvalue() == b'data: {"x": 1}\n\n'


def test_slot_skips_the_ring_ports_of_live_jobs(monkeypatch):
    job = "0005" + "0" * 12                      # its own slot is 5
    assert C._slot(job, used=set()) == 5
    assert C._slot(job, used={5, 6}) == 7
    assert C._slot("0063" + "0" * 12, used={99}) == 0    # wraps
    with pytest.raises(ValueError, match="every ring-port slot"):
        C._slot(job, used=set(range(100)))
    # the registry's live jobs: a recorded ring port, else the nonce's slot
    monkeypatch.setattr(J, "by_job", lambda: {
        "aaaa000000000000": [{"ring_port": C.RING_PORT + 5 * 20 + 1}],
        "0006000000000000": [{"rank": 0}]})
    assert C._used_slots() == {5, 6}
    assert C._slot(job) == 7


def test_jaccl_without_an_rdma_subnet_is_refused_not_rerouted(cache,
                                                             monkeypatch):
    """RDMA up on both Macs, but on different cables: refused, with each
    Mac's active devices -- never jaccl over some other device."""
    monkeypatch.setattr(C, "_resolve", lambda i, name="": "/m/x")
    monkeypatch.setattr(C, "shape_of", lambda p, w, s, vision=True, mtp=True: SHAPE)
    rd = lambda dev: {"available": True, "reason": "", "devices": [dev],
                      "active": [dev]}
    ia = {**info("Apple M4 Max", "192.0.2.1", rdma=rd("rdma_en4")),
          "thunderbolt": [{"iface": "en4", "ip": "192.0.2.1"}]}
    ib = {**info("Apple M3 Ultra", "198.51.100.2", rdma=rd("rdma_en7")),
          "thunderbolt": [{"iface": "en7", "ip": "198.51.100.2"},
                          {"iface": "en2", "ip": "192.0.2.2"}]}
    peer = SimpleNamespace(id="bbbb", name="B", host="192.0.2.2",
                           key="192.0.2.2:8765", state="answering",
                           link="thunderbolt", node={"cluster": ib})
    out = C.launch({"action": "load", "identity": "abc",
                    "nodes": ["aaaa", "bbbb"], "split": "tensor",
                    "link": "jaccl"},
                   me={"id": "aaaa", "name": "A"}, peers=[peer],
                   local_info=ia, ui_port=1, serve_port=2,
                   post=lambda *a, **k: pytest.fail("posted"))
    assert "refused" in out and "no Thunderbolt subnet" in out["refused"]
    assert "rdma_en4" in out["refused"] and "rdma_en7" in out["refused"]
    assert links.rdma_device(ia, ib) is None


# --- one machine never holds two jobs' shares -------------------------------
# A 397B pipeline job's rank 1 held 49.5 GiB on an M3 Ultra (84 GiB working
# set) when a second job's rank 1 started loading the same share beside it:
# Metal wedged and the Mac rebooted, and rank 0 on the other machine sat
# idle in a collective with nobody to tear it down.

def test_prepare_refuses_a_share_beside_what_another_job_holds(cache):
    big = dict(SHAPE, tensor_per_rank_bytes=50 * GIB)
    ws = info("Apple M3 Ultra", "192.0.2.2", ws=84 * GIB)
    held = [("job 41648583878fcdfc rank 1", 54699, int(49.5 * GIB))]
    code, doc = C.prepare(spec(), resolve=lambda i: "/m/x", info=ws,
                          shape=lambda p, w, s, vision=True, mtp=True: big,
                          registry=dict,
                          held=lambda job: held)
    assert not doc["ok"], doc
    r = doc["refused"]
    assert "50.0 GiB" in r and "84.0 GiB" in r and "49.5 GiB of it held now" in r
    assert "pid 54699" in r and "41648583878fcdfc" in r
    code, doc = C.prepare(spec(), resolve=lambda i: "/m/x", info=ws,
                          shape=lambda p, w, s, vision=True, mtp=True: big,
                          registry=dict,
                          held=lambda job: [])
    assert doc["ok"], doc


def test_prepare_counts_what_another_job_holds_once(cache):
    """Memory available now already leaves out what other ranks hold: a
    rank's 11 GiB share beside a 79.4 GiB rank, on a 128 GiB Mac with 29
    GiB available, was refused at -50.3 GiB left (held subtracted twice)."""
    m = dict(info("Apple M4 Max", "192.0.2.2", ws=120 * GIB),
             available_bytes=29 * GIB)
    held = [("job 416d59fd3aa52908 rank 0", 51487, int(79.4 * GIB))]

    def go(need):
        return C.prepare(spec(), resolve=lambda i: "/m/x", info=m,
                         shape=lambda p, w, s, vision=True, mtp=True:
                         dict(SHAPE, tensor_per_rank_bytes=need),
                         registry=dict, held=lambda job: held)[1]
    assert go(11 * GIB)["ok"]
    # what is available now still bounds it: 29 GiB less the 6 GiB margin
    r = go(24 * GIB)["refused"]
    assert "29.0 GiB is available now; the 29.0 GiB left" in r, r


def test_what_is_held_counts_live_ranks_and_servers_not_the_stopping(
        cache, monkeypatch):
    from knurlogic.machine import servers
    monkeypatch.setattr(C, "_alive", lambda job, pid: pid != 4)
    monkeypatch.setattr(servers, "is_our_server", lambda pid: True)
    reg = {"aaaa0000/1": {"job": "aaaa0000", "rank": 1, "pid": 1},
           "bbbb0000/0": {"job": "bbbb0000", "rank": 0, "pid": 2,
                          "stopping": 1.0},
           "cccc0000/1": {"job": "cccc0000", "rank": 1, "pid": 3},
           "dddd0000/1": {"job": "dddd0000", "rank": 1, "pid": 4}}
    sreg = {8080: {"pid": 5}, 8081: {"pid": 1, "job": "aaaa0000"}}
    got = C.held_here("cccc0000", reg=reg, sreg=sreg,
                      rss=lambda pid: pid * GIB)
    assert sorted(p for _, p, _ in got) == [1, 5]
    assert ("the server on port 8080", 5, 5 * GIB) in got


def test_prepare_refuses_while_another_jobs_rank_is_loading(cache):
    other = "0123456789abcdef"
    reg = {f"{other}/1": {"job": other, "rank": 1, "pid": 1}}
    write_json(J.marker_path(other, 1), {"phase": "loading"})
    code, doc = C.prepare(spec(), resolve=lambda i: "/m/x",
                          info=info("Apple M3 Ultra", "192.0.2.2"),
                          shape=lambda p, w, s, vision=True, mtp=True: SHAPE,
                          registry=lambda: reg,
                          held=lambda job: [])
    assert not doc["ok"] and "one load at a time" in doc["refused"]
    write_json(J.marker_path(other, 1), {"phase": "ready"})
    code, doc = C.prepare(spec(), resolve=lambda i: "/m/x",
                          info=info("Apple M3 Ultra", "192.0.2.2"),
                          shape=lambda p, w, s, vision=True, mtp=True: SHAPE,
                          registry=lambda: reg,
                          held=lambda job: [])
    assert doc["ok"], doc


SLOW = ("import signal, sys, time\n"
        "def bye(*a):\n"
        "    time.sleep(float(sys.argv[1]))\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, bye)\n"
        "print('up', flush=True)\n"
        "time.sleep(120)\n")


def slow_rank(job, rank, exit_s):
    """A fake rank that takes `exit_s` to go after SIGTERM (a rank giving
    back 50 GiB of Metal memory), registered as this page's."""
    p = subprocess.Popen([sys.executable, "-c", SLOW, str(exit_s),
                          "--job", job], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "up"
    C._PROCS[(job, rank)] = p
    reg = J.registry()
    reg[f"{job}/{rank}"] = {"job": job, "rank": rank, "world": 2,
                            "pid": p.pid, "machines": ["A", "B"],
                            "t": time.time()}
    J.save_registry(reg)
    return p


def test_stop_then_immediately_launch_waits_for_the_old_rank_to_be_gone(
        cache, monkeypatch):
    """The smoke test's sequence: stop a job and launch the next at once.
    The old rank takes a while to exit; the new rank must not be spawned
    (nor the old job reported stopped) while it is still there."""
    monkeypatch.setattr(C, "_local_info",
                        lambda: info("Apple M3 Ultra", "192.0.2.2"))
    old, new = "41648583878fcdfc", "db0cb292aefeba55"
    p = slow_rank(old, 1, 1.5)
    stopper = threading.Thread(
        target=lambda: C.stop(old, propagate=False, grace=5), daemon=True)
    stopper.start()
    assert wait(lambda: J.registry().get(f"{old}/1", {}).get("stopping"), 5)
    # while it exits: "stopping", never "stopped", and its memory is not
    # counted as staying (start waits for it instead)
    doc = {d["job"]: d for d in C.jobs_document()}
    assert doc[old]["phase"] == "stopping" and doc[old]["exiting"] == [p.pid]
    code, got = prep(spec(job=new))
    assert got["ok"], got
    seen = {}

    def spawn(cmd, **kw):
        seen["old_alive"] = p.poll() is None and alive(p.pid)
        return subprocess.Popen([sys.executable, "-c",
                                 "import time; time.sleep(60)",
                                 "--job", new], **kw)
    code, out = C.start(new, spawn=spawn)
    assert code == 200, out
    assert seen == {"old_alive": False}
    stopper.join(10)
    assert [d["phase"] for d in C.jobs_document() if d["job"] == old] \
        == ["stopped"]
    C.stop(new, propagate=False, grace=1)
    C.ENDED.clear()


def test_prepare_waits_for_a_stopping_rank_then_goes(cache, monkeypatch):
    """A relaunch's prepare, while a previous job's rank on this machine
    is still exiting: it waits, bounded, instead of granting the new job a
    share -- and a ring port -- the old rank still holds, then goes once
    the old rank is gone."""
    monkeypatch.setattr(C, "EXIT_WAIT_S", 3.0)
    old, new = "41648583878fcdfc", "db0cb292aefeba55"
    p = slow_rank(old, 1, 0.3)
    reg = J.registry()
    reg[f"{old}/1"]["stopping"] = time.time()
    J.save_registry(reg)
    p.send_signal(signal.SIGTERM)          # its bye handler exits in 0.3s
    code, doc = C.prepare(spec(job=new), resolve=lambda i: "/m/x",
                         info=info("Apple M3 Ultra", "192.0.2.2"),
                         shape=lambda p, w, s, vision=True, mtp=True: SHAPE,
                         held=lambda job: [])
    assert doc["ok"], doc
    p.wait(timeout=5)
    C._PROCS.clear()
    J.save_registry({})


def test_prepare_names_a_stuck_old_rank_in_its_refusal(cache, monkeypatch):
    monkeypatch.setattr(C, "EXIT_WAIT_S", 0.3)
    old, new = "41648583878fcdfc", "db0cb292aefeba55"
    p = slow_rank(old, 1, 60)
    reg = J.registry()
    reg[f"{old}/1"]["stopping"] = time.time()
    J.save_registry(reg)
    code, doc = C.prepare(spec(job=new), resolve=lambda i: "/m/x",
                         info=info("Apple M3 Ultra", "192.0.2.2"),
                         shape=lambda p, w, s, vision=True, mtp=True: SHAPE,
                         held=lambda job: [])
    assert not doc["ok"], doc
    assert old in doc["refused"] and "still exiting" in doc["refused"]
    p.kill()
    p.wait()
    C._PROCS.clear()
    J.save_registry({})


def test_slot_conflict_finds_an_alive_rank_of_another_job_on_the_port(
        monkeypatch):
    """A quick relaunch's `_slot` only scans this machine's own registry
    (`_used_slots`), so a coordinator elsewhere can still hand a new job
    the same ring/jaccl slot a previous job's still-live rank holds
    here. `prepare`/`start` catch it on the port instead."""
    s = spec(job="db0cb292aefeba55", rank=1,
             hosts=["192.0.2.1:47201", "192.0.2.2:47202"])
    reg = {"old/1": {"job": "41648583878fcdfc", "rank": 1, "pid": 999,
                     "ring_port": 47202}}
    monkeypatch.setattr(C, "_alive", lambda job, pid: True)
    assert C._slot_conflict(s, reg) == ("41648583878fcdfc", 1, 999)
    monkeypatch.setattr(C, "_alive", lambda job, pid: False)
    assert C._slot_conflict(s, reg) is None
    assert C._slot_conflict(s, {}) is None
    # a different slot never conflicts
    other = dict(reg, **{"old/1": {**reg["old/1"], "ring_port": 47222}})
    monkeypatch.setattr(C, "_alive", lambda job, pid: True)
    assert C._slot_conflict(s, other) is None


def test_start_refuses_when_a_stopped_rank_will_not_go(cache, monkeypatch):
    monkeypatch.setattr(C, "_local_info",
                        lambda: info("Apple M3 Ultra", "192.0.2.2"))
    old, new = "41648583878fcdfc", "db0cb292aefeba55"
    p = slow_rank(old, 1, 60)
    reg = J.registry()
    reg[f"{old}/1"]["stopping"] = time.time()
    J.save_registry(reg)
    code, got = prep(spec(job=new))
    assert got["ok"], got
    spawned = []
    code, out = C.start(new, spawn=lambda *a, **k: spawned.append(a),
                        wait_s=0.5)
    assert code == 409 and not spawned
    assert f"pid {p.pid}" in out["error"] and "still exiting" in out["error"]
    p.kill()
    p.wait()
    C._PROCS.clear()
    J.save_registry({})


def test_a_stop_whose_rank_outlives_the_reap_is_finished_by_the_watcher(
        cache, monkeypatch):
    job = "41648583878fcdfc"
    monkeypatch.setattr(C, "_peer_pages", dict)
    p = slow_rank(job, 1, 3)
    out = C.stop(job, propagate=False, grace=0.2, reap=0.1)
    # SIGKILL takes a Popen at once; a slow reaper is modelled by the
    # record: still exiting means still recorded, and not "stopped"
    if out["exiting"]:
        assert [d["phase"] for d in C.jobs_document()] == ["stopping"]
        assert wait(lambda: not alive(p.pid), 10)
        C.watch_once()
    assert J.registry() == {}
    assert [d["phase"] for d in C.jobs_document()] == ["stopped"]
    C.ENDED.clear()


# --- a peer's page: what it says is evidence, its silence is not ------------
# docs/design/liveness.md: a page is not its rank (ranks outlive a page
# restart), so only what a page reports stops a job; a peer machine that
# goes silent is caught by the ranks' bell watchers (engine/split/liveness).

JOB = "ab12cd34ef567890"
RECS = [{"job": JOB, "rank": 0, "pid": 1, "t": 0.0,
         "nodes": [{"rank": 0, "id": "a", "name": "M4"},
                   {"rank": 1, "id": "b", "name": "M3"}]}]
PAGES = dict(pages={"b": "192.0.2.1:8765"}, me="a")


def _down(page, j):
    raise C.transport.PeerUnreachable("timed out")


def test_a_page_that_does_not_answer_never_stops_a_job(monkeypatch, caplog):
    """2026-10-04: a slow M4 page (28 s) stopped a job whose ranks were
    fine; 2026-10-15: a page redeploy did. However long a page is silent
    -- or unknown to this page -- the job is kept, with one warning."""
    monkeypatch.setattr(C, "peers_fn", list)
    with caplog.at_level("WARNING"):
        for _ in range(5):
            assert C.peer_verdict(JOB, RECS, ask=_down, **PAGES) == ("", None)
        assert C.peer_verdict(JOB, RECS, ask=_down, pages={},
                              me="a") == ("", None)
    assert sum("is not evidence" in r.message for r in caplog.records) == 1


def test_peer_verdict_acts_at_once_on_what_a_page_says(monkeypatch):
    monkeypatch.setattr(C, "peers_fn", list)

    def says(doc):
        return lambda page, j: doc
    assert C.peer_verdict(JOB, RECS, ask=says({"ranks_here": [1]}),
                          **PAGES) == ("", None)
    # preparing (from Prepare until its registry record exists): held
    assert C.peer_verdict(JOB, RECS, ask=says({"ranks_here": [],
                                               "prepared": True}),
                          **PAGES) == ("", None)
    why, kind = C.peer_verdict(JOB, RECS, ask=says(
        {"ranks_here": [], "prepared": False}), **PAGES)
    assert "M3 no longer runs its rank" in why and kind == "machine"
    # a rank that exited there: that page's own reason and kind
    why, kind = C.peer_verdict(JOB, RECS, ask=says(
        {"ranks_here": [], "exited": "rank 1 on M3 (pid 9) exited: out of "
         "memory: OOM", "exited_kind": "memory"}), **PAGES)
    assert why.startswith("M3: rank 1 on M3") and kind == "memory"
    # the job ended there: its reason
    assert C.peer_verdict(JOB, RECS, ask=says(
        {"ranks_here": [], "ended": "rank 1 exited"}), **PAGES) \
        == ("M3 stopped the job: rank 1 exited", None)
    # with the kind that page recorded: recovery treats it the same here
    assert C.peer_verdict(JOB, RECS, ask=says(
        {"ranks_here": [], "ended": "rank 1 exited: its ring lost a peer",
         "ended_kind": "machine"}), **PAGES)[1] == "machine"


def test_watch_keeps_a_job_whose_peer_page_is_silent(monkeypatch):
    """The watcher end to end: ranks alive here, the peer page silent ->
    nothing stopped, so recovery has nothing to relaunch; a rank dead here
    -> stopped as before."""
    from knurlogic.cluster import recovery as R
    recs = [dict(RECS[0], pid=4242, t=990.0)]
    monkeypatch.setattr(J, "by_job", lambda: {JOB: recs})
    monkeypatch.setattr(C, "_peer_pages", lambda: {"b": "192.0.2.1:8765"})
    monkeypatch.setattr(C, "peers_fn", list)
    monkeypatch.setattr(C, "_ask_job", _down)
    from knurlogic.machine import identity
    monkeypatch.setattr(identity, "identity", lambda: {"id": "a"})
    stops = []
    monkeypatch.setattr(C, "stop", lambda job, **kw: stops.append(kw))
    alive = {"v": True}
    monkeypatch.setattr(C, "_alive", lambda j, pid: alive["v"])
    # (under the join deadline: no marker here, so the rank reads as joining)
    for t in (1000.0, 1010.0, 1000.0 + J.JOIN_S - 20):
        assert C.watch_once(now=t) == []
    assert stops == [] and JOB not in R.MODELS
    alive["v"] = False
    out = C.watch_once(now=1000.0 + J.JOIN_S - 10)
    assert out and out[0][0] == JOB and stops


def test_peer_verdict_reads_the_peer_lists_boot_id(monkeypatch):
    """A new boot_id holding no rank is a restart; one still holding its
    rank is a restarted page whose rank lived on."""
    from knurlogic.cluster.peers import Peer
    rec = Peer(host="192.0.2.1", port=8765, id="b", name="M3",
               state="answering", last_seen=100.0, boot_id="one")
    monkeypatch.setattr(C, "peers_fn", lambda: [rec])
    up = (lambda page, j: {"ranks_here": [1]})
    assert C.peer_verdict(JOB, RECS, ask=up, **PAGES) == ("", None)
    rec.boot_id = "two"
    # the page restarted, its rank (its own session) did not: kept
    assert C.peer_verdict(JOB, RECS, ask=up, **PAGES) == ("", None)
    assert C._PEER_BOOT[(JOB, "b")] == "two"
    rec.boot_id = "three"
    why, _ = C.peer_verdict(JOB, RECS, ask=(
        lambda page, j: {"ranks_here": [], "prepared": False}), **PAGES)
    assert "M3 restarted" in why


def test_job_state_says_why_a_rank_here_exited(cache, monkeypatch):
    """A rank whose process is gone, before this page's watcher stopped the
    job: JobState carries the watcher's own reason, so a peer page stops the
    job with it at once -- whether or not this page's watcher runs."""
    job = "41648583878fcdfc"
    log = cache / "rank1.log"
    log.write_text("RuntimeError: [metal::malloc] Resource limit exceeded\n")
    J.save_registry({f"{job}/1": {"job": job, "rank": 1, "pid": 999999,
                                  "machine": "M3", "log": str(log)}})
    monkeypatch.setattr(C, "_alive", lambda j, pid: False)
    doc = C.job_state(job)
    assert doc["ranks_here"] == []
    assert doc["exited"].startswith("rank 1 on M3 (pid 999999) exited")
    assert doc["exited_kind"] == C.exit_reason("x", [{"log": str(log)}])[1]
    J.save_registry({})


def test_the_job_route_says_what_this_page_runs(two_pages):
    p = two_pages
    code, doc = C.peer_step("JobState", {"job": p.job})
    assert code == 200 and doc["ranks_here"] == [0] and not doc["ended"]
    code, doc = C.peer_step("JobState", {"job": "../x"})
    assert code == 400


def test_a_vanished_peer_page_keeps_the_job(
        two_pages, monkeypatch):
    """B's page process dies (a redeploy, a crash) while its rank, in its
    own session, runs on: A keeps the job however long B's page is gone. A
    machine that dies is the ranks' bell watchers' to catch."""
    p = two_pages
    assert C.watch_once() == []                 # B answers: healthy
    p.page_b.kill()                             # B's page is gone
    p.page_b.wait()
    for _ in range(3):
        assert C.watch_once() == []
        time.sleep(0.5)
    assert alive(p.rank0["pid"])


def test_a_peer_page_that_stopped_the_job_has_this_one_stop_too(
        two_pages, monkeypatch):
    """B's page (in its own process) watches A's: A forgetting the job
    without telling B is caught from B's side."""
    p = two_pages
    # A drops its ranks without propagating: B must notice by itself
    C.stop(p.job, reason="rank 0 on A exited", propagate=False, grace=1)
    assert wait(lambda: not alive(p.rank1["pid"]), 30)


# --- the cable: two Thunderbolt cables, one failing link init ---------------
# jaccl on one cable (192.0.2.x) can fail QP RTR (errno 96), even in a bare
# two-rank mlx test, while the other (198.51.100.x) works; a launch must not
# keep picking the lowest shared subnet.

def two_cable_infos():
    rd = {"available": True, "reason": "", "devices": ["rdma_en4",
                                                       "rdma_en7"],
          "active": ["rdma_en4", "rdma_en7"]}
    a = {**info("Apple M4 Max", "127.0.0.1", rdma=rd), "thunderbolt": [
        {"iface": "en4", "ip": "127.0.0.1"},
        {"iface": "en7", "ip": "127.0.1.1"}]}
    b = {**info("Apple M3 Ultra", "127.0.0.2", rdma=rd), "thunderbolt": [
        {"iface": "en4", "ip": "127.0.0.2"},
        {"iface": "en7", "ip": "127.0.1.2"}]}
    return a, b


def test_link_init_failure_reads_jaccl_and_ring_lines():
    assert "RTR failed with errno 96" in C.link_init_failure(
        "loading\nValueError: [jaccl] Changing queue pair to RTR failed "
        "with errno 96\n")
    assert C.link_init_failure("[ring] Couldn't connect (error: 61)")
    assert C.link_init_failure("rank 1 exited") == ""
    # a running peer that died is not a link that failed to come up: the
    # job must not move to another cable, and recovery waits for machines
    lost = "RuntimeError: [ring] connection to a peer was lost"
    assert C.link_init_failure(lost) == ""


def test_a_failing_cable_goes_last_for_that_pair_only(monkeypatch):
    a, b = two_cable_infos()
    a, b = {**a, "id": "aaaa"}, {**b, "id": "bbbb"}
    monkeypatch.setattr(links, "BAD_CABLES", {})
    assert links.shared_subnets(a, b, rdma=True) == ["127.0.0", "127.0.1"]
    links.BAD_CABLES[links.machine_pair(a, b)] = {"127.0.0": "RTR errno 96"}
    assert links.shared_subnets(a, b, rdma=True) == ["127.0.1", "127.0.0"]
    assert links.rdma_device(a, b) == "rdma_en7"
    assert links.ring_ips([a, b]) == ["127.0.1.1", "127.0.1.2"]
    c = {**b, "id": "cccc"}
    assert links.shared_subnet(a, c) == "127.0.0"


def jaccl_launch(monkeypatch, req, post=None, follow=None):
    a, b = two_cable_infos()
    monkeypatch.setattr(C, "_resolve", lambda i, name="": "/m/x")
    monkeypatch.setattr(C, "shape_of", lambda p, w, s, vision=True, mtp=True: SHAPE)
    peer = SimpleNamespace(id="bbbb", name="B", host="127.0.0.2",
                           key="127.0.0.2:8765", state="answering",
                           link="thunderbolt", node={"cluster": b})
    got = []

    def fake_post(page, kind, doc, **kw):
        got.append((kind, doc))
        if kind == "Prepare":
            return {"ok": True}
        if kind == "Start":
            return {"started": doc["job"]}
        return {}
    monkeypatch.setattr(C, "prepare", lambda spec: (200, {"ok": True}))
    monkeypatch.setattr(C, "start", lambda job, **k: (200, {"started": job}))
    out = C.launch({"action": "load", "identity": "abc",
                    "nodes": ["aaaa", "bbbb"], "split": "pipeline",
                    "link": "jaccl", **req},
                   me={"id": "aaaa", "name": "A"}, peers=[peer],
                   local_info=a, ui_port=1, serve_port=2,
                   post=post or fake_post,
                   follow=follow or (lambda job, ctx: None))
    return out, got


def test_a_launch_can_name_its_cable(cache, monkeypatch):
    monkeypatch.setattr(links, "BAD_CABLES", {})
    followed = []
    out, got = jaccl_launch(monkeypatch, {"cable": "127.0.1.x"},
                            follow=lambda j, c: followed.append(j))
    assert out.get("job") and out["cable"] == "127.0.1", out
    assert "named by the launch" in out["cable_note"]
    spec = next(d for k, d in got if k == "Prepare")
    assert spec["cable"] == "127.0.1"
    assert spec["ibv_devices"] == [[None, "rdma_en7"], ["rdma_en7", None]]
    assert spec["coordinator"].startswith("127.0.1.")
    assert not followed                  # a named cable is never moved off
    out, _ = jaccl_launch(monkeypatch, {"cable": "10.9.9"})
    assert "refused" in out and "127.0.0, 127.0.1" in out["refused"]


def test_the_default_is_the_lowest_cable_and_says_what_is_next(
        cache, monkeypatch):
    monkeypatch.setattr(links, "BAD_CABLES", {})
    followed = []
    out, _ = jaccl_launch(monkeypatch, {},
                          follow=lambda j, c: followed.append((j, c)))
    assert out["cable"] == "127.0.0"
    assert "127.0.1 next if link init fails" in out["cable_note"]
    assert followed and followed[0][1]["tried"] == ("127.0.0",)


def test_a_link_init_failure_relaunches_on_the_next_cable(cache,
                                                          monkeypatch):
    monkeypatch.setattr(links, "BAD_CABLES", {})
    ctxs = []
    out, _ = jaccl_launch(monkeypatch, {},
                          follow=lambda j, c: ctxs.append((j, c)))
    job, ctx = ctxs[0]
    # a failure that is not the link's: nothing moves
    assert C.failover(job, ctx, "rank 1 on B (pid 9) exited") is None
    why = ("rank 1 on B (pid 9) exited: link init failed: ValueError: "
           "[jaccl] Changing queue pair to RTR failed with errno 96")
    C.ENDED[job] = {"reason": why, "t": time.time(), "port": 2,
                    "machines": ["A", "B"]}
    new = C.failover(job, ctx, why)
    assert new["job"] and new["job"] != job and new["cable"] == "127.0.1"
    assert "moved from 127.0.0" in new["cable_note"]
    assert "errno 96" in new["cable_note"]
    assert C.ENDED[job]["relaunched"] == new["job"]
    assert "relaunched on cable 127.0.1" in C.ENDED[job]["reason"]
    assert "127.0.0" in links.BAD_CABLES[frozenset(("aaaa", "bbbb"))]
    # the next launch this session starts on the working cable, and says so
    later, _ = jaccl_launch(monkeypatch, {})
    assert later["cable"] == "127.0.1"
    assert "127.0.0 failed link init earlier" in later["cable_note"]
    # the relaunch failing too: nothing left to try, said
    ctx2 = ctxs[1][1]
    last = C.failover(new["job"], ctx2, why)
    assert "every shared Thunderbolt cable failed" in last["refused"]
    C.ENDED.clear()


def test_the_follower_stops_at_a_failure_or_gives_up(cache, monkeypatch):
    monkeypatch.setattr(links, "BAD_CABLES", {})
    ctxs = []
    jaccl_launch(monkeypatch, {}, follow=lambda j, c: ctxs.append((j, c)))
    job, ctx = ctxs[0]
    seen = []
    monkeypatch.setattr(C, "failover",
                        lambda j, c, why: seen.append(why) or "moved")
    ctx = dict(ctx, post=lambda page, kind, doc: {"ended": "rank 1 exited: "
                                           "[jaccl] RTR failed"})
    assert C._follow(job, ctx, sleep=lambda s: None) == "moved"
    assert seen == ["rank 1 exited: [jaccl] RTR failed"]
    t = iter(range(0, 10 ** 6, 100))
    ctx = dict(ctx, post=lambda page, kind, doc: {"ranks_here": [1]})
    assert C._follow(job, ctx, clock=lambda: next(t),
                     sleep=lambda s: None) is None


def test_a_rank_that_fails_jaccl_init_moves_the_job_to_the_next_cable(
        tmp_path, monkeypatch, owned_procs):
    """Two real pages and fake ranks: rank 1 on B fails link init on the
    first cable; B's page stops the job naming the jaccl line, and A (the
    coordinator) relaunches on the second cable, where it runs."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "A"))
    monkeypatch.setenv("PYTHONPATH", SRC + os.pathsep
                       + os.environ.get("PYTHONPATH", ""))
    monkeypatch.setenv("FAKE_BAD_CABLE", "127.0.0")
    monkeypatch.setitem(identity._ID, "id", "aaaa")
    monkeypatch.setitem(identity._ID, "name", "A")
    monkeypatch.setattr(C, "_resolve",
                        lambda i, name="": "/fake/artifact" if i == "abc" else None)
    monkeypatch.setattr(C, "shape_of", lambda p, w, s, vision=True, mtp=True: SHAPE)
    monkeypatch.setattr(links, "BAD_CABLES", {})
    monkeypatch.setattr(C, "FAILOVER_POLL_S", 0.2)
    info_a, info_b = two_cable_infos()
    monkeypatch.setattr(C, "_local_info", lambda: info_a)
    sys.path.insert(0, str(HERE))
    import cluster_fake_page
    monkeypatch.setattr(C, "RANK_ARGV", [cluster_fake_page.fake_argv])
    monkeypatch.setattr(C, "_WATCHER", _started())
    ui_a, ui_b = free_port(), free_port()
    srv = ThreadingHTTPServer(("127.0.0.1", ui_a), page_server.make_handler({}))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    env = {**os.environ, "XDG_CACHE_HOME": str(tmp_path / "B")}
    page_b = owned_procs(subprocess.Popen(
        [sys.executable, str(HERE / "cluster_fake_page.py"), str(ui_b),
         "bbbb", "B", json.dumps(info_b), "aaaa", f"127.0.0.1:{ui_a}"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    assert "up" in page_b.stdout.readline()
    peer = SimpleNamespace(id="bbbb", name="B", host="127.0.0.1",
                           key=f"127.0.0.1:{ui_b}", state="answering",
                           link="thunderbolt", node={"cluster": info_b})
    from knurlogic.cluster.peers import Peer
    known = Peer(host="127.0.0.1", port=ui_b, id="bbbb", name="B",
                 state="answering")
    monkeypatch.setattr(page_nodes, "PEERS", SimpleNamespace(
        all=lambda: [known], introduce=lambda *a, **k: None))
    jobs = []
    try:
        out = C.launch({"action": "load", "identity": "abc",
                        "nodes": ["aaaa", "bbbb"], "split": "pipeline",
                        "link": "jaccl"},
                       me={"id": "aaaa", "name": "A"}, peers=[peer],
                       local_info=info_a, ui_port=ui_a,
                       serve_port=free_port())
        assert out.get("job") and out["cable"] == "127.0.0", out
        jobs.append(out["job"])
        moved = wait(lambda: (C.ENDED.get(out["job"]) or {}).get(
            "relaunched"), 40)
        assert moved, C.ENDED.get(out["job"])
        jobs.append(moved)
        e = C.ENDED[out["job"]]
        assert "errno 96" in e["reason"] and "127.0.1" in e["reason"]
        assert wait(lambda: (J.read_marker(moved, 0) or {}).get("phase")
                    == "ready", 20)
        doc = next(d for d in C.jobs_document() if d["job"] == moved)
        assert doc["cable"] == "127.0.1"
        assert "moved from 127.0.0" in doc["cable_note"]
        assert "127.0.0" in links.BAD_CABLES[frozenset(("aaaa", "bbbb"))]
    finally:
        for j in jobs:
            C.stop(j, grace=1)
        reap_registries(tmp_path / "A", tmp_path / "B")
        srv.shutdown()
        C.ENDED.clear()


def test_a_stopped_job_never_claims_the_port_the_next_job_serves_on(monkeypatch):
    # a new job reuses :8080: its card must not say "stopped" while loading
    monkeypatch.setattr(C, "jobs_document", lambda: [
        {"job": "new", "phase": "ready", "port": 8080, "split": "pipeline"},
        {"job": "old", "phase": "stopped", "port": 8080}])
    doc = page_loads.with_jobs({"resident": [
        {"runtime": "knurlogic", "where": "http://198.51.100.2:8080"}]})
    assert doc["resident"][0]["cluster"]["job"] == "new"


def test_an_unload_after_a_page_restart_still_tells_the_other_pages(
        two_pages):
    """SPECS is this page process's memory: after a restart the job's
    machines come from its rank records, so the stop still reaches B."""
    p = two_pages
    C.SPECS.clear()                             # the page restarted
    out = C.stop(p.job, grace=1)
    assert out["told"] == ["B"]
    assert wait(lambda: not alive(p.rank1["pid"]), 30)


def test_a_source_tree_reads_its_version_from_pyproject(tmp_path):
    import knurlogic
    pkg = tmp_path / "src" / "knurlogic"
    pkg.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["x"]\n\n[project]\nname = "knurlogic"\n'
        'version = "9.8.7"\n\n[tool.x]\nversion = "0"\n')
    assert knurlogic._source_version(pkg / "__init__.py") == "9.8.7"
    # an installed package (no src/ parent) defers to its metadata
    site = tmp_path / "site" / "knurlogic"
    site.mkdir(parents=True)
    assert knurlogic._source_version(site / "__init__.py") is None


def test_same_build_different_version_string_is_accepted_with_a_note(cache):
    other = dict(VERSIONS, knurlogic="0.1.0")
    code, doc = prep(spec(versions=other))
    assert code == 200 and doc["ok"], doc
    assert "0.1.0.dev0" in doc["note"]


# --- control-plane contract (docs/design/orchestration.md) --------------------

def test_a_retried_start_answers_started_again(cache, monkeypatch):
    monkeypatch.setattr(C, "_local_info",
                        lambda: info("Apple M3 Ultra", "192.0.2.2"))
    monkeypatch.setattr(C, "_ensure_watcher", lambda: None)
    job = "db0cb292aefeba55"
    code, got = prep(spec(job=job))
    assert got["ok"] and got["v"] == [1, 0], got
    procs = []

    def spawn(cmd, **kw):
        procs.append(subprocess.Popen(["sleep", "30"]))
        return procs[0]
    try:
        code, first = C.start(job, spawn=spawn, wait_s=0.5)
        assert code == 200 and first["started"] == job, first
        code, again = C.start(job, spawn=spawn, wait_s=0.5)
        assert code == 200 and len(procs) == 1
        assert (again["rank"], again["pid"]) == (first["rank"], first["pid"])
        assert again["v"] == [1, 0]
        # an unknown job is still the same 404
        assert C.start("ffffffffffffffff", spawn=spawn)[0] == 404
    finally:
        for p in procs:
            p.kill()
            p.wait()
        C._PROCS.clear()
        J.save_registry({})


def test_a_start_stays_prepared_until_its_rank_is_recorded(cache,
                                                          monkeypatch):
    """A peer page asking JobState while this page's Start is under way
    (waiting on an exiting rank, spawning) must see the job held: with no
    clock in peer_verdict, "not prepared, no rank" stops the job at once
    (docs/design/liveness.md). A Start retried meanwhile spawns nothing."""
    monkeypatch.setattr(C, "_local_info",
                        lambda: info("Apple M3 Ultra", "192.0.2.2"))
    monkeypatch.setattr(C, "_ensure_watcher", lambda: None)
    job = "db0cb292aefeba55"
    assert prep(spec(job=job))[1]["ok"]
    seen, procs, go = [], [], threading.Event()

    def spawn(cmd, **kw):
        seen.append(C.job_state(job))
        go.wait(10)
        procs.append(subprocess.Popen(["sleep", "30"]))
        return procs[-1]
    out: dict = {}
    first = threading.Thread(target=lambda: out.setdefault(
        "first", C.start(job, spawn=spawn, wait_s=0.5)))
    retry = threading.Thread(target=lambda: out.setdefault(
        "retry", C.start(job, spawn=spawn, wait_s=0.5)))
    try:
        first.start()
        assert wait(lambda: seen, 10)
        assert seen[0]["prepared"] and not seen[0]["ranks_here"]
        retry.start()
        time.sleep(0.2)
        go.set()
        first.join(10)
        retry.join(10)
        assert out["first"][0] == 200 and out["retry"][0] == 200
        assert len(procs) == 1
        assert out["retry"][1]["pid"] == out["first"][1]["pid"]
        doc = C.job_state(job)
        assert not doc["prepared"] and doc["ranks_here"] == [1]
        assert job not in C.PREPARED
    finally:
        go.set()
        for p in procs:
            p.kill()
            p.wait()
        C._PROCS.clear()
        J.save_registry({})


def test_a_start_that_fails_to_spawn_is_no_longer_prepared(cache,
                                                           monkeypatch):
    monkeypatch.setattr(C, "_local_info",
                        lambda: info("Apple M3 Ultra", "192.0.2.2"))
    job = "db0cb292aefeba55"
    assert prep(spec(job=job))[1]["ok"]

    def spawn(cmd, **kw):
        raise OSError("no such file")
    code, out = C.start(job, spawn=spawn, wait_s=0.5)
    assert code == 500 and "no such file" in out["error"]
    assert job not in C.PREPARED and not C.job_state(job)["prepared"]


def test_failure_kind_is_read_before_the_wording():
    from knurlogic.cluster import recovery
    assert recovery.kind("something odd", "machine") == "machine"
    assert recovery.kind("unloaded", "memory") == "memory"
    assert recovery.kind("unloaded") == "requested"      # the regex fallback
    assert recovery.kind("x", "nonsense") == "failure"


def test_stop_carries_its_kind_to_the_ended_record_and_the_peers(
        cache, monkeypatch):
    monkeypatch.setattr(C, "_peer_pages", lambda: {"b": "y:1"})
    job = "41648583878fcdfc"
    C.SPECS[job] = spec(job=job)
    told = []
    out = C.stop(job, reason="B went away", kind="machine",
                 post=lambda page, kind, doc, **k: told.append(doc) or {})
    assert out["v"] == [1, 0] and out["exiting"] == []
    assert told[0]["kind"] == "machine"
    assert C.ENDED[job]["kind"] == "machine"
    assert C.job_state(job)["ended_kind"] == "machine"
    C.ENDED.clear()


def test_a_rank_whose_poll_cannot_answer_is_judged_by_its_process():
    """Popen.poll() answers None while another thread waits on the process,
    dead or not; a killed rank read as alive that way and the watch saw a
    healthy job (now and then, in a Linux container)."""
    job = "91648583878fcdfc"
    dead = SimpleNamespace(pid=99999999, poll=lambda: None)
    C._PROCS[(job, 0)] = dead
    try:
        assert not C._alive(job, dead.pid)
    finally:
        C._PROCS.pop((job, 0))


def test_a_late_stop_keeps_the_failovers_relaunch(cache, monkeypatch):
    """The coordinator's failover can move a job to the next cable before
    the failing rank's stop reaches it; that stop must not erase where the
    job went (it did, now and then, in a Linux container)."""
    monkeypatch.setattr(C, "_peer_pages", dict)
    job = "81648583878fcdfc"
    C.SPECS[job] = spec(job=job)
    C.ENDED[job] = {"reason": "link init failed; relaunched on cable "
                    "127.0.1 as job 9", "relaunched": "9", "t": 1.0}
    C.stop(job, reason="rank 1 on B (pid 3) exited",
           post=lambda *a, **k: {})
    assert C.ENDED[job]["relaunched"] == "9"
    assert "relaunched on cable" in C.ENDED[job]["reason"]
    C.ENDED.clear()


def test_an_unload_is_recorded_as_requested(cache, monkeypatch):
    """The page drops a launch's card for a requested stop and shows a
    failure otherwise (picker.js followLaunch reads the ended kind): an
    unload, which carries no kind of its own, is recorded as requested."""
    monkeypatch.setattr(C, "_peer_pages", dict)
    job = "51648583878fcdfc"
    C.SPECS[job] = spec(job=job)
    C.stop(job, reason="unloaded", post=lambda *a, **k: {})
    assert C.ENDED[job]["kind"] == "requested"
    job2 = "61648583878fcdfc"
    C.SPECS[job2] = spec(job=job2)
    C.stop(job2, reason="rank 1 on B (pid 3) exited",
           post=lambda *a, **k: {})
    assert C.ENDED[job2]["kind"] is None          # a failure stays one
    C.ENDED.clear()


def test_a_job_being_unloaded_is_requested_while_its_ranks_exit(
        cache, monkeypatch):
    """Between the stop's start and the ended record (up to the 25 s
    terminate), the page lists the job as stopping: an unload must read
    as requested there too, or the card shows FAILED for good."""
    job = "71648583878fcdfc"
    J.save_registry({f"{job}/1": {"job": job, "rank": 1, "pid": 999999,
                                  "split": "tensor", "link": "ring",
                                  "stopping": 1.0, "stop_reason": "unloaded",
                                  "stop_kind": "requested"}})
    row = next(r for r in C.jobs_document() if r["job"] == job)
    assert row["phase"] == "stopping" and row["kind"] == "requested"
    assert C.job_state(job)["ended_kind"] == "requested"
    J.save_registry({})


def test_failover_beyond_two_machines_is_a_logged_no_op(monkeypatch, caplog):
    monkeypatch.setattr(links, "BAD_CABLES", {})
    order = [{"id": c, "name": c.upper()} for c in "abc"]
    ctx = {"order": order, "net": "127.0.0", "tried": ("127.0.0",),
           "req": {}, "args": {}}
    why = ("rank 1 on B (pid 9) exited: link init failed: ValueError: "
           "[jaccl] Changing queue pair to RTR failed with errno 96")
    assert C.failover("ab12cd34ef567890", ctx, why) is None
    assert links.BAD_CABLES == {}
    assert "no cable failover beyond two" in caplog.text


def test_no_link_refuses_naming_the_shared_cables(cache, monkeypatch):
    monkeypatch.setattr(links, "BAD_CABLES", {})
    out, got = jaccl_launch(monkeypatch, {"link": ""})
    assert out["refused"].startswith("link is tcp | rdma; the machines "
                                     "share 127.0.0, 127.0.1")
    assert not got


def test_no_link_is_tcp_over_the_one_shared_cable(cache, monkeypatch):
    monkeypatch.setattr(links, "BAD_CABLES", {})
    real = links.shared_subnets
    monkeypatch.setattr(links, "shared_subnets",
                        lambda a, b, rdma=False: real(a, b, rdma)[:1])
    out, got = jaccl_launch(monkeypatch, {"link": ""})
    assert "refused" not in out and "error" not in out


def test_ring_chunk_is_the_smallest_ranks_room_and_a_set_wins():
    got = [{"ok": True, "prefill_chunk": 2048},
           {"ok": True, "prefill_chunk": 1024}]
    assert C.ring_chunk({}, got) == (1024, C.RING_CHUNK_WHY)
    assert C.ring_chunk({"KNURLOGIC_PREFILL_CHUNK": "256"}, got) == (256, "set")
    assert C.ring_chunk({}, got + [{"ok": True}])[0] == C.PREFILL_CHUNK


def test_start_runs_the_rings_chunk_and_says_why(cache, monkeypatch):
    monkeypatch.setattr(C, "_local_info",
                        lambda: info("Apple M3 Ultra", "192.0.2.2"))
    code, got = prep(spec())
    assert got["ok"], got
    cmds = []

    def spawn(cmd, **k):
        cmds.append(cmd)
        raise OSError("not really")
    C.start("ab12cd34ef567890", spawn=spawn, wait_s=0.5,
            ring={"prefill_chunk": 1024, "prefill_why": C.RING_CHUNK_WHY})
    cmd = cmds[0]
    assert cmd[cmd.index("--prefill-chunk") + 1] == "1024"
    assert cmd[cmd.index("--prefill-why") + 1] == C.RING_CHUNK_WHY
    C._PROCS.clear()


def test_a_stopped_job_does_not_claim_the_next_server_on_its_port(monkeypatch):
    # a single-Mac load on 8080 after a stopped cluster job there showed as
    # that job ("pipeline over RDMA · stopped"), beside its own launch card
    monkeypatch.setattr(C, "jobs_document", lambda: [
        {"job": "77865d0000000000", "phase": "stopped", "port": 8080,
         "split": "pipeline", "link": "rdma", "machines": ["A", "B"]}])
    doc = page_loads.with_jobs({"resident": [
        {"runtime": "knurlogic", "where": "http://127.0.0.1:8080",
         "state": "loading"}]})
    assert not (doc["resident"][0].get("cluster") or {}).get("job")


def test_a_failed_beat_does_not_end_the_heartbeat(tmp_path, monkeypatch):
    """One beat raising (a full disk, a probe bug past its own guard) is
    logged and the marker thread keeps beating."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(J, "MARK_S", 0.01)
    m = J.Marker("0123abcd", 0)
    calls = []

    def beat():
        calls.append(1)
        if len(calls) == 2:
            raise TypeError("not JSON serializable")
    m.beat = beat
    m.start()
    end = time.time() + 5
    while len(calls) < 5 and time.time() < end:
        time.sleep(0.01)
    assert len(calls) >= 5


# --- a rank mid-step is never SIGKILLed ---------------------------------------

def test_mid_step_reads_the_marker_of_that_pid():
    doc = {"pid": 7, "busy": True, "phase": "ready"}
    assert J.mid_step("j", 1, 7, read=lambda j, r: doc)
    assert not J.mid_step("j", 1, 8, read=lambda j, r: doc)   # pid reused
    assert not J.mid_step("j", 1, 7, read=lambda j, r: {**doc, "busy": False})
    assert J.mid_step("j", 1, 7, read=lambda j, r: {"pid": 7,
                                                    "phase": "stuck"})
    assert not J.mid_step("j", 1, 7, read=lambda j, r: None)


def test_terminate_never_sigkills_a_busy_rank(monkeypatch):
    import signal
    sent = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: sent.append((pid, sig)))
    killed = J.terminate([11, 12], grace=0, alive=lambda p: True,
                         spare=lambda p: p == 11)
    assert killed == [12]
    assert (11, signal.SIGKILL) not in sent and (11, signal.SIGTERM) in sent
    assert (12, signal.SIGKILL) in sent


def test_terminate_spares_a_stuck_pid_once(monkeypatch, caplog):
    import signal
    sent = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: sent.append((pid, sig)))
    monkeypatch.setattr(J, "_STUCK", set())
    for _ in range(3):
        assert J.terminate([11], grace=0, alive=lambda p: True,
                           spare=lambda p: True) == []
    assert sent == [(11, signal.SIGTERM)]
    assert sum("did not stop" in r.message for r in caplog.records) == 1


def test_the_watcher_leaves_a_job_whose_live_ranks_are_mid_step(
        cache, monkeypatch):
    job = "41648583878fcdfc"
    J.save_registry({f"{job}/1": {"job": job, "rank": 1, "pid": 4321,
                                  "stopping": 1.0}})
    monkeypatch.setattr(C, "_alive", lambda j, p: True)
    monkeypatch.setattr(J, "mid_step", lambda j, r, p: True)
    stops = []
    monkeypatch.setattr(C, "stop", lambda *a, **k: stops.append(a))
    assert C.watch_once() == [] and stops == []
    monkeypatch.setattr(J, "mid_step", lambda j, r, p: False)
    C.watch_once()
    assert stops == [(job,)]
    J.save_registry({})


def test_a_recorded_rank_whose_command_line_is_unreadable_is_alive(
        monkeypatch):
    from knurlogic.machine import processes
    monkeypatch.setattr(processes, "command", lambda pid: "")
    monkeypatch.setattr(processes, "running", lambda pid: True)
    assert J.is_rank(4321, "ab12cd34ef567890")
    monkeypatch.setattr(processes, "running", lambda pid: False)
    assert not J.is_rank(4321, "ab12cd34ef567890")
    monkeypatch.setattr(processes, "command",
                        lambda pid: "python other --job 0000000000000000")
    monkeypatch.setattr(processes, "running", lambda pid: True)
    assert not J.is_rank(4321, "ab12cd34ef567890")     # the pid was reused
