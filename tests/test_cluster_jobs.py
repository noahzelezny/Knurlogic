"""Cluster jobs: the page-to-page two-phase launch and the failure path.

No model loads anywhere. The pure parts (placement, the RDMA probe, the
prepare checks, the stall verdict) are tested directly. The failure path
runs for real across processes: this test process is page A (the
coordinator, with its page served on a thread), tests/cluster_fake_page.py
is page B in its own process with its own cache dir, and each page spawns
tests/cluster_fake_rank.py as its rank -- which writes real markers and,
as rank 0, answers through the real scheduler abort and SIGTERM path."""

import json
import os
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

from knurlogic.cluster import jobs as J
from knurlogic.cluster import links
from knurlogic.interfaces import cluster_jobs as C
from knurlogic.interfaces import ui
from knurlogic.machine import identity

GIB = 1 << 30
HERE = Path(__file__).resolve().parent
SRC = str(HERE.parent / "src")
VERSIONS = {"knurlogic": "0.1.0.dev0", "mlx": "0.31.2",
            "build": "0123456789ab+mlx0.31.2"}


def info(chip, tb, ws=64 * GIB, rdma=None):
    return {"chip": chip, "p_core_ghz": None, "bandwidth_gbs": None,
            "thunderbolt": [{"iface": "en4", "ip": tb}],
            "rdma": rdma or {"available": False, "reason": "off",
                             "devices": [], "active": []},
            "versions": dict(VERSIONS), "jaccl_selfheal": False,
            "working_set_bytes": ws}


SHAPE = {"layer_bytes": [GIB] * 8, "other_bytes": GIB,
         "tensor_per_rank_bytes": 4 * GIB, "refusals": []}


# --- placement ---------------------------------------------------------------

def test_placement_leader_is_the_newest_chip_and_it_is_deterministic():
    ms = [{"name": "M3", **info("Apple M3 Ultra", "10.0.0.1")},
          {"name": "M4", **info("Apple M4 Max", "10.0.0.2")}]
    a = C.placement(ms, SHAPE, "tensor")
    assert a["order"] == ["M4", "M3"] and a["leader"] == "M4"
    assert [s["bytes"] for s in a["shares"]] == [4 * GIB] * 2
    assert C.placement(list(reversed(ms)), SHAPE, "tensor") == a


def test_pipeline_placement_gives_rank_0_the_last_layers():
    ms = [{"name": "A", **info("Apple M4 Max", "10.0.0.1", ws=20 * GIB)},
          {"name": "B", **info("Apple M3 Ultra", "10.0.0.2", ws=9 * GIB)}]
    p = C.placement(ms, SHAPE, "pipeline")
    assert p["order"] == ["A", "B"] and sum(p["layers"]) == 8
    assert p["shares"][0]["bounds"][1] == 8          # rank 0 ends the model
    assert p["layers"][0] > p["layers"][1]           # more room, more layers
    assert "rank 0 A leads" in p["reason"]


def test_placement_refuses_a_share_that_does_not_fit():
    ms = [{"name": "A", **info("Apple M4 Max", "10.0.0.1", ws=2 * GIB)},
          {"name": "B", **info("Apple M3 Ultra", "10.0.0.2")}]
    with pytest.raises(ValueError, match="A: its tensor share"):
        C.placement(ms, SHAPE, "tensor")


def test_placement_leaves_the_step_margin():
    """A share that fits the working set but not its step margin (5%, at
    least 4 GiB) is refused; the reason says what each rank leaves."""
    ms = [{"name": "A", **info("Apple M4 Max", "10.0.0.1", ws=7 * GIB)},
          {"name": "B", **info("Apple M3 Ultra", "10.0.0.2")}]
    with pytest.raises(ValueError, match="4.0 GiB step margin"):
        C.placement(ms, SHAPE, "tensor")              # 4 of 7 GiB: 3 left
    ms[0]["working_set_bytes"] = 8 * GIB
    t = C.placement(ms, SHAPE, "tensor")
    assert "A leaves 4.0 GiB" in t["reason"]
    # pipeline: 8 x 1 GiB layers + 1 GiB each; after the 4 GiB margins
    # 12 and 6 GiB hold 7 and 1 layers (the bare working sets would 11 + 5)
    ms = [{"name": "A", **info("Apple M4 Max", "10.0.0.1", ws=12 * GIB)},
          {"name": "B", **info("Apple M3 Ultra", "10.0.0.2", ws=6 * GIB)}]
    p = C.placement(ms, SHAPE, "pipeline")
    assert p["layers"] == [7, 1]
    assert all(s["bytes"] <= m - 4 * GIB for s, m in
               zip(p["shares"], [12 * GIB, 6 * GIB]))
    assert "leaves 4.0 GiB" in p["reason"]


def test_ring_ips_pick_the_shared_thunderbolt_subnet():
    a = {"name": "A", "thunderbolt": [{"ip": "10.0.1.1"}, {"ip": "10.0.0.1"}]}
    b = {"name": "B", "thunderbolt": [{"ip": "10.0.0.2"}]}
    assert C._ring_ips([a, b]) == ["10.0.0.1", "10.0.0.2"]
    with pytest.raises(ValueError, match="no Thunderbolt address"):
        C._ring_ips([a, {"name": "C", "thunderbolt": []}])


def test_rdma_device_is_the_one_on_the_peers_subnet():
    a = {"thunderbolt": [{"iface": "en7", "ip": "10.0.1.1"},
                         {"iface": "en4", "ip": "10.0.0.1"}],
         "rdma": {"active": ["rdma_en4", "rdma_en7"]}}
    b = {"thunderbolt": [{"iface": "en2", "ip": "10.0.0.2"}],
         "rdma": {"active": ["rdma_en2"]}}
    assert C._rdma_device(a, b) == "rdma_en4"
    assert C._rdma_device(b, a) == "rdma_en2"


def test_two_cables_both_ends_on_one_subnet():
    # the M3/M4 rig: two cables, 10.0.0.x (M3 en4 - M4 en3) and 10.0.1.x
    # (M3 en7 - M4 en2). Each side picking "a device on the peer's subnet"
    # took M4 en2 and M3 en4 -- two different cables; jaccl failed RTR.
    m3 = {"name": "M3", "thunderbolt": [{"iface": "en4", "ip": "10.0.0.1"},
                                        {"iface": "en7", "ip": "10.0.1.1"}],
          "rdma": {"active": ["rdma_en4", "rdma_en7"]}}
    m4 = {"name": "M4", "thunderbolt": [{"iface": "en2", "ip": "10.0.1.2"},
                                        {"iface": "en3", "ip": "10.0.0.2"}],
          "rdma": {"active": ["rdma_en2", "rdma_en3"]}}
    assert C._rdma_device(m4, m3) == "rdma_en3"
    assert C._rdma_device(m3, m4) == "rdma_en4"
    assert C._ring_ips([m4, m3]) == ["10.0.0.2", "10.0.0.1"]
    assert C._ring_ips([m4, m3], rdma=True) == ["10.0.0.2", "10.0.0.1"]
    # RDMA down on 10.0.0.x at one end: both move to 10.0.1.x
    m4d = {**m4, "rdma": {"active": ["rdma_en2"]}}
    assert C._rdma_device(m4d, m3) == "rdma_en2"
    assert C._rdma_device(m3, m4d) == "rdma_en7"
    assert C._ring_ips([m4d, m3], rdma=True) == ["10.0.1.2", "10.0.1.1"]


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
         "hosts": ["10.0.0.1:47200", "10.0.0.2:47201"], "ibv_devices": None,
         "coordinator": "", "layers": [], "prefill_chunk": 512,
         "tune": "balanced", "port": 0, "working_set_gib": 60.0,
         "bandwidth_gbs": 0, "sets": {}, "versions": dict(VERSIONS),
         "jaccl_timeout_ms": 0,
         "nodes": [{"rank": 0, "id": "a", "name": "A", "page": "x:1"},
                   {"rank": 1, "id": "b", "name": "B", "page": "y:1"}]}
    s.update(kw)
    return s


def prep(s, **kw):
    return C.prepare(s, resolve=kw.get("resolve", lambda i: "/m/x"),
                     info=kw.get("info", info("Apple M3 Ultra", "10.0.0.2")),
                     shape=lambda p, w, sp: kw.get("shape", SHAPE),
                     registry=lambda: {})


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "A"))
    return tmp_path


def test_prepare_accepts_and_remembers(cache):
    code, doc = prep(spec())
    assert code == 200 and doc["ok"]
    assert "ab12cd34ef567890" in C.PREPARED
    C.PREPARED.clear()


@pytest.mark.parametrize("change,why", [
    ({"versions": {"knurlogic": "0.0.9", "mlx": "0.31.2"}}, "knurlogic 0.1.0"),
    ({"hosts": ["10.0.0.1:1", "10.9.9.9:2"]}, "not one of"),
    ({"link": "jaccl", "ibv_devices": [[None, "rdma_en4"], ["rdma_en2", None]],
      "coordinator": "10.0.0.1:1"}, "RDMA on"),
])
def test_prepare_refuses_with_its_reason(cache, change, why):
    code, doc = prep(spec(**change))
    assert code == 200 and not doc["ok"] and why in doc["refused"]


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
    monkeypatch.setattr(C, "_chip", lambda: "Apple M4 Max")
    monkeypatch.setattr(C, "_selfheal", lambda: False)
    monkeypatch.setattr(links, "thunderbolt", lambda: [])
    monkeypatch.setattr(links, "rdma", lambda: {"available": False})
    v = C.node_info(0)["versions"]
    assert v["build"] == C.build_fingerprint()
    assert v["build"].endswith("+mlx" + (v["mlx"] or "none"))


def test_prepare_refuses_a_share_that_does_not_fit_here(cache):
    code, doc = prep(spec(), info=info("Apple M3 Ultra", "10.0.0.2",
                                       ws=2 * GIB))
    assert not doc["ok"] and "working set" in doc["refused"]
    s = spec(split="pipeline", layers=[2, 6])        # rank 1: layers 0..5
    code, doc = prep(s, info=info("Apple M3 Ultra", "10.0.0.2", ws=6 * GIB))
    assert not doc["ok"] and "7.0 GiB" in doc["refused"]
    # fits the working set, not the step margin: 4 GiB in 7 leaves 3
    code, doc = prep(spec(), info=info("Apple M3 Ultra", "10.0.0.2",
                                       ws=7 * GIB))
    assert not doc["ok"] and "4.0 GiB step margin" in doc["refused"]
    code, doc = prep(spec(), info=info("Apple M3 Ultra", "10.0.0.2",
                                       ws=8 * GIB))
    assert doc["ok"], doc
    C.PREPARED.clear()


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
    monkeypatch.setattr(ui, "launch_knobs",
                        lambda: frozenset({"kv_bits"}))
    code, doc = prep(spec(sets={"kv_bits": "8", "evil": "x"}))
    assert not doc["ok"] and "evil" in doc["refused"]
    code, doc = prep(spec(sets={"kv_bits": "8; rm"}))
    assert not doc["ok"] and "kv_bits" in doc["refused"]
    code, doc = prep(spec(sets={"kv_bits": "8"}))
    assert doc["ok"]
    assert C.PREPARED["ab12cd34ef567890"]["spec"]["sets"] == {"kv_bits": "8"}
    C.PREPARED.clear()


def test_stop_tells_only_pages_this_page_knows(cache, monkeypatch):
    """The spec's page addresses are never posted to: a node is told at
    the address this page's PEERS store has for its id, or not at all."""
    from knurlogic.cluster.peers import Peer
    s = spec(nodes=[{"rank": 0, "id": "me", "name": "A", "page": "x:1"},
                    {"rank": 1, "id": "b", "name": "B",
                     "page": "evil.example:80"},
                    {"rank": 2, "id": "c", "name": "C", "page": "z:1"}])
    monkeypatch.setattr(identity, "identity", lambda: {"id": "me"})
    peers = [Peer(host="10.0.0.2", port=8765, id="b", name="B",
                  state="answering")]
    monkeypatch.setattr(ui, "PEERS", SimpleNamespace(all=lambda: peers))
    C.SPECS["ab12cd34ef567890"] = s
    posted = []
    out = C.stop("ab12cd34ef567890", post=lambda u, d: posted.append(u),
                 grace=0)
    assert posted == ["http://10.0.0.2:8765" + C.STOP_PATH]
    assert out["told"] == ["B"]
    C.ENDED.pop("ab12cd34ef567890", None)


# --- the stall verdict -------------------------------------------------------

def test_verdict_idle_is_not_stalled_but_busy_and_still_is(monkeypatch):
    w = J.Watch()
    ranks = [{"rank": 0, "pid": 1, "machine": "A"}]
    idle = {"phase": "ready", "step": 5, "busy": False}
    busy = {"phase": "ready", "step": 5, "busy": True}
    alive = lambda p: True
    assert w.verdict("j", ranks, alive, now=0, read=lambda j, r: idle) == ""
    assert w.verdict("j", ranks, alive, now=1e6, read=lambda j, r: idle) == ""
    assert w.verdict("j", ranks, alive, now=1e6 + J.STALL_S - 1,
                     read=lambda j, r: busy) == ""
    why = w.verdict("j", ranks, alive, now=1e6 + J.STALL_S + 1,
                    read=lambda j, r: busy)
    assert "stalled" in why
    moved = {**busy, "step": 6}
    assert w.verdict("j", ranks, alive, now=2e6, read=lambda j, r: moved) == ""


def test_verdict_a_long_prefill_is_one_step_of_many_chunks_not_a_stall():
    """397B over the M4+M3: a 30k-token prompt missing the prompt cache is
    one step of ~2 minutes; the page stopped the job as stalled at 122 s
    while both ranks were computing (2026-09-27). Chunks are progress."""
    w = J.Watch()
    ranks = [{"rank": 0, "pid": 1, "machine": "A"}]
    alive = lambda p: True
    doc = lambda c: (lambda j, r: {"phase": "ready", "step": 9, "busy": True,
                                   "chunk": c})
    assert w.verdict("j", ranks, alive, now=0, read=doc(0)) == ""
    for k in range(1, 6):          # a chunk every 60 s, no step for 300 s
        assert w.verdict("j", ranks, alive, now=60 * k, read=doc(k)) == ""
    assert "stalled" in w.verdict("j", ranks, alive,
                                  now=300 + J.STALL_S + 1, read=doc(5))


def test_verdict_a_request_waiting_on_a_slow_load_is_not_a_stall():
    """Qwen3.8-Flash-Next-6bit read over SMB on the M4: a request arrived
    while the ranks were still loading (busy, step 0) and the page stopped
    the job as stalled at 122 s, mid-load (2026-09-27). Only a loaded ring
    can stall; loading has its own clock (the pid, the join deadline)."""
    w = J.Watch()
    ranks = [{"rank": 0, "pid": 1, "machine": "A"}]
    alive = lambda p: True
    loading = lambda j, r: {"phase": "loading", "step": 0, "busy": True}
    assert w.verdict("j", ranks, alive, now=0, read=loading) == ""
    assert w.verdict("j", ranks, alive, now=10 * J.STALL_S,
                     read=loading) == ""
    # once ready, the clock starts from there
    ready = lambda j, r: {"phase": "ready", "step": 0, "busy": True}
    t = 10 * J.STALL_S + 1
    assert w.verdict("j", ranks, alive, now=t, read=ready) == ""
    assert "stalled" in w.verdict("j", ranks, alive, now=t + J.STALL_S + 1,
                                  read=ready)


def test_a_prefill_chunk_bumps_the_marker(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    m = J.Marker("ab12cd34ef567890", 0)
    monkeypatch.setitem(J.CURRENT, "marker", m)
    J.chunk_done()
    J.chunk_done()
    m.beat()
    assert J.read_marker("ab12cd34ef567890", 0)["chunk"] == 2


def test_verdict_dead_pid_and_never_joined():
    w = J.Watch()
    ranks = [{"rank": 1, "pid": 7, "machine": "B", "t": 0}]
    assert "exited" in w.verdict("j", ranks, lambda p: False, now=1)
    why = w.verdict("j", ranks, lambda p: True, now=J.JOIN_S + 5,
                    read=lambda j, r: {"phase": "joining", "since": 0})
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
    J.after_load()
    assert os.environ["JACCL_COLLECTIVE_TIMEOUT_MS"] == "60000"


def test_rank_env_sets_the_timeout_to_zero_for_the_load_only_with_the_fork():
    s = spec(link="jaccl", jaccl_timeout_ms=60000, coordinator="10.0.0.1:9")
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
    from knurlogic.interfaces.http.openai import _status_of
    e = _status_of(RingFailed("a rank exited"))
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
def two_pages(tmp_path, monkeypatch):
    """Page A (here) and page B (a process), each with its own cache."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "A"))
    monkeypatch.setenv("PYTHONPATH", SRC + os.pathsep
                       + os.environ.get("PYTHONPATH", ""))
    monkeypatch.setitem(identity._ID, "id", "aaaa")
    monkeypatch.setitem(identity._ID, "name", "A")
    monkeypatch.setattr(C, "_resolve",
                        lambda i: "/fake/artifact" if i == "abc" else None)
    monkeypatch.setattr(C, "shape_of", lambda p, w, s: SHAPE)
    info_a = info("Apple M4 Max", "127.0.0.1")
    info_b = info("Apple M3 Ultra", "127.0.0.1")
    monkeypatch.setattr(C, "_local_info", lambda: info_a)
    sys.path.insert(0, str(HERE))
    import cluster_fake_page
    monkeypatch.setattr(C, "RANK_ARGV", [cluster_fake_page.fake_argv])
    monkeypatch.setattr(C, "_WATCHER", [1])        # A is watched by hand
    ui_a, ui_b = free_port(), free_port()
    srv = ThreadingHTTPServer(("127.0.0.1", ui_a), ui.make_handler({}))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    env = {**os.environ, "XDG_CACHE_HOME": str(tmp_path / "B")}
    page_b = subprocess.Popen(
        [sys.executable, str(HERE / "cluster_fake_page.py"), str(ui_b),
         "bbbb", "B", json.dumps(info_b), "aaaa", f"127.0.0.1:{ui_a}"],
        env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert "up" in page_b.stdout.readline()
    peer = SimpleNamespace(id="bbbb", name="B", host="127.0.0.1",
                           key=f"127.0.0.1:{ui_b}", state="answering",
                           link="thunderbolt", node={"cluster": info_b})
    from knurlogic.cluster.peers import Peer
    known = Peer(host="127.0.0.1", port=ui_b, id="bbbb", name="B",
                 state="answering")
    monkeypatch.setattr(ui, "PEERS", SimpleNamespace(
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
    C.stop(job, propagate=False, grace=1)
    for pid in (rank0["pid"], rank1["pid"]):
        try:
            os.kill(pid, 9)
        except OSError:
            pass
    page_b.kill()
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
    doc = ui.with_jobs({"resident": [
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


def test_the_verdict_names_a_ranks_machine_from_the_job_when_unrecorded():
    w = J.Watch()
    why = w.verdict("j", [{"rank": 1, "pid": 5, "machines": ["A", "B"]}],
                    lambda p: False, now=1)
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
    out = ui._stop(p.port)              # the Unload button on rank 0's row
    assert out["stopped"] and out["told"] == ["B"]
    assert not alive(p.rank0["pid"])
    assert wait(lambda: not alive(p.rank1["pid"]), 20)
    assert p.port not in servers.registry()


def test_peer_cluster_routes_are_gated_like_peer_loaded(two_pages):
    p = two_pages
    req = urllib.request.Request(
        f"http://127.0.0.1:{p.ui_b}{C.STOP_PATH}",
        data=json.dumps({"job": p.job}).encode(), method="POST",
        headers={"Content-Type": "application/json",
                 "Origin": f"http://127.0.0.1:{p.ui_b}"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=10)
    assert e.value.code == 403
    assert alive(p.rank1["pid"])
    code, doc = C.peer_route(C.PREPARE_PATH, json.dumps(
        {**spec(), "path": "/etc"}).encode())
    assert code == 400 and "identity" in doc["error"]


def test_a_ranks_working_set_is_the_wired_limit_not_the_ram():
    gib = 1 << 30
    assert C.gpu_working_set(96 * gib, 84 * gib) == 84 * gib
    assert C.gpu_working_set(96 * gib, 0) == 96 * gib
    assert C.gpu_working_set(0, 84 * gib) == 84 * gib
    assert C.gpu_working_set(0, 0) == 0


def test_every_rank_syncs_the_gpu_fast():
    for link, files in (("ring", {"hostfile": "/j/h.json"}),
                        ("jaccl", {"ibv": "/j/ibv.json", "hostfile": ""})):
        s = {**spec(), "link": link}
        assert C.rank_env(s, files, selfheal=False)[
            "MLX_METAL_FAST_SYNCH"] == "1"


class _Capture:
    """Just enough of a request handler for ui._send_json."""

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
    ui._stream(h, base + "/v1/chat/completions", b"{}", base=base)
    assert h.code == 503, h.doc()
    err = h.doc()["error"]
    assert err["code"] == "cluster_failed" and err["type"] == "server_error"
    assert "rank 0 on A " in err["message"] and "exited" in err["message"]
    # and once the job is stopped, still that reason (not a 502)
    h = _Capture()
    ui._stream(h, base + "/v1/chat/completions", b"{}", base=base)
    assert h.code == 503 and "rank 0 on A " in h.doc()["error"]["message"]


def test_an_unreachable_server_that_is_no_job_is_still_a_502(cache):
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    h = _Capture()
    ui._stream(h, base + "/v1/chat/completions", b"{}", base=base)
    assert h.code == 502


def test_a_peers_dead_rank_0_is_a_503_with_the_peers_stop_reason(
        monkeypatch):
    base = "http://10.0.0.2:8080"
    dead = f"http://127.0.0.1:{free_port()}"
    monkeypatch.setitem(ui._PEER_TARGETS, base, {
        "machine": "M4", "relay": dead, "job": "ab12cd34ef567890"})

    def survey():
        ui._PEER_JOBS["ab12cd34ef567890"] = {
            "job": "ab12cd34ef567890", "phase": "stopped",
            "reason": "rank 0 on M4 (pid 7) exited"}
    monkeypatch.setattr(ui, "refresh_targets", survey)
    h = _Capture()
    ui._stream(h, ui.upstream(base, "/v1/chat/completions"), b"{}",
               base=base)
    assert h.code == 503
    assert h.doc()["error"]["code"] == "cluster_failed"
    assert "rank 0 on M4 (pid 7) exited" in h.doc()["error"]["message"]
    ui._PEER_JOBS.clear()


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
    monkeypatch.setattr(ui, "cluster_failure",
                        lambda b: asked.append(b) or "rank 1 on B exited")
    base = _sse_upstream([b'{"x": 1}'], done)
    h = _Capture()
    ui._stream(h, base + "/v1/chat/completions", b"{}", base=base)
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
    monkeypatch.setattr(ui, "cluster_failure",
                        lambda b: "rank 1 on B exited")
    relayed = json.dumps(ui.cluster_failed("rank 0 on A exited")).encode()
    base = _sse_upstream([b'{"x": 1}', relayed], False)
    h = _Capture()
    ui._stream(h, base + "/v1/chat/completions", b"{}", base=base)
    out = h.wfile.getvalue().decode()
    assert out.count("cluster_failed") == 1
    assert "rank 0 on A exited" in out and "rank 1 on B" not in out


def test_a_cut_stream_that_is_no_cluster_job_just_ends(monkeypatch):
    monkeypatch.setattr(ui, "cluster_failure", lambda b: "")
    base = _sse_upstream([b'{"x": 1}'], False)
    h = _Capture()
    ui._stream(h, base + "/v1/chat/completions", b"{}", base=base)
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
    monkeypatch.setattr(C, "_resolve", lambda i: "/m/x")
    monkeypatch.setattr(C, "shape_of", lambda p, w, s: SHAPE)
    rd = lambda dev: {"available": True, "reason": "", "devices": [dev],
                      "active": [dev]}
    ia = {**info("Apple M4 Max", "10.0.0.1", rdma=rd("rdma_en4")),
          "thunderbolt": [{"iface": "en4", "ip": "10.0.0.1"}]}
    ib = {**info("Apple M3 Ultra", "10.0.1.2", rdma=rd("rdma_en7")),
          "thunderbolt": [{"iface": "en7", "ip": "10.0.1.2"},
                          {"iface": "en2", "ip": "10.0.0.2"}]}
    peer = SimpleNamespace(id="bbbb", name="B", host="10.0.0.2",
                           key="10.0.0.2:8765", state="answering",
                           link="thunderbolt", node={"cluster": ib})
    out = C.launch({"action": "load", "identity": "abc",
                    "nodes": ["aaaa", "bbbb"], "split": "tensor",
                    "link": "jaccl"},
                   me={"id": "aaaa", "name": "A"}, peers=[peer],
                   local_info=ia, ui_port=1, serve_port=2,
                   post=lambda *a, **k: pytest.fail("posted"))
    assert "refused" in out and "no Thunderbolt subnet" in out["refused"]
    assert "rdma_en4" in out["refused"] and "rdma_en7" in out["refused"]
    assert C._rdma_device(ia, ib) is None


# --- one machine never holds two jobs' shares (the M3 OOM, 2026-09-27) ------
# A 397B pipeline job's rank 1 held 49.5 GiB on the M3 (84 GiB working
# set) when a second job's rank 1 started loading the same share beside it:
# Metal wedged and the Mac rebooted, and the M4's rank 0 sat idle in a
# collective for three hours with nobody to tear it down.

def test_prepare_refuses_a_share_beside_what_another_job_holds(cache):
    big = dict(SHAPE, tensor_per_rank_bytes=50 * GIB)
    ws = info("Apple M3 Ultra", "10.0.0.2", ws=84 * GIB)
    held = [("job 41648583878fcdfc rank 1", 54699, int(49.5 * GIB))]
    code, doc = C.prepare(spec(), resolve=lambda i: "/m/x", info=ws,
                          shape=lambda p, w, s: big, registry=lambda: {},
                          held=lambda job: held)
    assert not doc["ok"], doc
    r = doc["refused"]
    assert "50.0 GiB" in r and "84.0 GiB" in r and "49.5 GiB of it held now" in r
    assert "pid 54699" in r and "41648583878fcdfc" in r
    code, doc = C.prepare(spec(), resolve=lambda i: "/m/x", info=ws,
                          shape=lambda p, w, s: big, registry=lambda: {},
                          held=lambda job: [])
    assert doc["ok"], doc
    C.PREPARED.clear()


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
    J._write(J.marker_path(other, 1), {"phase": "loading"})
    code, doc = C.prepare(spec(), resolve=lambda i: "/m/x",
                          info=info("Apple M3 Ultra", "10.0.0.2"),
                          shape=lambda p, w, s: SHAPE, registry=lambda: reg,
                          held=lambda job: [])
    assert not doc["ok"] and "one load at a time" in doc["refused"]
    J._write(J.marker_path(other, 1), {"phase": "ready"})
    code, doc = C.prepare(spec(), resolve=lambda i: "/m/x",
                          info=info("Apple M3 Ultra", "10.0.0.2"),
                          shape=lambda p, w, s: SHAPE, registry=lambda: reg,
                          held=lambda job: [])
    assert doc["ok"], doc
    C.PREPARED.clear()


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
                        lambda: info("Apple M3 Ultra", "10.0.0.2"))
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


def test_start_refuses_when_a_stopped_rank_will_not_go(cache, monkeypatch):
    monkeypatch.setattr(C, "_local_info",
                        lambda: info("Apple M3 Ultra", "10.0.0.2"))
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
    monkeypatch.setattr(C, "_peer_pages", lambda: {})
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


# --- a peer that vanishes takes the job with it ------------------------------

def test_peer_verdict_counts_an_unreachable_page_for_peer_gone_s(
        monkeypatch):
    monkeypatch.setattr(C, "_PEER_OK", {})
    job = "ab12cd34ef567890"
    recs = [{"job": job, "rank": 0, "pid": 1, "t": 0.0,
             "nodes": [{"rank": 0, "id": "a", "name": "M4"},
                       {"rank": 1, "id": "b", "name": "M3"}]}]

    def down(page, j):
        raise OSError("no route to host")
    kw = dict(pages={"b": "10.0.0.1:8765"}, me="a")
    assert C.peer_verdict(job, recs, now=100, ask=down, **kw) == ""
    assert C.peer_verdict(job, recs, now=110, ask=down, **kw) == ""
    why = C.peer_verdict(job, recs, now=100 + J.PEER_GONE_S, ask=down, **kw)
    assert "M3's page has not answered" in why and "20 s" in why
    # an answer with the rank running resets the clock
    up = (lambda page, j: {"ranks_here": [1]})
    assert C.peer_verdict(job, recs, now=130, ask=up, **kw) == ""
    assert C.peer_verdict(job, recs, now=145, ask=down, **kw) == ""
    # answering without the rank counts the same as not answering
    gone = (lambda page, j: {"ranks_here": [], "prepared": False})
    assert "no longer runs its rank" in C.peer_verdict(
        job, recs, now=151, ask=gone, **kw)
    # a page that says the job ended there: at once, with its reason
    ended = (lambda page, j: {"ranks_here": [], "ended": "rank 1 exited"})
    assert C.peer_verdict(job, recs, now=131, ask=ended, **kw) \
        == "M3 stopped the job: rank 1 exited"
    # a peer this page no longer knows is unreachable too
    monkeypatch.setattr(C, "_PEER_OK", {})
    assert C.peer_verdict(job, recs, now=0, ask=up, pages={}, me="a") == ""
    assert "M3" in C.peer_verdict(job, recs, now=J.PEER_GONE_S, ask=up,
                                  pages={}, me="a")


def test_the_job_route_says_what_this_page_runs(two_pages):
    p = two_pages
    code, doc = C.peer_route(C.JOB_PATH, json.dumps({"job": p.job}).encode())
    assert code == 200 and doc["ranks_here"] == [0] and not doc["ended"]
    code, doc = C.peer_route(C.JOB_PATH, b'{"job": "../x"}')
    assert code == 400


def test_a_vanished_peer_page_has_the_idle_rank_0_torn_down(
        two_pages, monkeypatch):
    """The M3 rebooted; the M4's rank 0 was idle in a collective, so no
    stall verdict, and its page watched only its own ranks. Now it watches
    the job's other pages too."""
    p = two_pages
    monkeypatch.setattr(J, "PEER_GONE_S", 1.0)
    assert C.watch_once() == []                 # B answers: healthy
    p.page_b.kill()                             # B's page is gone
    p.page_b.wait()
    assert C.watch_once() == []                 # the clock starts
    time.sleep(1.2)
    stopped = C.watch_once()
    assert stopped and stopped[0][0] == p.job
    assert "B's page has not answered" in stopped[0][1]
    assert not alive(p.rank0["pid"])
    ended = [d for d in C.jobs_document() if d["job"] == p.job]
    assert ended and ended[0]["phase"] == "stopped"


def test_a_peer_page_that_stopped_the_job_has_this_one_stop_too(
        two_pages, monkeypatch):
    """B's page (in its own process) watches A's: A forgetting the job
    without telling B is caught from B's side."""
    p = two_pages
    monkeypatch.setattr(J, "PEER_GONE_S", 0.5)
    # A drops its ranks without propagating: B must notice by itself
    C.stop(p.job, reason="rank 0 on A exited", propagate=False, grace=1)
    assert wait(lambda: not alive(p.rank1["pid"]), 30)
