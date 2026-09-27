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
VERSIONS = {"knurlogic": "0.1.0.dev0", "mlx": "0.31.2"}


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
    ms = [{"name": "M3", **info("Apple M3 Ultra", "192.0.2.1")},
          {"name": "M4", **info("Apple M4 Max", "192.0.2.2")}]
    a = C.placement(ms, SHAPE, "tensor")
    assert a["order"] == ["M4", "M3"] and a["leader"] == "M4"
    assert [s["bytes"] for s in a["shares"]] == [4 * GIB] * 2
    assert C.placement(list(reversed(ms)), SHAPE, "tensor") == a


def test_pipeline_placement_gives_rank_0_the_last_layers():
    ms = [{"name": "A", **info("Apple M4 Max", "192.0.2.1", ws=20 * GIB)},
          {"name": "B", **info("Apple M3 Ultra", "192.0.2.2", ws=5 * GIB)}]
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


def test_ring_ips_pick_the_shared_thunderbolt_subnet():
    a = {"name": "A", "thunderbolt": [{"ip": "198.51.100.1"}, {"ip": "192.0.2.1"}]}
    b = {"name": "B", "thunderbolt": [{"ip": "192.0.2.2"}]}
    assert C._ring_ips([a, b]) == ["192.0.2.1", "192.0.2.2"]
    with pytest.raises(ValueError, match="no Thunderbolt address"):
        C._ring_ips([a, {"name": "C", "thunderbolt": []}])


def test_rdma_device_is_the_one_on_the_peers_subnet():
    a = {"thunderbolt": [{"iface": "en7", "ip": "198.51.100.1"},
                         {"iface": "en4", "ip": "192.0.2.1"}],
         "rdma": {"active": ["rdma_en4", "rdma_en7"]}}
    b = {"thunderbolt": [{"iface": "en2", "ip": "192.0.2.2"}],
         "rdma": {"active": ["rdma_en2"]}}
    assert C._rdma_device(a, b) == "rdma_en4"
    assert C._rdma_device(b, a) == "rdma_en2"


def test_two_cables_both_ends_on_one_subnet():
    # the M3/M4 rig: two cables, 192.0.2.x (M3 en4 - M4 en3) and 198.51.100.x
    # (M3 en7 - M4 en2). Each side picking "a device on the peer's subnet"
    # took M4 en2 and M3 en4 -- two different cables; jaccl failed RTR.
    m3 = {"name": "M3", "thunderbolt": [{"iface": "en4", "ip": "192.0.2.1"},
                                        {"iface": "en7", "ip": "198.51.100.1"}],
          "rdma": {"active": ["rdma_en4", "rdma_en7"]}}
    m4 = {"name": "M4", "thunderbolt": [{"iface": "en2", "ip": "198.51.100.2"},
                                        {"iface": "en3", "ip": "192.0.2.2"}],
          "rdma": {"active": ["rdma_en2", "rdma_en3"]}}
    assert C._rdma_device(m4, m3) == "rdma_en3"
    assert C._rdma_device(m3, m4) == "rdma_en4"
    assert C._ring_ips([m4, m3]) == ["192.0.2.2", "192.0.2.1"]
    assert C._ring_ips([m4, m3], rdma=True) == ["192.0.2.2", "192.0.2.1"]
    # RDMA down on 192.0.2.x at one end: both move to 198.51.100.x
    m4d = {**m4, "rdma": {"active": ["rdma_en2"]}}
    assert C._rdma_device(m4d, m3) == "rdma_en2"
    assert C._rdma_device(m3, m4d) == "rdma_en7"
    assert C._ring_ips([m4d, m3], rdma=True) == ["198.51.100.2", "198.51.100.1"]


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
         "tune": "balanced", "port": 0, "working_set_gib": 60.0,
         "bandwidth_gbs": 0, "sets": {}, "versions": dict(VERSIONS),
         "jaccl_timeout_ms": 0,
         "nodes": [{"rank": 0, "id": "a", "name": "A", "page": "x:1"},
                   {"rank": 1, "id": "b", "name": "B", "page": "y:1"}]}
    s.update(kw)
    return s


def prep(s, **kw):
    return C.prepare(s, resolve=kw.get("resolve", lambda i: "/m/x"),
                     info=kw.get("info", info("Apple M3 Ultra", "192.0.2.2")),
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
    ({"hosts": ["192.0.2.1:1", "203.0.113.9:2"]}, "not one of"),
    ({"link": "jaccl", "ibv_devices": [[None, "rdma_en4"], ["rdma_en2", None]],
      "coordinator": "192.0.2.1:1"}, "RDMA on"),
])
def test_prepare_refuses_with_its_reason(cache, change, why):
    code, doc = prep(spec(**change))
    assert code == 200 and not doc["ok"] and why in doc["refused"]


def test_prepare_refuses_a_share_that_does_not_fit_here(cache):
    code, doc = prep(spec(), info=info("Apple M3 Ultra", "192.0.2.2",
                                       ws=2 * GIB))
    assert not doc["ok"] and "working set" in doc["refused"]
    s = spec(split="pipeline", layers=[2, 6])        # rank 1: layers 0..5
    code, doc = prep(s, info=info("Apple M3 Ultra", "192.0.2.2", ws=6 * GIB))
    assert not doc["ok"] and "7.0 GiB" in doc["refused"]


def test_prepare_says_not_on_and_rejects_bad_shapes(cache):
    code, doc = prep(spec(), resolve=lambda i: None)
    assert not doc["ok"] and doc["refused"].startswith("not on")
    for bad in ({"job": "../../etc"}, {"world": 1}, {"split": "data"},
                {"hosts": ["a:1"]}, {"path": "/etc"}):
        code, doc = prep(spec(**bad))
        assert code == 400


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
         "bbbb", "B", json.dumps(info_b)], env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert "up" in page_b.stdout.readline()
    peer = SimpleNamespace(id="bbbb", name="B", host="127.0.0.1",
                           key=f"127.0.0.1:{ui_b}", state="answering",
                           link="thunderbolt", node={"cluster": info_b})
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
