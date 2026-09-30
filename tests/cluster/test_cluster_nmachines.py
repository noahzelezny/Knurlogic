"""Clusters of more than two Macs: the jaccl device matrix over every pair,
the per-pair refusal, the TCP ring's addresses, the experimental note.
Fake infos and fake pages; nothing starts."""

from types import SimpleNamespace

import pytest

from knurlogic.cluster import launch as C

GIB = 1 << 30
SHAPE = {"layer_bytes": [GIB] * 8, "other_bytes": GIB,
         "tensor_per_rank_bytes": 4 * GIB, "refusals": []}
VERSIONS = {"knurlogic": "0.1.0.dev0", "mlx": "0.31.2",
            "build": "0123456789ab+mlx0.31.2"}


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "A"))
    return tmp_path


def mesh_infos(n, missing=()):
    """n machines, every pair on its own /24 (10.0.<pair>.x) over its own
    interface, except the pairs in `missing`."""
    ms = []
    for i in range(n):
        tb, act = [], []
        for j in range(n):
            if i == j or (min(i, j), max(i, j)) in missing:
                continue
            k = min(i, j) * 16 + max(i, j)
            tb.append({"iface": f"en{10 + j}", "ip": f"10.0.{k}.{i + 1}",
                       "gbps": 80})
            act.append(f"rdma_en{10 + j}")
        ms.append({"chip": "Apple M4 Max", "p_core_ghz": None,
                   "bandwidth_gbs": None, "thunderbolt": tb,
                   "rdma": {"available": True, "reason": "",
                            "devices": list(act), "active": list(act)},
                   "versions": dict(VERSIONS), "jaccl_selfheal": False,
                   "working_set_bytes": 64 * GIB})
    return ms


def n_launch(monkeypatch, infos, link, split="pipeline", req=None):
    monkeypatch.setattr(C, "_resolve", lambda i, name="": "/m/x")
    monkeypatch.setattr(C, "shape_of", lambda p, w, s: SHAPE)
    monkeypatch.setattr(C, "BAD_CABLES", {})
    monkeypatch.setattr(C, "prepare", lambda spec: (200, {"ok": True}))
    monkeypatch.setattr(C, "start", lambda job: (200, {"started": job}))
    peers = [SimpleNamespace(id=f"m{i}", name=f"M{i}", host=f"127.0.0.{i}",
                             key=f"127.0.0.{i}:8765", state="answering",
                             link="thunderbolt", node={"cluster": m})
             for i, m in enumerate(infos) if i]
    got = []

    def post(page, kind, doc, **kw):
        got.append((kind, doc))
        return {"ok": True} if kind == "Prepare" \
            else {"started": doc["job"]}
    out = C.launch({"action": "load", "identity": "abc",
                    "nodes": [f"m{i}" for i in range(len(infos))],
                    "split": split, "link": link, **(req or {})},
                   me={"id": "m0", "name": "M0"}, peers=peers,
                   local_info=infos[0], ui_port=1, serve_port=2, post=post,
                   follow=lambda j, c: None)
    return out, got


@pytest.mark.parametrize("n", [3, 4])
def test_jaccl_joins_every_pair_and_says_it_is_experimental(
        cache, monkeypatch, n):
    out, got = n_launch(monkeypatch, mesh_infos(n), "jaccl")
    assert out.get("job"), out
    assert C.RDMA_N_NOTE in out["alerts"]
    assert "experimental and untested" in C.RDMA_N_NOTE
    specs = [d for k, d in got if k == "Prepare"]
    assert len(specs) == n - 1            # this page's rank runs locally
    ids = [x["id"] for x in specs[0]["nodes"]]
    ibv = specs[0]["ibv_devices"]
    for i in range(n):
        assert ibv[i][i] is None
        for j in range(n):
            if i != j:                    # rank i's device facing rank j
                assert ibv[i][j] == f"rdma_en{10 + int(ids[j][1:])}", ibv
    assert all(s["ibv_devices"] == ibv for s in specs)


def test_jaccl_with_a_missing_edge_refuses_naming_the_pair(cache, monkeypatch):
    out, _ = n_launch(monkeypatch, mesh_infos(4, missing={(1, 3)}), "jaccl")
    assert "job" not in out
    assert "M1 and M3" in out["refused"] and "TCP" in out["refused"]


@pytest.mark.parametrize("n", [3, 4])
def test_a_tcp_ring_launches_ignores_a_cable_and_has_no_rdma_note(
        cache, monkeypatch, n):
    out, got = n_launch(monkeypatch, mesh_infos(n), "tcp",
                        req={"cable": "198.51.100"})
    assert out.get("job"), out
    assert "alerts" not in out and out["cable"] == ""
    assert "cable ignored" in out["cable_note"]
    spec = next(d for k, d in got if k == "Prepare")
    assert spec["ibv_devices"] is None and len(spec["hosts"]) == n
    assert len(out["machines"]) == n
    assert len([1 for k, d in got if k == "Start"]) == n - 1


@pytest.mark.parametrize("n", [3, 4])
def test_ring_ips_take_an_address_a_neighbour_shares(n):
    ms = mesh_infos(n)
    ips = C._ring_ips(ms)
    assert len(ips) == n
    for r, ip in enumerate(ips):
        near = {C._subnet(t["ip"]) for k in ((r - 1) % n, (r + 1) % n)
                for t in ms[k]["thunderbolt"]}
        assert C._subnet(ip) in near


def _busy_leader(busy, free):
    """Rank 0's prepare (the coordinator is rank 0 here) refuses `busy`
    ports the way a machine serving there does, suggesting `free`."""
    seen = []

    def prepare(spec):
        seen.append(spec["port"])
        if spec["port"] in busy:
            return 200, {"ok": False, "free_port": free,
                         "refused": f"port {spec['port']} is taken"}
        return 200, {"ok": True}
    return seen, prepare


def _port_launch(monkeypatch, prep, req=None):
    from knurlogic.machine import servers
    monkeypatch.setattr(servers, "free_port", lambda start, taken=(): start)
    monkeypatch.setattr(C, "_resolve", lambda i, name="": "/m/x")
    monkeypatch.setattr(C, "shape_of", lambda p, w, s: SHAPE)
    monkeypatch.setattr(C, "BAD_CABLES", {})
    monkeypatch.setattr(C, "prepare", prep)
    monkeypatch.setattr(C, "start", lambda job: (200, {"started": job}))
    infos = mesh_infos(2)
    peers = [SimpleNamespace(id="m1", name="M1", host="127.0.0.1",
                             key="127.0.0.1:8765", state="answering",
                             link="thunderbolt", node={"cluster": infos[1]})]
    return C.launch({"action": "load", "identity": "abc",
                     "nodes": ["m0", "m1"], "split": "pipeline",
                     "link": "tcp", **(req or {})},
                    me={"id": "m0", "name": "M0"}, peers=peers,
                    local_info=infos[0], ui_port=1, serve_port=8080,
                    post=lambda page, kind, d: {"ok": True, "started": d.get("job")},
                    follow=lambda j, c: None)


def test_unnamed_port_retries_once_on_the_suggested_free_one(monkeypatch):
    seen, prep = _busy_leader({8080}, 8083)
    out = _port_launch(monkeypatch, prep)
    assert seen == [8080, 8083] and out["port"] == 8083


def test_named_port_that_is_taken_stays_refused(monkeypatch):
    seen, prep = _busy_leader({8080}, 8083)
    out = _port_launch(monkeypatch, prep, {"port": 8080})
    assert seen == [8080] and out["refused"].startswith("nothing started")


def test_free_port_skips_served_and_bound_ports(monkeypatch):
    import socket
    from knurlogic.machine import servers
    monkeypatch.setattr(servers, "registry",
                        lambda: {8080: {"pid": 1, "artifact": "x"}})
    monkeypatch.setattr(servers, "is_our_server", lambda pid: True)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        bound = s.getsockname()[1]
        assert servers.free_port(bound) != bound
    assert servers.free_port(8080) != 8080
