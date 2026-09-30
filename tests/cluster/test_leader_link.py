"""A cluster job's leader is reachable from the job's other machines (it
binds loopback AND its link address, never every interface), and people
see one link vocabulary -- tcp | rdma -- while ring | jaccl stay mlx's
backend argument inside a job (older recovery records still read)."""

import socket

import pytest

from knurlogic.cluster import launch as C
from knurlogic.interfaces import http as H
from knurlogic.interfaces.http.server import browser_refusal


def _spec(rank, link="ring", **kw):
    return {"rank": rank, "world": 2, "split": "pipeline", "link": link,
            "job": "ab", "hosts": ["10.0.0.2:47200", "10.0.0.1:47201"],
            "prefill_chunk": 512, "port": 8080 if rank == 0 else 0,
            "serve_hosts": ["127.0.0.1", "10.0.0.2"], **kw}


def test_the_leader_binds_loopback_and_its_link_address_only():
    a0 = C.rank_argv("/m", _spec(0), {})
    assert a0[a0.index("--host") + 1] == "127.0.0.1,10.0.0.2"
    assert "0.0.0.0" not in a0
    # the other ranks serve no API: no --host
    assert "--host" not in C.rank_argv("/m", _spec(1), {})


def test_the_leader_url_is_its_link_address():
    assert C.leader_url(_spec(0)) == "http://10.0.0.2:8080/v1"
    assert C.leader_url({"serve_hosts": ["127.0.0.1"], "port": 8080}) == ""
    assert C.leader_url({}) == ""


def test_a_prepare_refuses_serve_hosts_that_are_not_addresses():
    base = dict(_spec(0), job="ab" * 8, identity="x",
                nodes=[{"name": "A"}, {"name": "B"}])
    assert "serve_hosts" not in (C.check_spec(base) or "")
    for bad in (["0.0.0.0 --evil"], "127.0.0.1", [], ["host.example"]):
        why = C.check_spec(dict(base, serve_hosts=bad))
        assert why == "serve_hosts is a list of IP addresses", bad


def test_bind_all_serves_every_address_and_skips_one_that_is_gone():
    made = []

    def make(app, host, port):
        if host == "10.9.9.9":
            raise OSError(49, "Can't assign requested address")
        made.append(host)
        return object()
    out = H.bind_all(None, "127.0.0.1,10.9.9.9,::1", 8080, make=make)
    assert made == ["127.0.0.1", "::1"] and len(out) == 2
    # single-Mac serve: one address, as before
    made.clear()
    H.bind_all(None, "127.0.0.1", 8080, make=make)
    assert made == ["127.0.0.1"]


def test_bind_all_really_listens_on_each_address():
    from knurlogic.interfaces.http.server import make_server
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    srvs = H.bind_all(None, "127.0.0.1", port, make=make_server)
    try:
        assert srvs[0].server_address == ("127.0.0.1", port)
    finally:
        for x in srvs:
            x.server_close()


def test_the_model_server_answers_a_request_to_its_link_address():
    # an agent on the other machine calling http://10.0.0.2:8080
    # (no Origin)
    assert browser_refusal({"Host": "10.0.0.2:8080"}) is None


def test_single_mac_serve_still_defaults_to_loopback():
    import argparse
    from knurlogic.interfaces import serve as S
    seen = {}

    def run(path, host, *a, **k):
        seen["host"] = host
        return 0
    S_run = S.run
    try:
        S.run = run
        S.main(["some-model"])
    finally:
        S.run = S_run
    assert seen["host"] == "127.0.0.1"


# --- one vocabulary --------------------------------------------------------

@pytest.mark.parametrize("given,backend", [
    ("tcp", "ring"), ("rdma", "jaccl"),          # what people say
    ("ring", "ring"), ("jaccl", "jaccl"),        # older records, callers
    ("udp", None), (None, None)])
def test_one_mapping_to_mlx_backends(given, backend):
    assert C.backend(given) == backend


@pytest.mark.parametrize("given,shown", [
    ("ring", "tcp"), ("jaccl", "rdma"), ("tcp", "tcp"), ("rdma", "rdma")])
def test_people_see_tcp_or_rdma(given, shown):
    assert C.link_name(given) == shown


def test_a_launch_takes_either_name_and_answers_in_tcp_rdma():
    # the refusal before any machine is asked names the user's words
    out = C.launch({"nodes": ["a", "b"], "split": "pipeline",
                    "link": "udp"}, me={}, peers=[], local_info={},
                   ui_port=0, serve_port=0)
    assert out["error"] == "split is tensor|pipeline, link is tcp|rdma"


def test_the_recovery_record_reads_ring_jaccl_and_shows_tcp_rdma(
        monkeypatch):
    from knurlogic.cluster import recovery as R
    monkeypatch.setattr(R, "MODELS", {})
    monkeypatch.setattr(R, "save", lambda: None)
    monkeypatch.setattr(R, "ensure_thread", lambda: None)
    monkeypatch.setattr(R, "_sync", lambda key: None)
    R.track_cluster("a" * 16, req={"identity": "abc", "link": "jaccl",
                                   "split": "pipeline"},
                    args={}, order=[{"name": "A"}], port=8080,
                    leader_here=True)
    rec = next(iter(R.MODELS.values()))
    assert rec["link"] == "rdma"
    # the relaunch request keeps what launch() was given, which it reads
    assert C.backend(rec["req"]["link"]) == "jaccl"
    # a record written before this change says jaccl; people see rdma
    rec.update(link="jaccl", pending=True, state="recovering")
    assert R.not_serving()[0]["link"] == "rdma"
