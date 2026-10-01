"""A fake peer page for tests/cluster (not a test module): the REAL page
handler and cluster routes (interfaces/page/server, cluster/launch) in its
own process, with its own cache dir (XDG_CACHE_HOME) and identity, and only
the machine facts faked: its artifact, the model's shape, its cluster
block, and the rank it spawns (tests/support/cluster_fake_rank.py).

argv: <port> <id> <name> <info json> [<peer id> <peer address>]
(the peer: this page's PEERS record of the other page -- a stop is only
ever posted to an address PEERS knows)

env (what tests/support/fake_cluster.py sets):
  FAKE_PEERS=<json [[id, "host:port"], ...]>  the REAL Peers store, each
      named with --peer, so this page does the real liveness probing
  FAKE_FAST=1         short liveness clocks (answering 1.5 s, gone 4 s)
  FAKE_PROTOCOL_MAJOR=<n>  this page speaks that protocol major
  FAKE_DENY_PEER_GATE=1    the peer gate refuses every connection, as it
      does a Wi-Fi arrival (the status GET still answers)
  FAKE_NO_PEERS=1     (fake_cluster) this page is told of no peer
  FAKE_JOB_ID=<hex>   the job id a launch from this page mints
  FAKE_SERVE_PORT=<n>  the port a job this page coordinates serves on
"""
import json
import os
import sys
from http.server import ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))


class DenyGate:
    """The peer gate as a connection from Wi-Fi meets it."""

    def allows(self, _ip):
        return False

    def refusal(self, _ip):
        return b"refused"


def fast():
    from knurlogic.cluster import jobs, peers, recovery, transport
    recovery.BACKOFF_S = (0.5, 1.0, 2.0)
    recovery.TICK_S = 0.5
    peers.REFRESH_S = 0.5
    peers.ANSWERING_S = 1.5
    jobs.PEER_GONE_S = 4.0
    transport.TIMEOUTS["status"] = 1.0


def main():
    port, nid, name, info = (int(sys.argv[1]), sys.argv[2], sys.argv[3],
                             json.loads(sys.argv[4]))
    from knurlogic.cluster import launch as C
    from knurlogic.cluster import protocol
    from knurlogic.interfaces.page import documents
    from knurlogic.interfaces.page import server as page_server
    from knurlogic.machine import identity
    identity._ID.update(id=nid, name=name, id_source="test")
    if os.environ.get("FAKE_FAST"):
        fast()
    if os.environ.get("FAKE_JOB_ID"):
        from types import SimpleNamespace
        C.secrets = SimpleNamespace(
            token_hex=lambda n: os.environ["FAKE_JOB_ID"])
    if os.environ.get("FAKE_PROTOCOL_MAJOR"):
        protocol.VERSION = (int(os.environ["FAKE_PROTOCOL_MAJOR"]), 0)
    C._resolve = lambda ident, name="": "/fake/artifact" if ident == "abc" else None
    C.shape_of = lambda path, world, split, vision=True: SHAPE
    C._local_info = lambda: info
    C.node_info = lambda working_set_bytes=0, ttl=30.0: info
    C.RANK_ARGV[0] = fake_argv
    C.WATCH_S = 0.3
    if os.environ.get("FAKE_PEERS"):
        from knurlogic.cluster.peers import Peers
        manual = []
        for _pid, addr in json.loads(os.environ["FAKE_PEERS"]):
            host, pport = addr.rsplit(":", 1)
            manual.append((host, int(pport)))
        page_server.PEERS = Peers({"id": nid, "name": name}, port,
                                  manual=manual, persist=False).start()
    elif len(sys.argv) > 6:
        from types import SimpleNamespace

        from knurlogic.cluster.peers import Peer
        host, pport = sys.argv[6].rsplit(":", 1)
        peer = Peer(host=host, port=int(pport), id=sys.argv[5],
                    state="answering")
        page_server.PEERS = SimpleNamespace(all=lambda: [peer],
                                   introduce=lambda *a, **k: None)
    routes = {}
    serve_port = int(os.environ.get("FAKE_SERVE_PORT") or 0)
    if serve_port:
        routes = documents.routes(
            status_fn=page_server._status_fn,
            loaded_fn=page_server._loaded_fn(),
            load_fn=page_server._load_fn(serve_port))
        full = routes["/status.json"]
        routes["/status.json"] = lambda q, _n=0: (
            documents._json(page_server._status_light())
            if (q.get("light") or [""])[0] else full(q, _n))
        page_server._SERVE_PORT["ui"] = port
        C.start_watching_existing()
    else:
        routes = documents.routes(
            status_fn=page_server._status_fn,
            loaded_fn=page_server._loaded_fn())
        full = routes["/status.json"]
        routes["/status.json"] = lambda q, _n=0: (
            documents._json(page_server._status_light())
            if (q.get("light") or [""])[0] else full(q, _n))
        page_server._SERVE_PORT["ui"] = port
    srv = ThreadingHTTPServer(("127.0.0.1", port), page_server.make_handler(
        routes, gate_for_peers=DenyGate()
        if os.environ.get("FAKE_DENY_PEER_GATE") else None))
    print("fake page up", flush=True)
    srv.serve_forever()


SHAPE = {"layer_bytes": [1 << 30] * 8, "other_bytes": 1 << 30,
         "tensor_per_rank_bytes": 4 << 30, "refusals": []}


def fake_argv(path, spec, files):
    return [sys.executable, os.path.join(HERE, "cluster_fake_rank.py"),
            path, "knurlogic", "serve", "--rank", str(spec["rank"]),
            "--job", spec["job"], "--port", str(spec.get("port") or 0),
            "--cable", str(spec.get("cable") or ""),
            # a test's process: a real page never lists it
            # (machine/servers.is_test_process)
            "--knurlogic-test"]


if __name__ == "__main__":
    main()
