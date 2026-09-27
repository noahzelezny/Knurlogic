"""A fake peer page for tests/test_cluster_jobs.py (not a test module): the
REAL page handler and cluster routes (interfaces/ui, interfaces/cluster_jobs)
in its own process, with its own cache dir (XDG_CACHE_HOME) and identity,
and only the machine facts faked: its artifact, the model's shape, its
cluster block, and the rank it spawns (tests/cluster_fake_rank.py).

argv: <port> <id> <name> <info json> [<peer id> <peer address>]
(the peer: this page's PEERS record of the other page -- a stop is only
ever posted to an address PEERS knows)
"""
import json
import os
import sys
from http.server import ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    port, nid, name, info = (int(sys.argv[1]), sys.argv[2], sys.argv[3],
                             json.loads(sys.argv[4]))
    from knurlogic.interfaces import cluster_jobs as C
    from knurlogic.interfaces import ui
    from knurlogic.machine import identity
    identity._ID.update(id=nid, name=name, id_source="test")
    C._resolve = lambda ident: "/fake/artifact" if ident == "abc" else None
    C.shape_of = lambda path, world, split: SHAPE
    C._local_info = lambda: info
    C.RANK_ARGV[0] = fake_argv
    C.WATCH_S = 0.3
    if len(sys.argv) > 6:
        from types import SimpleNamespace

        from knurlogic.cluster.peers import Peer
        host, pport = sys.argv[6].rsplit(":", 1)
        peer = Peer(host=host, port=int(pport), id=sys.argv[5],
                    state="answering")
        ui.PEERS = SimpleNamespace(all=lambda: [peer],
                                   introduce=lambda *a, **k: None)
    srv = ThreadingHTTPServer(("127.0.0.1", port), ui.make_handler({}))
    print("fake page up", flush=True)
    srv.serve_forever()


SHAPE = {"layer_bytes": [1 << 30] * 8, "other_bytes": 1 << 30,
         "tensor_per_rank_bytes": 4 << 30, "refusals": []}


def fake_argv(path, spec, files):
    return [sys.executable, os.path.join(HERE, "cluster_fake_rank.py"),
            path, "knurlogic", "serve", "--rank", str(spec["rank"]),
            "--job", spec["job"], "--port", str(spec.get("port") or 0)]


if __name__ == "__main__":
    main()
