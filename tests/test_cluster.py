"""Resolving over several nodes, the aggregate status, and reading exo's
node inventory (against a stub exo) as one witness of which machines exist.
"""

import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knurlogic.cluster import exo as exo_nodes
from knurlogic.machine import status
from knurlogic.machine.artifact import Artifact
from knurlogic.tuning.resolve import Node, resolve, resolve_cluster

GIB = 1 << 30

STATE = {
    "nodeMemory": {
        "nodeA": {"ramTotal": {"inBytes": 128 * GIB},
                  "ramAvailable": {"inBytes": 84 * GIB},
                  "swapTotal": {"inBytes": 0},
                  "swapAvailable": {"inBytes": 0}},
        "nodeB": {"ramTotal": {"inBytes": 64 * GIB},
                  "ramAvailable": {"inBytes": 40 * GIB},
                  "swapTotal": {"inBytes": 0},
                  "swapAvailable": {"inBytes": 0}},
    },
    "nodeIdentities": {"nodeA": {"friendlyName": "studio"},
                       "nodeB": {"friendlyName": "laptop"}},
}


def _art(**kw):
    base = dict(path=Path("/nonexistent/art"), model_type="qwen4_exp_text",
                model_file="model.py", bytes_on_disk=110 * GIB,
                hidden_size=2560, moe_intermediate_size=640,
                vq_modules={"m": {"d": 4, "K": 2048}})
    base.update(kw)
    return Artifact(**base)


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class _Stub(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _j(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/state"):
            return self._j(STATE)
        self.send_error(404)

    def log_message(self, *a):
        pass


def _serve(handler, port):
    srv = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# --- the resolver over nodes -----------------------------------------------

def test_single_box_call_is_unchanged():
    """Everything that existed took one budget and got one Resolution back."""
    r = resolve(_art(bytes_on_disk=70 * GIB), 96 * GIB)
    assert r.env["VQ_MOE_GEMMSEG_RTILE"] == "32"
    assert hasattr(r, "as_exports") and not hasattr(r, "nodes")


def test_cluster_resolves_each_node_against_its_own_box():
    """The point of the shape change: a 128 GB box and a 64 GB box do not get
    the same knobs, and a single budget cannot express that."""
    c = resolve(_art(), [Node("big", 128 * GIB, holds_bytes=90 * GIB),
                         Node("small", 64 * GIB, holds_bytes=62 * GIB)])
    assert set(c.nodes) == {"big", "small"}
    # 38 GiB of headroom against 2 GiB: the tight box gets its PER-NODE
    # memory knobs resolved down and the roomy one keeps its own.
    assert int(c.nodes["small"].env["VQ_DECODE_CHUNK"]) < \
        int(c.nodes["big"].env["VQ_DECODE_CHUNK"])


def test_the_prompt_chunk_is_one_value_on_every_rank():
    """Ring-wide, not per node. This test used to assert 2048 on the big box
    and 512 on the small one -- which is the desync recorded live on a
    pipeline ring (GLM-5.3 at 2048 on one rank, 4096 on the other). The tightest
    node's chunk is everyone's, and the big node is told why. Every rank is
    512 by default, so the ring rule is exercised where widths can differ:
    tune=fast on a measured-wide family, one roomy rank and one tight."""
    c = resolve(_art(), [Node("big", 128 * GIB, holds_bytes=90 * GIB),
                         Node("small", 64 * GIB, holds_bytes=62 * GIB)])
    assert c.nodes["big"].env["KNURLOGIC_PREFILL_CHUNK"] == \
        c.nodes["small"].env["KNURLOGIC_PREFILL_CHUNK"] == "512"
    c = resolve(_art(model_type="qwen3_5"),
                [Node("big", 128 * GIB, holds_bytes=48 * GIB),
                 Node("small", 64 * GIB, holds_bytes=62 * GIB)], tune="fast")
    big, small = c.nodes["big"].env, c.nodes["small"].env
    assert big["KNURLOGIC_PREFILL_CHUNK"] == small["KNURLOGIC_PREFILL_CHUNK"] == "512"
    assert any("every rank must match" in n for n in c.nodes["big"].notes)


def test_shards_are_assumed_proportionally_and_it_says_so():
    """An assumption that is not labelled is indistinguishable from a
    measurement later."""
    c = resolve(_art(bytes_on_disk=120 * GIB),
                {"a": 120 * GIB, "b": 60 * GIB})
    assert any("ASSUMED" in n for n in c.nodes["a"].notes)
    assert any("proportional" in n for n in c.notes)


def test_a_cluster_too_small_for_the_artifact_says_so_once():
    c = resolve(_art(bytes_on_disk=200 * GIB), [Node("a", 64 * GIB),
                                                Node("b", 64 * GIB)])
    assert any("does not fit even sharded" in w for w in c.warnings)


def test_declared_placement_beats_the_assumption():
    c = resolve_cluster(_art(bytes_on_disk=100 * GIB),
                        [Node("a", 96 * GIB, holds_bytes=94 * GIB),
                         Node("b", 96 * GIB, holds_bytes=6 * GIB)])
    assert not any("ASSUMED" in n for n in c.nodes["a"].notes)
    assert int(c.nodes["a"].env["VQ_DECODE_CHUNK"]) < \
        int(c.nodes["b"].env["VQ_DECODE_CHUNK"]), (
        "the node holding 94 of 100 GiB is the tight one, and declared "
        "placement is what makes that visible")


# --- the status shape -------------------------------------------------------

def test_one_node_keeps_the_old_toplevel_contract():
    """A client written against one box must not break when the shape grows."""
    s = status.snapshot(node="local")
    d = status.aggregate([s])
    assert d["memory"] == s["memory"] and d["uptime_seconds"] is not None
    assert d["nodes"] == [s] and d["cluster"]["nodes_total"] == 1


def test_rollup_sums_and_excludes_nodes_that_did_not_answer():
    """A sum over silent nodes is a smaller number that looks like headroom."""
    def m(x):
        return {"available": True, "active_bytes": x, "cache_bytes": 0,
                "peak_bytes": 0, "working_set_bytes": 2 * x,
                "total_bytes": 2 * x, "headroom_bytes": x,
                "process_rss_bytes": 0}
    up = status.snapshot(node="a", memory_fn=lambda: m(10))
    down = status.snapshot(node="b", reachable=False, memory_fn=lambda: m(10))
    c = status.aggregate([up, down])["cluster"]
    assert c["memory"]["active_bytes"] == 10
    assert (c["nodes_reachable"], c["nodes_total"]) == (1, 2)


# --- exo as a witness of which machines exist ------------------------------

def test_inventory_reads_exos_own_node_memory():
    port = _free_port()
    srv = _serve(_Stub, port)
    try:
        nodes = exo_nodes.inventory(f"http://127.0.0.1:{port}")
    finally:
        srv.shutdown()
    assert [n.name for n in nodes] == ["studio", "laptop"]
    assert nodes[0].ram_total == 128 * GIB


def test_box_wide_numbers_are_not_labelled_as_weights():
    """A number that covers the whole machine must not be printed under a
    label that says 'weights': that is how a runtime gets blamed for
    everything else on the box. `scope` is what decides the wording."""
    def m(scope):
        return {"available": True, "scope": scope, "active_bytes": 10 * GIB,
                "cache_bytes": 0, "peak_bytes": 0,
                "working_set_bytes": 20 * GIB, "total_bytes": 20 * GIB,
                "headroom_bytes": 10 * GIB, "process_rss_bytes": 0}
    box = status.render_cluster(status.aggregate(
        [status.snapshot(node=n, memory_fn=lambda: m("box"))
         for n in ("a", "b")]))
    proc = status.render_cluster(status.aggregate(
        [status.snapshot(node=n, memory_fn=lambda: m("process"))
         for n in ("a", "b")]))
    assert "in use on the box" in box and "weights" not in box
    assert "weights + live" in proc


def test_settings_json_says_running_would_be_and_how_to_get_it(tmp_path):
    """The three questions a settings panel has to keep apart.

    A slider that looks like it retunes a loaded model would be a lie -- the
    runtime reads its environment at import and the import already happened.
    So the document has to carry the running value, the value another tune
    WOULD give, and the fact that it takes a restart.
    """
    from knurlogic.interfaces.page import documents
    from knurlogic.tuning.resolve import resolve

    a = _art(bytes_on_disk=72 * GIB)
    live = resolve(a, 84 * GIB, tune="balanced")
    doc = documents.settings_document(
        a, live_env=dict(live.env), live_tune="balanced",
        live_working_set=84 * GIB,
        resolve_fn=lambda ws, t: resolve(a, ws, tune=t))

    same = doc({})
    assert not any(k["changed"] for k in same["knobs"]), (
        "asking for the running tune must not report a pending change")
    # Reach is per knob, not per page: some apply now, some need a restart,
    # and some do nothing on this artifact at all.
    assert {k["reach"] for k in same["knobs"]} <= {"live", "restart",
                                                   "no-effect"}

    fast = doc({"tune": ["fast"]})
    changed = {k["name"]: (k["running"], k["would_be"])
               for k in fast["knobs"] if k["changed"]}
    assert "VQLAB_CACHE_LIMIT_GB" in changed
    assert fast["exports"].startswith("export ")

    # Every knob shown carries the sentence that explains it. A settings UI
    # that lists names and values is a config file with a stylesheet.
    assert all(k["what"] for k in fast["knobs"]), \
        [k["name"] for k in fast["knobs"] if not k["what"]]


def test_a_misspelled_tune_from_a_url_falls_back_instead_of_500ing():
    """Query strings are user input; a typo must not take the page down."""
    from knurlogic.interfaces.page import documents
    from knurlogic.tuning.resolve import resolve
    a = _art(bytes_on_disk=72 * GIB)
    doc = documents.settings_document(a, live_env={}, live_tune="balanced",
                                live_working_set=84 * GIB,
                                resolve_fn=lambda ws, t: resolve(a, ws, tune=t))
    assert doc({"tune": ["turbo"]})["asked"]["tune"] == "balanced"
    assert doc({"working_set_gib": ["not-a-number"]})[
        "asked"]["working_set_bytes"] == 84 * GIB
