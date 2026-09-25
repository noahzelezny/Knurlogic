"""What kind of machine a node is -- and when to admit we do not know.

The bug these pin is the one the project has paid for twice already: an
answer computed HERE being reported for somewhere else. A cluster page that
labels every node with `system_profiler` output is the same shape as a
version check run with bare `python3` inside a loop over env paths -- every
iteration answers for the local box.
"""
from knurlogic.machine import status, wired


def test_a_modern_identifier_cannot_name_the_product():
    """`Mac15,14` IS a Mac Studio and nothing in the string says so. Apple
    dropped the product prefix, so guessing from the identifier has to
    return nothing rather than a coin flip dressed as a fact."""
    assert wired._kind_from_identifier("Mac15,14") == ""
    assert wired._kind_from_identifier("Mac16,7") == ""


def test_old_identifiers_still_name_the_product():
    assert wired._kind_from_identifier("MacBookPro18,3") == "laptop"
    assert wired._kind_from_identifier("Macmini9,1") == "mini"
    assert wired._kind_from_identifier("iMac21,1") == "imac"


def test_a_remote_node_is_read_from_what_it_reported():
    assert wired.kind_from("Noah's Mac Studio", "Mac15,14")["kind"] == "studio"
    assert wired.kind_from("NozzleBook Pro", "Mac16,7")["kind"] == "laptop"
    assert wired.kind_from("kitchen mini", "")["kind"] == "mini"


def test_an_unknown_machine_stays_unknown():
    """The page draws a dashed box for this, which is the right outcome:
    a machine nobody could identify should look unidentified."""
    assert wired.kind_from("worker-07", "Mac15,14")["kind"] == ""
    assert wired.kind_from("", "")["kind"] == ""


def test_a_remote_snapshot_never_inherits_the_local_machine():
    """`machine_fn` is the seam that keeps this honest -- without it every
    node in a snapshot would carry whatever box built the snapshot."""
    remote = status.snapshot(node="elsewhere", reachable=True,
                             memory_fn=lambda: {"available": False},
                             machine_fn=lambda: wired.kind_from("elsewhere"))
    assert remote["machine"]["kind"] == ""
    assert remote["machine"]["model"] == ""     # not this box's model name


def test_the_local_snapshot_asks_the_host_directly():
    local = status.snapshot(memory_fn=lambda: {"available": False})
    assert set(local["machine"]) == {"kind", "model", "model_id", "chip"}


def test_metrics_ride_in_the_local_snapshot_and_a_peer_gets_none_invented():
    from knurlogic.machine import metrics
    metrics._hist.clear()      # an earlier snapshot in this process sampled
    local = status.snapshot(memory_fn=lambda: {"available": False},
                            memory_map={"installed_bytes": 100,
                                        "used_bytes": 25})
    m = local["metrics"]
    assert m["now"]["memory_pct"] == 25.0
    assert m["history"][-1] is m["now"]
    assert len(m["history"]) <= metrics.HISTORY
    # a node built from someone else's numbers draws no lines it did not send
    peer = status.snapshot(memory_fn=lambda: {"available": False},
                           machine_fn=lambda: {"kind": ""})
    assert "metrics" not in peer


def test_a_missing_reading_is_none_not_zero(monkeypatch):
    from knurlogic.machine import metrics
    monkeypatch.setattr(metrics.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(OSError()))
    s = metrics.sample()
    assert s["gpu_pct"] is None and s["swap_bytes"] is None
    assert s["memory_pct"] is None


def test_temperature_reads_none_when_the_sensor_api_is_gone(monkeypatch):
    from knurlogic.machine import metrics
    monkeypatch.setattr(metrics, "_hid",
                        lambda: (_ for _ in ()).throw(OSError("moved")))
    assert metrics._temp_c() is None
