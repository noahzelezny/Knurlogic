"""What kind of machine a node is -- and when to admit we do not know.

The bug these pin is the one the project has paid for twice already: an
answer computed HERE being reported for somewhere else. A cluster page that
labels every node with `system_profiler` output is the same shape as a
version check run with bare `python3` inside a loop over env paths -- every
iteration answers for the local box.
"""
from knurlogic import status, wired


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
    assert wired.kind_from("Studio A", "Mac15,14")["kind"] == "studio"
    assert wired.kind_from("Laptop B", "Mac16,7")["kind"] == "laptop"
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
    assert set(local["machine"]) == {"kind", "model", "model_id"}
