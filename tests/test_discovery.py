"""cluster/discovery: Bonjour through dns_sd.h. The live test registers and
browses on the LOCAL-ONLY interface -- nothing leaves the machine -- and
needs mDNSResponder, so it runs on macOS only."""
import sys
import time

import pytest

from knurlogic.cluster.discovery import (LOCAL_ONLY, Discovery, txt_decode,
                                         txt_encode)


def test_txt_round_trips_and_truncates_at_255():
    d = {"id": "abc", "name": "Noah's Mac Studio", "schema": "2"}
    assert txt_decode(txt_encode(d)) == d
    assert len(txt_encode({"k": "x" * 400})) == 256


@pytest.mark.skipif(sys.platform != "darwin", reason="needs mDNSResponder")
def test_a_registered_page_is_found_resolved_and_its_txt_read():
    changes = []
    # its own service type: a live page on this Mac browses _knurlogic._tcp
    # and would list the test's record as a machine called "Test"
    d = Discovery(on_change=changes.append, if_index=LOCAL_ONLY,
                  service="_knurlogic-test._tcp")
    try:
        assert d.register("kl-pytest 0f0f0f", 18998,
                          {"id": "0f0f0f0f0f0f", "name": "Test"})
        assert d.browse()
        d.start()
        deadline = time.time() + 10
        while time.time() < deadline and not d.found:
            time.sleep(0.1)
        [svc] = [s for s in d.snapshot() if s["name"] == "kl-pytest 0f0f0f"]
        assert (svc["host"], svc["port"]) == ("127.0.0.1", 18998)
        assert svc["txt"]["id"] == "0f0f0f0f0f0f"
        assert changes and not d.errors
    finally:
        d.stop()


def test_a_ref_queued_twice_is_freed_once_and_its_callback_goes():
    """A resolve can call back twice in one batch; freeing its ref twice
    would hand mDNSResponder a dead pointer (segfault-capable)."""
    import ctypes
    import types
    from knurlogic.cluster.discovery import Discovery
    d = Discovery()
    a, b = ctypes.c_void_p(1), ctypes.c_void_p(2)
    d._add_ref(a, "cb-a")
    d._add_ref(b, "cb-b")
    freed = []
    lib = types.SimpleNamespace(DNSServiceRefDeallocate=freed.append)
    d._free += [a, a, None]
    d._drain_free(lib)
    assert freed == [a]
    assert d._refs == [b] and list(d._keep) == [id(b)]


def test_a_long_instance_name_is_cut_on_a_character_boundary():
    from knurlogic.cluster.discovery import _label
    name = "Noah’s Mac Studio — " + "é" * 40
    b = _label(name)
    assert len(b) <= 63
    b.decode("utf-8")                      # never split mid-character
