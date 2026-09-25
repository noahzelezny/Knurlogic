"""cluster/discovery: Bonjour through dns_sd.h. The live test registers and
browses on the LOCAL-ONLY interface -- nothing leaves the machine -- and
needs mDNSResponder, so it runs on macOS only."""
import sys
import time

import pytest

from knurlogic.cluster.discovery import (LOCAL_ONLY, Discovery, txt_decode,
                                         txt_encode)


def test_txt_round_trips_and_truncates_at_255():
    d = {"id": "abc", "name": "Studio A", "schema": "2"}
    assert txt_decode(txt_encode(d)) == d
    assert len(txt_encode({"k": "x" * 400})) == 256


@pytest.mark.skipif(sys.platform != "darwin", reason="needs mDNSResponder")
def test_a_registered_page_is_found_resolved_and_its_txt_read():
    changes = []
    d = Discovery(on_change=changes.append, if_index=LOCAL_ONLY)
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
