"""The request ledger (machine/ledger.py; docs/design/fleet.md): ULIDs,
labels cut at their byte limits, a ring bounded by days and bytes,
rollups by label."""
import re
import time

import pytest

from knurlogic.machine import ledger as L


def _row(i, ts, **kw):
    r = {"id": L.ulid(ts), "ts_start": ts, "ts_end": ts + 1, "model": "m",
         "api": "chat", "key_id": "anonymous", "prompt_tokens": 10,
         "output_tokens": 2, "finish": "stop", "status": 200}
    r.update(kw)
    return r


def test_ulids_sort_by_time_and_are_crockford():
    a, b = L.ulid(1_000_000.0), L.ulid(1_000_001.0)
    assert re.match(r"^[0-9A-HJKMNP-TV-Z]{26}$", a) and a < b
    assert len({L.ulid() for _ in range(1000)}) == 1000


def test_labels_are_cut_on_a_character_boundary():
    assert L.label("x" * 200, 128) == "x" * 128
    assert L.label("é" * 100, 32) == "é" * 16          # 2 bytes each
    assert L.label("aé" * 20, 4) == "aéa"               # never half a char
    assert L.label("", 128) is None and L.label(None, 128) is None
    got = L.labels({"X-Client": "c/1", "X-Client-Role": "r" * 40})
    assert got == {"client": "c/1", "session": None, "run": None,
                   "role": "r" * 32}


def test_a_folded_header_reads_as_one_line():
    # a header line that starts with a space continues the one before it;
    # http.server keeps the CRLF in the value
    assert L.label("s2\r\n X-Cache-Keep:latest", 128) == \
        "s2 X-Cache-Keep:latest"
    assert L.label("pm\x00-1\x1b", 128) == "pm-1"
    assert L.label(" \r\n ", 128) is None


def test_the_ledger_lives_in_knurlogic_home(tmp_path, monkeypatch):
    monkeypatch.setenv("KNURLOGIC_HOME", str(tmp_path))
    assert L.path() == tmp_path / "ledger.db"
    L.record(_row(0, time.time()))
    assert L.ledger().file == tmp_path / "ledger.db"
    assert len(L.ledger().rows()) == 1


def test_old_rows_go_after_the_days(tmp_path):
    led = L.Ledger(tmp_path / "l.db", days=30)
    now = time.time()
    led.insert(_row(0, now - 31 * 86400))
    led.insert(_row(1, now))
    led.prune()
    assert [r["ts_start"] for r in led.rows()] == [now]


def test_the_oldest_go_when_over_the_bytes(tmp_path):
    led = L.Ledger(tmp_path / "l.db", mib=0.25)
    now = time.time()
    for i in range(4000):
        led.insert(_row(i, now - 4000 + i, client="c" * 100,
                        session="s" * 100))
    led.prune()
    with led._lock:
        used = led._used(led._conn())
    assert used <= 0.25 * 1024 * 1024
    rows = led.rows(limit=10_000)
    assert 0 < len(rows) < 4000
    # the newest are kept
    assert max(r["ts_start"] for r in rows) == now - 1


@pytest.mark.parametrize("finish", ["stop", "length", "tool_calls",
                                    "error", "cancelled", "refused"])
def test_a_row_per_finish_kind(tmp_path, finish):
    led = L.Ledger(tmp_path / "l.db")
    led.insert(_row(0, time.time(), finish=finish))
    assert led.rows()[0]["finish"] == finish


def test_rollups_by_session_and_run(tmp_path):
    led = L.Ledger(tmp_path / "l.db")
    now = time.time()
    for s, r, fin in (("a", "1", "stop"), ("a", "2", "error"),
                      ("b", "1", "cancelled")):
        led.insert(_row(0, now, session=s, run=r, finish=fin))
    by = {x["session"]: x for x in led.summary(group="session")}
    assert by["a"]["requests"] == 2 and by["a"]["errors"] == 1
    assert by["b"]["cancelled"] == 1 and by["a"]["output_tokens"] == 4
    assert {x["run"]: x["requests"] for x in led.summary(group="run")} == \
        {"1": 2, "2": 1}
    with pytest.raises(ValueError):
        led.summary(group="prompt; DROP TABLE requests")
