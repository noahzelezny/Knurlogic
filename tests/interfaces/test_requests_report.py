"""Request reporting: the scheduler's running/waiting counts, and where they
travel (/status.json, /v1/residency, /loaded.json rows, the MCP's state),
plus Retry-After on the 503s. No model: the scheduler is not started."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from knurlogic.engine.runtime.scheduler import (  # noqa: E402,E501
    Job,
    OutOfMemory,
    RingFailed,
    Scheduler,
)


class Host:
    state, path, error = "ready", "/m/a", ""

    def status(self):
        return {"state": self.state, "model": self.path, "memory_bytes": 1}


def _job(age=0.0):
    j = Job(request=None, args=None)
    j.submitted = time.perf_counter() - age
    return j


def _row(job):
    from knurlogic.engine.runtime.scheduler import _Row
    return _Row(job, None, [])


def test_idle():
    s = Scheduler(Host(), completion_batch_size=4)
    assert s.requests() == {"in_flight": 0, "pending": 0, "capacity": 4,
                            "oldest_pending_s": 0.0, "holding": None}


def test_counts_queued_waiting_and_past_the_batch():
    s = Scheduler(Host(), completion_batch_size=2)
    for uid in range(3):                   # the third waits for a slot
        s._rows[uid] = _row(_job(age=1.0))
    s._waiting.append(_job(age=7.0))
    s._jobs.put(_job(age=0.5))
    gone = _job(age=99.0)
    gone.cancelled = True
    s._waiting.append(gone)                # a cancelled one is not waiting
    r = s.requests()
    assert r["in_flight"] == 2 and r["capacity"] == 2
    assert r["pending"] == 3
    assert 6.5 < r["oldest_pending_s"] < 60


def test_holding_reason():
    s = Scheduler(Host(), completion_batch_size=2)
    s._waiting.append(_job())
    assert s._why_waiting(room=False) == "memory"
    s._holding = s._why_waiting(room=False)
    assert s.requests()["holding"] == "memory"
    s.host.state = "loading"
    assert s._why_waiting(room=False) == "loading"
    s.host.state = "ready"
    held = _job()
    held.waiting_on = {1}
    s._waiting.append(held)
    assert s._why_waiting(room=True) == "memory"
    s._waiting.clear()
    assert s._why_waiting(room=True) is None
    for uid in range(3):
        s._rows[uid] = _row(_job())
    s._holding = None
    assert s.requests()["holding"] == "batch_full"


def test_residency_row_carries_requests():
    from knurlogic.interfaces.http import residency as res_api
    s = Scheduler(Host(), completion_batch_size=8)
    row = res_api.residency(s.host, s)["data"][0]
    assert row["requests"]["capacity"] == 8
    assert row["requests"]["pending"] == 0


def test_status_json_requests_reach_loaded_rows(monkeypatch):
    from knurlogic.machine import loaded
    req = {"in_flight": 1, "pending": 2, "capacity": 4,
           "oldest_pending_s": 3.0, "holding": "memory"}
    monkeypatch.setattr(loaded, "_get", lambda url, timeout=1.5: {
        "schema": 2, "artifact": {"name": "m", "path": "/m"},
        "memory": {"active_bytes": 5}, "requests": req})
    r = loaded._knurlogic("http://127.0.0.1:1")[0]
    assert r.__dict__["requests"] == req


def test_mcp_state_lists_requests_per_model(monkeypatch):
    from knurlogic.interfaces import mcp
    from knurlogic.interfaces.page import server as page_server
    from knurlogic.machine import loaded
    req = {"in_flight": 1, "pending": 0, "capacity": 4,
           "oldest_pending_s": 0.0, "holding": None}
    monkeypatch.setattr(loaded, "survey", lambda: {"resident": [
        {"name": "m", "where": "http://127.0.0.1:9", "requests": req},
        {"name": "x", "where": "http://127.0.0.1:11434", "requests": None}],
        "runtimes": ["knurlogic"], "memory": {}})
    monkeypatch.setattr(page_server, "children", lambda: [])
    monkeypatch.setattr(mcp, "_me_name", lambda: "here")
    st = mcp.state()           # no page (conftest): this Mac's survey
    assert st["requests"] == [dict(req, model="m", machine="here",
                                   where="http://127.0.0.1:9")]


def test_the_503s_say_when_to_retry():
    from knurlogic.interfaces.http.openai import _status_of
    assert _status_of(OutOfMemory("x")).retry_after == 10
    assert _status_of(RingFailed("x")).retry_after == 30
    e = _status_of(RuntimeError("no model to serve: gone"))
    assert e.status == 503 and e.retry_after == 5
    assert _status_of(ValueError("bad")).retry_after is None


def test_messages_refusal_sends_retry_after():
    from knurlogic.interfaces.http import messages as M
    sent = []

    def transport(_oai):
        raise M.TransportError(503, "no model is loaded", 5)
    h = M.handler_over(transport, "m")
    h(b"{}", lambda b: None, lambda *a: sent.append(a))
    assert sent == [(503, "application/json", {"Retry-After": "5"})]


def test_the_warm_up_row_is_not_a_request():
    s = Scheduler(Host(), completion_batch_size=4)
    s._rows[0] = _row(_job())
    assert s.requests()["in_flight"] == 1
    s._warming = True
    assert s.requests()["in_flight"] == 0
