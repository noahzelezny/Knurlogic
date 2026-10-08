"""The telemetry contract, server side (docs/design/telemetry.md,
`telemetry: 1`; the ledger per docs/design/fleet.md): X-Client-* labels
stored with each request, X-Request-Id (the client's, else a ULID) naming its ledger row,
usage.knurlogic.{request_id, timing} on every API, the SSE
knurlogic.progress event, and /v1/models advertising the version.

The model server runs in front of a fake scheduler that answers each job
from a script, as the real one does (progress, deltas, done with usage)."""
import json
import queue
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace as NS

import pytest

from knurlogic.engine.runtime.request import Delta
from knurlogic.interfaces.http import server as S
from knurlogic.interfaces.http import telemetry as T
from knurlogic.machine import ledger as L

ULID = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")
TIMING = {"queue_ms", "prefill_ms", "decode_ms", "prefill_tps", "decode_tps"}
LABELS = {"X-Client": "client/0.0.1", "X-Client-Session": "sess-1",
          "X-Client-Run": "worker-2", "X-Client-Role": "pm"}


class _Sched:
    """Answers a job as the scheduler does: prefill progress, the answer,
    then usage with the timing and the request's id (scheduler._done)."""

    def __init__(self):
        self.host = NS(state="ready", path=None, tokenizer=None)
        self.fail = False

    def ahead(self, job):
        return 0

    def submit(self, job):
        job.submitted = time.perf_counter()
        if self.fail:
            job.outbox.put(("error", RuntimeError("the engine broke")))
            return job
        job.outbox.put(("progress", (512, 1024)))
        job.outbox.put(("progress", (1024, 1024)))
        job.outbox.put(("delta", Delta(content="hello", finish="stop")))
        job.outbox.put(("done", {
            "prompt_tokens": 1024, "completion_tokens": 3,
            "prompt_tokens_details": {"cached_tokens": 256},
            "knurlogic": {"request_id": job.request_id, "timing": {
                "queue_s": 0.002, "queue_ms": 2.0, "prefill_ms": 400.0,
                "decode_ms": 50.0, "prefill_tps": 1920.0,
                "decode_tps": 40.0}}}))
        return job


@pytest.fixture
def server():
    sched = _Sched()
    app = S.App(sched, served=lambda: {"id": "tiny"})
    srv = S.make_server(app, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", sched
    srv.shutdown()


def _post(url, body, headers=None):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode() if not isinstance(body, bytes)
        else body, headers={"Content-Type": "application/json",
                            **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.headers.get("X-Request-Id"), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("X-Request-Id"), e.read()


def _row(rid, wait=2.0):
    """The ledger row; written at the handler's close, just after the
    last byte the client read."""
    end = time.time() + wait
    while time.time() < end:
        rows = [r for r in L.ledger().rows(limit=1000) if r["id"] == rid]
        if rows:
            return rows[0]
        time.sleep(0.02)
    raise AssertionError(f"no ledger row {rid}")


def _sse(raw: bytes) -> list:
    """[(event or None, data dict or str)] of an SSE body."""
    out = []
    for block in raw.decode().split("\n\n"):
        ev, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()
        if data is None:
            continue
        try:
            data = json.loads(data)
        except ValueError:
            pass
        out.append((ev, data))
    return out


MSG = [{"role": "user", "content": "hi"}]

# (path, body, how the final usage is found) for each API, both ways
APIS = {
    "chat": ("/v1/chat/completions", {"messages": MSG},
             lambda d: d["usage"]),
    "completions": ("/v1/completions", {"prompt": "hi"},
                    lambda d: d["usage"]),
    "messages": ("/v1/messages", {"messages": MSG, "max_tokens": 9},
                 lambda d: d["usage"]),
    "responses": ("/v1/responses", {"input": "hi"}, lambda d: d["usage"]),
    "ollama": ("/api/chat", {"messages": MSG, "stream": False},
               lambda d: d["usage"]),
}


@pytest.mark.parametrize("api", sorted(APIS))
def test_every_api_names_its_ledger_row(server, api):
    url, _ = server
    path, body, usage_of = APIS[api]
    code, rid, raw = _post(url + path, body, LABELS)
    assert code == 200, raw
    assert ULID.match(rid)
    kn = usage_of(json.loads(raw))["knurlogic"]
    assert kn["request_id"] == rid
    assert TIMING <= set(kn["timing"])
    row = _row(rid)
    assert row["api"] == api and row["model"] == "tiny"
    assert (row["client"], row["session"], row["run"], row["role"]) == \
        ("client/0.0.1", "sess-1", "worker-2", "pm")
    assert row["key_id"] == "anonymous"
    assert row["prompt_tokens"] == 1024 and row["output_tokens"] == 3
    assert row["cached_tokens"] == 256
    assert row["prefill_tps"] == 1920.0 and row["queue_ms"] == 2.0
    assert row["finish"] == "stop" and row["status"] == 200


@pytest.mark.parametrize("api", sorted(APIS))
def test_a_client_request_id_is_echoed_and_names_the_row(server, api):
    url, _ = server
    path, body, usage_of = APIS[api]
    mine = f"client-{api}-7"
    code, rid, raw = _post(url + path, body, {"X-Request-Id": mine})
    assert code == 200 and rid == mine, raw
    assert usage_of(json.loads(raw))["knurlogic"]["request_id"] == mine
    assert _row(mine)["api"] == api


def _final_usage(api, raw):
    if api == "ollama":
        last = json.loads(raw.decode().strip().splitlines()[-1])
        return last["usage"]
    evs = _sse(raw)
    if api == "messages":
        return [d for e, d in evs if e == "message_delta"][-1]["usage"]
    if api == "responses":
        return [d for e, d in evs
                if e == "response.completed"][-1]["response"]["usage"]
    return [d for e, d in evs
            if isinstance(d, dict) and d.get("usage")][-1]["usage"]


@pytest.mark.parametrize("api", sorted(APIS))
def test_every_api_streamed_carries_the_id_and_timing(server, api):
    url, _ = server
    path, body, _ = APIS[api]
    body = dict(body, stream=True,
                stream_options={"include_usage": True})
    code, rid, raw = _post(url + path, body)
    assert code == 200 and ULID.match(rid)
    kn = _final_usage(api, raw)["knurlogic"]
    assert kn["request_id"] == rid and TIMING <= set(kn["timing"])
    row = _row(rid)
    assert row["client"] is None and row["finish"] == "stop"


def _progress(raw):
    return [d for e, d in _sse(raw) if e == "knurlogic.progress"]


def test_progress_events_on_an_opted_in_openai_stream(server):
    url, _ = server
    code, rid, raw = _post(url + "/v1/chat/completions",
                           {"messages": MSG, "stream": True},
                           {"X-Client": "client/0.0.1"})
    ev = _progress(raw)
    assert [(e["phase"], e["done"], e["total"]) for e in ev] == \
        [("prefill", 512, 1024), ("prefill", 1024, 1024)]
    assert all(e["request_id"] == rid and e["queue"] == {"ahead": 0}
               for e in ev)
    assert ev[0]["tps"] is None           # one chunk seen: no rate yet
    assert b": keepalive 512/1024" in raw          # the comment stays
    # before the first token
    assert raw.index(b"knurlogic.progress") < raw.index(b"hello")


def test_no_progress_events_for_an_openai_client_that_did_not_opt_in(
        server):
    """The OpenAI SDK hands an unknown event's data on as a chunk (it has
    no `choices`), so an anonymous OpenAI-shaped stream gets none."""
    url, _ = server
    for path, body in (("/v1/chat/completions", {"messages": MSG}),
                       ("/v1/responses", {"input": "hi"})):
        _, _, raw = _post(url + path, dict(body, stream=True))
        assert _progress(raw) == [] and b"knurlogic.progress" not in raw


def test_responses_stream_carries_progress_when_opted_in(server):
    url, _ = server
    _, rid, raw = _post(url + "/v1/responses", {"input": "hi",
                                                "stream": True},
                        {"X-Client": "client/0.0.1"})
    ev = _progress(raw)
    assert len(ev) == 2 and ev[0]["type"] == "knurlogic.progress"


def test_messages_stream_always_carries_progress(server):
    """The Anthropic SDK skips event names it does not know."""
    url, _ = server
    _, rid, raw = _post(url + "/v1/messages",
                        {"messages": MSG, "max_tokens": 9, "stream": True})
    ev = _progress(raw)
    assert [e["done"] for e in ev] == [512, 1024]
    assert all(e["request_id"] == rid for e in ev)
    assert [e for e, _ in _sse(raw)].count("ping") == 2   # still pinged


def test_ollama_and_non_streamed_requests_get_no_progress(server):
    url, _ = server
    _, _, raw = _post(url + "/api/chat", {"messages": MSG},
                      {"X-Client": "c/1"})
    assert b"knurlogic.progress" not in raw
    _, _, raw = _post(url + "/v1/chat/completions", {"messages": MSG},
                      {"X-Client": "c/1"})
    assert b"knurlogic.progress" not in raw


def test_the_openai_sdk_reads_an_anonymous_stream(server):
    openai = pytest.importorskip("openai")
    url, _ = server
    client = openai.OpenAI(base_url=url + "/v1", api_key="x")
    chunks = list(client.chat.completions.create(
        model="tiny", messages=MSG, stream=True,
        stream_options={"include_usage": True}))
    text = "".join(c.choices[0].delta.content or "" for c in chunks
                   if c.choices)
    assert text == "hello"
    assert all(hasattr(c, "choices") for c in chunks)


def test_queued_streams_say_how_many_are_ahead(monkeypatch):
    monkeypatch.setattr(S, "PROGRESS_S", 0.01)
    monkeypatch.setattr(S, "QUEUED_KEEPALIVE_S", 0.02)

    class Job:
        request_id = "01J0000000000000000000000Q"
        cancelled = False

        def __init__(self):
            self.outbox = queue.Queue()

        def cancel(self):
            self.cancelled = True

    from knurlogic.interfaces.http.openai import Reply
    job = Job()
    reply = Reply(job, {"chat": True, "model": "m", "progress": True})
    gen = S._queued(job, reply, NS(ahead=lambda j: 3))
    out = [next(gen) for _ in range(4)]
    ev = [json.loads(c.split(b"data: ")[1]) for c in out
          if c.startswith(b"event: knurlogic.progress")]
    assert ev and ev[0]["phase"] == "queue" and \
        ev[0]["queue"] == {"ahead": 3}
    assert ev[0]["request_id"] == job.request_id
    assert b": keepalive queued\n\n" in out
    gen.close()
    assert job.cancelled


def test_an_error_and_a_refusal_are_rows_too(server):
    url, sched = server
    code, rid, _ = _post(url + "/v1/chat/completions", b"{nope", LABELS)
    assert code == 400 and ULID.match(rid)
    row = _row(rid)
    assert row["finish"] == "error" and row["status"] == 400
    assert row["session"] == "sess-1"
    sched.fail = True
    code, rid, _ = _post(url + "/v1/messages",
                         {"messages": MSG, "max_tokens": 9})
    assert code == 500 and _row(rid)["finish"] == "error"


def test_a_stream_the_client_left_is_cancelled(server):
    """A stream that never reaches its done event closes as cancelled."""
    rec = T.Request("chat", {})
    job = NS(cancelled=True)
    rec.jobs.append(job)
    rec.status = 200
    assert rec.row()["finish"] == "cancelled"


def test_other_routes_mint_no_id(server):
    url, _ = server
    code, rid, _ = _post(url + "/v1/nothing", {})
    assert code == 404 and rid is None


def test_labels_are_cut_to_their_byte_limits_never_refused(server):
    url, _ = server
    long = {"X-Client": "c" * 300, "X-Client-Session": "s" * 129,
            "X-Client-Run": "r" * 128, "X-Client-Role": "role" * 20}
    code, rid, _ = _post(url + "/v1/chat/completions", {"messages": MSG},
                         long)
    assert code == 200
    row = _row(rid)
    assert row["client"] == "c" * 128 and row["session"] == "s" * 128
    assert row["run"] == "r" * 128 and row["role"] == ("role" * 20)[:32]


def test_models_advertises_the_telemetry_version(server):
    url, _ = server
    with urllib.request.urlopen(url + "/v1/models", timeout=5) as r:
        doc = json.loads(r.read())
    assert doc["knurlogic"]["telemetry"] == T.VERSION == 1


def test_usage_rolls_up_by_session_and_run(server):
    url, _ = server
    for sess, run in (("a", "1"), ("a", "2"), ("b", "1")):
        _, rid, _ = _post(url + "/v1/chat/completions", {"messages": MSG},
                          {"X-Client-Session": sess, "X-Client-Run": run})
        _row(rid)
    with urllib.request.urlopen(url + "/v1/usage?group=session",
                                timeout=5) as r:
        doc = json.loads(r.read())
    by = {s["session"]: s for s in doc["summary"]}
    assert by["a"]["requests"] == 2 and by["b"]["requests"] == 1
    assert by["a"]["output_tokens"] == 6
    with urllib.request.urlopen(url + "/v1/usage?group=run",
                                timeout=5) as r:
        by = {s["run"]: s["requests"]
              for s in json.loads(r.read())["summary"]}
    assert by == {"1": 2, "2": 1}
    try:
        urllib.request.urlopen(url + "/v1/usage?group=prompt", timeout=5)
        raise AssertionError("an unknown group is refused")
    except urllib.error.HTTPError as e:
        assert e.code == 400


def test_no_request_text_reaches_the_ledger(server):
    url, _ = server
    sentinel = "SENTINEL-7c1f-PROMPT-TEXT"
    _, rid, _ = _post(url + "/v1/chat/completions",
                      {"messages": [{"role": "user", "content": sentinel}]})
    _row(rid)
    L.ledger().close()
    blob = b"".join(p.read_bytes() for p in L.path().parent.iterdir()
                    if p.name.startswith("ledger.db"))
    assert sentinel.encode() not in blob
