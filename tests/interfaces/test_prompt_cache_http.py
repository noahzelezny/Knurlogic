"""The disk prompt cache over HTTP: POST /v1/prompt-cache/save, and the
ledger's disk_tokens (engine/serve/prompt_disk; usage.knurlogic.cache.disk)."""
import json
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace as NS

import pytest

from knurlogic.engine.runtime.request import Delta
from knurlogic.interfaces.http import server as S
from knurlogic.machine import ledger as L


class _Sched:
    def __init__(self):
        self.host = NS(state="ready", path="/models/tiny", tokenizer=None)
        self.saves = 0

    def ahead(self, job):
        return 0

    def save_prompt_cache(self):
        self.saves += 1
        done = threading.Event()
        done.set()
        return NS(done=done, error="", result={
            "saved": 2, "kept": 1, "skipped": 0, "bytes": 1024,
            "why": [], "seconds": 0.1})

    def submit(self, job):
        job.submitted = time.perf_counter()
        job.outbox.put(("delta", Delta(content="hi", finish="stop")))
        job.outbox.put(("done", {
            "prompt_tokens": 1000, "completion_tokens": 1,
            "prompt_tokens_details": {"cached_tokens": 900},
            "knurlogic": {"request_id": job.request_id, "timing": {},
                          "cache": {"used": 900, "disk": {
                              "tokens": 900, "read_ms": 12.5}}}}))
        return job


@pytest.fixture
def server():
    sched = _Sched()
    srv = S.make_server(S.App(sched, served=lambda: {"id": "tiny"}),
                        "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", sched
    srv.shutdown()


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.headers.get("X-Request-Id"), json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, None, json.loads(e.read())


def test_a_save_on_request(server):
    url, sched = server
    st, _, body = _post(url + "/v1/prompt-cache/save", {})
    assert st == 200 and sched.saves == 1
    assert body["object"] == "prompt_cache.save" and body["model"] == "tiny"
    assert body["saved"] == 2 and "why" not in body


def test_disk_tokens_reach_the_ledger(server):
    url, _ = server
    st, rid, body = _post(url + "/v1/chat/completions",
                          {"messages": [{"role": "user", "content": "x"}]})
    assert st == 200
    assert body["usage"]["knurlogic"]["cache"]["disk"]["tokens"] == 900
    end = time.time() + 2
    while time.time() < end:
        rows = [r for r in L.ledger().rows(limit=100) if r["id"] == rid]
        if rows:
            break
        time.sleep(0.02)
    assert rows[0]["disk_tokens"] == 900


def test_an_older_ledger_gains_the_column(tmp_path):
    f = tmp_path / "old.db"
    db = sqlite3.connect(str(f))
    db.executescript(L._SCHEMA.replace(",\n  disk_tokens INTEGER", ""))
    db.close()
    led = L.Ledger(f)
    led.insert({"id": "a", "ts_start": 1.0, "key_id": "anonymous",
                "disk_tokens": 5})
    assert led.rows(limit=5)[0]["disk_tokens"] == 5
