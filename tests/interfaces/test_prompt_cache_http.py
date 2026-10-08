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


# --- sessions: drop, pin, the registry, a session's save -----------------------

def _cmd(result):
    done = threading.Event()
    done.set()
    return NS(done=done, error="", result=result)


class _SessionSched(_Sched):
    def __init__(self, entries=(), key_id=None):
        super().__init__()
        self.calls = []
        self.entries, self.key_id = list(entries), key_id

    def save_prompt_cache(self, session=None):
        self.calls.append(("save", session))
        return _cmd({"saved": 1, "kept": 0, "skipped": 0, "bytes": 1,
                     "why": [], "entries": 1, "not_worth": 0})

    def drop_prompt_cache(self, session):
        self.calls.append(("drop", session))
        return _cmd({"session": session, "memory": 2, "disk": 3})

    def pin_prompt_cache(self, session, pinned):
        self.calls.append(("pin", session, pinned))
        return _cmd({"session": session, "pinned": pinned, "entries": 1})

    def list_prompt_cache(self):
        return _cmd({"entries": self.entries, "key_id": self.key_id,
                     "model": "tiny"})


def _serve(sched):
    srv = S.make_server(S.App(sched, served=lambda: {"id": "tiny"}),
                        "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def test_drop_pin_and_a_sessions_save_reach_the_scheduler():
    sched = _SessionSched()
    srv, url = _serve(sched)
    try:
        st, _, body = _post(url + "/v1/prompt-cache/drop", {"session": "a"})
        assert st == 200 and body["memory"] == 2 and body["disk"] == 3
        st, _, body = _post(url + "/v1/prompt-cache/pin",
                            {"session": "a", "pinned": False})
        assert st == 200 and body["pinned"] is False
        st, _, body = _post(url + "/v1/prompt-cache/save", {"session": "a"})
        assert st == 200 and body["session"] == "a"
        assert body["not_worth"] == 0
        st, _, _ = _post(url + "/v1/prompt-cache/drop", {})
        assert st == 400
        st, _, _ = _post(url + "/v1/prompt-cache/pin",
                         {"session": "a", "pinned": "yes"})
        assert st == 400
        assert sched.calls == [("drop", "a"), ("pin", "a", False),
                               ("save", "a")]
    finally:
        srv.shutdown()


def test_the_registry_lists_memory_and_disk_once_each():
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models.cache import KVCache

    from knurlogic.engine.serve import prompt_disk as D
    key = D.identity("/nonexistent/tiny")
    d = D.root() / D.key_id(key)

    def kv():
        c = KVCache()
        k = mx.ones((1, 2, 4, 32))
        c.update_and_fetch(k, k)
        return [c]
    own = {"session": "s1", "role": "main", "run": "r"}
    D.save_entry(d, key, [1, 2, 3, 4], kv(), "user", 1, 0, owner=own,
                 model="tiny")
    D.save_entry(d, key, [5, 2, 3, 4], kv(), "user", 1, 1, owner=own,
                 model="tiny")
    both = {"session": "s1", "role": "main", "run": "r", "tokens": 4,
            "bytes": 10, "in_memory": True, "on_disk": True,
            "saved_at": 1.0, "pinned": False,
            "hash": D.tokens_hash([1, 2, 3, 4])}
    mem_only = dict(both, session="s2", on_disk=False,
                    hash=D.tokens_hash([9, 9]))
    sched = _SessionSched([both, mem_only], D.key_id(key))
    srv, url = _serve(sched)
    try:
        with urllib.request.urlopen(url + "/v1/prompt-cache") as r:
            rows = json.loads(r.read())["data"]
    finally:
        srv.shutdown()
    assert len(rows) == 3
    by = {r["hash"]: r for r in rows}
    b = by[D.tokens_hash([1, 2, 3, 4])]
    assert b["in_memory"] and b["on_disk"] and b["model"] == "tiny"
    m = by[D.tokens_hash([9, 9])]
    assert m["in_memory"] and not m["on_disk"]
    o = by[D.tokens_hash([5, 2, 3, 4])]
    assert not o["in_memory"] and o["on_disk"] and o["session"] == "s1"
    assert o["role"] == "main" and o["tokens"] == 4 and o["bytes"] > 0


def test_the_owner_labels_and_retention_reach_the_job():
    seen = []

    class Sched(_Sched):
        def submit(self, job):
            seen.append(job)
            return super().submit(job)
    srv, url = _serve(Sched())
    body = json.dumps({"messages": [{"role": "user",
                                     "content": "x"}]}).encode()
    try:
        urllib.request.urlopen(urllib.request.Request(
            url + "/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "X-Client-Session": "x" * 300, "X-Client-Role": "sub",
                     "X-Cache-Retain": "pin"}), timeout=10).read()
        urllib.request.urlopen(urllib.request.Request(
            url + "/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"}), timeout=10).read()
    finally:
        srv.shutdown()
    a, b = seen
    assert a.session == "x" * 128 and a.role == "sub" and a.pin
    assert b.session is None and not b.pin
