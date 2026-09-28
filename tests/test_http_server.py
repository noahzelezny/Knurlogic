"""knurlogic's own HTTP server on the tiny model, over real sockets: the
OpenAI shapes, SSE framing, errors, the Anthropic surface in-process, and
Scout's endpoints. Templates and thinking are the conformance suite's job
(tests/api, on a real model); this is the wire."""
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

mx = pytest.importorskip("mlx.core")

from test_batch_drafting import _tiny  # noqa: E402
from test_scheduler import Host, Tok  # noqa: E402


class DTok(Tok):
    def decode(self, ids):
        return "".join(f"<{i}>" for i in ids)


@pytest.fixture(scope="module")
def url():
    """One server for the module, never stopped (see test_scheduler)."""
    import mlx.nn as nn
    from knurlogic.engine.runtime.scheduler import Scheduler
    from knurlogic.engine.serve import state
    from knurlogic.interfaces.http import scout
    from knurlogic.interfaces.http.server import App, make_server
    model, head, prompts = _tiny(512)
    mx.eval([v if isinstance(v, mx.array) else v.parameters()
             for v in vars(head).values()
             if isinstance(v, (mx.array, nn.Module))])
    state.DRAFT.update(head=head, on=True)
    host = Host(model, DTok(prompts))
    host.path, host.loaded_at = "/models/tiny", 0.0
    host.status = lambda: {"state": "ready", "model": host.path, "error": ""}
    sched = Scheduler(host, prefill_step_size=16).start()
    app = App(sched, served=lambda: {"id": "tiny", "capabilities": ["text"],
                                     "size_bytes": 1},
              concurrency=lambda: scout.concurrency(sched))
    app.translate = None                      # no template to translate
    srv = make_server(app, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", prompts
    state.DRAFT.update(head=None, on=False)


def _msg(ids):
    return [{"role": "user", "content": " ".join(map(str, ids))}]


def post(url, path, body):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def sse(raw: bytes):
    out = []
    for ev in raw.split(b"\n\n"):
        for line in ev.split(b"\n"):
            if line.startswith(b"data:"):
                d = line[5:].strip()
                out.append("[DONE]" if d == b"[DONE]" else json.loads(d))
    return out


def test_a_chat_completion_has_the_openai_shape_and_the_hint(url):
    u, prompts = url
    code, headers, raw = post(u, "/v1/chat/completions",
                              {"messages": _msg(prompts[0]),
                               "max_tokens": 6})
    assert code == 200
    r = json.loads(raw)
    assert r["object"] == "chat.completion"
    ch = r["choices"][0]
    assert ch["message"]["role"] == "assistant" and ch["message"]["content"]
    assert ch["finish_reason"] == "length"
    assert r["usage"]["completion_tokens"] == 6
    assert r["usage"]["prompt_tokens"] == len(prompts[0])
    assert headers.get("X-Knurlogic-Concurrency", "").startswith("rows=")


def test_streaming_frames_deltas_then_usage_then_done(url):
    u, prompts = url
    code, headers, raw = post(u, "/v1/chat/completions",
                              {"messages": _msg(prompts[1]), "max_tokens": 5,
                               "stream": True,
                               "stream_options": {"include_usage": True}})
    assert code == 200 and "text/event-stream" in headers["Content-Type"]
    ev = sse(raw)
    assert ev[-1] == "[DONE]"
    chunks = [e for e in ev if isinstance(e, dict)]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    text = "".join((c["choices"][0]["delta"].get("content") or "")
                   for c in chunks if c["choices"])
    assert text.count("<") == 5
    assert chunks[-1]["usage"]["completion_tokens"] == 5
    finals = [c["choices"][0]["finish_reason"] for c in chunks if c["choices"]]
    assert finals[-1] == "length"


def test_a_text_completion(url):
    u, prompts = url
    code, _, raw = post(u, "/v1/completions",
                        {"prompt": " ".join(map(str, prompts[1])),
                         "max_tokens": 3})
    r = json.loads(raw)
    assert code == 200 and r["object"] == "text_completion"
    assert r["choices"][0]["text"].count("<") == 3


def test_a_stop_string_across_tokens_is_honoured(url):
    u, prompts = url
    _, _, raw = post(u, "/v1/chat/completions",
                     {"messages": _msg(prompts[2]), "max_tokens": 8})
    free = json.loads(raw)["choices"][0]["message"]["content"]
    cut = free.index(">", free.index(">") + 1) + 1
    stop = free[cut:cut + 3]                 # spans a token boundary
    _, _, raw = post(u, "/v1/chat/completions",
                     {"messages": _msg(prompts[2]), "max_tokens": 8,
                      "stop": stop})
    r = json.loads(raw)
    assert r["choices"][0]["message"]["content"] == free[:cut]
    assert r["choices"][0]["finish_reason"] == "stop"


@pytest.mark.parametrize("body,param", [
    ({"n": 2}, "n"),
    ({"temperature": -1}, "temperature"),
    ({"max_tokens": "many"}, "max_tokens"),
    ({"stop": 5}, "stop"),
    ({"messages": []}, "messages"),
])
def test_bad_requests_get_openai_error_objects(url, body, param):
    u, prompts = url
    full = {"messages": _msg(prompts[0]), **body}
    code, _, raw = post(u, "/v1/chat/completions", full)
    e = json.loads(raw)["error"]
    assert code == 400 and e["param"] == param
    assert e["type"] == "invalid_request_error" and e["message"]


def test_max_completion_tokens_is_accepted(url):
    u, prompts = url
    _, _, raw = post(u, "/v1/chat/completions",
                     {"messages": _msg(prompts[0]),
                      "max_completion_tokens": 2})
    assert json.loads(raw)["usage"]["completion_tokens"] == 2


def test_a_render_failure_is_a_400_before_any_stream(url):
    u, _ = url
    code, headers, raw = post(u, "/v1/chat/completions",
                              {"messages": [{"role": "user",
                                             "content": "FAIL"}],
                               "stream": True})
    assert code == 400 and "render" in json.loads(raw)["error"]["message"]


def test_images_to_a_text_model_are_refused(url):
    u, prompts = url
    code, _, raw = post(u, "/v1/chat/completions", {"messages": [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;"
                                                "base64,AAAA"}},
            {"type": "text", "text": "hi"}]}]})
    assert code == 400 and "image" in raw.decode().lower()


def test_seeded_requests_repeat_and_differ_by_seed(url):
    u, prompts = url

    def one(seed):
        _, _, raw = post(u, "/v1/chat/completions",
                         {"messages": _msg(prompts[0]), "max_tokens": 10,
                          "temperature": 1.0, "seed": seed})
        return json.loads(raw)["choices"][0]["message"]["content"]
    assert one(5) == one(5) and one(5) != one(6)


def test_messages_is_served_in_process(url):
    u, prompts = url
    code, _, raw = post(u, "/v1/messages",
                        {"model": "tiny", "max_tokens": 4,
                         "messages": [{"role": "user", "content":
                                       " ".join(map(str, prompts[0]))}]})
    r = json.loads(raw)
    assert code == 200 and r["type"] == "message"
    assert r["content"][0]["type"] == "text" and r["content"][0]["text"]
    assert r["usage"]["output_tokens"] == 4


def test_messages_streams_in_process(url):
    u, prompts = url
    code, _, raw = post(u, "/v1/messages",
                        {"model": "tiny", "max_tokens": 4, "stream": True,
                         "messages": [{"role": "user", "content":
                                       " ".join(map(str, prompts[0]))}]})
    kinds = [e.get("type") for e in sse(raw) if isinstance(e, dict)]
    assert kinds[0] == "message_start" and kinds[-1] == "message_stop"
    assert "content_block_delta" in kinds


def test_models_and_health(url):
    u, _ = url
    with urllib.request.urlopen(u + "/v1/models") as r:
        d = json.loads(r.read())
    assert d["object"] == "list" and d["data"][0]["id"] == "tiny"
    assert d["data"][0]["capabilities"] == ["text"]
    assert isinstance(d["data"][0]["context_length"], int)
    with urllib.request.urlopen(u + "/health") as r:
        assert json.loads(r.read())["status"] == "ok"


def test_concurrent_requests_all_answer(url):
    u, prompts = url
    out = []

    def one(p):
        code, _, raw = post(u, "/v1/chat/completions",
                            {"messages": _msg(p), "max_tokens": 6})
        out.append((code, json.loads(raw)))
    ts = [threading.Thread(target=one, args=(p,)) for p in prompts * 2]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(out) == 6 and all(c == 200 for c, _ in out)


def _raw(url, head: bytes) -> bytes:
    import socket
    from urllib.parse import urlparse
    u = urlparse(url)
    s = socket.create_connection((u.hostname, u.port), timeout=30)
    s.sendall(head)
    out = b""
    while True:
        b = s.recv(65536)
        if not b:
            break
        out += b
    s.close()
    return out


def test_a_body_over_the_cap_is_a_413_before_it_is_read(url):
    u, _ = url
    out = _raw(u, b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                  b"Content-Type: application/json\r\n"
                  b"Content-Length: 99999999999\r\n\r\n{")
    assert out.split(b" ", 2)[1] == b"413"
    assert b"maximum" in out and b"--max-request-mib" in out


def test_a_malformed_content_length_is_a_400_not_a_dropped_connection(url):
    u, _ = url
    out = _raw(u, b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                  b"Content-Length: lots\r\n\r\n")
    assert out.split(b" ", 2)[1] == b"400" and b"not a number" in out


def _req(url, path, headers, body=b'{"messages": []}'):
    req = urllib.request.Request(url + path, data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def test_a_foreign_web_page_cannot_drive_the_server(url):
    """A page open in the browser posts text/plain (no preflight) with its
    Origin: refused, and nothing grants it CORS."""
    u, _ = url
    code, headers, body = _req(u, "/v1/ensure",
                               {"Content-Type": "text/plain",
                                "Origin": "https://evil.example"})
    assert code == 403 and b"--allow-origin" in body
    assert "Access-Control-Allow-Origin" not in headers


def test_the_servers_own_page_and_plain_clients_are_answered(url):
    u, prompts = url
    host = u.split("//", 1)[1]
    ok = {"Content-Type": "application/json"}
    body = json.dumps({"messages": _msg(prompts[0]),
                       "max_tokens": 1}).encode()
    assert _req(u, "/v1/chat/completions", ok, body)[0] == 200
    assert _req(u, "/v1/chat/completions",
                dict(ok, Origin=f"http://{host}"), body)[0] == 200


def test_a_rebound_domain_is_refused_by_host(url):
    u, _ = url
    code, _, body = _req(u, "/health", {"Host": "attacker.example:80"},
                         body=None)
    assert code == 403 and b"DNS-rebinding" in body


def test_host_is_local_accepts_this_machine_only():
    from knurlogic.interfaces.http.server import host_is_local
    for h in ("localhost:8080", "127.0.0.1:1", "[::1]:80", "10.0.0.2:8098",
              "mac.local", "foo.localhost"):
        assert host_is_local(h), h
    for h in ("evil.example", "evil.example:8080", "127.0.0.1.nip.io"):
        assert not host_is_local(h), h
    # this machine's name is accepted exactly, never as a first label: a
    # rebinding domain named after the machine must not pass
    import socket
    me = socket.gethostname().lower().split(".")[0]
    assert host_is_local(f"{me}:8080")
    assert not host_is_local(f"{me}.attacker.example:8080")


@pytest.mark.parametrize("msgs", [["hi"], [{"content": "no role"}],
                                  [{"role": "user", "content": 5}],
                                  [{"role": "user", "content": ["x"]}]])
def test_malformed_messages_are_a_400(url, msgs):
    u, _ = url
    code, _, raw = post(u, "/v1/chat/completions", {"messages": msgs})
    assert code == 400 and json.loads(raw)["error"]["param"] == "messages"


def test_a_route_that_raises_answers_500_with_a_body(url, monkeypatch):
    u, _ = url
    from knurlogic.interfaces.http import openai as O
    monkeypatch.setattr(O, "models_document",
                        lambda *a: 1 / 0)
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(u + "/v1/models", timeout=30)
    assert e.value.code == 500
    assert "ZeroDivisionError" in json.loads(e.value.read())["error"]["message"]


def test_the_cluster_gate_refuses_what_it_does_not_allow(url, monkeypatch):
    u, _ = url
    from knurlogic.interfaces.http import server as S

    class Closed:
        def allows(self, ip):
            return False

        def refusal(self, ip):
            return b"thunderbolt only"
    handler = [c for c in S.Handler.__subclasses__()][-1]
    monkeypatch.setattr(handler.app, "gate", Closed())
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(u + "/health", timeout=30)
    assert e.value.code == 403 and e.value.read() == b"thunderbolt only"


def test_an_allowed_host_name_is_answered():
    from knurlogic.interfaces.http.server import browser_refusal
    h = {"Host": "studio.tail1234.ts.net:8080"}
    assert "--allow-host studio.tail1234.ts.net" in browser_refusal(h)
    assert browser_refusal(h, allow_hosts=("studio.tail1234.ts.net",)) \
        is None


def test_a_silent_request_samples_as_the_model_recommends(tmp_path):
    """Silence meant greedy, which Qwen's thinking models are documented
    to loop under; the model's generation_config fills what the request
    leaves out, and the answer says which."""
    import json
    from knurlogic.interfaces.http import openai as O
    from knurlogic.machine.artifact import sampling_defaults
    (tmp_path / "generation_config.json").write_text(json.dumps(
        {"do_sample": True, "temperature": 0.6, "top_p": 0.95,
         "top_k": 20}))
    model = sampling_defaults(tmp_path)
    assert model == {"temp": 0.6, "top_p": 0.95, "top_k": 20}
    body = {"messages": [{"role": "user", "content": "hi"}]}
    job, ctx = O.build_job(body, chat=True, sampling_defaults=model)
    assert job.sampling == {"temp": 0.6, "top_p": 0.95, "top_k": 20}
    assert ctx["sampling"]["from_model"] == ["temperature", "top_p",
                                             "top_k"]
    # what the request says wins, key by key
    job, ctx = O.build_job(dict(body, temperature=0), chat=True,
                           sampling_defaults=model)
    assert job.sampling["temp"] == 0.0 and job.sampling["top_p"] == 0.95
    # a model that says do_sample false, or nothing, stays greedy
    (tmp_path / "generation_config.json").write_text(json.dumps(
        {"do_sample": False, "temperature": 0.6}))
    assert sampling_defaults(tmp_path) == {}
    job, _ = O.build_job(body, chat=True, sampling_defaults={})
    assert job.sampling == {"temp": 0.0}


def test_count_tokens_is_the_prompt_the_model_would_see(url):
    """Claude Code asks /v1/messages/count_tokens to manage its context."""
    u, _ = url
    body = {"model": "x", "max_tokens": 8,
            "messages": [{"role": "user", "content": "1 2 3"}]}
    code, _, raw = post(u, "/v1/messages/count_tokens", body)
    n = json.loads(raw)["input_tokens"]
    code2, _, raw2 = post(u, "/v1/messages/count_tokens", dict(
        body, messages=[{"role": "user", "content": "1 2 3 4 5 6"}]))
    assert code == code2 == 200
    assert json.loads(raw2)["input_tokens"] == n + 3 > 3


def test_models_carries_the_context_window(tmp_path):
    """/v1/models says how long a context the model takes, from its
    config (text_config's for a multimodal wrapper); the page offers
    max_tokens up to it. 0 when the config does not say."""
    import json
    from knurlogic.interfaces.http import openai as O
    from knurlogic.machine.artifact import context_length
    assert context_length(tmp_path) == 0
    (tmp_path / "config.json").write_text(json.dumps(
        {"text_config": {"max_position_embeddings": 262144}}))
    assert context_length(tmp_path) == 262144
    (tmp_path / "config.json").write_text(json.dumps(
        {"max_position_embeddings": 32768,
         "text_config": {"max_position_embeddings": 4096}}))
    n = context_length(tmp_path)
    assert n == 32768
    doc = O.models_document({"id": "m"}, {}, n)
    assert doc["data"][0]["context_length"] == 32768
    assert O.models_document({"id": "m"})["data"][0]["context_length"] == 0


def test_the_request_counter_does_not_lose_concurrent_increments(
        monkeypatch):
    """Found by Qwen3.8-Flash-Next-6bit (cluster shootout 2026-09-27):
    `self.requests += 1` ran unlocked on every handler thread. The count
    now goes through one locked step; many threads, no lost update."""
    import threading
    from types import SimpleNamespace
    from knurlogic.interfaces.http import server as S
    app = S.App.__new__(S.App)
    app.requests = 0
    app._count_lock = threading.Lock()
    ts = [threading.Thread(target=lambda: [app._count() for _ in range(5000)])
          for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert app.requests == 40000
