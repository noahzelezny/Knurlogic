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
              concurrency=lambda: scout.concurrency(sched),
              image_limit=scout.image_limit)
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
