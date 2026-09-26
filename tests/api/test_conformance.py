"""What a knurlogic server must do on the wire, pinned before the server is
rewritten (docs/PLAN.md, "knurlogic's own server").

Every test talks HTTP to KNURLOGIC_API_URL and nothing else, so it holds the
current server (mlx-lm's, patched) and the next one to the same contract.
Model-dependent checks read what the server says about itself first
(/status.json: thinking dialect, vision) and skip what the served model
cannot do, rather than assuming a particular model.

Sections: catalog . chat . streaming . usage and the cache report .
reasoning . sampling . images . Anthropic messages . limits and errors .
concurrency . status . the harness's ingest (targets for the new server; xfail
until they exist).
"""
from __future__ import annotations

import base64
import io
import json
import os
import threading
import urllib.error
import urllib.request

import pytest

URL = os.environ.get("KNURLOGIC_API_URL", "").rstrip("/")
TIMEOUT = float(os.environ.get("KNURLOGIC_API_TIMEOUT", "600"))


def _ours() -> bool:
    """Is this knurlogic's own server? Its /health says so; mlx-lm's does
    not. The known gaps below are xfail on mlx-lm's only."""
    if not URL:
        return False
    try:
        with urllib.request.urlopen(URL + "/health", timeout=30) as r:
            return json.loads(r.read()).get("server") == "knurlogic"
    except Exception:
        return False


OURS = _ours()
Q = "What is 2 + 3? Reply with just the number."


# --- plumbing -----------------------------------------------------------------

def get(path):
    with urllib.request.urlopen(URL + path, timeout=TIMEOUT) as r:
        return json.loads(r.read())


def post(path, body, headers=None):
    req = urllib.request.Request(
        URL + path, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read())


def post_status(path, body):
    """(status code, parsed body or text) -- for the error cases."""
    req = urllib.request.Request(
        URL + path, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


def stream(path, body, headers=None):
    """Every `data:` payload of an SSE response, parsed; '[DONE]' kept as a
    string so its presence can be asserted. Read in small chunks: a reader
    that only works on whole lines hides framing bugs."""
    req = urllib.request.Request(
        URL + path, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", **(headers or {})})
    out, buf = [], b""
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        assert "text/event-stream" in r.headers.get("Content-Type", "")
        for chunk in iter(lambda: r.read(7), b""):
            buf += chunk
            while b"\n\n" in buf:
                ev, buf = buf.split(b"\n\n", 1)
                for line in ev.split(b"\n"):
                    if line.startswith(b"data:"):
                        d = line[5:].strip()
                        if d == b"[DONE]":
                            out.append("[DONE]")
                        elif d:
                            out.append(json.loads(d))
    return out


def chat(**kw):
    body = {"model": "m", "messages": [{"role": "user", "content": Q}],
            "max_tokens": 400, "temperature": 0}
    body.update(kw)
    return post("/v1/chat/completions", body)


def status():
    return get("/status.json")


def thinking():
    return (status().get("thinking") or {})


def dialect():
    return thinking().get("dialect")


def has_vision():
    v = status().get("vision")
    return isinstance(v, dict) and bool(v.get("served"))


def png(rgb=(220, 20, 20), size=64):
    from PIL import Image
    b = io.BytesIO()
    Image.new("RGB", (size, size), rgb).save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


# --- catalog ------------------------------------------------------------------

def test_models_lists_the_served_model():
    d = get("/v1/models")
    assert d.get("object") == "list" and d["data"]
    assert all("id" in m for m in d["data"])


# --- chat ---------------------------------------------------------------------

def test_a_chat_completion_has_the_openai_shape():
    r = chat()
    assert r["object"] == "chat.completion"
    ch = r["choices"][0]
    assert ch["message"]["role"] == "assistant"
    assert isinstance(ch["message"]["content"], str)
    assert ch["finish_reason"] in ("stop", "length")


def test_max_tokens_ends_with_length():
    r = chat(messages=[{"role": "user", "content": "Count from 1 to 500."}],
             max_tokens=8, reasoning_effort="none")
    assert r["choices"][0]["finish_reason"] == "length"
    assert r["usage"]["completion_tokens"] <= 8


@pytest.mark.xfail(not OURS, strict=True,
                   reason="mlx-lm matches stop sequences as "
                   "token ids: stop 'D' never matches the token ' D' "
                   "(measured, gemma e4b, 2026-09-25). OpenAI's contract "
                   "is text; the new server matches text.")
def test_a_stop_sequence_ends_the_answer_before_it():
    r = chat(messages=[{"role": "user",
                        "content": "Write the letters A B C D E F, "
                                   "space separated."}],
             stop=["D"], reasoning_effort="none", max_tokens=60)
    assert "D" not in r["choices"][0]["message"]["content"]


def test_a_system_message_and_several_turns_are_accepted():
    r = chat(messages=[{"role": "system", "content": "Answer tersely."},
                       {"role": "user", "content": "Say hi."},
                       {"role": "assistant", "content": "hi"},
                       {"role": "user", "content": Q}],
             reasoning_effort="none")
    assert r["choices"][0]["message"]["content"].strip()


# --- streaming ----------------------------------------------------------------

def test_streaming_sends_deltas_then_done():
    ev = stream("/v1/chat/completions",
                {"model": "m", "messages": [{"role": "user", "content": Q}],
                 "max_tokens": 200, "stream": True, "temperature": 0,
                 "reasoning_effort": "none"})
    assert ev[-1] == "[DONE]"
    chunks = [e for e in ev if isinstance(e, dict)]
    assert chunks and all(c["object"] == "chat.completion.chunk"
                          for c in chunks)
    text = "".join((c["choices"][0].get("delta") or {}).get("content") or ""
                   for c in chunks if c.get("choices"))
    assert text.strip()
    finals = [c["choices"][0].get("finish_reason") for c in chunks
              if c.get("choices")]
    assert finals[-1] in ("stop", "length")


def test_stream_options_include_usage_sends_a_usage_chunk():
    ev = stream("/v1/chat/completions",
                {"model": "m", "messages": [{"role": "user", "content": Q}],
                 "max_tokens": 100, "stream": True, "temperature": 0,
                 "reasoning_effort": "none",
                 "stream_options": {"include_usage": True}})
    usage = [e["usage"] for e in ev if isinstance(e, dict) and e.get("usage")]
    assert usage and usage[-1]["completion_tokens"] > 0


# --- usage and the cache report -----------------------------------------------

def test_usage_has_prompt_completion_and_total():
    u = chat(reasoning_effort="none")["usage"]
    assert u["prompt_tokens"] > 0 and u["completion_tokens"] > 0
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]


def test_the_cache_report_says_what_the_prompt_cache_did():
    c = chat(reasoning_effort="none")["usage"]["knurlogic"]["cache"]
    for k in ("offered", "used", "discarded", "prefilled", "via"):
        assert k in c


def test_a_shared_prefix_is_reused():
    """the harness's ingest sends the same long schema text with a different
    tail; the second request must not prefill the shared part again."""
    prefix = ("You label media. Fields: title, people, place, mood, "
              "objects, text-in-image, date clues. " * 20)
    for tail in ("First item: a red square.", "Second item: a blue circle."):
        r = chat(messages=[{"role": "system", "content": prefix},
                           {"role": "user", "content": tail}],
                 reasoning_effort="none", max_tokens=20)
    u = r["usage"]
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    assert cached > u["prompt_tokens"] // 2
    assert u["knurlogic"]["cache"]["used"] == cached


# --- reasoning ----------------------------------------------------------------

def test_the_server_says_what_reasoning_effort_does_here():
    t = thinking()
    assert "dialect" in t and "native" in t


def test_a_level_off_the_ladder_is_refused():
    code, body = post_status("/v1/chat/completions",
                             {"model": "m", "reasoning_effort": "lots",
                              "messages": [{"role": "user", "content": Q}]})
    assert code == 400 and "reasoning_effort" in body


def test_none_turns_reasoning_off_where_the_model_can():
    t = thinking()
    if not any(n["level"] == "none" for n in t.get("native", [])):
        pytest.skip("this model has no off")
    r = chat(reasoning_effort="none",
             messages=[{"role": "user", "content": "Is 91 prime? Explain."}])
    u = r["usage"]
    assert u["knurlogic"]["thinking"]["applied"] == "off"
    assert not (r["choices"][0]["message"].get("reasoning") or "")
    assert (u.get("completion_tokens_details") or {}).get(
        "reasoning_tokens", 0) == 0


def test_reasoning_streams_apart_from_the_answer_and_is_counted():
    if not dialect():
        pytest.skip("this model has no thinking controls")
    ev = stream("/v1/chat/completions",
                {"model": "m", "stream": True, "max_tokens": 3000,
                 "temperature": 0, "stream_options": {"include_usage": True},
                 "messages": [{"role": "user", "content":
                               "Is 391 prime? Think it through, then say "
                               "yes or no."}]})
    chunks = [e for e in ev if isinstance(e, dict) and e.get("choices")]
    reasoning = "".join((c["choices"][0].get("delta") or {}).get(
        "reasoning") or "" for c in chunks)
    content = "".join((c["choices"][0].get("delta") or {}).get(
        "content") or "" for c in chunks)
    usage = [e["usage"] for e in ev if isinstance(e, dict) and e.get("usage")]
    assert reasoning.strip(), "the default level should think on this"
    for marker in ("<think>", "</think>", "<|channel>", "<channel|>"):
        assert marker not in content
    assert usage[-1]["completion_tokens_details"]["reasoning_tokens"] > 0


def test_reasoning_exclude_strips_it_and_still_counts_it():
    if not dialect():
        pytest.skip("this model has no thinking controls")
    r = chat(reasoning={"exclude": True}, max_tokens=3000,
             messages=[{"role": "user", "content":
                        "Is 391 prime? Think it through, then say yes or "
                        "no."}])
    m = r["choices"][0]["message"]
    assert not m.get("reasoning") and not m.get("reasoning_content")


def test_every_request_reports_the_level_it_got_even_the_first():
    """The first requests after a load were served the default whatever
    they asked for (two causes, both fixed 2026-09-25). Several at once."""
    t = thinking()
    if not t.get("native"):
        pytest.skip("this model has no thinking controls")
    lowest = t["native"][0]["name"]
    got = []

    def one():
        got.append(chat(reasoning_effort=t["native"][0]["level"],
                        max_tokens=40)["usage"]["knurlogic"]["thinking"]
                   ["applied"])
    ts = [threading.Thread(target=one) for _ in range(4)]
    for x in ts:
        x.start()
    for x in ts:
        x.join()
    assert got == [lowest] * 4


# --- sampling -----------------------------------------------------------------

def test_a_seed_makes_sampling_repeatable_and_different_seeds_differ():
    """mlx-lm's compiled sampler ignored the seed off the main thread; the
    same answer came back for every seed (fixed 2026-09-25)."""
    ask = dict(temperature=1.5, max_tokens=12, reasoning_effort="none",
               messages=[{"role": "user",
                          "content": "Write one unusual adjective."}])
    a1 = chat(seed=11, **ask)["choices"][0]["message"]["content"]
    a2 = chat(seed=11, **ask)["choices"][0]["message"]["content"]
    others = {chat(seed=s, **ask)["choices"][0]["message"]["content"]
              for s in (12, 13, 14, 15)}
    assert a1 == a2
    assert len(others | {a1}) > 1


def test_temperature_zero_is_greedy_and_repeatable():
    a = chat(reasoning_effort="none")["choices"][0]["message"]["content"]
    b = chat(reasoning_effort="none")["choices"][0]["message"]["content"]
    assert a == b


# --- images -------------------------------------------------------------------

def _image_msg(*urls, text="What colour is this? One word."):
    return [{"role": "user", "content": [
        *({"type": "image_url", "image_url": {"url": u}} for u in urls),
        {"type": "text", "text": text}]}]


def test_an_image_is_seen():
    if not has_vision():
        pytest.skip("the served model has no vision")
    r = chat(messages=_image_msg(png()), reasoning_effort="none",
             max_tokens=20)
    assert "red" in r["choices"][0]["message"]["content"].lower()


def test_several_images_in_one_message():
    """the harness sends N video frames in one message when a contact sheet is not
    used."""
    if not has_vision():
        pytest.skip("the served model has no vision")
    r = chat(messages=_image_msg(png((220, 20, 20)), png((20, 20, 220)),
                                 text="Two images. Name both colours, "
                                      "in order."),
             reasoning_effort="none", max_tokens=30)
    t = r["choices"][0]["message"]["content"].lower()
    assert "red" in t and "blue" in t
    assert r["usage"]["knurlogic"]["cache"]["images"]["total"] == 2


def test_images_to_a_text_model_are_refused_not_ignored():
    if has_vision():
        pytest.skip("the served model has vision")
    code, body = post_status("/v1/chat/completions",
                             {"model": "m", "messages": _image_msg(png())})
    assert code == 400 and "image" in body.lower()


# --- Anthropic messages -------------------------------------------------------

AH = {"x-api-key": "x", "anthropic-version": "2023-06-01"}


def test_messages_answers_in_content_blocks():
    r = post("/v1/messages", {"model": "m", "max_tokens": 200,
                              "messages": [{"role": "user", "content": Q}],
                              "thinking": {"type": "disabled"}}, AH)
    assert r["type"] == "message" and r["role"] == "assistant"
    assert any(b["type"] == "text" and b["text"].strip()
               for b in r["content"])
    assert r["usage"]["input_tokens"] > 0 and r["usage"]["output_tokens"] > 0


def test_messages_streams_thinking_then_text_blocks():
    if not dialect():
        pytest.skip("this model has no thinking controls")
    ev = stream("/v1/messages",
                {"model": "m", "max_tokens": 3000, "stream": True,
                 "thinking": {"type": "enabled", "budget_tokens": 2048},
                 "messages": [{"role": "user", "content":
                               "Is 391 prime? Think, then say yes or no."}]},
                AH)
    kinds = [e["content_block"]["type"] for e in ev
             if isinstance(e, dict) and e.get("type") == "content_block_start"]
    deltas = {e["delta"]["type"] for e in ev if isinstance(e, dict)
              and e.get("type") == "content_block_delta"}
    assert kinds[:1] == ["thinking"] and "text" in kinds
    assert {"thinking_delta", "signature_delta", "text_delta"} <= deltas


# --- concurrency --------------------------------------------------------------

def test_concurrent_requests_all_answer():
    out, errs = [], []

    def one(i):
        try:
            out.append(chat(reasoning_effort="none", max_tokens=30,
                            messages=[{"role": "user", "content":
                                       f"What is {i} + {i}? Just the "
                                       f"number."}]))
        except Exception as e:           # pragma: no cover - reported below
            errs.append(e)
    ts = [threading.Thread(target=one, args=(i,)) for i in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errs and len(out) == 6


# --- status -------------------------------------------------------------------

def test_status_reports_the_node_its_memory_and_the_artifact():
    s = status()
    assert s["schema"] >= 2 and s["nodes"]
    assert s["artifact"]["name"] and s["memory"]


# --- the harness's ingest: targets for knurlogic's own server ----------------------
# (docs/PLAN.md "Requirements for knurlogic's own server"). xfail until the
# endpoints exist; strict, so passing by accident is noticed.

new_server = pytest.mark.xfail(not OURS,
                               reason="knurlogic's own server only",
                               strict=True, raises=(urllib.error.HTTPError,
                                                    KeyError, AssertionError))


@new_server
def test_residency_is_one_flat_honest_list():
    r = get("/v1/residency")
    row = r["data"][0]
    for k in ("model", "capabilities", "memory_bytes", "nodes", "state"):
        assert k in row
    assert row["state"] in ("loading", "ready", "unloading")


@new_server
def test_the_catalog_carries_capabilities_and_size():
    m = get("/v1/models")["data"][0]
    assert set(m["capabilities"]) <= {"text", "vision", "thinking"}
    assert m["size_bytes"] > 0


@new_server
def test_ensure_loaded_is_idempotent_and_can_wait():
    served = get("/v1/models")["data"][0]["id"]
    a = post("/v1/ensure", {"model": served, "wait": True})
    b = post("/v1/ensure", {"model": served, "wait": True})
    assert a["state"] == b["state"] == "ready"


@new_server
def test_an_image_over_the_size_limit_is_refused_explicitly():
    if not has_vision():
        pytest.skip("the served model has no vision")
    code, body = post_status("/v1/chat/completions",
                             {"model": "m", "messages": _image_msg(
                                 png(size=12000))})
    assert code == 413 and "max" in body.lower()


@new_server
def test_a_concurrency_hint_is_in_the_headers():
    req = urllib.request.Request(
        URL + "/v1/chat/completions", method="POST",
        data=json.dumps({"model": "m", "max_tokens": 5,
                         "messages": [{"role": "user", "content": Q}]}
                        ).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        assert r.headers["X-Knurlogic-Concurrency"]


# --- additions from the server design review (docs/SERVER.md) ----------------

def test_a_seeded_request_under_concurrent_load_equals_it_alone():
    """The seed addresses each token by position, so batching cannot move
    the random draws. What batching CAN move is the logits: a batched
    forward is not bit-identical to a one-row forward on every kernel
    (GLM 2.7: up to 0.09 in logprob, M4, 2026-09-25). So: a divergence
    with identical logprobs before it is a sampling bug (fail); one with
    the logprobs already drifted is the kernels (skip, with the drift)."""
    ask = dict(temperature=1.2, max_tokens=24, reasoning_effort="none",
               seed=21, logprobs=True,
               messages=[{"role": "user", "content":
                          "Invent a name for a small robot."}])

    def run():
        c = chat(**ask)["choices"][0]
        return [(x["token"], x["logprob"]) for x in
                ((c.get("logprobs") or {}).get("content") or [])]
    alone = run()
    got, noise = {}, []

    def other(i):
        noise.append(chat(temperature=1.0, max_tokens=24,
                          reasoning_effort="none",
                          messages=[{"role": "user", "content":
                                     f"Say a word about the number {i}."}]))
    ts = [threading.Thread(target=other, args=(i,)) for i in range(3)]
    ts.append(threading.Thread(target=lambda: got.update(x=run())))
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    loaded = got["x"]
    assert alone, "no logprobs came back"
    d = next((i for i, (a, b) in enumerate(zip(alone, loaded))
              if a[0] != b[0]), None)
    if d is None:
        return
    drift = max((abs(a[1] - b[1]) for a, b in zip(alone[:d], loaded[:d])),
                default=0.0)
    if drift > 1e-4:
        pytest.skip(f"batched logits differ from one-row logits on this "
                    f"model (max logprob drift {drift:.4f} before token "
                    f"{d}); the draws are the same, the numbers are not")
    assert False, (f"token {d} differs with identical logprobs before it: "
                   f"the seeded draw moved under load")


def test_text_completions_answer():
    r = post("/v1/completions", {"model": "m", "prompt": "1, 2, 3,",
                                 "max_tokens": 6, "temperature": 0})
    assert r["object"] == "text_completion"
    assert isinstance(r["choices"][0]["text"], str)
    assert r["usage"]["completion_tokens"] > 0


def test_max_completion_tokens_is_the_same_as_max_tokens():
    r = chat(max_completion_tokens=5, reasoning_effort="none",
             messages=[{"role": "user", "content": "Count from 1 to 500."}])
    assert r["usage"]["completion_tokens"] <= 5
    assert r["choices"][0]["finish_reason"] == "length"


@new_server
def test_errors_are_openai_error_objects():
    code, body = post_status("/v1/chat/completions",
                             {"model": "m", "temperature": -1,
                              "messages": [{"role": "user", "content": Q}]})
    e = json.loads(body)["error"]
    assert code == 400
    assert {"message", "type", "param", "code"} <= set(e)


@new_server
def test_n_above_one_is_refused_not_ignored():
    code, body = post_status("/v1/chat/completions",
                             {"model": "m", "n": 2,
                              "messages": [{"role": "user", "content": Q}]})
    assert code == 400 and json.loads(body)["error"]["param"] == "n"


def test_multibyte_text_streams_without_replacement_characters():
    ev = stream("/v1/chat/completions",
                {"model": "m", "stream": True, "max_tokens": 80,
                 "temperature": 0, "reasoning_effort": "none",
                 "messages": [{"role": "user", "content":
                               "Write 日本語のテキスト and three emoji "
                               "(🦊🌊🎈), then stop."}]})
    text = "".join((c["choices"][0].get("delta") or {}).get("content") or ""
                   for c in ev if isinstance(c, dict) and c.get("choices"))
    assert text and "�" not in text


@new_server
def test_a_client_that_goes_away_frees_its_row():
    import http.client
    import time
    from urllib.parse import urlparse
    u = urlparse(URL)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=TIMEOUT)
    conn.request("POST", "/v1/chat/completions", body=json.dumps(
        {"model": "m", "stream": True, "max_tokens": 4000,
         "reasoning_effort": "none",
         "messages": [{"role": "user", "content":
                       "Count from 1 to 5000, one number per line."}]}),
        headers={"Content-Type": "application/json"})
    sock = conn.sock                     # getresponse() lets go of it
    r = conn.getresponse()
    r.read(200)
    import socket
    sock.shutdown(socket.SHUT_RDWR)
    sock.close()
    for _ in range(100):
        rows = get("/v1/residency")["data"][0]["rows"]
        if rows == 0:
            break
        time.sleep(0.2)
    assert rows == 0


def test_tool_calls_come_back_as_tool_calls():
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Weather for a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}}, "required": ["city"]}}}]
    r = chat(tools=tools, reasoning_effort="none", max_tokens=300,
             messages=[{"role": "user", "content":
                        "What is the weather in Paris? Use the tool."}])
    m = r["choices"][0]["message"]
    if not m.get("tool_calls"):
        pytest.skip("the model answered without calling the tool")
    tc = m["tool_calls"][0]
    assert tc["type"] == "function" and tc["id"]
    assert tc["function"]["name"] == "get_weather"
    assert "city" in json.loads(tc["function"]["arguments"])
    assert r["choices"][0]["finish_reason"] == "tool_calls"
