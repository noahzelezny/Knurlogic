"""The Anthropic Messages adapter, against a stub OpenAI engine.

A harness pointed at a server with ANTHROPIC_BASE_URL speaks this shape.
mlx-lm's server answers only /v1/chat/completions, so without this
`knurlogic serve` cannot back one -- exo can, which is the whole difference.

The stub answers in the OpenAI shape with values nothing else could produce,
so a translation that quietly invented content would fail rather than pass.
"""

import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knurlogic.interfaces import messages as M

MARKER = "from-the-stub-engine"


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]; s.close(); return p


class _Engine(BaseHTTPRequestHandler):
    """Answers /v1/chat/completions, streaming or not, like mlx-lm would."""
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n) or b"{}")
        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            for chunk in (
                {"choices": [{"delta": {"content": MARKER}}]},
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "id": "call_1",
                     "function": {"name": "read", "arguments": '{"path":'}}]}}]},
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "function": {"arguments": '"a.py"}'}}]}}]},
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                 "usage": {"prompt_tokens": 7, "completion_tokens": 11}},
            ):
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.close_connection = True
            return
        body = json.dumps({
            "choices": [{"message": {"role": "assistant", "content": MARKER},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5},
            "echo": req,
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def _engine():
    port = _free_port()
    srv = ThreadingHTTPServer(("127.0.0.1", port), _Engine)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{port}/v1/chat/completions"


def _call(url, req):
    out, started = [], {}

    def write(b):
        out.append(b)

    def start(code, ctype):
        started.update(code=code, ctype=ctype)

    M.handler(url, model="test-model")(json.dumps(req).encode(), write, start)
    return started, b"".join(out)


# --- translation ------------------------------------------------------------

def test_tool_results_become_openai_tool_messages():
    """The shapes genuinely differ: Anthropic puts a tool result inside a
    USER turn, OpenAI makes it its own message keyed to the call id."""
    o = M.to_openai({
        "model": "m", "system": "be brief",
        "messages": [
            {"role": "user", "content": "read foo.py"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "read",
                 "input": {"path": "foo.py"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1",
                 "content": "print(1)"}]}]})
    assert o["messages"][0] == {"role": "system", "content": "be brief"}
    call = o["messages"][2]["tool_calls"][0]
    assert call["function"]["name"] == "read"
    assert json.loads(call["function"]["arguments"]) == {"path": "foo.py"}
    assert o["messages"][3] == {"role": "tool", "tool_call_id": "t1",
                                "content": "print(1)"}


def test_tools_are_translated_into_function_schemas():
    o = M.to_openai({"messages": [], "tools": [
        {"name": "read", "description": "read a file",
         "input_schema": {"type": "object",
                          "properties": {"path": {"type": "string"}}}}]})
    f = o["tools"][0]["function"]
    assert f["name"] == "read" and f["parameters"]["properties"]["path"]


def test_a_tool_call_sets_stop_reason_tool_use():
    out = M.from_openai({"choices": [{"message": {
        "content": None, "tool_calls": [
            {"id": "c1", "function": {"name": "read",
                                      "arguments": '{"path":"a"}'}}]},
        "finish_reason": "tool_calls"}]}, "m")
    assert out["stop_reason"] == "tool_use"
    block = out["content"][0]
    assert block["type"] == "tool_use" and block["input"] == {"path": "a"}


def test_invalid_tool_json_is_passed_through_not_dropped():
    """A model emitting bad JSON is a model problem. Dropping the call would
    turn it into a silent one."""
    out = M.from_openai({"choices": [{"message": {"tool_calls": [
        {"id": "c1", "function": {"name": "read", "arguments": "{not json"}}]},
        "finish_reason": "tool_calls"}]}, "m")
    assert out["content"][0]["input"]["_raw"] == "{not json"


# --- over the wire ----------------------------------------------------------

def test_non_streaming_round_trip():
    srv, url = _engine()
    try:
        started, body = _call(url, {"model": "m", "max_tokens": 16,
                                    "messages": [{"role": "user",
                                                  "content": "hi"}]})
    finally:
        srv.shutdown()
    assert started["code"] == 200
    out = json.loads(body)
    assert out["type"] == "message" and out["role"] == "assistant"
    assert out["content"][0]["text"] == MARKER, "must be the engine's answer"
    assert out["usage"] == {"input_tokens": 3, "output_tokens": 5}


def test_streaming_emits_the_event_order_a_harness_parses():
    """The order is the contract. A stream that delivered the right tokens in
    the wrong frames would look fine in a terminal and break a harness."""
    srv, url = _engine()
    try:
        started, body = _call(url, {"model": "m", "stream": True,
                                    "messages": [{"role": "user",
                                                  "content": "hi"}]})
    finally:
        srv.shutdown()
    assert started["ctype"] == "text/event-stream"
    events = [l[7:] for l in body.decode().splitlines()
              if l.startswith("event: ")]
    assert events[0] == "message_start"
    assert events[-1] == "message_stop"
    assert events[-2] == "message_delta"
    assert events.count("content_block_start") == 2      # text, then tool_use
    assert events.count("content_block_stop") == 2

    data = [json.loads(l[6:]) for l in body.decode().splitlines()
            if l.startswith("data: ")]
    text = "".join(d["delta"]["text"] for d in data
                   if d.get("delta", {}).get("type") == "text_delta")
    assert text == MARKER
    # Tool arguments stream as raw JSON fragments and are reassembled by the
    # client, so they must be forwarded verbatim rather than re-serialised.
    partial = "".join(d["delta"]["partial_json"] for d in data
                      if d.get("delta", {}).get("type") == "input_json_delta")
    assert json.loads(partial) == {"path": "a.py"}
    assert data[-2]["delta"]["stop_reason"] == "tool_use"


def test_a_dead_engine_answers_an_anthropic_shaped_error():
    started, body = _call("http://127.0.0.1:1/v1/chat/completions",
                          {"messages": []})
    assert started["code"] == 502
    assert json.loads(body)["error"]["type"] == "api_error"


# --- which tool dialect an artifact speaks ----------------------------------
# Tool calling is not one format. Across 54 artifacts on this machine:
# 40 qwen3_coder (<tool_call><function=NAME><parameter=P>), 7 glm47,
# 3 gemma4, 1 json_tools, 3 whose template never mentions tools. The engine
# picks a parser by inferring it from the chat template, and when the
# inference misses it returns None -- at which point tool calls arrive as
# prose and a harness sees a model that describes the function it would call
# instead of calling it.

def test_the_template_is_read_from_either_place_an_artifact_keeps_it(tmp_path):
    """Newer exports put it in chat_template.jinja and leave the tokenizer
    config's field empty; older ones do the opposite. Reading one answers
    'no template' for half the artifacts here."""
    from knurlogic.machine.artifact import Artifact
    (tmp_path / "config.json").write_text('{"model_type":"x"}')
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "OLD STYLE {{ messages }}"}))
    a = Artifact.load(tmp_path)
    assert "OLD STYLE" in a.chat_template()

    (tmp_path / "chat_template.jinja").write_text("NEW STYLE {{ messages }}")
    assert "NEW STYLE" in Artifact.load(tmp_path).chat_template()


def test_a_template_asking_for_tools_with_no_parser_is_flagged():
    """The failure this catches is silent: nothing errors, the model just
    talks about calling functions."""
    from knurlogic.engine import seam as engine
    ts = engine.tool_support(
        "You have tools. Emit <weird_custom_tag>name</weird_custom_tag>.")
    assert ts["mentions_tools"] is True
    assert ts["parser"] is None


def test_the_agentic_dialect_is_recognised():
    """<tool_call>\\n<function=NAME>\\n<parameter=P> is the Qwen3-Coder /
    agentic-harness form, not the JSON that plain Qwen emits."""
    from knurlogic.engine import seam as engine
    ts = engine.tool_support(
        "reply in the following format:\n\n<tool_call>\n<function=example>\n"
        "<parameter=p>v</parameter>\n</function>\n</tool_call>")
    assert ts["parser"] == "qwen3_coder"


def test_no_template_is_not_reported_as_no_tools():
    from knurlogic.engine import seam as engine
    ts = engine.tool_support("")
    assert ts["has_template"] is False and ts["mentions_tools"] is False


# --- pointing a client at the server ----------------------------------------

def test_connect_offers_the_scoped_config_and_not_the_global_one():
    """Writing ~/.claude/settings.json would route every session on the
    machine at a local model, including the ones with nothing to do with it.
    A config change nobody can see is how you debug the wrong thing."""
    from knurlogic.interfaces import connect
    out = connect.render("http://127.0.0.1:8080", "some-artifact")
    assert ".claude/settings.json" in out
    assert "NOT your global" in out


def test_the_timeout_is_raised_because_a_local_model_is_slower():
    """The default timeout is the first thing to bite on a long tool loop."""
    from knurlogic.interfaces import connect
    env = connect.env_lines("http://x", "m")
    assert int(env["API_TIMEOUT_MS"]) > 600_000
    assert env["ANTHROPIC_BASE_URL"] == "http://x"
    # All three model slots, or the harness falls back to a hosted name that
    # this server has never heard of.
    assert all(env[f"ANTHROPIC_DEFAULT_{k}_MODEL"] == "m"
               for k in ("OPUS", "SONNET", "HAIKU"))
