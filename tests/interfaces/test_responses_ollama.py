"""The OpenAI Responses and Ollama adapters, against a stub OpenAI engine
(as test_messages does for Anthropic): the translation in both directions,
non-streaming and streaming, tools, images, thinking, refusals."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from knurlogic.interfaces.http import messages as M
from knurlogic.interfaces.http import ollama as OL
from knurlogic.interfaces.http import responses as R

MARKER = "from-the-stub-engine"
USAGE = {"prompt_tokens": 7, "completion_tokens": 11,
         "knurlogic": {"timing": {"queue_s": 0.1, "ttft_s": 0.5,
                                  "prefill_tok_s": 70.0,
                                  "decode_tok_s": 20.0}}}
CALL = {"id": "call_1", "type": "function",
        "function": {"name": "read", "arguments": '{"path":"a.py"}'}}

STREAM = (
    {"choices": [{"delta": {"reasoning_content": "hm"}}]},
    {"choices": [{"delta": {"content": MARKER}}]},
    {"choices": [{"delta": {"tool_calls": [
        {"index": 0, "id": "call_1",
         "function": {"name": "read", "arguments": '{"path":'}}]}}]},
    {"choices": [{"delta": {"tool_calls": [
        {"index": 0, "function": {"arguments": '"a.py"}'}}]}}]},
    {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    {"choices": [], "usage": USAGE},
)


def _engine(oai: dict):
    engine.seen = oai
    if oai.get("stream"):
        return [f"data: {json.dumps(c)}\n\n".encode() for c in STREAM] + \
            [b"data: [DONE]\n\n"]
    return {"choices": [{"message": {
        "role": "assistant", "content": MARKER, "reasoning_content": "hm",
        "tool_calls": [CALL]}, "finish_reason": "tool_calls"}],
        "usage": USAGE}


engine = _engine


def _call(handler, req, transport=_engine):
    out, started = [], {}
    handler(transport, "served")(json.dumps(req).encode(), out.append,
                                 lambda code, ctype, headers=None:
                                 started.update(code=code, ctype=ctype))
    return started, b"".join(out)


def _events(raw):
    return [json.loads(line[5:]) for line in raw.decode().split("\n")
            if line.startswith("data:")]


def _responses(req, transport=_engine):
    return _call(R.handler_over, req, transport)


def _ollama(req, generate=False, transport=_engine):
    out, started = [], {}
    OL.handler_over(transport, "served", generate)(
        json.dumps(req).encode(), out.append,
        lambda code, ctype, headers=None: started.update(code=code,
                                                         ctype=ctype))
    return started, b"".join(out)


# --- Responses --------------------------------------------------------------

def test_responses_translation_of_input_items_and_tools():
    o = R.to_chat({
        "model": "m", "instructions": "be brief", "max_output_tokens": 9,
        "temperature": 0.3, "reasoning": {"effort": "low"},
        "tools": [{"type": "function", "name": "read", "description": "d",
                   "parameters": {"type": "object", "properties": {}}}],
        "input": [
            {"role": "developer", "content": "rules"},
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "look"},
                {"type": "input_image", "image_url": "data:image/png;base64,QQ"}]},
            {"type": "function_call", "call_id": "c1", "name": "read",
             "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1",
             "output": "print(1)"}]})
    m = o["messages"]
    assert m[0] == {"role": "system", "content": "be brief"}
    assert m[1] == {"role": "system", "content": "rules"}
    assert m[2]["content"][1]["image_url"]["url"].startswith("data:image")
    assert m[3]["tool_calls"][0]["id"] == "c1"
    assert m[4] == {"role": "tool", "tool_call_id": "c1",
                    "content": "print(1)"}
    assert o["max_tokens"] == 9 and o["temperature"] == 0.3
    assert o["reasoning_effort"] == "low"
    assert o["tools"][0]["function"]["name"] == "read"


def test_responses_non_streaming_shape():
    s, raw = _responses({"model": "m", "input": "hi", "stream": False})
    r = json.loads(raw)
    assert s["code"] == 200 and r["object"] == "response"
    assert r["status"] == "completed" and r["id"].startswith("resp_")
    kinds = [i["type"] for i in r["output"]]
    assert kinds == ["reasoning", "message", "function_call"]
    assert r["output"][0]["summary"][0]["text"] == "hm"
    assert r["output"][1]["content"][0]["text"] == MARKER
    fc = r["output"][2]
    assert (fc["call_id"], fc["name"]) == ("call_1", "read")
    assert json.loads(fc["arguments"]) == {"path": "a.py"}
    assert r["usage"] == {"input_tokens": 7, "output_tokens": 11,
                          "total_tokens": 18,
                          "input_tokens_details": {"cached_tokens": 0},
                          "output_tokens_details": {"reasoning_tokens": 0}}


def test_responses_streaming_event_order():
    s, raw = _responses({"input": "hi", "stream": True})
    ev = _events(raw)
    names = [e["type"] for e in ev]
    assert s["ctype"] == "text/event-stream"
    assert names[0] == "response.created" and names[-1] == \
        "response.completed"
    for want in ("response.reasoning_summary_text.delta",
                 "response.output_text.delta", "response.output_text.done",
                 "response.function_call_arguments.delta",
                 "response.function_call_arguments.done",
                 "response.output_item.done"):
        assert want in names
    assert [e["sequence_number"] for e in ev] == list(range(len(ev)))
    done = ev[-1]["response"]
    assert [i["type"] for i in done["output"]] == \
        ["reasoning", "message", "function_call"]
    assert done["output"][1]["content"][0]["text"] == MARKER
    assert json.loads(done["output"][2]["arguments"]) == {"path": "a.py"}
    assert done["usage"]["input_tokens"] == 7


@pytest.mark.parametrize("extra", [{"previous_response_id": "resp_1"},
                                   {"store": True},
                                   {"tools": [{"type": "web_search"}]},
                                   {"input": []}])
def test_responses_refusals_are_plain_400s(extra):
    s, raw = _responses({"input": "hi", **extra})
    assert s["code"] == 400
    assert json.loads(raw)["error"]["type"] == "invalid_request_error"


def test_responses_engine_refusal_keeps_its_status():
    def dead(oai):
        raise M.TransportError(503, "no model", retry_after=5)
    s, raw = _responses({"input": "hi"}, dead)
    assert s["code"] == 503 and json.loads(raw)["error"]["message"] == \
        "no model"


def test_responses_length_is_incomplete():
    def cut(oai):
        return {"choices": [{"message": {"content": "x"},
                             "finish_reason": "length"}], "usage": {}}
    r = json.loads(_responses({"input": "hi"}, cut)[1])
    assert r["status"] == "incomplete"
    assert r["incomplete_details"] == {"reason": "max_output_tokens"}


# --- Ollama -----------------------------------------------------------------

def test_ollama_chat_translation_images_tools_options():
    o = OL.to_chat({
        "model": "m", "stream": False,
        "options": {"temperature": 0.2, "num_predict": 5, "stop": ["x"]},
        "messages": [
            {"role": "user", "content": "what is this", "images": ["QQ=="]},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "read", "arguments": {"p": 1}}}]},
            {"role": "tool", "tool_name": "read", "content": "ok"}],
        "tools": [{"type": "function", "function": {"name": "read"}}]})
    m = o["messages"]
    assert m[0]["content"][1]["image_url"]["url"] == \
        "data:image/png;base64,QQ=="
    assert json.loads(m[1]["tool_calls"][0]["function"]["arguments"]) == \
        {"p": 1}
    assert m[2]["tool_call_id"] == m[1]["tool_calls"][0]["id"]
    assert o["max_tokens"] == 5 and o["temperature"] == 0.2
    assert o["stop"] == ["x"] and o["stream"] is False


def test_ollama_chat_streams_ndjson_by_default_and_ends_done():
    s, raw = _ollama({"model": "m", "messages": [
        {"role": "user", "content": "hi"}]})
    assert s["ctype"] == "application/x-ndjson"
    lines = [json.loads(x) for x in raw.decode().splitlines()]
    assert engine.seen["stream"] is True
    assert lines[0]["message"]["thinking"] == "hm"
    assert lines[1]["message"]["content"] == MARKER and not lines[1]["done"]
    call = [x for x in lines if x.get("message", {}).get("tool_calls")]
    assert call[0]["message"]["tool_calls"][0]["function"] == \
        {"name": "read", "arguments": {"path": "a.py"}}
    last = lines[-1]
    assert last["done"] and last["done_reason"] == "stop"
    assert last["prompt_eval_count"] == 7 and last["eval_count"] == 11
    assert last["prompt_eval_duration"] == 400_000_000       # ttft - queue
    assert last["eval_duration"] == 500_000_000              # 10 / 20 tok/s


def test_ollama_chat_non_streaming_single_object():
    s, raw = _ollama({"model": "m", "stream": False, "messages": [
        {"role": "user", "content": "hi"}]})
    r = json.loads(raw)
    assert s["ctype"] == "application/json" and r["done"] is True
    assert r["message"]["role"] == "assistant"
    assert r["message"]["content"] == MARKER
    assert r["eval_count"] == 11 and r["total_duration"] > 0


def test_ollama_generate_non_streaming_and_streaming():
    r = json.loads(_ollama({"prompt": "hi", "system": "s", "stream": False},
                           generate=True)[1])
    assert r["response"] == MARKER and r["done"]
    assert engine.seen["messages"][0] == {"role": "system", "content": "s"}
    _, raw = _ollama({"prompt": "hi"}, generate=True)
    lines = [json.loads(x) for x in raw.decode().splitlines()]
    assert "".join(x["response"] for x in lines if not x["done"]) == MARKER
    assert lines[-1]["done"] and lines[-1]["response"] == ""


@pytest.mark.parametrize("req,gen", [({"messages": []}, False),
                                     ({"messages": "x"}, False),
                                     ({"prompt": "x", "raw": True}, True)])
def test_ollama_bad_requests_are_400_with_error_string(req, gen):
    s, raw = _ollama(req, gen)
    assert s["code"] == 400 and isinstance(json.loads(raw)["error"], str)


def test_ollama_model_documents(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(
        {"model_type": "gemma4", "quantization": {"bits": 8}}))
    served = {"id": "m", "size_bytes": 5, "created": 100,
              "capabilities": ["text", "vision"]}
    t = OL.tags_document(served, tmp_path)["models"][0]
    assert t["name"] == "m" and t["size"] == 5
    assert t["details"]["family"] == "gemma4"
    assert t["details"]["quantization_level"] == "Q8"
    s = OL.show_document(served, tmp_path, 4096, {"dialect": "x"})
    assert s["capabilities"] == ["completion", "tools", "vision", "thinking"]
    assert s["model_info"]["gemma4.context_length"] == 4096
