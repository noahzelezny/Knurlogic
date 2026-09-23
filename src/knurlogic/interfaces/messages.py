"""The Anthropic Messages API, as a translation over the engine's OpenAI one.

WHY THIS EXISTS. A coding harness -- Claude Code among them -- speaks the
Anthropic Messages shape and is pointed at a server with ANTHROPIC_BASE_URL.
exo implements `/v1/messages` and can therefore back one; mlx-lm's server
answers only `/v1/chat/completions`, so `knurlogic serve` could not, and the
gap is a translation rather than an inference problem.

IT IS A TRANSLATION, NOT A SECOND INFERENCE PATH. The engine already owns
chat templates, stop sequences, streaming and tool-call parsing, and a second
implementation of any of that would drift from the first. So this converts a
request into the OpenAI shape, hands it to the endpoint the engine is already
serving, and converts what comes back. Knurlogic's job stays what it was.

WHAT IT CANNOT PROMISE. A harness leans hard on tool-calling: whether a given
model emits well-formed tool calls at all is a property of the model, not of
this file. The translation being correct is necessary and nowhere near
sufficient, and `doctor` should say that rather than this module implying it.
"""

from __future__ import annotations

import json
import time
import uuid

STOP_REASON = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    None: "end_turn",
}


def _text_of(content) -> str:
    """Anthropic content is a string or a list of blocks."""
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content or []
                   if isinstance(b, dict) and b.get("type") == "text")


def to_openai(req: dict) -> dict:
    """Anthropic Messages request -> OpenAI chat request."""
    out_msgs = []

    system = req.get("system")
    if system:
        out_msgs.append({"role": "system", "content": _text_of(system)})

    for m in req.get("messages", []):
        role, content = m.get("role"), m.get("content")
        if isinstance(content, str):
            out_msgs.append({"role": role, "content": content})
            continue

        # A user turn may carry tool RESULTS, which OpenAI models as separate
        # messages with role "tool" -- one per result, each keyed to the call.
        results = [b for b in content or []
                   if isinstance(b, dict) and b.get("type") == "tool_result"]
        for b in results:
            payload = b.get("content")
            out_msgs.append({
                "role": "tool",
                "tool_call_id": b.get("tool_use_id"),
                "content": (payload if isinstance(payload, str)
                            else _text_of(payload)),
            })

        text = _text_of(content)
        calls = [b for b in content or []
                 if isinstance(b, dict) and b.get("type") == "tool_use"]
        if calls:
            out_msgs.append({
                "role": "assistant",
                "content": text or None,
                "tool_calls": [{
                    "id": c.get("id"),
                    "type": "function",
                    "function": {"name": c.get("name"),
                                 "arguments": json.dumps(c.get("input") or {})},
                } for c in calls],
            })
        elif text or not results:
            out_msgs.append({"role": role, "content": text})

    body = {
        "model": req.get("model", "local"),
        "messages": out_msgs,
        "stream": bool(req.get("stream")),
    }
    if req.get("max_tokens"):
        body["max_tokens"] = req["max_tokens"]
    for k in ("temperature", "top_p"):
        if req.get(k) is not None:
            body[k] = req[k]
    if req.get("stop_sequences"):
        body["stop"] = req["stop_sequences"]
    if req.get("tools"):
        body["tools"] = [{
            "type": "function",
            "function": {"name": t.get("name"),
                         "description": t.get("description", ""),
                         "parameters": t.get("input_schema")
                         or {"type": "object", "properties": {}}},
        } for t in req["tools"]]
    return body


def _blocks_from_choice(msg: dict) -> list:
    blocks = []
    if msg.get("content"):
        blocks.append({"type": "text", "text": msg["content"]})
    for c in msg.get("tool_calls") or []:
        fn = c.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            # A model that emits invalid JSON is a model problem, not a
            # translation problem -- pass the raw text through rather than
            # dropping the call silently.
            args = {"_raw": fn.get("arguments")}
        blocks.append({"type": "tool_use", "id": c.get("id") or _id("toolu"),
                       "name": fn.get("name"), "input": args})
    return blocks or [{"type": "text", "text": ""}]


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def from_openai(resp: dict, model: str) -> dict:
    """OpenAI chat completion -> Anthropic Messages response."""
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    usage = resp.get("usage") or {}
    blocks = _blocks_from_choice(msg)
    stop = STOP_REASON.get(choice.get("finish_reason"), "end_turn")
    if any(b["type"] == "tool_use" for b in blocks):
        stop = "tool_use"
    return {
        "id": _id("msg"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": usage.get("prompt_tokens", 0),
                  "output_tokens": usage.get("completion_tokens", 0)},
    }


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def stream(openai_lines, model: str):
    """OpenAI SSE chunks -> Anthropic SSE events.

    The event ORDER is the contract a harness parses against, so it is built
    explicitly rather than emitted as deltas arrive: message_start, then one
    content_block per piece of content, then message_delta with the stop
    reason, then message_stop. A tool call is its own block whose input
    arrives as `input_json_delta`, which is why arguments are forwarded as
    raw JSON text instead of being parsed and re-serialised here.
    """
    msg_id = _id("msg")
    yield _sse("message_start", {
        "type": "message_start",
        "message": {"id": msg_id, "type": "message", "role": "assistant",
                    "model": model, "content": [], "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0}}})

    index, open_block, stop = 0, None, "end_turn"
    tool_open = {}
    usage = {"input_tokens": 0, "output_tokens": 0}

    for line in openai_lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8", "replace")
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except ValueError:
            continue
        u = chunk.get("usage") or {}
        if u:
            usage = {"input_tokens": u.get("prompt_tokens", 0),
                     "output_tokens": u.get("completion_tokens", 0)}
        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}

        if delta.get("content"):
            if open_block != "text":
                if open_block is not None:
                    yield _sse("content_block_stop",
                               {"type": "content_block_stop", "index": index})
                    index += 1
                yield _sse("content_block_start", {
                    "type": "content_block_start", "index": index,
                    "content_block": {"type": "text", "text": ""}})
                open_block = "text"
            yield _sse("content_block_delta", {
                "type": "content_block_delta", "index": index,
                "delta": {"type": "text_delta", "text": delta["content"]}})

        for tc in delta.get("tool_calls") or []:
            i = tc.get("index", 0)
            fn = tc.get("function") or {}
            if i not in tool_open:
                if open_block is not None:
                    yield _sse("content_block_stop",
                               {"type": "content_block_stop", "index": index})
                    index += 1
                tool_open[i] = index
                open_block = "tool"
                stop = "tool_use"
                yield _sse("content_block_start", {
                    "type": "content_block_start", "index": index,
                    "content_block": {"type": "tool_use",
                                      "id": tc.get("id") or _id("toolu"),
                                      "name": fn.get("name") or "",
                                      "input": {}}})
            if fn.get("arguments"):
                yield _sse("content_block_delta", {
                    "type": "content_block_delta", "index": tool_open[i],
                    "delta": {"type": "input_json_delta",
                              "partial_json": fn["arguments"]}})

        if choice.get("finish_reason"):
            stop = STOP_REASON.get(choice["finish_reason"], stop)

    if open_block is not None:
        yield _sse("content_block_stop",
                   {"type": "content_block_stop", "index": index})
    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop, "stop_sequence": None},
        "usage": {"output_tokens": usage["output_tokens"]}})
    yield _sse("message_stop", {"type": "message_stop"})


def handler(chat_url: str, model: str, timeout: float = 3600.0):
    """A `/v1/messages` handler that calls the engine's own OpenAI endpoint.

    Self-request on purpose: the engine already owns chat templating, stop
    sequences and tool parsing, and re-implementing any of it here would give
    the two endpoints different behaviour for the same model.
    """
    import urllib.request

    def call(body: bytes, write, start_response):
        try:
            req = json.loads(body or b"{}")
        except ValueError:
            start_response(400, "application/json")
            write(json.dumps({"type": "error", "error": {
                "type": "invalid_request_error",
                "message": "body must be JSON"}}).encode())
            return

        want_stream = bool(req.get("stream"))
        oai = to_openai(req)
        r = urllib.request.Request(
            chat_url, data=json.dumps(oai).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            resp = urllib.request.urlopen(r, timeout=timeout)
        except Exception as e:
            start_response(502, "application/json")
            write(json.dumps({"type": "error", "error": {
                "type": "api_error",
                "message": f"engine at {chat_url}: {e}"}}).encode())
            return

        if not want_stream:
            out = from_openai(json.loads(resp.read().decode()),
                              req.get("model", model))
            start_response(200, "application/json")
            write(json.dumps(out).encode())
            return

        start_response(200, "text/event-stream")
        for event in stream(resp, req.get("model", model)):
            write(event)

    return call
