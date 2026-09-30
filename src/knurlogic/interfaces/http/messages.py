"""The Anthropic Messages API, as a translation over the engine's OpenAI one.

A coding harness speaks the Anthropic Messages shape (ANTHROPIC_BASE_URL).
This converts each request into the OpenAI shape, hands it to that surface
in-process (`handler_over`), and converts what comes back; chat templates,
stop sequences, streaming and tool-call parsing stay owned by the OpenAI
surface. Whether a model emits well-formed tool calls is a property of the
model, not of this translation.
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
    "compaction": "compaction",
    None: "end_turn",
}


def anthropic_images_to_openai(content) -> list:
    """Anthropic Messages `content` blocks -> OpenAI `content` parts.

    This module translates a Claude-shaped request onto the
    engine's OpenAI surface (`serve.py`'s `messages_fn`), and until now that
    translation dropped image blocks on the floor -- text only. An Anthropic
    image block is `{"type": "image", "source": {"type": "base64",
    "media_type": "...", "data": "..."}}` (or `{"type": "url", "url": ...}`,
    which the engine's own image loader refuses -- P0's `images.decode`
    does not fetch http(s), and this function does not either: it only
    reshapes the block, `served_vision`'s caller does the fetching-or-not).
    OpenAI's chat surface wants `{"type": "image_url",
    "image_url": {"url": "data:<media_type>;base64,<data>"}}`.

    Anything already OpenAI-shaped, or any block this function does not
    recognise as an Anthropic image, passes through unchanged -- this is a
    shim for ONE block type, not a general content-block rewriter, and a
    text block silently dropped would be a worse bug than one image block
    left in a shape the engine already refuses with a clear error.
    """
    if not isinstance(content, list):
        return content
    out = []
    for block in content:
        if not isinstance(block, dict):
            out.append(block)
            continue
        if block.get("type") == "image" and isinstance(block.get("source"), dict):
            src = block["source"]
            if src.get("type") == "base64" and src.get("data"):
                media = src.get("media_type") or "image/png"
                out.append({"type": "image_url",
                            "image_url": {"url": f"data:{media};base64,{src['data']}"}})
                continue
            if src.get("type") == "url" and src.get("url"):
                out.append({"type": "image_url",
                            "image_url": {"url": src["url"]}})
                continue
        out.append(block)
    return out


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

    # A system message inside `messages` (Claude Code sends one mid-way) is
    # folded into the leading one: chat templates such as Qwen's refuse a
    # system message anywhere but first.
    late_system = []
    for m in req.get("messages", []):
        role, content = m.get("role"), m.get("content")
        if role == "system":
            late_system.append(content if isinstance(content, str)
                               else _text_of(content))
            continue
        if isinstance(content, str):
            out_msgs.append({"role": role, "content": content})
            continue

        # A compaction block (the server's own, resent) is a cut point:
        # its own message, for context_management/context_edits to fold on. The
        # rest of the message follows it as usual.
        for b in content or []:
            if isinstance(b, dict) and b.get("type") == "compaction":
                out_msgs.append({"role": "assistant", "content": None,
                                 "compaction": str(b.get("content") or "")})
        content = [b for b in content or []
                   if not (isinstance(b, dict)
                           and b.get("type") == "compaction")]
        if not content:
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
        elif any(isinstance(b, dict) and b.get("type") == "image"
                 for b in content or []):
            # Images keep their place among the text, in order: a harness
            # that sends [text, image, text] means exactly that sequence.
            parts = []
            for b in anthropic_images_to_openai(list(content or [])):
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text":
                    parts.append({"type": "text", "text": b.get("text", "")})
                elif b.get("type") == "image_url":
                    parts.append(b)
            out_msgs.append({"role": role, "content": parts})
        elif text or not results:
            out_msgs.append({"role": role, "content": text})

    late = "\n\n".join(t for t in late_system if t)
    if late:
        if out_msgs and out_msgs[0]["role"] == "system":
            out_msgs[0]["content"] = out_msgs[0]["content"] + "\n\n" + late
        else:
            out_msgs.insert(0, {"role": "system", "content": late})

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
    # Thinking: "disabled" asks for none; "enabled" leaves the model's own
    # default. budget_tokens is NOT honoured -- knurlogic uses each model's
    # native controls, never a token budget (engine/serve/thinking.py).
    # An explicit reasoning_effort, if a client sends one, is passed on.
    t = req.get("thinking")
    if isinstance(t, dict) and t.get("type") == "disabled":
        body["reasoning_effort"] = "none"
    # Anthropic returns thinking blocks only when thinking is enabled; so
    # does this, by asking the engine to leave reasoning out otherwise.
    enabled = isinstance(t, dict) and t.get("type") == "enabled"
    body["reasoning"] = {"exclude": not enabled}
    if body.get("stream"):
        body["stream_options"] = {"include_usage": True}
    if req.get("reasoning_effort"):
        body["reasoning_effort"] = req["reasoning_effort"]
    # performed by the server (context_management/compaction), whichever
    # API asked
    if req.get("context_management") is not None:
        body["context_management"] = req["context_management"]
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
    if msg.get("compaction") is not None:
        # first, as the API places it: everything before it is what it
        # summarizes, and the client resends it with the rest
        blocks.append({"type": "compaction", "content": msg["compaction"]})
    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
    if reasoning:
        # A local model signs nothing; the signature is empty, which a
        # client echoing the block back sends to us, not to Anthropic.
        blocks.append({"type": "thinking", "thinking": reasoning,
                       "signature": ""})
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


def _usage(usage: dict) -> dict:
    """OpenAI usage -> Anthropic's: input_tokens is what was NOT read from
    the prompt cache, cache_read_input_tokens what was (a harness adds the
    two for the context it is using); a summary pass is an iteration;
    usage.knurlogic (context tokens against the window, ...) passed on."""
    cached = int((usage.get("prompt_tokens_details") or {})
                 .get("cached_tokens", 0) or 0)
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    out = {"input_tokens": max(prompt - cached, 0),
           "cache_creation_input_tokens": 0,
           "cache_read_input_tokens": cached,
           "output_tokens": int(usage.get("completion_tokens", 0) or 0)}
    kn = usage.get("knurlogic") or {}
    if kn.get("compaction"):
        c = kn["compaction"]
        out["iterations"] = [
            {"type": "compaction",
             "input_tokens": max(c.get("input_tokens", 0)
                                 - c.get("cache_read_input_tokens", 0), 0),
             "cache_read_input_tokens": c.get("cache_read_input_tokens", 0),
             "output_tokens": c.get("output_tokens", 0)},
            {"type": "message", "input_tokens": out["input_tokens"],
             "cache_read_input_tokens": cached,
             "output_tokens": out["output_tokens"]}]
    if kn:
        out["knurlogic"] = kn
    return out


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
    if choice.get("finish_reason") == "compaction":
        # paused after compaction: the summary alone, no text block
        blocks = [b for b in blocks if b["type"] == "compaction"]
    out = {
        "id": _id("msg"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": stop,
        "stop_sequence": None,
        # what the engine applied and reused (usage.knurlogic), passed on
        # so a GLM "thinking disabled" that could only go to low is SAID
        **({"knurlogic": usage["knurlogic"]} if usage.get("knurlogic")
           else {}),
        "usage": _usage(usage),
    }
    if resp.get("context_management"):
        out["context_management"] = resp["context_management"]
    return out


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
    managed = None

    for line in openai_lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8", "replace")
        line = line.strip()
        if line.startswith(": keepalive"):
            # a summary pass (or a long prefill) under way: say so, as the
            # API does, so the client does not time out
            yield _sse("ping", {"type": "ping"})
            continue
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except ValueError:
            continue
        if isinstance(chunk.get("error"), dict):
            # the engine failed mid-stream: say so, never end_turn with a
            # truncated answer
            err = chunk["error"]
            yield _sse("error", {"type": "error", "error": {
                "type": "api_error",
                "message": err.get("message") or "the engine failed"}})
            return
        u = chunk.get("usage") or {}
        if u:
            usage = _usage(u)
        if chunk.get("context_management"):
            managed = chunk["context_management"]
        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}

        if delta.get("compaction") is not None:
            # whole, as the API streams it: start, one compaction_delta,
            # stop -- before any other block
            if open_block is not None:
                yield _sse("content_block_stop",
                           {"type": "content_block_stop", "index": index})
                index += 1
            yield _sse("content_block_start", {
                "type": "content_block_start", "index": index,
                "content_block": {"type": "compaction", "content": ""}})
            yield _sse("content_block_delta", {
                "type": "content_block_delta", "index": index,
                "delta": {"type": "compaction_delta",
                          "content": delta["compaction"]}})
            yield _sse("content_block_stop",
                       {"type": "content_block_stop", "index": index})
            index += 1
            open_block = None

        thought = delta.get("reasoning_content") or delta.get("reasoning")
        if thought:
            if open_block != "thinking":
                if open_block is not None:
                    yield _sse("content_block_stop",
                               {"type": "content_block_stop", "index": index})
                    index += 1
                yield _sse("content_block_start", {
                    "type": "content_block_start", "index": index,
                    "content_block": {"type": "thinking", "thinking": "",
                                      "signature": ""}})
                open_block = "thinking"
            yield _sse("content_block_delta", {
                "type": "content_block_delta", "index": index,
                "delta": {"type": "thinking_delta", "thinking": thought}})

        if delta.get("content"):
            if open_block == "thinking":
                # a thinking block closes with its (empty) signature
                yield _sse("content_block_delta", {
                    "type": "content_block_delta", "index": index,
                    "delta": {"type": "signature_delta", "signature": ""}})
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
                if open_block == "thinking":
                    yield _sse("content_block_delta", {
                        "type": "content_block_delta", "index": index,
                        "delta": {"type": "signature_delta",
                                  "signature": ""}})
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

    if open_block == "thinking":
        yield _sse("content_block_delta", {
            "type": "content_block_delta", "index": index,
            "delta": {"type": "signature_delta", "signature": ""}})
    if open_block is not None:
        yield _sse("content_block_stop",
                   {"type": "content_block_stop", "index": index})
    final = {"type": "message_delta",
             "delta": {"stop_reason": stop, "stop_sequence": None},
             "usage": usage}
    if managed:
        final["context_management"] = managed
    yield _sse("message_delta", final)
    yield _sse("message_stop", {"type": "message_stop"})


class TransportError(Exception):
    def __init__(self, status: int, message: str, retry_after: int = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


def handler_over(transport, model: str):
    """A `/v1/messages` handler over any OpenAI chat transport:
    transport(openai_body) -> the response dict, or (streaming) an
    iterable of SSE lines; TransportError for a refusal. knurlogic's own
    server passes its OpenAI surface directly -- no loopback request."""

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
        try:
            resp = transport(oai)
        except TransportError as e:
            if e.retry_after:
                start_response(e.status, "application/json",
                               {"Retry-After": str(int(e.retry_after))})
            else:
                start_response(e.status, "application/json")
            write(json.dumps({"type": "error", "error": {
                "type": ("invalid_request_error" if e.status < 500
                         else "api_error"),
                "message": str(e)}}).encode())
            return

        if not want_stream:
            out = from_openai(resp, req.get("model", model))
            start_response(200, "application/json")
            write(json.dumps(out).encode())
            return

        start_response(200, "text/event-stream")
        for event in stream(resp, req.get("model", model)):
            write(event)

    return call
