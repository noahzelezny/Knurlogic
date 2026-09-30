"""The OpenAI Responses API (/v1/responses), as a translation over the
engine's OpenAI chat surface -- the same shape as messages.py: the request
becomes a chat request handed to the in-process transport, and what comes
back (one completion, or SSE chunks) becomes a Response or the Responses
event stream. Templates, thinking, vision, stop handling and tool-call
parsing stay owned by the chat path.

Not supported, refused with a 400: `previous_response_id` and `store: true`
(the server keeps no conversation; resend the input), and tools that are
not functions (web_search, file_search, ...: the server runs none). An
omitted `store` is treated as false.
"""

from __future__ import annotations

import json
import time
import uuid

from .messages import TransportError


class BadRequest(ValueError):
    def __init__(self, message: str, param: str = None):
        super().__init__(message)
        self.param = param


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def _parts(content, role: str):
    """Responses content (string or parts) -> chat content."""
    if isinstance(content, str) or content is None:
        return content or ""
    parts = []
    for p in content:
        t = p.get("type") if isinstance(p, dict) else None
        if t in ("input_text", "output_text", "text"):
            parts.append({"type": "text", "text": p.get("text", "")})
        elif t == "input_image":
            url = p.get("image_url")
            if isinstance(url, dict):
                url = url.get("url")
            if not url:
                raise BadRequest("an input_image needs an image_url "
                                 "(file_id images are not supported)",
                                 "input")
            parts.append({"type": "image_url", "image_url": {"url": url}})
        elif t == "refusal":
            parts.append({"type": "text", "text": p.get("refusal", "")})
        else:
            raise BadRequest(f"content part type {t!r} is not supported",
                             "input")
    if all(p["type"] == "text" for p in parts):
        return "".join(p["text"] for p in parts)
    return parts


def to_chat(req: dict) -> dict:
    """A Responses request -> an OpenAI chat request."""
    if not isinstance(req, dict):
        raise BadRequest("the request body must be a JSON object")
    if req.get("previous_response_id") is not None:
        raise BadRequest("previous_response_id is not supported: this "
                         "server stores no responses; resend the whole "
                         "input", "previous_response_id")
    if req.get("store") is True:
        raise BadRequest("store is not supported: this server stores no "
                         "responses; send store: false", "store")
    msgs = []
    if req.get("instructions"):
        msgs.append({"role": "system", "content": str(req["instructions"])})
    inp = req.get("input")
    if isinstance(inp, str):
        msgs.append({"role": "user", "content": inp})
    elif isinstance(inp, list) and inp:
        for i, it in enumerate(inp):
            if not isinstance(it, dict):
                raise BadRequest(f"input[{i}] must be an object", "input")
            t = it.get("type") or ("message" if "role" in it else None)
            if t == "message":
                role = it.get("role")
                msgs.append({"role": "system" if role == "developer"
                             else role,
                             "content": _parts(it.get("content"), role)})
            elif t == "function_call":
                call = {"id": it.get("call_id") or it.get("id") or _id("call"),
                        "type": "function",
                        "function": {"name": it.get("name"),
                                     "arguments": it.get("arguments") or "{}"}}
                if msgs and msgs[-1]["role"] == "assistant" and \
                        msgs[-1].get("tool_calls") is not None:
                    msgs[-1]["tool_calls"].append(call)
                else:
                    msgs.append({"role": "assistant", "content": None,
                                 "tool_calls": [call]})
            elif t == "function_call_output":
                out = it.get("output")
                msgs.append({"role": "tool",
                             "tool_call_id": it.get("call_id"),
                             "content": out if isinstance(out, str)
                             else json.dumps(out)})
            elif t == "reasoning":
                continue            # the model's earlier thinking, not resent
            else:
                raise BadRequest(f"input item type {t!r} is not supported",
                                 "input")
    else:
        raise BadRequest("input must be a string or a non-empty list of "
                         "items", "input")
    body = {"model": req.get("model") or "default", "messages": msgs,
            "stream": bool(req.get("stream"))}
    if req.get("max_output_tokens") is not None:
        body["max_tokens"] = req["max_output_tokens"]
    for k in ("temperature", "top_p"):
        if req.get(k) is not None:
            body[k] = req[k]
    r = req.get("reasoning")
    if isinstance(r, dict) and r.get("effort"):
        body["reasoning_effort"] = r["effort"]
    if body["stream"]:
        body["stream_options"] = {"include_usage": True}
    tools = []
    for t in req.get("tools") or []:
        if not isinstance(t, dict) or t.get("type") != "function":
            raise BadRequest("only function tools are supported; this "
                             "server runs no built-in tools", "tools")
        tools.append({"type": "function", "function": {
            "name": t.get("name"), "description": t.get("description", ""),
            "parameters": t.get("parameters")
            or {"type": "object", "properties": {}}}})
    if tools:
        body["tools"] = tools
    return body


def _usage(u: dict) -> dict:
    u = u or {}
    inp, out = int(u.get("prompt_tokens", 0) or 0), \
        int(u.get("completion_tokens", 0) or 0)
    cached = int((u.get("prompt_tokens_details") or {})
                 .get("cached_tokens", 0) or 0)
    return {"input_tokens": inp,
            "input_tokens_details": {"cached_tokens": cached},
            "output_tokens": out,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": inp + out}


def _message_item(text: str, item_id: str = None, done: bool = True) -> dict:
    return {"id": item_id or _id("msg"), "type": "message",
            "status": "completed" if done else "in_progress",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text,
                         "annotations": []}] if done else []}


def _reasoning_item(text: str, item_id: str = None) -> dict:
    return {"id": item_id or _id("rs"), "type": "reasoning",
            "summary": [{"type": "summary_text", "text": text}]}


def _call_item(name: str, args: str, call_id: str = None,
               item_id: str = None, done: bool = True) -> dict:
    return {"id": item_id or _id("fc"), "type": "function_call",
            "status": "completed" if done else "in_progress",
            "call_id": call_id or _id("call"), "name": name or "",
            "arguments": args}


def _envelope(rid: str, model: str, created: int, req: dict,
              status: str, output: list, usage=None, incomplete=None) -> dict:
    return {"id": rid, "object": "response", "created_at": created,
            "status": status, "error": None,
            "incomplete_details": incomplete, "model": model,
            "instructions": req.get("instructions"), "output": output,
            "parallel_tool_calls": True, "store": False,
            "temperature": req.get("temperature"),
            "top_p": req.get("top_p"),
            "max_output_tokens": req.get("max_output_tokens"),
            "tools": req.get("tools") or [], "tool_choice": "auto",
            "usage": usage}


def _finish(reason, status: str = "completed"):
    if reason == "length":
        return "incomplete", {"reason": "max_output_tokens"}
    return status, None


def from_chat(resp: dict, req: dict, model: str) -> dict:
    """An OpenAI chat completion -> a Response."""
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    out = []
    thought = msg.get("reasoning_content") or msg.get("reasoning")
    if thought:
        out.append(_reasoning_item(thought))
    if msg.get("content") or not (out or msg.get("tool_calls")):
        out.append(_message_item(msg.get("content") or ""))
    for c in msg.get("tool_calls") or []:
        fn = c.get("function") or {}
        out.append(_call_item(fn.get("name"), fn.get("arguments") or "{}",
                              c.get("id")))
    status, inc = _finish(choice.get("finish_reason"))
    return _envelope(_id("resp"), model, int(time.time()), req, status, out,
                     _usage(resp.get("usage")), inc)


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def stream(lines, req: dict, model: str):
    """OpenAI SSE chunks -> Responses events. One output item is open at a
    time (reasoning, then the message, then each call, as they arrive);
    each is opened, fed deltas, and closed with its `.done` events."""
    rid, created = _id("resp"), int(time.time())
    seq = [0]
    output, usage, finish = [], {}, None

    def ev(name, **data):
        data.update(type=name, sequence_number=seq[0])
        seq[0] += 1
        return _sse(name, data)

    base = _envelope(rid, model, created, req, "in_progress", [])
    yield ev("response.created", response=base)
    yield ev("response.in_progress", response=base)

    cur = None      # {"kind", "id", "text", "index", ...}

    def open_item(kind, **kw):
        nonlocal cur
        idx = len(output)
        item_id = _id({"reasoning": "rs", "message": "msg",
                       "call": "fc"}[kind])
        cur = dict(kind=kind, id=item_id, index=idx, text="", **kw)
        output.append(None)
        if kind == "reasoning":
            item = {"id": item_id, "type": "reasoning", "summary": []}
            yield ev("response.output_item.added", output_index=idx,
                     item=item)
            yield ev("response.reasoning_summary_part.added",
                     item_id=item_id, output_index=idx, summary_index=0,
                     part={"type": "summary_text", "text": ""})
        elif kind == "message":
            yield ev("response.output_item.added", output_index=idx,
                     item=_message_item("", item_id, done=False))
            yield ev("response.content_part.added", item_id=item_id,
                     output_index=idx, content_index=0,
                     part={"type": "output_text", "text": "",
                           "annotations": []})
        else:
            yield ev("response.output_item.added", output_index=idx,
                     item=_call_item(kw["name"], "", kw["call_id"], item_id,
                                     done=False))

    def close_item():
        nonlocal cur
        if cur is None:
            return
        k, i, idx, text = cur["kind"], cur["id"], cur["index"], cur["text"]
        if k == "reasoning":
            part = {"type": "summary_text", "text": text}
            yield ev("response.reasoning_summary_text.done", item_id=i,
                     output_index=idx, summary_index=0, text=text)
            yield ev("response.reasoning_summary_part.done", item_id=i,
                     output_index=idx, summary_index=0, part=part)
            item = {"id": i, "type": "reasoning", "summary": [part]}
        elif k == "message":
            part = {"type": "output_text", "text": text, "annotations": []}
            yield ev("response.output_text.done", item_id=i,
                     output_index=idx, content_index=0, text=text)
            yield ev("response.content_part.done", item_id=i,
                     output_index=idx, content_index=0, part=part)
            item = _message_item(text, i)
        else:
            yield ev("response.function_call_arguments.done", item_id=i,
                     output_index=idx, arguments=text)
            item = _call_item(cur["name"], text, cur["call_id"], i)
        output[idx] = item
        yield ev("response.output_item.done", output_index=idx, item=item)
        cur = None

    tool_open = {}
    for line in lines:
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
        if isinstance(chunk.get("error"), dict):
            err = chunk["error"]
            msg = err.get("message") or "the engine failed"
            failed = _envelope(rid, model, created, req, "failed", output)
            failed["error"] = {"code": "server_error", "message": msg}
            yield ev("error", code="server_error", message=msg, param=None)
            yield ev("response.failed", response=failed)
            return
        if chunk.get("usage"):
            usage = chunk["usage"]
        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        thought = delta.get("reasoning_content") or delta.get("reasoning")
        if thought:
            if cur is None or cur["kind"] != "reasoning":
                yield from close_item()
                yield from open_item("reasoning")
            cur["text"] += thought
            yield ev("response.reasoning_summary_text.delta",
                     item_id=cur["id"], output_index=cur["index"],
                     summary_index=0, delta=thought)
        if delta.get("content"):
            if cur is None or cur["kind"] != "message":
                yield from close_item()
                yield from open_item("message")
            cur["text"] += delta["content"]
            yield ev("response.output_text.delta", item_id=cur["id"],
                     output_index=cur["index"], content_index=0,
                     delta=delta["content"])
        for tc in delta.get("tool_calls") or []:
            i, fn = tc.get("index", 0), tc.get("function") or {}
            if i not in tool_open:
                yield from close_item()
                call_id = tc.get("id") or _id("call")
                yield from open_item("call", name=fn.get("name") or "",
                                     call_id=call_id)
                tool_open[i] = cur
            if fn.get("arguments"):
                c = tool_open[i]
                c["text"] += fn["arguments"]
                yield ev("response.function_call_arguments.delta",
                         item_id=c["id"], output_index=c["index"],
                         delta=fn["arguments"])
        if choice.get("finish_reason"):
            finish = choice["finish_reason"]
    yield from close_item()
    status, inc = _finish(finish)
    final = _envelope(rid, model, created, req, status, output,
                      _usage(usage), inc)
    yield ev("response.incomplete" if inc else "response.completed",
             response=final)


def _error_body(message: str, status: int, param: str = None) -> dict:
    return {"error": {"message": message, "param": param,
                      "type": ("invalid_request_error" if status < 500
                               else "server_error"), "code": None}}


def handler_over(transport, model: str):
    """A `/v1/responses` handler over the same in-process OpenAI chat
    transport messages.handler_over takes."""

    def call(body: bytes, write, start_response):
        def fail(status, message, param=None, headers=None):
            start_response(status, "application/json", headers or {})
            write(json.dumps(_error_body(message, status, param)).encode())
        try:
            req = json.loads(body or b"{}")
        except ValueError:
            return fail(400, "body must be JSON")
        try:
            oai = to_chat(req)
        except BadRequest as e:
            return fail(400, str(e), e.param)
        try:
            resp = transport(oai)
        except TransportError as e:
            return fail(e.status, str(e), headers=(
                {"Retry-After": str(int(e.retry_after))}
                if e.retry_after else None))
        name = req.get("model") or model
        if not req.get("stream"):
            start_response(200, "application/json")
            return write(json.dumps(from_chat(resp, req, name)).encode())
        start_response(200, "text/event-stream")
        for event in stream(resp, req, name):
            write(event)

    return call
