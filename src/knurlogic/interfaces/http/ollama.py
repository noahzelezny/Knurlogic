"""The Ollama API (/api/chat, /api/generate, /api/tags, /api/show,
/api/version), as a translation over the engine's OpenAI chat surface, like
messages.py. Ollama clients default to port 11434; point them here with
OLLAMA_HOST=http://127.0.0.1:8080.

Differences worth knowing: `stream` defaults to true (NDJSON lines); the
`model` a request names is echoed, the one served answers; `raw`, `format`,
`template`, `context` and `keep_alive` are not used; durations are
nanoseconds derived from the engine's own timing (usage.knurlogic.timing),
and load_duration is 0 (the model is already loaded).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .messages import TransportError

#: Ollama option -> the chat request's parameter
OPTIONS = {"temperature": "temperature", "top_p": "top_p", "top_k": "top_k",
           "min_p": "min_p", "seed": "seed", "num_predict": "max_tokens",
           "stop": "stop", "repeat_penalty": "repetition_penalty",
           "presence_penalty": "presence_penalty",
           "frequency_penalty": "frequency_penalty"}
DONE_REASON = {"length": "length"}


class BadRequest(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _content(text: str, images) -> object:
    if not images:
        return text
    parts: list = [{"type": "text", "text": text}] if text else []
    for img in images:
        if not isinstance(img, str):
            raise BadRequest("images must be base64 strings")
        url = img if img.startswith("data:") else \
            f"data:image/png;base64,{img}"
        parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts


def _tool_calls(calls: list, n: int) -> list:
    return [{"id": f"call_{n + i}", "type": "function",
             "function": {"name": (c.get("function") or {}).get("name"),
                          "arguments": json.dumps(
                              (c.get("function") or {}).get("arguments")
                              or {})}} for i, c in enumerate(calls)]


def _messages(msgs) -> list:
    if not isinstance(msgs, list):
        raise BadRequest("messages must be a list")
    out: list = []
    pending: list = []
    n = 0
    for i, m in enumerate(msgs):
        if not isinstance(m, dict) or not isinstance(m.get("role"), str):
            raise BadRequest(f"messages[{i}] must be an object with a "
                             f"string role")
        role, text = m["role"], m.get("content") or ""
        if role == "tool":
            # Ollama keys a result by tool_name; the chat path by call id
            k = next((j for j, c in enumerate(pending)
                      if c["function"]["name"] == m.get("tool_name")), 0)
            cid = pending.pop(k)["id"] if pending else f"call_{n}"
            out.append({"role": "tool", "tool_call_id": cid,
                        "content": text})
            continue
        msg = {"role": role, "content": _content(text, m.get("images"))}
        if m.get("tool_calls"):
            calls = _tool_calls(m["tool_calls"], n)
            n += len(calls)
            pending += calls
            msg["tool_calls"] = calls
            msg["content"] = text or None
        out.append(msg)
    return out


def to_chat(req: dict, generate: bool = False) -> dict:
    """An Ollama /api/chat (or /api/generate) request -> an OpenAI chat
    request."""
    if not isinstance(req, dict):
        raise BadRequest("the request body must be a JSON object")
    if generate:
        if req.get("raw"):
            raise BadRequest("raw is not supported")
        if not isinstance(req.get("prompt", ""), str):
            raise BadRequest("prompt must be a string")
        msgs = []
        if req.get("system"):
            msgs.append({"role": "system", "content": req["system"]})
        msgs.append({"role": "user", "content": _content(
            req.get("prompt") or "", req.get("images"))})
    else:
        msgs = _messages(req.get("messages"))
        if not msgs:
            raise BadRequest("messages must be a non-empty list")
    stream = req.get("stream")
    body = {"model": req.get("model") or "default", "messages": msgs,
            "stream": True if stream is None else bool(stream)}
    if body["stream"]:
        body["stream_options"] = {"include_usage": True}
    opts = req.get("options") or {}
    if not isinstance(opts, dict):
        raise BadRequest("options must be an object")
    for k, v in opts.items():
        if k in OPTIONS and v is not None:
            # Ollama: -1 (or -2) means no limit
            if k == "num_predict" and isinstance(v, int) and v < 0:
                continue
            body[OPTIONS[k]] = v
    think = req.get("think")
    if think is False or think in ("none",):
        body["reasoning_effort"] = "none"
    elif isinstance(think, str):
        body["reasoning_effort"] = think
    body["reasoning"] = {"exclude": think is False}
    if req.get("tools"):
        body["tools"] = [t for t in req["tools"] if isinstance(t, dict)]
    return body


def _durations(usage: dict, started: float) -> dict:
    """Ollama's counts and nanosecond durations from the engine's timing."""
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    out = int(usage.get("completion_tokens", 0) or 0)
    t = (usage.get("knurlogic") or {}).get("timing") or {}
    ttft = float(t.get("ttft_s") or 0)
    queue = float(t.get("queue_s") or 0)
    # prefill_tok_s counts only the tokens the prompt cache did not supply,
    # so the time is taken from the timings, not from the prompt size
    pre = max(ttft - queue, 0)
    dec = ((out - 1) / t["decode_tok_s"]) if t.get("decode_tok_s") else 0
    ns = lambda s: int(s * 1e9)             # noqa: E731
    total = ttft + dec if ttft else time.time() - started
    return {"total_duration": ns(total), "load_duration": 0,
            "prompt_eval_count": prompt, "prompt_eval_duration": ns(pre),
            "eval_count": out, "eval_duration": ns(dec)}


def _args(raw) -> dict:
    try:
        v = json.loads(raw or "{}")
    except ValueError:
        return {"_raw": raw}
    return v if isinstance(v, dict) else {"_value": v}


def _ollama_calls(calls: list) -> list:
    return [{"function": {"name": (c.get("function") or {}).get("name"),
                          "arguments": _args((c.get("function") or {})
                                             .get("arguments"))}}
            for c in calls]


def _piece(content: str, thinking: str, calls: list, generate: bool,
           model: str, done=None, usage=None, started=0.0) -> dict:
    out: dict = {"model": model, "created_at": _now()}
    if generate:
        out["response"] = content
        if thinking:
            out["thinking"] = thinking
    else:
        msg: dict = {"role": "assistant", "content": content}
        if thinking:
            msg["thinking"] = thinking
        if calls:
            msg["tool_calls"] = _ollama_calls(calls)
        out["message"] = msg
    out["done"] = done is not None
    if done is not None:
        out["done_reason"] = "stop" if calls else done
        out.update(_durations(usage or {}, started))
        kn = (usage or {}).get("knurlogic")
        if kn:
            out["usage"] = {"knurlogic": kn}   # request_id, timing, ...
    return out


def from_chat(resp: dict, model: str, generate: bool, started: float) -> dict:
    """An OpenAI chat completion -> one Ollama response object."""
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    return _piece(msg.get("content") or "",
                  msg.get("reasoning_content") or msg.get("reasoning") or "",
                  msg.get("tool_calls") or [], generate, model,
                  DONE_REASON.get(choice.get("finish_reason"), "stop"),
                  resp.get("usage"), started)


def stream(lines, model: str, generate: bool, started: float):
    """OpenAI SSE chunks -> Ollama NDJSON lines, ending in the done one."""
    usage: dict = {}
    calls: dict = {}
    finish = "stop"
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
            yield (json.dumps({"error": chunk["error"].get("message")
                               or "the engine failed"}) + "\n").encode()
            return
        usage = chunk.get("usage") or usage
        choice = (chunk.get("choices") or [{}])[0]
        d = choice.get("delta") or {}
        text = d.get("content") or ""
        thought = d.get("reasoning_content") or d.get("reasoning") or ""
        for tc in d.get("tool_calls") or []:
            # arguments arrive in fragments; Ollama sends a call whole
            c = calls.setdefault(tc.get("index", 0), {"function": {
                "name": "", "arguments": ""}})
            fn = tc.get("function") or {}
            c["function"]["name"] = fn.get("name") or c["function"]["name"]
            c["function"]["arguments"] += fn.get("arguments") or ""
        if text or thought:
            yield (json.dumps(_piece(text, thought, [], generate,
                                     model)) + "\n").encode()
        if choice.get("finish_reason"):
            finish = DONE_REASON.get(choice["finish_reason"], "stop")
    if calls and not generate:
        yield (json.dumps(_piece("", "", [calls[k] for k in sorted(calls)],
                                 generate, model)) + "\n").encode()
    yield (json.dumps(_piece("", "", [], generate, model, finish, usage,
                             started)) + "\n").encode()


def handler_over(transport, model: str, generate: bool):
    """A `/api/chat` (or `/api/generate`) handler over the in-process
    OpenAI chat transport."""

    def call(body: bytes, write, start_response):
        def fail(status, message, headers=None):
            start_response(status, "application/json", headers or {})
            write(json.dumps({"error": message}).encode())
        started = time.time()
        try:
            req = json.loads(body or b"{}")
            oai = to_chat(req, generate)
        except ValueError as e:              # JSON errors and BadRequest
            return fail(400, str(e) if isinstance(e, BadRequest)
                        else "body must be JSON")
        try:
            resp = transport(oai)
        except TransportError as e:
            return fail(e.status, str(e), (
                {"Retry-After": str(int(e.retry_after))}
                if e.retry_after else None))
        name = req.get("model") or model
        if not oai["stream"]:
            start_response(200, "application/json")
            return write(json.dumps(from_chat(resp, name, generate,
                                              started)).encode())
        start_response(200, "application/x-ndjson")
        for line in stream(resp, name, generate, started):
            write(line)

    return call


# --------------------------------------------------- the model documents

def _config(path) -> dict:
    try:
        return json.loads((Path(path) / "config.json").read_text())
    except (OSError, ValueError, TypeError):
        return {}


def details(path) -> dict:
    cfg = _config(path)
    text: Any = cfg.get("text_config") if isinstance(cfg.get("text_config"),
                                                     dict) else cfg
    q = cfg.get("quantization") or text.get("quantization") or {}
    bits = q.get("bits") if isinstance(q, dict) else None
    family = cfg.get("model_type") or text.get("model_type") or ""
    return {"parent_model": "", "format": "safetensors", "family": family,
            "families": [family] if family else [],
            "parameter_size": "",
            "quantization_level": f"Q{bits}" if bits else "F16"}


def tags_document(served: dict, path) -> dict:
    """/api/tags: the one served model."""
    name = served.get("id", "")
    mtime = int(served.get("created") or 0)
    stamp = datetime.fromtimestamp(mtime, timezone.utc).isoformat() \
        if mtime else _now()
    return {"models": [{"name": name, "model": name, "modified_at": stamp,
                        "size": int(served.get("size_bytes") or 0),
                        "digest": "", "details": details(path)}]}


def show_document(served: dict, path, context_length: int = 0,
                  thinking: dict = None) -> dict:
    """/api/show: the served model's details and capabilities."""
    d = details(path)
    caps = ["completion", "tools"]
    if "vision" in (served.get("capabilities") or []):
        caps.append("vision")
    if thinking and thinking.get("dialect"):
        caps.append("thinking")
    info = {"general.architecture": d["family"]}
    if context_length:
        info[f"{d['family']}.context_length"] = int(context_length)
    return {"modelfile": "", "parameters": "", "template": "",
            "details": d, "model_info": info, "capabilities": caps,
            "modified_at": _now()}
