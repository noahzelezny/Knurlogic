"""The OpenAI surface of knurlogic's own server: /v1/chat/completions,
/v1/completions, /v1/models. A request becomes a scheduler Job; the Job's
outbox becomes one response or an SSE stream.

Errors are OpenAI's shape, `{"error": {message, type, param, code}}`, and
a request is refused before any header is sent whenever the refusal is
known by then -- which includes everything admission can refuse, because
the handler waits for the Job's first event before choosing the status.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Callable, Iterator, Optional

from knurlogic.engine.runtime import prompt as P
from knurlogic.engine.runtime.scheduler import Job

DEFAULT_MAX_TOKENS = 512


class ApiError(Exception):
    def __init__(self, status: int, message: str, *, type_: str = None,
                 param: str = None, code: str = None):
        super().__init__(message)
        self.status = status
        self.type = type_ or ("invalid_request_error" if status < 500
                              else "server_error")
        self.param, self.code = param, code

    def body(self) -> dict:
        return {"error": {"message": str(self), "type": self.type,
                          "param": self.param, "code": self.code}}


def _num(body, name, kind, lo=None, hi=None, default=None):
    v = body.get(name, default)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, kind):
        raise ApiError(400, f"{name} must be a number"
                       if kind != int else f"{name} must be an integer",
                       param=name)
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        rng = f"between {lo} and {hi}" if hi is not None else f"at least {lo}"
        raise ApiError(400, f"{name} must be {rng}", param=name)
    return v


def build_job(body: dict, *, chat: bool, translate: Callable = None,
              has_vision: Callable[[], bool] = lambda: False) -> tuple:
    """(Job, context) from a request body; ApiError to refuse."""
    if not isinstance(body, dict):
        raise ApiError(400, "the request body must be a JSON object")
    n = _num(body, "n", int, lo=1)
    if n not in (None, 1):
        raise ApiError(400, "n > 1 is not supported: send the request n "
                            "times (each is batched)", param="n")
    max_tokens = body.get("max_completion_tokens", body.get("max_tokens"))
    if max_tokens is None:
        max_tokens = DEFAULT_MAX_TOKENS
    elif isinstance(max_tokens, bool) or not isinstance(max_tokens, int) \
            or max_tokens < 0:
        raise ApiError(400, "max_tokens must be a non-negative integer",
                       param="max_tokens")
    temp = _num(body, "temperature", (int, float), lo=0, default=0.0)
    sampling = {"temp": float(temp)}
    for name, key, kind, lo, hi in (
            ("top_p", "top_p", (int, float), 0, 1),
            ("top_k", "top_k", int, 0, None),
            ("min_p", "min_p", (int, float), 0, 1),
            ("xtc_probability", "xtc_probability", (int, float), 0, 1),
            ("xtc_threshold", "xtc_threshold", (int, float), 0, 0.5)):
        v = _num(body, name, kind, lo, hi)
        if v is not None:
            sampling[key] = v
    seed = _num(body, "seed", int)
    if seed is not None:
        sampling["seed"] = seed
    penalties = {}
    for name, kind in (("repetition_penalty", (int, float)),
                       ("repetition_context_size", int),
                       ("presence_penalty", (int, float)),
                       ("presence_context_size", int),
                       ("frequency_penalty", (int, float)),
                       ("frequency_context_size", int)):
        v = _num(body, name, kind)
        if v is not None and v != 0:
            penalties[name] = v
    if body.get("logit_bias"):
        try:
            penalties["logit_bias"] = {int(k): float(v) for k, v in
                                       body["logit_bias"].items()}
        except (AttributeError, TypeError, ValueError):
            raise ApiError(400, "logit_bias must map token ids to numbers",
                           param="logit_bias")
    stops = body.get("stop") or []
    if isinstance(stops, str):
        stops = [stops]
    if not isinstance(stops, list) or not all(isinstance(s, str)
                                              for s in stops):
        raise ApiError(400, "stop must be a string or a list of strings",
                       param="stop")
    logprobs = bool(body.get("logprobs", False))
    top = _num(body, "top_logprobs", int, 0, 20) or 0

    ctx = {"stream": bool(body.get("stream", False)),
           "include_usage": bool((body.get("stream_options") or {}).get(
               "include_usage")),
           "model": body.get("model") or "default",
           "thinking": None, "exclude": False, "chat": chat}
    if chat:
        msgs = body.get("messages")
        if not isinstance(msgs, list) or not msgs:
            raise ApiError(400, "messages must be a non-empty list",
                           param="messages")
        from knurlogic.engine.vision import request as vreq
        if vreq.has_images(msgs):
            if not has_vision():
                raise ApiError(400, "this request has images but the "
                                    "served model has no vision; send text "
                                    "only", param="messages")
        kwargs = body.get("chat_template_kwargs")
        if translate is not None:
            from knurlogic.engine.serve import thinking
            try:
                kwargs, ctx["thinking"] = translate(body, kwargs)
            except ValueError as e:
                raise ApiError(400, str(e), param="reasoning_effort")
            ctx["exclude"] = thinking.excluded(body)
        req = P.ChatRequest("chat", "", msgs, body.get("tools") or None,
                            body.get("role_mapping"))
        args = P.PromptArgs(kwargs)
    else:
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            raise ApiError(400, "prompt must be a string", param="prompt")
        req, args = P.ChatRequest("text", prompt), P.PromptArgs()
    job = Job(req, args, max_tokens=max_tokens, sampling=sampling,
              penalties=penalties, stops=stops, logprobs=logprobs,
              top_logprobs=top)
    return job, ctx


def _status_of(err: BaseException) -> ApiError:
    if isinstance(err, ApiError):
        return err
    if isinstance(err, P.PromptError):
        return ApiError(400, str(err))
    from knurlogic.engine.vision import ImageTooLarge, VisionError
    name = type(err).__name__
    if isinstance(err, ImageTooLarge):
        return ApiError(413, str(err), param="messages",
                        code="image_too_large")
    if isinstance(err, VisionError):
        return ApiError(400, str(err), param="messages")
    if "no model" in str(err):
        return ApiError(503, str(err), type_="server_error")
    return ApiError(500, f"{name}: {err}")


class Reply:
    """Turns one Job's outbox into the wire format."""

    def __init__(self, job: Job, ctx: dict, *, decode: Callable = None,
                 served: str = ""):
        self.job, self.ctx = job, ctx
        self.decode = decode or (lambda t: "")
        self.id = (f"chatcmpl-{uuid.uuid4().hex}" if ctx["chat"]
                   else f"cmpl-{uuid.uuid4().hex}")
        self.created = int(time.time())
        self.model = served or ctx["model"]

    def first(self, timeout: Optional[float] = None):
        """The Job's first event, which decides the status: a refusal at
        tokenize or render (before prefill starts) is an HTTP error, not a
        200 carrying one. Prefill progress counts as a first event, so a
        long prompt streams keepalives instead of holding the headers."""
        return self.job.outbox.get(timeout=timeout)

    # ---------------------------------------------------------- one-shot

    def complete(self, first) -> dict:
        reasoning, content, calls, lps, finish, usage = "", "", [], [], \
            "stop", {}
        ev = first
        while True:
            kind, val = ev
            if kind == "progress":
                pass
            elif kind == "delta":
                reasoning += val.reasoning
                content += val.content
                calls += val.tool_calls
                lps += val.logprobs
                finish = val.finish or finish
            elif kind == "done":
                usage = val
                break
            elif kind == "error":
                raise _status_of(val)
            ev = self.job.outbox.get()
        choice = {"index": 0, "finish_reason": finish}
        if self.ctx["chat"]:
            msg = {"role": "assistant", "content": content}
            if reasoning and not self.ctx["exclude"]:
                msg["reasoning"] = msg["reasoning_content"] = reasoning
            if calls:
                msg["tool_calls"] = [_call(c) for c in calls]
            choice["message"] = msg
        else:
            choice["text"] = content
        if lps:
            choice["logprobs"] = {"content": [self._lp(x) for x in lps]}
        return {"id": self.id, "object": ("chat.completion"
                                          if self.ctx["chat"]
                                          else "text_completion"),
                "created": self.created, "model": self.model,
                "choices": [choice], "usage": self._usage(usage)}

    # ----------------------------------------------------------- stream

    def events(self, first) -> Iterator[bytes]:
        """SSE bytes, ending in [DONE]. The caller cancels the Job if a
        write fails."""
        obj = "chat.completion.chunk" if self.ctx["chat"] else \
            "text_completion"
        if self.ctx["chat"]:
            yield self._sse(obj, {"role": "assistant", "content": ""}, None)
        ev = first
        while True:
            kind, val = ev
            if kind == "progress":
                yield f": keepalive {val[0]}/{val[1]}\n\n".encode()
            elif kind == "delta":
                d = {}
                if self.ctx["chat"]:
                    if val.content:
                        d["content"] = val.content
                    if val.reasoning and not self.ctx["exclude"]:
                        d["reasoning"] = d["reasoning_content"] = \
                            val.reasoning
                    if val.tool_calls:
                        d["tool_calls"] = [_call(c, stream=True)
                                           for c in val.tool_calls]
                else:
                    d = {"text": val.content} if val.content else {}
                if d or val.finish:
                    yield self._sse(obj, d, val.finish,
                                    [self._lp(x) for x in val.logprobs])
            elif kind == "done":
                if self.ctx["include_usage"]:
                    yield _data({"id": self.id, "object": obj,
                                 "created": self.created,
                                 "model": self.model, "choices": [],
                                 "usage": self._usage(val)})
                break
            elif kind == "error":
                e = _status_of(val)
                yield _data(e.body())
                break
            ev = self.job.outbox.get()
        yield b"data: [DONE]\n\n"

    def _sse(self, obj, delta, finish, lps=None) -> bytes:
        ch = {"index": 0, "finish_reason": finish}
        if self.ctx["chat"]:
            ch["delta"] = delta
        else:
            ch.update(delta or {"text": ""})
        if lps:
            ch["logprobs"] = {"content": lps}
        return _data({"id": self.id, "object": obj, "created": self.created,
                      "model": self.model, "choices": [ch]})

    def _lp(self, x) -> dict:
        tok, lp, top = x
        out = {"token": self.decode(tok), "id": tok, "logprob": lp}
        if top:
            out["top_logprobs"] = [{"token": self.decode(t), "id": t,
                                    "logprob": v} for t, v in top]
        return out

    def _usage(self, usage: dict) -> dict:
        u = dict(usage or {})
        if self.ctx.get("thinking") is not None:
            u.setdefault("knurlogic", {})["thinking"] = self.ctx["thinking"]
        return u


def _call(c: dict, stream: bool = False) -> dict:
    out = {"id": c["id"], "type": "function", "function": c["function"]}
    if stream:
        out["index"] = c["index"]
    return out


def _data(obj: Any) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode()


def models_document(served: dict) -> dict:
    """/v1/models: the one served model, with what the harness asked for."""
    return {"object": "list", "data": [
        {"id": served["id"], "object": "model",
         "created": int(served.get("created") or 0),
         "owned_by": "knurlogic",
         "capabilities": served.get("capabilities") or ["text"],
         "size_bytes": int(served.get("size_bytes") or 0)}]}
