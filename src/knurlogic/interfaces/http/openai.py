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
from collections.abc import Callable, Iterator
from typing import Any

from knurlogic.engine.runtime import prompt as P
from knurlogic.engine.runtime.scheduler import Job


class ApiError(Exception):
    def __init__(self, status: int, message: str, *, type_: str = None,
                 param: str = None, code: str = None,
                 retry_after: int = None, memory: dict = None):
        super().__init__(message)
        #: the scheduler's memory terms behind a 503 (Scheduler._memory)
        self.memory = memory
        self.status = status
        #: seconds, sent as Retry-After (the 503s: busy memory, no model,
        #: a cluster stopping); None sends none
        self.retry_after = retry_after
        self.type = type_ or ("invalid_request_error" if status < 500
                              else "server_error")
        self.param, self.code = param, code

    def body(self) -> dict:
        err = {"message": str(self), "type": self.type,
               "param": self.param, "code": self.code}
        if self.memory is not None:
            err["memory"] = self.memory
        return {"error": err}


def _num(body, name, kind, lo=None, hi=None, default=None):
    v = body.get(name, default)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, kind):
        raise ApiError(400, f"{name} must be a number"
                       if kind is not int else f"{name} must be an integer",
                       param=name)
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        rng = f"between {lo} and {hi}" if hi is not None else f"at least {lo}"
        raise ApiError(400, f"{name} must be {rng}", param=name)
    return v


def _non_thinking(body: dict, nt: dict, sampling: dict, penalties: dict,
                  ctx: dict) -> None:
    """Thinking is off: each parameter the request left to the model takes
    the makers' thinking-off value (Qwen3.5: 0.7 / 0.8 / top_k 20 /
    presence 1.5) instead of generation_config's thinking set."""
    filled = ctx["sampling"]["from_model"]
    for name, key in (("temperature", "temp"), ("top_p", "top_p"),
                      ("top_k", "top_k"), ("min_p", "min_p")):
        if key in nt and (name in filled or name not in body):
            sampling[key] = float(nt[key]) if key == "temp" else nt[key]
            if name not in filled:
                filled.append(name)
    if "presence_penalty" in nt and "presence_penalty" not in body:
        penalties["presence_penalty"] = nt["presence_penalty"]
        filled.append("presence_penalty")
    ctx["sampling"]["applied"] = {k: v for k, v in sampling.items()
                                  if k != "seed"}
    if "presence_penalty" in penalties:
        ctx["sampling"]["applied"]["presence_penalty"] = \
            penalties["presence_penalty"]
    ctx["sampling"]["mode"] = "non-thinking"


def _tool_choice(tc):
    """OpenAI chat's tool_choice, checked: None, "auto", "none",
    "required", or {"type": "function", "function": {"name": str}}."""
    if tc is None or tc in ("auto", "none", "required"):
        return tc
    if isinstance(tc, dict) and tc.get("type") == "function" \
            and isinstance((tc.get("function") or {}).get("name"), str):
        return {"type": "function",
                "function": {"name": tc["function"]["name"]}}
    raise ApiError(400, 'tool_choice must be "auto", "none", "required" or '
                        '{"type": "function", "function": {"name": ...}}',
                   param="tool_choice")


def build_job(body: dict, *, chat: bool, translate: Callable = None,
              has_vision: Callable[[], bool] = lambda: False,
              sampling_defaults: dict | None = None) -> tuple:
    """(Job, context) from a request body; ApiError to refuse.

    `sampling_defaults`: the served model's recommended sampling
    (machine/artifact.sampling_defaults), used for each parameter the
    request leaves out. Silence used to mean greedy, which Qwen's thinking
    models are documented to degrade and loop under; a model that says
    nothing (or do_sample false) is still greedy."""
    if not isinstance(body, dict):
        raise ApiError(400, "the request body must be a JSON object")
    n = _num(body, "n", int, lo=1)
    if n not in (None, 1):
        raise ApiError(400, "n > 1 is not supported: send the request n "
                            "times (each is batched)", param="n")
    max_tokens = body.get("max_completion_tokens", body.get("max_tokens"))
    if max_tokens is None:
        pass                # the scheduler gives the rest of the window
    elif isinstance(max_tokens, bool) or not isinstance(max_tokens, int) \
            or max_tokens < 0:
        raise ApiError(400, "max_tokens must be a non-negative integer",
                       param="max_tokens")
    model = dict(sampling_defaults or {})
    non_thinking = model.pop("non_thinking", None)
    temp = _num(body, "temperature", (int, float), lo=0)
    filled = []
    if temp is None:
        temp = model.get("temp", 0.0)
        if "temp" in model:
            filled.append("temperature")
    sampling = {"temp": float(temp)}
    for name, key, kind, lo, hi in (
            ("top_p", "top_p", (int, float), 0, 1),
            ("top_k", "top_k", int, 0, None),
            ("min_p", "min_p", (int, float), 0, 1),
            ("xtc_probability", "xtc_probability", (int, float), 0, 1),
            ("xtc_threshold", "xtc_threshold", (int, float), 0, 0.5)):
        v = _num(body, name, kind, lo, hi)
        if v is None and key in model:
            v = model[key]
            filled.append(name)
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
                           param="logit_bias") from None
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
           "thinking": None, "exclude": False, "chat": chat,
           # what sampled this request, and which of it was the model's
           # recommendation rather than the request's
           "sampling": {"applied": {k: v for k, v in sampling.items()
                                    if k != "seed"},
                        "from_model": filled}}
    if chat:
        msgs = body.get("messages")
        if not isinstance(msgs, list) or not msgs:
            raise ApiError(400, "messages must be a non-empty list",
                           param="messages")
        for i, m in enumerate(msgs):
            if not isinstance(m, dict) or not isinstance(m.get("role"), str):
                raise ApiError(400, f"messages[{i}] must be an object with "
                                    f"a string role", param="messages")
            c = m.get("content")
            if c is not None and not isinstance(c, (str, list)):
                raise ApiError(400, f"messages[{i}].content must be a "
                                    f"string or a list of parts",
                               param="messages")
            if isinstance(c, list) and not all(isinstance(p, dict)
                                               for p in c):
                raise ApiError(400, f"messages[{i}].content parts must be "
                                    f"objects", param="messages")
        from knurlogic.engine.vision import request as vreq
        if vreq.has_images(msgs):
            if not has_vision():
                from knurlogic.engine.serve.vision import no_vision_why
                raise ApiError(400, no_vision_why(), param="messages")
        kwargs = body.get("chat_template_kwargs")
        if translate is not None:
            from knurlogic.engine.serve import thinking
            try:
                kwargs, ctx["thinking"] = translate(body, kwargs)
            except ValueError as e:
                raise ApiError(400, str(e), param="reasoning_effort") from e
            ctx["exclude"] = thinking.excluded(body)
            if non_thinking and (ctx["thinking"] or {}).get("applied") == "off":
                _non_thinking(body, non_thinking, sampling, penalties, ctx)
        req = P.ChatRequest("chat", "", msgs, body.get("tools") or None,
                            body.get("role_mapping"),
                            _tool_choice(body.get("tool_choice")))
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
    from knurlogic.engine.runtime.memory_guard import OutOfMemory
    from knurlogic.engine.runtime.scheduler import RingFailed
    if isinstance(err, RingFailed):
        return ApiError(503, str(err), type_="server_error",
                        code="cluster_failed", retry_after=30)
    if isinstance(err, OutOfMemory):
        return ApiError(503, str(err), type_="server_error",
                        code="insufficient_memory", retry_after=10,
                        memory=getattr(err, "memory", None))
    if "no model" in str(err):
        return ApiError(503, str(err), type_="server_error", retry_after=5)
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
        #: the ledger's open Request (telemetry.py), or None
        self.record = ctx.get("record")
        self._pace = None      # (perf_counter, done) at the first progress

    def _closed(self, usage, finish) -> None:
        if self.record is not None:
            self.record.done(usage, finish)

    def _failed(self) -> None:
        if self.record is not None:
            self.record.failed()

    def progress(self, phase: str, done: int = 0, total: int = 0,
                 ahead: int = 0) -> bytes:
        """A knurlogic.progress event (telemetry.md); tps over the prefill
        chunks seen so far."""
        from . import telemetry as T
        tps = None
        if phase == "prefill":
            now = time.perf_counter()
            if self._pace is None:
                self._pace = (now, done)
            elif now > self._pace[0]:
                tps = round((done - self._pace[1]) / (now - self._pace[0]),
                            1)
        return T.progress(self.job.request_id, phase, done, total, tps,
                          ahead)

    def first(self, timeout: float | None = None):
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
                self._failed()
                raise _status_of(val)
            ev = self.job.outbox.get()
        choice = {"index": 0, "finish_reason": finish}
        if self.ctx["chat"]:
            msg: dict = {"role": "assistant", "content": content}
            if reasoning and not self.ctx["exclude"]:
                msg["reasoning"] = msg["reasoning_content"] = reasoning
            if calls:
                msg["tool_calls"] = [_call(c) for c in calls]
                if not content.strip():
                    # the separator a model writes before its tool block
                    # (DeepSeek-V4's "\n\n") is not text to show
                    msg["content"] = ""
            if self.ctx.get("compaction") is not None:
                msg["compaction"] = self.ctx["compaction"]
            choice["message"] = msg
        else:
            choice["text"] = content
        if lps:
            choice["logprobs"] = {"content": [self._lp(x) for x in lps]}
        out = {"id": self.id, "object": ("chat.completion"
                                         if self.ctx["chat"]
                                         else "text_completion"),
               "created": self.created, "model": self.model,
               "choices": [choice], "usage": self._usage(usage)}
        self._closed(usage, finish)
        if self.ctx.get("applied"):
            out["context_management"] = {"applied_edits":
                                         self.ctx["applied"]}
        return out

    # ----------------------------------------------------------- stream

    def events(self, first) -> Iterator[bytes]:
        """SSE bytes, ending in [DONE]. The caller cancels the Job if a
        write fails."""
        obj = "chat.completion.chunk" if self.ctx["chat"] else \
            "text_completion"
        if self.ctx["chat"]:
            yield self._sse(obj, {"role": "assistant", "content": ""}, None)
            # what context management did, before the answer: the summary
            # as its own delta (a client keeps it and resends it), the
            # edits applied beside it
            if self.ctx.get("compaction") is not None:
                yield self._sse(obj, {"compaction": self.ctx["compaction"]},
                                None)
            if self.ctx.get("applied"):
                yield _data({"id": self.id, "object": obj,
                             "created": self.created, "model": self.model,
                             "choices": [], "context_management": {
                                 "applied_edits": self.ctx["applied"]}})
        ev = first
        # whitespace-only text is held until real text follows, and dropped
        # when a tool call comes first: the separator a model writes before
        # its tool block (DeepSeek-V4's "\n\n") is not text to show
        held, finish = "", "stop"
        while True:
            kind, val = ev
            if kind == "progress":
                yield f": keepalive {val[0]}/{val[1]}\n\n".encode()
                if self.ctx.get("progress"):
                    yield self.progress("prefill", val[0], val[1])
            elif kind == "delta":
                d = {}
                if self.ctx["chat"]:
                    text = val.content
                    if val.tool_calls:
                        held, text = "", text if text.strip() else ""
                    elif text and not text.strip():
                        held, text = held + text, ""
                    elif text or val.finish:
                        held, text = "", held + text
                    if text:
                        d["content"] = text
                    if val.reasoning and not self.ctx["exclude"]:
                        d["reasoning"] = d["reasoning_content"] = \
                            val.reasoning
                    if val.tool_calls:
                        d["tool_calls"] = [_call(c, stream=True)
                                           for c in val.tool_calls]
                else:
                    d = {"text": val.content} if val.content else {}
                finish = val.finish or finish
                if d or val.finish:
                    yield self._sse(obj, d, val.finish,
                                    [self._lp(x) for x in val.logprobs])
            elif kind == "done":
                self._closed(val, finish)
                if self.ctx["include_usage"]:
                    yield _data({"id": self.id, "object": obj,
                                 "created": self.created,
                                 "model": self.model, "choices": [],
                                 "usage": self._usage(val)})
                break
            elif kind == "error":
                self._failed()
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
        if self.ctx.get("sampling") is not None:
            u.setdefault("knurlogic", {})["sampling"] = self.ctx["sampling"]
        # how full the context is, so a harness can decide when to ask
        # for compaction: this prompt against the model's window
        if self.ctx["chat"] and "prompt_tokens" in u:
            u.setdefault("knurlogic", {})["context"] = {
                "tokens": int(u["prompt_tokens"]),
                "window": int(self.ctx.get("window") or 0)}
        if self.ctx.get("iteration"):
            u.setdefault("knurlogic", {})["compaction"] = \
                self.ctx["iteration"]
        return u


def _call(c: dict, stream: bool = False) -> dict:
    out = {"id": c["id"], "type": "function", "function": c["function"]}
    if stream:
        out["index"] = c["index"]
    return out


def _data(obj: Any) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode()


def models_document(served: dict, sampling: dict | None = None,
                    context_length: int = 0,
                    thinking: dict | None = None,
                    load_state: str = "ready",
                    memory_short: str | None = None) -> dict:
    """/v1/models: the one served model, with its capabilities and size, the
    sampling a request that says nothing gets (the model's recommendation;
    {} is greedy), its context window (0: the config does not say), and
    the thinking levels its template has (engine/serve/thinking.levels:
    dialect, default, native [{level on the reasoning_effort ladder, the
    template's own name}]) when known.

    `status` is "loading" until the host's warm-up is done ("ready"), or
    "failed": requests queue while it loads, so a client that polls
    /v1/models and then times its first request would time the load. The
    answer stays 200 -- the page's liveness probe and router read this
    endpoint while a model loads -- and OpenAI clients ignore the field.
    A ready model whose scheduler could not admit a minimal prompt
    (Scheduler.memory_short) is not ready: its status is that reason."""
    status = {"ready": "ready", "failed": "failed"}.get(load_state, "loading")
    if status == "ready" and memory_short:
        status = memory_short
    m = {"id": served["id"], "object": "model", "status": status,
         "created": int(served.get("created") or 0),
         "owned_by": "knurlogic",
         "capabilities": served.get("capabilities") or ["text"],
         "size_bytes": int(served.get("size_bytes") or 0),
         "sampling_defaults": dict(sampling or {}),
         "context_length": int(context_length or 0)}
    if thinking is not None:
        m["thinking"] = thinking
    # the telemetry contract this server speaks (docs/design/telemetry.md)
    return {"object": "list", "data": [m], "knurlogic": {"telemetry": 1}}
