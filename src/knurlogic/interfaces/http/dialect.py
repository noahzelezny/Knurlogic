"""What the API dialects translated over the OpenAI chat surface share
(messages.py: Anthropic, responses.py: OpenAI Responses, ollama.py):

  handler_over(...)   the handler skeleton: parse the body, translate it,
                      hand it to the in-process transport, answer whole or
                      as a stream, and refuse in the dialect's own shape
  BadRequest          a request the translation refuses (a 400), with the
                      parameter at fault when there is one
  sse(event, data)    one named server-sent event (openai.sse_data is the
                      data-only form the chat surface streams)
  new_id(prefix)      an id in the OpenAI / Anthropic style: prefix_<24 hex>
  image_part(url)     an image as an OpenAI chat content part
  cached_tokens(u)    the prompt tokens an OpenAI usage read from the cache

The transport is the chat surface itself: transport(openai_body) -> the
response dict, or (streaming) an iterable of SSE lines; it raises
openai.ApiError for a refusal (status, message, retry_after).
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable

from .openai import ApiError


class BadRequest(ValueError):
    def __init__(self, message: str, param: str | None = None):
        super().__init__(message)
        self.param = param


def sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def image_part(url: str) -> dict:
    return {"type": "image_url", "image_url": {"url": url}}


def cached_tokens(usage: dict) -> int:
    return int((usage.get("prompt_tokens_details") or {})
               .get("cached_tokens", 0) or 0)


def handler_over(transport: Callable, *, translate: Callable,
                 error_body: Callable, answer: Callable, events: Callable,
                 streams: Callable = lambda req, _oai: bool(req.get("stream")),
                 stream_type: str = "text/event-stream") -> Callable:
    """A dialect's handler(body, write, start_response) over `transport`.

      translate(req) -> the OpenAI chat body; BadRequest refuses it
      error_body(message, status, param) -> the dialect's error document
      answer(resp, req, started) -> the whole reply's document
      events(resp, req, started) -> the stream's chunks (bytes)
      streams(req, oai) -> whether to answer as a stream

    `started` is the time the request arrived (Ollama reports durations)."""

    def call(body: bytes, write, start_response):
        started = time.time()

        def fail(status, message, param=None, headers=None):
            start_response(status, "application/json",
                           *([headers] if headers else []))
            write(json.dumps(error_body(message, status, param)).encode())
        try:
            req = json.loads(body or b"{}")
        except ValueError:
            return fail(400, "body must be JSON")
        try:
            oai = translate(req)
        except BadRequest as e:
            return fail(400, str(e), e.param)
        try:
            resp = transport(oai)
        except ApiError as e:
            return fail(e.status, str(e), headers=(
                {"Retry-After": str(int(e.retry_after))}
                if e.retry_after else None))
        if not streams(req, oai):
            start_response(200, "application/json")
            return write(json.dumps(answer(resp, req, started)).encode())
        start_response(200, stream_type)
        for chunk in events(resp, req, started):
            write(chunk)

    return call
