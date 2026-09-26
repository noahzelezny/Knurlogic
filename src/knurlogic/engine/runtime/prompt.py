"""A request's messages -> the prompt's tokens, cut into segments.

Segments are where the prompt cache stores checkpoints: the system prompt,
the conversation, and a short thinking tail the template may open. A later
request that shares the system prompt restores from the first checkpoint
instead of prefilling it again. The rules are the ones mlx-lm's server used,
rewritten here with knurlogic's fix folded in (the system segment is found on
templates where the empty user turn renders as a pure prefix -- GLM).

`tokenize` has the signature vision's `VisionServe.tokenize` calls as its
`real`: (gen, tokenizer, request, args) -> (prompt, segments, types,
initial state), so images go through the same function as text.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, List, Optional

from knurlogic.engine.serve import segments as _segments
from knurlogic.engine.serve import thinking


@dataclass
class ChatRequest:
    """What the prompt is built from. `request_type` "text" is a plain
    completion of `prompt`; "chat" renders `messages` (and `tools`)."""
    request_type: str = "chat"
    prompt: str = ""
    messages: List[dict] = field(default_factory=list)
    tools: Optional[list] = None
    role_mapping: Optional[dict] = None


@dataclass
class PromptArgs:
    """The rendering arguments: the server's defaults, then the request's."""
    chat_template_kwargs: Optional[dict] = None


class PromptError(ValueError):
    """The request cannot be rendered (400)."""


def flatten(messages: List[dict]) -> List[dict]:
    """A copy with list-of-parts content joined into one string (images are
    already placeholders by now) and tool-call arguments decoded, which is
    what chat templates expect."""
    out = copy.deepcopy(messages)
    for m in out:
        c = m.get("content")
        if isinstance(c, list):
            texts = [p.get("text", "") for p in c
                     if isinstance(p, dict) and p.get("type") == "text"]
            if len(texts) != len(c):
                raise PromptError("a message part is not text, and this "
                                  "model reads text only")
            m["content"] = "".join(texts)
        elif c is None:
            m["content"] = ""
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            if isinstance(fn.get("arguments"), str):
                try:
                    fn["arguments"] = json.loads(fn["arguments"])
                except ValueError:
                    pass
    return out


def tokenize(gen, tokenizer, request: ChatRequest, args: PromptArgs):
    """(prompt, segments, segment types, initial state). `gen` is unused
    (it is the slot vision passes its caller in)."""
    if request.request_type != "chat":
        p = list(tokenizer.encode(request.prompt))
        return p, [p], ["assistant"], "normal"
    if not getattr(tokenizer, "has_chat_template", True):
        raise PromptError("this model has no chat template; use "
                          "/v1/completions with a prompt")
    kw = dict(args.chat_template_kwargs or {})
    close = bool(kw.pop(thinking.CLOSE, False))
    messages = flatten(request.messages)
    render = dict(kw, tools=request.tools) if request.tools else dict(kw)
    try:
        if close:
            # the generation prompt ends with the think block already
            # closed; _Closing reads the flag from this call's kwargs
            prompt = list(thinking._Closing(tokenizer).apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                **{**render, thinking.CLOSE: True}))
        else:
            prompt = list(tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                **render))
    except Exception as e:
        raise PromptError(f"the chat template could not render this "
                          f"request: {type(e).__name__}: {e}") from e

    state = "normal"
    if getattr(tokenizer, "has_thinking", False):
        if tokenizer.rfind_think_start(prompt) > \
                tokenizer.rfind_think_end(prompt):
            state = "reasoning"
    if not messages or messages[-1].get("role") != "user":
        return prompt, [prompt], ["assistant"], state

    segs: List[List[int]] = []
    types: List[str] = []
    sys_end = 0
    n_sys = 0
    for m in messages:
        if m.get("role") != "system":
            break
        n_sys += 1
    if n_sys:
        try:
            sys_tokens = list(tokenizer.apply_chat_template(
                messages[:n_sys] + [{"role": "user", "content": ""}],
                add_generation_prompt=False, tokenize=True, **render))
        except Exception:
            sys_tokens = []
        # where the system render and the prompt first differ ...
        for i, (a, b) in enumerate(zip(sys_tokens, prompt)):
            if a != b:
                sys_end = i
                break
        if 0 < sys_end < len(prompt):
            segs.append(prompt[:sys_end])
            types.append("system")

    tail = len(prompt)
    if getattr(tokenizer, "has_thinking", False):
        at = tokenizer.rfind_think_start(prompt, start=tail - 11)
        if at >= 0:
            tail = at
    if sys_end < tail:
        segs.append(prompt[sys_end:tail])
        types.append("user")
    if tail < len(prompt):
        segs.append(prompt[tail:])
        types.append("assistant")
    if not segs:
        segs, types = [prompt], ["assistant"]
    # ... or, where the empty turn renders as a pure prefix, where the
    # system render ends (engine/serve/segments.py)
    segs, types = _segments.split_system(tokenizer, messages, prompt, segs,
                                         types, render)
    return prompt, segs, types, state
