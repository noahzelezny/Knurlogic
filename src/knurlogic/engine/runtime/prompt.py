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
import re
from dataclasses import dataclass, field
from typing import List, Optional

from knurlogic.engine import templates as _templates
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


class _Mark:
    """Marks a text part vision put in place of an image (its control tokens
    are real). An object, not a flag: a client's JSON can set any key to
    true, but cannot make this -- and it survives flatten's deepcopy."""
    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


PLACEHOLDER = "knurlogic_placeholder"
MARK = _Mark()
_ZWSP = "\u200b"


def control_strings(tokenizer) -> Optional[re.Pattern]:
    """The tokenizer's control tokens as they are spelled in text --
    added tokens that read as markup (`<|im_end|>`, `<start_of_turn>`,
    `<think>`, `[gMASK]`) -- as one pattern, longest first; None if none.

    A tokenizer turns those spellings into the control token wherever they
    appear, message content included. So a message quoting `<|im_end|>`
    ended its own turn, and `<|im_start|>system` inside a user message or a
    tool result opened a system turn (measured on Qwen3.8 Flash: models
    reviewing this code stopped mid-thought when they quoted one)."""
    cached = getattr(tokenizer, "_knurlogic_controls", False)
    if cached is not False:
        return cached
    # mlx-lm's TokenizerWrapper forwards these to the HF tokenizer (its
    # `_tokenizer`); a fast HF tokenizer's own `_tokenizer` is the Rust
    # one, which has neither -- so ask the object itself
    names = set()
    try:
        for t in (getattr(tokenizer, "added_tokens_decoder", None)
                  or {}).values():
            names.add(getattr(t, "content", str(t)))
        names.update(getattr(tokenizer, "all_special_tokens", None) or [])
    except (AttributeError, TypeError, ValueError):
        pass    # any tokenizer shape: an odd one just yields fewer names
    # (DeepSeek's `｜DSML｜`, the tool-call markup token, is bracketed by
    # full-width bars: a tool result quoting DSML would open a call)
    names = sorted((n for n in names if isinstance(n, str) and len(n) >= 3
                    and ((n[0] in "<[" and n[-1] in ">]")
                         or (n[0] == n[-1] == "｜"))),
                   key=len, reverse=True)
    pat = re.compile("|".join(map(re.escape, names))) if names else None
    try:
        tokenizer._knurlogic_controls = pat
    except (AttributeError, TypeError):
        pass    # a tokenizer that refuses attributes: recompute next time
    return pat


def neutralize(text: str, pattern, keep=()) -> str:
    """Control-token spellings in `text` made plain text: a zero-width space
    after the first character. The model reads the same characters; the
    tokenizer no longer matches the control token. `keep`: spellings left
    as they are."""
    if pattern is None or not text:
        return text
    return pattern.sub(lambda m: m[0] if m[0] in keep
                       else m[0][0] + _ZWSP + m[0][1:], text)


def _neutralize_values(v, pattern):
    if isinstance(v, str):
        return neutralize(v, pattern)
    if isinstance(v, dict):
        return {k: _neutralize_values(x, pattern) for k, x in v.items()}
    if isinstance(v, list):
        return [_neutralize_values(x, pattern) for x in v]
    return v


def flatten(messages: List[dict], tokenizer=None) -> List[dict]:
    """A copy with list-of-parts content joined into one string (images are
    already placeholders by now) and tool-call arguments decoded, which is
    what chat templates expect. Content and tool-call arguments are
    neutralized (control_strings): only the template and vision's
    placeholders write control tokens."""
    pat = control_strings(tokenizer) if tokenizer is not None else None
    # An assistant turn a client sends back may carry its reasoning inline
    # (<think>...</think>answer, as Ollama/llama.cpp-era clients store it);
    # the templates split on those tags to drop or wrap it, so they stay.
    # (the control spellings INSIDE the markers: gemma's is
    # "<|channel>thought", whose control token is "<|channel>")
    marks = [t for t in (getattr(tokenizer, "think_start", None),
                         getattr(tokenizer, "think_end", None))
             if isinstance(t, str) and t]
    think = set(pat.findall(" ".join(marks))) if pat is not None else set()
    out = copy.deepcopy(messages)
    for m in out:
        keep = think if m.get("role") == "assistant" else ()
        rc = m.get("reasoning_content")
        if isinstance(rc, str):
            m["reasoning_content"] = neutralize(rc, pat)
        c = m.get("content")
        if isinstance(c, list):
            texts = [p.get("text", "") if p.get(PLACEHOLDER) is MARK
                     else neutralize(p.get("text", ""), pat, keep) for p in c
                     if isinstance(p, dict) and p.get("type") == "text"]
            if len(texts) != len(c):
                raise PromptError("a message part is not text, and this "
                                  "model reads text only")
            m["content"] = "".join(texts)
        elif c is None:
            m["content"] = ""
        else:
            m["content"] = (neutralize(c, pat, keep) if isinstance(c, str)
                            else c)
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            if isinstance(fn.get("arguments"), str):
                try:
                    fn["arguments"] = json.loads(fn["arguments"])
                except ValueError:
                    pass
            if "arguments" in fn:
                fn["arguments"] = _neutralize_values(fn["arguments"], pat)
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
    # an artifact whose template is a known stub (DeepSeek-V4's mlx
    # conversion: no tools) renders with knurlogic's (engine/templates)
    _templates.install(tokenizer)
    kw = dict(args.chat_template_kwargs or {})
    close = bool(kw.pop(thinking.CLOSE, False))
    patched = _preserving_template(tokenizer)
    if patched is not None and "chat_template" not in kw:
        kw["chat_template"] = patched
    if PRESERVE not in kw and (patched is not None
                               or PRESERVE in _template_text(tokenizer)):
        kw[PRESERVE] = True
    messages = flatten(request.messages, tokenizer)
    # tool descriptions are rendered into the prompt too, and an MCP
    # server's are third-party text
    tools = _neutralize_values(request.tools, control_strings(tokenizer)) \
        if request.tools else None
    render = dict(kw, tools=tools) if tools else dict(kw)
    # The thinking probe renders on HTTP threads under this lock; a
    # tokenizer's template environment is not safe to share across threads.
    with thinking._render_lock:
        try:
            prompt = _render(tokenizer, messages, render, close)
        except Exception as e:  # a chat template is third-party code; its failure is the request's refusal
            raise PromptError(f"the chat template could not render this "
                              f"request: {type(e).__name__}: {e}") from e
        return _segment(tokenizer, messages, render, prompt)


#: Qwen3.6's template switch that renders every assistant turn's think
#: block, not only those after the last real user query. Without it, a
#: user message after tool calls (the next question, the compaction
#: summary ask) re-renders every earlier assistant turn without its
#: `<think>\n\n</think>\n\n`, so the history diverges right after the goal
#: and no stored checkpoint is a prefix -- a hybrid model re-prefills the
#: whole conversation (13.8k tokens, Qwen3.6-35B on an M4 Max). On, the
#: rendered history only grows; the model also sees earlier turns'
#: reasoning when a client sends it back, which Qwen3.6 is trained for.
#: A request's own chat_template_kwargs value wins.
PRESERVE = "preserve_thinking"


#: Qwen3.5's think-drop condition, the one Qwen3.6 put the switch in front
#: of (its template differs from 3.5's here and nowhere that matters). The
#: 3.5 templates (397B-A17B) have the same failure -- a user message after
#: tool calls drops every earlier turn's think block, so the next question
#: and the compaction summary re-prefill the whole history -- and no
#: switch; this adds Qwen3.6's own, spelled as 3.6 spells it. Trade-off:
#: the model sees its earlier turns' reasoning (what it wrote, which a
#: client sent back; empty blocks when it sent none), which 3.5 was not
#: trained on the way 3.6 is -- the price of not re-prefilling a hybrid
#: model's whole conversation once per user turn. A request's
#: `preserve_thinking: false` restores the template's own behaviour.
_DROP = re.compile(r"(\{%-?\s*if\s+)(loop\.index0\s*>\s*ns\.last_query_index)"
                   r"(\s*-?%\})")
_SWITCH = "(preserve_thinking is defined and preserve_thinking is true) or "


def _preserving_template(tokenizer) -> Optional[str]:
    """The template with Qwen3.6's preserve_thinking switch added, for a
    template that has Qwen3.5's think-drop and no switch; else None."""
    t = getattr(tokenizer, "chat_template", None)
    if not isinstance(t, str) or PRESERVE in t:
        return None
    cached = getattr(tokenizer, "_knurlogic_preserving", False)
    if cached is not False and cached[0] is t:
        return cached[1]
    new, n = _DROP.subn(lambda m: f"{m[1]}{_SWITCH}({m[2]}){m[3]}", t)
    out = new if n == 1 else None
    try:
        tokenizer._knurlogic_preserving = (t, out)
    except (AttributeError, TypeError):
        pass    # a tokenizer that refuses attributes: recompute next time
    return out


def _template_text(tokenizer) -> str:
    t = getattr(tokenizer, "chat_template", None)
    if isinstance(t, dict):
        t = " ".join(v for v in t.values() if isinstance(v, str))
    return t if isinstance(t, str) else ""


def _render(tokenizer, messages, render, close) -> list:
    if close:
        # the generation prompt ends with the think block already closed;
        # _Closing reads the flag from this call's kwargs
        return list(thinking._Closing(tokenizer).apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            **{**render, thinking.CLOSE: True}))
    return list(tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, **render))


#: the roles a prompt may end with and still be cut into segments
ANSWERED = ("user", "tool")


def _segment(tokenizer, messages, render, prompt):
    """Cut the rendered prompt into segments (see the module doc)."""
    state = "normal"
    if getattr(tokenizer, "has_thinking", False):
        if tokenizer.rfind_think_start(prompt) > \
                tokenizer.rfind_think_end(prompt):
            state = "reasoning"
    # a turn the model answers: a user's, or a tool result an agent sends
    # back (every agent turn after its first -- without a checkpoint there,
    # a hybrid model re-prefilled the whole conversation each turn). An
    # assistant message last is a prefill: one segment.
    if not messages or messages[-1].get("role") not in ANSWERED:
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
        except _templates.TEMPLATE_ERRORS:
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
    # The conversation checkpoint ends where the last message ends, before
    # the generation prompt's assistant header: the next turn, and the
    # compaction summary pass (this conversation plus a user turn), both
    # continue from there, and a hybrid model's checkpoint cannot be
    # trimmed back to it. Ending after `<|im_start|>assistant\n`, the
    # summary pass re-prefills the whole history (13.9k tokens, 40 s,
    # Qwen3.6-35B on an M4 Max).
    try:
        hist = list(tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=True, **render))
    except _templates.TEMPLATE_ERRORS:
        hist = []
    if sys_end < len(hist) < tail and prompt[:len(hist)] == hist:
        tail = len(hist)
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
