"""One request's text, from the executor's token events to what the client
reads (docs/design/server.md). No model and no mlx here: this is where
tokens become reasoning, answer and tool calls, where a stop string ends
the answer, and where the usage numbers are counted.

Control tokens (think/tool markers, end of turn) are matched as TOKEN
sequences by the engine's state machine and never reach the client. Stop
strings match the detokenized ANSWER text only, across token boundaries;
the last max(len(stop)) - 1 characters are held back so a stop never leaks
into a streamed delta. Tool calls are parsed by the tokenizer's own tool
parser when the call closes.

The object is fed on the scheduler's thread and read on the HTTP thread
only through the deltas it returns, so it holds no lock.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_MACHINES: dict = {}

#: The tags that close a think block. A model may end its thinking with
#: `</thinking>` where its template says `</think>` (seen on Qwen3.6); both
#: close the block, in the streamed and the non-streamed path alike. The
#: tokenizer's own think_end is one of these; the others are matched as the
#: token sequences their text encodes to.
THINK_CLOSE_TAGS = ("</think>", "</thinking>")
#: what may directly follow a close tag and merge with its last token
#: (">\n\n" is one token in some vocabularies)
_CLOSE_TAILS = ("", "\n", "\n\n", "\n\n\n", "\r\n")
#: what may directly precede it and merge with its first token (Qwen3.6
#: encodes " </" as one token, "</" as another)
_CLOSE_HEADS = ("", " ")


def _extra_think_closes(tokenizer) -> dict:
    """{token sequence: its text} for the alternative close tags, as the
    tokenizer encodes them bare and with a newline tail."""
    if getattr(tokenizer, "think_end", None) not in THINK_CLOSE_TAGS:
        return {}
    out: dict = {}
    for tag in THINK_CLOSE_TAGS:
        if tag == tokenizer.think_end:
            continue
        for head in _CLOSE_HEADS:
            for tail in _CLOSE_TAILS:
                try:
                    ids = tuple(tokenizer.encode(head + tag + tail,
                                                 add_special_tokens=False))
                except (AttributeError, TypeError, ValueError, KeyError):
                    continue        # a vocabulary that cannot encode it
                if ids:
                    out[ids] = head + tag + tail
    return out


def control_machine(tokenizer, initial: str = "normal"):
    """(ControlMachine, {token sequence: its text}) for the tokenizer's
    control tokens: end-of-turn, think markers, tool markers. The user's
    stop strings are NOT here -- they are text, matched by `Request`.
    Cached per tokenizer and initial state."""
    key = (id(tokenizer), initial)
    hit = _MACHINES.get(key)
    if hit is not None and hit[0] is tokenizer:
        return hit[1], hit[2]
    from .control import ControlMachine

    seqs: dict[tuple[int, ...], str] = {}
    ends: list = []
    for t in tokenizer.eos_token_ids:
        seqs[(t,)] = tokenizer.convert_ids_to_tokens(t)
        ends.append(((t,), None))
    edges: dict[str, list] = {"normal": list(ends)}
    if getattr(tokenizer, "has_thinking", False):
        ts, te = (tuple(tokenizer.think_start_tokens),
                  tuple(tokenizer.think_end_tokens))
        edges["normal"].append((ts, "reasoning"))
        extra = _extra_think_closes(tokenizer)
        edges["reasoning"] = [(te, "normal"),
                              *((q, "normal") for q in extra), *ends]
        seqs[ts], seqs[te] = tokenizer.think_start, tokenizer.think_end
        seqs.update(extra)
    if getattr(tokenizer, "has_tool_calling", False):
        ts = tuple(tokenizer.tool_call_start_tokens)
        te = tuple(tokenizer.tool_call_end_tokens or ())
        edges["normal"].append((ts, "tool"))
        edges["tool"] = ([(te, "normal")] if te else []) + ends
        seqs[ts] = tokenizer.tool_call_start
        if te:
            seqs[te] = tokenizer.tool_call_end
    sm = ControlMachine(edges, initial=initial)
    if len(_MACHINES) > 64:
        _MACHINES.clear()
    _MACHINES[key] = (tokenizer, sm, seqs)
    return sm, seqs


@dataclass
class Delta:
    """What one feed produced, for the wire."""
    reasoning: str = ""
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    #: set once, on the delta that ends the request
    finish: str | None = None
    #: (token, logprob, top) for each token whose text is in this delta,
    #: when the request asked for logprobs
    logprobs: list[tuple] = field(default_factory=list)

    def __bool__(self):
        return bool(self.reasoning or self.content or self.tool_calls
                    or self.finish or self.logprobs)


class Request:
    """One request's text state. `feed(token event)` -> Delta; `finish()`
    when the engine is done with the row (or the request is stopped)."""

    def __init__(self, detokenizer, *, sequences: dict[tuple, str],
                 stops: Sequence[str] = (), tool_parser: Callable = None,
                 tools: Any = None, logprobs: bool = False,
                 prompt_tokens: int = 0, no_tools: bool = False):
        self.detok = detokenizer
        self.detok.reset()
        self.seqs = dict(sequences)
        self.hold_n = max((len(s) for s in sequences), default=1)
        # a think close only matches while reasoning (control_machine), so
        # past the think block the answer is held only as long as the
        # other sequences need: "</thinking>"'s 3-4 tokens would otherwise
        # delay every streamed token of the reply
        self.hold_out = max((len(s) for s, t in sequences.items()
                             if t.strip() not in THINK_CLOSE_TAGS),
                            default=1)
        self.stops = [s for s in (stops or []) if s]
        self.stop_hold = max((len(s) for s in self.stops), default=1) - 1
        self.tool_parser = tool_parser
        self.tools = tools
        #: tool_choice "none": a tool call the model makes is dropped
        self.no_tools = no_tools
        self.want_logprobs = logprobs
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.made_tool_call = False
        self.finished: str | None = None
        #: the answer text withheld while a stop string could be starting
        self._pending = ""
        self._tool_text = ""
        self._prev_state = None
        #: [state, text, token, logprob, top] awaiting control matching
        self._buf: deque = deque()
        self._tool_idx = 0

    # --------------------------------------------------------------- feeding

    def feed(self, tok) -> Delta:
        """One Token event from the executor."""
        out = Delta()
        if self.finished:
            return out
        self.completion_tokens += 1
        if tok.state == "reasoning":
            self.reasoning_tokens += 1
        self.detok.add_token(tok.token)
        self._buf.append([tok.state, self.detok.last_segment, tok.token,
                          tok.logprob, tok.top_logprobs])
        if tok.match is not None:
            # The matched control sequence's tokens say nothing -- but a
            # BPE detokenizer holds a lone space and flushes it WITH the
            # next token's text, so text before the marker's own string
            # is the answer's, not the marker's.
            ents = list(self._buf)[-len(tok.match):]
            joined = "".join(e[1] for e in ents)
            marker = self.seqs.get(tuple(tok.match))
            keep = joined[:joined.index(marker)] \
                if marker and marker in joined else ""
            for e in ents:
                e[1] = ""
            if keep:
                # it belongs to the state BEFORE the marker
                buf = list(self._buf)
                n = len(buf) - len(ents)
                ents[0][0] = buf[n - 1][0] if n > 0 else \
                    (self._prev_state or "normal")
                ents[0][1] = keep
        hold = self.hold_n if tok.state == "reasoning" else self.hold_out
        while len(self._buf) >= hold:
            self._route(self._buf.popleft(), out)
            if self.finished:
                return out
        if tok.finish is not None:
            self._end(tok.finish, out)
        return out

    def finish(self, reason: str = "stop") -> Delta:
        """The row ended without a token saying so (removed, failed)."""
        out = Delta()
        if not self.finished:
            self._end(reason, out)
        return out

    # -------------------------------------------------------------- internal

    def _route(self, entry, out: Delta) -> None:
        state, text, token, lp, top = entry
        if self.want_logprobs and token is not None:
            out.logprobs.append((token, lp, top))
        if state == "reasoning":
            out.reasoning += text
        elif state == "tool":
            self._tool_text += text
        elif state == "normal":
            if self._prev_state == "tool":
                self._close_tool(out)
            self._answer(text, out)
        self._prev_state = state

    def _answer(self, text: str, out: Delta) -> None:
        if not self.stops:
            out.content += text
            return
        self._pending += text
        cut = min((i for i in (self._pending.find(s) for s in self.stops)
                   if i >= 0), default=-1)
        if cut >= 0:
            out.content += self._pending[:cut]
            self._pending = ""
            self._end("stop", out, flush=False)
            return
        keep = self.stop_hold
        if len(self._pending) > keep:
            n = len(self._pending) - keep
            out.content += self._pending[:n]
            self._pending = self._pending[n:]

    def _close_tool(self, out: Delta) -> None:
        text, self._tool_text = self._tool_text, ""
        if not text or self.no_tools:
            return
        self.made_tool_call = True
        out.tool_calls += self._parse_tool(text)

    def _parse_tool(self, text: str) -> list[dict]:
        if self.tool_parser is None:
            return []
        try:
            parsed = self.tool_parser(text, self.tools)
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning("a tool call did not parse (%s: %s); the text "
                           "was probably cut short", type(e).__name__, e)
            return []
        calls = []
        for tc in parsed if isinstance(parsed, list) else [parsed]:
            tc = dict(tc)
            cid = tc.pop("id", None) or f"call_{uuid.uuid4().hex[:24]}"
            args = tc.get("arguments")
            if not isinstance(args, str):
                tc["arguments"] = json.dumps(args, ensure_ascii=False)
            calls.append({"index": self._tool_idx, "id": cid,
                          "type": "function", "function": tc})
            self._tool_idx += 1
        return calls

    def _end(self, reason: str, out: Delta, flush: bool = True) -> None:
        if flush:
            # the tokens still held for control matching, then the
            # detokenizer's own tail
            while self._buf:
                self._route(self._buf.popleft(), out)
                if self.finished:
                    return
            self.detok.finalize()
            tail = self.detok.last_segment
            if tail:
                # text the detokenizer held for a token already routed
                self._route([self._prev_state or "normal", tail, None,
                             None, None], out)
                if self.finished:
                    return
            if self._prev_state == "tool" or self._tool_text:
                self._close_tool(out)
            out.content += self._pending
            self._pending = ""
        self._buf.clear()
        if reason == "stop" and self.made_tool_call:
            reason = "tool_calls"
        self.finished = reason
        out.finish = reason

    # ----------------------------------------------------------------- usage

    def usage(self, cache_report: dict | None = None) -> dict:
        u: dict = {"prompt_tokens": self.prompt_tokens,
             "completion_tokens": self.completion_tokens,
             "total_tokens": self.prompt_tokens + self.completion_tokens}
        if self.reasoning_tokens:
            u["completion_tokens_details"] = {
                "reasoning_tokens": self.reasoning_tokens}
        if cache_report is not None:
            u["prompt_tokens_details"] = {
                "cached_tokens": int(cache_report.get("used", 0))}
            u.setdefault("knurlogic", {})["cache"] = cache_report
        return u
