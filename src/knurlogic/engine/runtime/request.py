"""One request's text, from the executor's token events to what the client
reads (docs/design/server.md, build step 2). No model and no mlx here: this is
where tokens become reasoning, answer and tool calls, where a stop string
ends the answer, and where the usage numbers are counted.

  control tokens   think/tool markers and end-of-turn tokens are matched as
                   TOKEN sequences by the engine's state machine (built by
                   `control_machine`); their text never reaches the client.
                   A marker can be several tokens, so the last few tokens'
                   text is held until no marker can still be completing.
  reasoning split  a token's state says where its text goes: reasoning,
                   tool, or the answer.
  stop strings     the request's `stop` matches the detokenized ANSWER text
                   -- never reasoning, never a tool call -- across token
                   boundaries (mlx-lm matched them as token ids, so
                   `stop: "D"` missed a token " D"). The last
                   max(len(stop)) - 1 characters are held back, so a stop
                   never leaks into a streamed delta before it is seen.
  tool calls       the text between tool markers, parsed by the tokenizer's
                   own tool parser when the call closes.
  usage            prompt, completion (every token the engine emitted),
                   reasoning tokens, and the engine's cache report.

The object is fed on the scheduler's thread and read on the HTTP thread
only through the deltas it returns, so it holds no lock.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_MACHINES: dict = {}


def control_machine(tokenizer, initial: str = "normal"):
    """(SequenceStateMachine, {token sequence: its text}) for the tokenizer's
    control tokens: end-of-turn, think markers, tool markers. The user's
    stop strings are NOT here -- they are text, matched by `Request`.
    Cached per tokenizer and initial state."""
    key = (id(tokenizer), initial)
    hit = _MACHINES.get(key)
    if hit is not None and hit[0] is tokenizer:
        return hit[1], hit[2]
    import importlib
    SequenceStateMachine = importlib.import_module(
        "mlx_lm.generate").SequenceStateMachine

    seqs: Dict[Tuple[int, ...], str] = {}
    ends = []
    for t in tokenizer.eos_token_ids:
        seqs[(t,)] = tokenizer.convert_ids_to_tokens(t)
        ends.append(((t,), None))
    edges = {"normal": list(ends)}
    if getattr(tokenizer, "has_thinking", False):
        ts, te = (tuple(tokenizer.think_start_tokens),
                  tuple(tokenizer.think_end_tokens))
        edges["normal"].append((ts, "reasoning"))
        edges["reasoning"] = [(te, "normal"), *ends]
        seqs[ts], seqs[te] = tokenizer.think_start, tokenizer.think_end
    if getattr(tokenizer, "has_tool_calling", False):
        ts = tuple(tokenizer.tool_call_start_tokens)
        te = tuple(tokenizer.tool_call_end_tokens or ())
        edges["normal"].append((ts, "tool"))
        edges["tool"] = ([(te, "normal")] if te else []) + ends
        seqs[ts] = tokenizer.tool_call_start
        if te:
            seqs[te] = tokenizer.tool_call_end
    sm = SequenceStateMachine(edges, initial=initial)
    if len(_MACHINES) > 64:
        _MACHINES.clear()
    _MACHINES[key] = (tokenizer, sm, seqs)
    return sm, seqs


@dataclass
class Delta:
    """What one feed produced, for the wire."""
    reasoning: str = ""
    content: str = ""
    tool_calls: List[dict] = field(default_factory=list)
    #: set once, on the delta that ends the request
    finish: Optional[str] = None
    #: (token, logprob, top) for each token whose text is in this delta,
    #: when the request asked for logprobs
    logprobs: List[tuple] = field(default_factory=list)

    def __bool__(self):
        return bool(self.reasoning or self.content or self.tool_calls
                    or self.finish or self.logprobs)


class Request:
    """One request's text state. `feed(token event)` -> Delta; `finish()`
    when the engine is done with the row (or the request is stopped)."""

    def __init__(self, detokenizer, *, sequences: Dict[tuple, str],
                 stops: List[str] = (), tool_parser: Callable = None,
                 tools: Any = None, logprobs: bool = False,
                 prompt_tokens: int = 0):
        self.detok = detokenizer
        self.detok.reset()
        self.seqs = dict(sequences)
        self.hold_n = max((len(s) for s in sequences), default=1)
        self.stops = [s for s in (stops or []) if s]
        self.stop_hold = max((len(s) for s in self.stops), default=1) - 1
        self.tool_parser = tool_parser
        self.tools = tools
        self.want_logprobs = logprobs
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.made_tool_call = False
        self.finished: Optional[str] = None
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
        while len(self._buf) >= self.hold_n:
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
        if not text:
            return
        self.made_tool_call = True
        out.tool_calls += self._parse_tool(text)

    def _parse_tool(self, text: str) -> List[dict]:
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

    def usage(self, cache_report: Optional[dict] = None) -> dict:
        u = {"prompt_tokens": self.prompt_tokens,
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
