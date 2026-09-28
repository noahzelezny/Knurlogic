"""Server-side compaction: the harness asks, the server performs.

A chat request (OpenAI's shape; /v1/messages arrives here translated) goes
through `prepare` before it becomes a Job:

  1. every compaction the client resent is folded in (context_edits.view),
  2. the clearing edits are applied in the order given,
  3. a compact edit whose trigger the prompt has passed becomes a `Pending`
     summary pass -- or, with KNURLOGIC_COMPACT_AUTO on, a request that
     asked for nothing does too.

`summarize` runs that pass as a CONTINUATION of the conversation: the
history as the model already saw it (a prefix-cache hit), plus one user
turn asking for the summary and, in the same pass, a one-line finding per
dropped tool call. Only the summary's own tokens cost anything. The
request then runs on the compacted prompt, whose prefill is the fresh
prompt-cache entry the client's next turn hits.

Nothing is kept here: the summary goes back in the response and the client
resends it. Where the pass fails -- an error, an empty answer, one longer
than what it replaces -- the span is dropped behind the harness's marker and the
edit says `fallback: true`; the user's turn never fails for it.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

from knurlogic.interfaces import context_edits as E

logger = logging.getLogger(__name__)

#: a summary pass samples cool and does not think
SUMMARY_TEMPERATURE = 0.2
#: room for one finding line per dropped tool call, beyond the budget
TOKENS_PER_FINDING = 48


@dataclass
class Pending:
    """A summary pass to run: the view it summarizes, which span goes."""
    edit: E.Compact
    view: list
    plan: E.Plan
    uses: list
    budget: int
    dropped_tokens: int
    tokens: int                  # the view's prompt, before


@dataclass
class Outcome:
    """What the response says about context management."""
    applied: list = field(default_factory=list)
    compaction: Optional[str] = None
    pause: bool = False
    #: the summary pass's usage: {input_tokens, output_tokens,
    #: cache_read_input_tokens}
    iteration: Optional[dict] = None


def settings(env=None) -> dict:
    from knurlogic.tuning.settings import compact_settings
    return compact_settings(os.environ if env is None else env)


def _trigger(edit: E.Compact, window: int, cfg: dict) -> int:
    if edit.trigger is not None and not edit.auto:
        return edit.trigger
    frac = int(window * cfg["trigger"]) if window else 0
    if edit.auto:
        return frac or E.COMPACT_TRIGGER_DEFAULT
    # the API's default, unless the model's window is smaller than it
    return min(E.COMPACT_TRIGGER_DEFAULT, frac) if frac else \
        E.COMPACT_TRIGGER_DEFAULT


def prepare(body: dict, *, count: Callable[[list, list], int],
            window: int = 0, env=None) -> tuple:
    """(the body to run, Outcome, Pending or None). `count(messages,
    tools)` is the prompt's length in the served model's tokens.
    EditError: a context_management to refuse."""
    cfg = settings(env)
    edits = E.parse(body.get("context_management"))
    if not edits and cfg["auto"] and body.get("context_management") is None:
        edits = [E.Compact(auto=True)]
    compact = next((e for e in edits if isinstance(e, E.Compact)), None)
    keep = compact.keep if compact and compact.keep is not None \
        else cfg["keep"]
    msgs = body.get("messages") or []
    viewed, cuts = E.view(msgs, keep)
    out = Outcome()
    tools = body.get("tools") or []
    tokens = None

    def ntok(text: str) -> int:
        return count([{"role": "user", "content": text}], []) if text else 0

    for e in edits:
        if isinstance(e, E.ClearThinking):
            viewed, a = E.clear_thinking(viewed, e)
        elif isinstance(e, E.ClearTools):
            tokens = count(viewed, tools) if tokens is None else tokens
            viewed, a = E.clear_tool_uses(viewed, e, tokens, ntok)
            if a:
                tokens = None
        else:
            continue
        if a:
            out.applied.append(a)

    pending = None
    if compact is not None:
        tokens = count(viewed, tools) if tokens is None else tokens
        p = E.plan(viewed, keep)
        if tokens >= _trigger(compact, window, cfg) and p is not None:
            kept = E.compacted(viewed, p, "")
            dropped = max(tokens - count(kept, tools), 0)
            pending = Pending(compact, viewed, p,
                              E.tool_uses(viewed[p.start:p.end]),
                              E.budget(dropped, cfg["summary_min"],
                                       cfg["summary_max"]),
                              dropped, tokens)
    run = dict(body, messages=viewed)
    run.pop("context_management", None)
    return run, out, pending


def summary_body(body: dict, pending: Pending) -> dict:
    """The summary pass: the conversation as it stands, plus the ask.
    Same model, tools and template kwargs as the request, so the rendered
    history is the prefix the prompt cache holds."""
    ask = E.prompt(pending.budget, pending.uses, pending.edit.instructions)
    b = {k: v for k, v in body.items()
         if k in ("model", "tools", "chat_template_kwargs", "role_mapping")}
    b.update(messages=list(pending.view) + [{"role": "user", "content": ask}],
             max_tokens=pending.budget
             + TOKENS_PER_FINDING * len(pending.uses) + 256,
             temperature=SUMMARY_TEMPERATURE, stream=False,
             reasoning_effort="none", reasoning={"exclude": True})
    return b


def summarize(body: dict, pending: Pending, out: Outcome,
              generate: Callable[[dict], dict], env=None) -> dict:
    """Run the pass (`generate(openai_body) -> completion dict`), fill
    `out`, and return the body to run on the compacted prompt."""
    cfg = settings(env)
    p, uses = pending.plan, pending.uses
    text, usage, why = "", {}, None
    try:
        resp = generate(summary_body(body, pending))
        msg = ((resp.get("choices") or [{}])[0].get("message") or {})
        text = msg.get("content") or ""
        usage = resp.get("usage") or {}
    except Exception as e:                       # fail soft, as the harness does
        why = f"{type(e).__name__}: {e}"
        logger.warning("compaction summary failed: %s", why)
    summary, found = E.parse_output(text, len(uses))
    dropped_chars = sum(len(E._text(m.get("content")) or "")
                        for m in pending.view[p.start:p.end])
    if why is None and not summary:
        why = "the summary pass returned no text"
    elif why is None and dropped_chars and len(summary) > dropped_chars:
        why = "the summary was longer than what it replaces"
    fallback = why is not None
    if fallback:
        summary, found = E.fallback_summary(p.dropped), {}
    final, distilled, cleared = E.render(summary, uses, found,
                                         distill=cfg["distill"])
    compacted = E.compacted(pending.view, p, final)
    cached = int((usage.get("prompt_tokens_details") or {})
                 .get("cached_tokens", 0) or 0)
    out.compaction = final
    out.pause = pending.edit.pause
    out.iteration = {"type": "compaction",
                     "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
                     "output_tokens": int(usage.get("completion_tokens", 0)
                                          or 0),
                     "cache_read_input_tokens": cached}
    applied = {"type": E.COMPACT, "summarized_messages": p.dropped,
               "kept_messages": len(pending.view) - p.end,
               "cleared_input_tokens": pending.dropped_tokens,
               "distilled_tool_uses": distilled,
               "cleared_tool_uses": cleared,
               "summary_budget": pending.budget,
               "input_tokens_before": pending.tokens}
    if pending.edit.auto:
        applied["automatic"] = True
    if fallback:
        applied.update(fallback=True, reason=why)
    out.applied.append(applied)
    return dict(body, messages=compacted)


# ------------------------------------------------------------ responses

def attach(resp: dict, out: Outcome) -> dict:
    """An OpenAI completion with what context management did: the summary
    as `message.compaction`, `context_management.applied_edits`, and the
    summary pass's usage under usage.knurlogic.compaction."""
    if out.compaction is not None:
        msg = resp["choices"][0].setdefault("message", {})
        msg["compaction"] = out.compaction
    if out.applied:
        resp["context_management"] = {"applied_edits": out.applied}
    if out.iteration:
        resp.setdefault("usage", {}).setdefault("knurlogic", {})[
            "compaction"] = out.iteration
    return resp


def paused(out: Outcome, *, id_: str, created: int, model: str,
           context: Optional[dict] = None) -> dict:
    """The whole response when the edit said pause_after_compaction: the
    summary alone, finish_reason "compaction"."""
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
             "knurlogic": {"compaction": out.iteration}}
    if context:
        usage["knurlogic"]["context"] = context
    return attach({"id": id_, "object": "chat.completion", "created": created,
                   "model": model,
                   "choices": [{"index": 0, "finish_reason": "compaction",
                                "message": {"role": "assistant",
                                            "content": None}}],
                   "usage": usage}, out)
