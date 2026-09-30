"""Server-side compaction: the harness asks, the server performs.

`prepare` folds in resent compactions (context_edits.view), applies the
clearing edits in order, and turns a compact edit whose trigger the prompt
has passed (or any request, with KNURLOGIC_COMPACT_AUTO on) into a
`Pending` summary pass. `summarize` runs that pass as a continuation of the
conversation, so only the summary's own tokens cost anything. Nothing is
kept server-side; if the pass fails, the span is dropped behind a backstop
marker with `fallback: true` and the user's turn still runs.

Design: docs/design/compaction.md.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from knurlogic.context_management import context_edits as E

logger = logging.getLogger(__name__)

#: a summary pass samples cool and does not think
SUMMARY_TEMPERATURE = 0.2
#: room for one finding line per dropped tool call, beyond the dropped span
TOKENS_PER_FINDING = 48


@dataclass
class Pending:
    """A summary pass to run: the view it summarizes, which span goes."""
    edit: E.Compact
    view: list
    plan: E.Plan
    uses: list
    dropped_tokens: int
    tokens: int                  # the view's prompt, before


@dataclass
class Outcome:
    """What the response says about context management."""
    applied: list = field(default_factory=list)
    compaction: str | None = None
    pause: bool = False
    #: the summary pass's usage: {input_tokens, output_tokens,
    #: cache_read_input_tokens}
    iteration: dict | None = None


def settings(env=None) -> dict:
    """The operator's compaction settings. With no `env`: this process's
    environment under the knurlogic-wide ones (machine/preferences), read
    per request so a change in Settings -> Knurlogic applies to the next
    request of every running server."""
    from knurlogic.tuning.settings import compact_settings
    if env is not None:
        return compact_settings(env)
    from knurlogic.machine import preferences
    return compact_settings(preferences.compaction_env())


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
                              dropped, tokens)
    run = dict(body, messages=viewed)
    run.pop("context_management", None)
    return run, out, pending


def summary_body(body: dict, pending: Pending) -> dict:
    """The summary pass: the conversation as it stands, plus the ask.
    Same model, tools and template kwargs as the request, so the rendered
    history is the prefix the prompt cache holds."""
    ask = E.prompt(pending.uses, pending.edit.instructions)
    b = {k: v for k, v in body.items()
         if k in ("model", "tools", "chat_template_kwargs", "role_mapping")}
    b.update(messages=list(pending.view) + [{"role": "user", "content": ask}],
             # never longer than what it replaces: the dropped span, plus a
             # finding line per dropped tool call
             max_tokens=pending.dropped_tokens
             + TOKENS_PER_FINDING * len(pending.uses),
             temperature=SUMMARY_TEMPERATURE, stream=False,
             reasoning_effort="none", reasoning={"exclude": True})
    return b


def summarize(body: dict, pending: Pending, out: Outcome,
              generate: Callable[[dict], dict], env=None) -> dict:
    """Run the pass (`generate(openai_body) -> completion dict`), fill
    `out`, and return the body to run on the compacted prompt."""
    cfg = settings(env)
    p, uses = pending.plan, pending.uses
    text, why = "", None
    usage: dict = {}
    try:
        resp = generate(summary_body(body, pending))
        msg = ((resp.get("choices") or [{}])[0].get("message") or {})
        text = msg.get("content") or ""
        usage = resp.get("usage") or {}
    # fail soft  # the model call fails soft, logged below
    except Exception as e:
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
           context: dict | None = None) -> dict:
    """The whole response when the edit said pause_after_compaction: the
    summary alone, finish_reason "compaction"."""
    usage: dict = {"prompt_tokens": 0, "completion_tokens": 0,
                   "total_tokens": 0,
                   "knurlogic": {"compaction": out.iteration}}
    if context:
        usage["knurlogic"]["context"] = context
    return attach({"id": id_, "object": "chat.completion", "created": created,
                   "model": model,
                   "choices": [{"index": 0, "finish_reason": "compaction",
                                "message": {"role": "assistant",
                                            "content": None}}],
                   "usage": usage}, out)
