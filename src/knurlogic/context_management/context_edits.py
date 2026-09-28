"""Context edits over OpenAI-shaped messages: the history surgery behind
`context_management`, without a model.

WHAT A HARNESS SENDS. Anthropic's `context_management.edits` (checked
against platform.claude.com, 2026-09-28):

    {"type": "clear_thinking_20251015", "keep": "all" | {"type":
     "thinking_turns", "value": N}}
    {"type": "clear_tool_uses_20250919", "trigger": {"type": "input_tokens"
     | "tool_uses", "value": N}, "keep": {"type": "tool_uses", "value": 3},
     "clear_at_least": {"type": "input_tokens", "value": N},
     "exclude_tools": [...], "clear_tool_inputs": false}
    {"type": "compact_20260112", "trigger": {"type": "input_tokens",
     "value": 150000}, "pause_after_compaction": false, "instructions": null}

The same object is accepted on /v1/chat/completions. knurlogic adds one
field to the compact edit, `keep: {"type": "messages", "value": N}` (the
tail kept verbatim; the operator's default otherwise).

NOTHING IS STORED. A compaction goes back to the client -- a `compaction`
content block (Anthropic) or `message.compaction` (OpenAI) -- and the
client resends it with the rest of its history. On input the server folds
the history over each compaction it finds (`view`): what came before it
becomes the first user message, the summary and the tail that was kept,
exactly as the model saw it when the summary was written, so the rewrite
is the same every turn and its prompt a prefix-cache hit.

THE INVARIANTS are the harness's (scout/tasks/agent_loop_compaction.py): the
leading system message and the first user message (the goal) are kept;
the last N messages are kept; the kept tail never starts on a tool result
(it is widened back to the call that asked for it); nothing is done unless
at least two messages would go.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

COMPACT = "compact_20260112"
CLEAR_TOOLS = "clear_tool_uses_20250919"
CLEAR_THINKING = "clear_thinking_20251015"

#: the API's own default trigger for a compact edit
COMPACT_TRIGGER_DEFAULT = 150_000
CLEAR_TOOLS_TRIGGER_DEFAULT = 100_000
CLEARED = "[tool result cleared to bound context]"
#: a system message whose content is exactly this form is a cut point too,
#: so any OpenAI client can round-trip a compaction as plain text
WRAP = ("<knurlogic:compaction>", "</knurlogic:compaction>")
#: how the summary is shown to the model -- fixed, so the rendered prompt
#: is byte-identical every turn (a prefix-cache hit)
SUMMARY_HEAD = ("This conversation was compacted to bound its context. "
                "Summary of the earlier part:\n\n")
MIN_DROP = 2


class EditError(ValueError):
    """A context_management the server cannot honour: a 400 naming it."""


@dataclass
class Compact:
    trigger: Optional[int] = None        # input tokens; None: the default
    pause: bool = False
    instructions: Optional[str] = None
    keep: Optional[int] = None           # messages; None: the operator's
    auto: bool = False                   # the server's default, unasked


@dataclass
class ClearTools:
    trigger_type: str = "input_tokens"
    trigger: int = CLEAR_TOOLS_TRIGGER_DEFAULT
    keep: int = 3
    clear_at_least: int = 0
    exclude: set = field(default_factory=set)
    clear_inputs: bool = False


@dataclass
class ClearThinking:
    keep: object = 1                     # "all" or a number of turns


def _value(obj, name, kinds, default):
    """`{"type": ..., "value": N}` -> N, checked."""
    if obj is None:
        return default
    if not isinstance(obj, dict) or not isinstance(obj.get("value"), int) \
            or isinstance(obj.get("value"), bool) or obj["value"] < 0:
        raise EditError(f"{name} must be {{\"type\": ..., \"value\": a "
                        f"non-negative integer}}")
    if obj.get("type") not in kinds:
        raise EditError(f"{name}.type must be one of {sorted(kinds)}")
    return obj["value"]


def parse(cm) -> list:
    """`context_management` -> [edit], in order; EditError to refuse. An
    unknown edit type is refused by name, never ignored."""
    if cm is None:
        return []
    if not isinstance(cm, dict):
        raise EditError("context_management must be an object")
    edits = cm.get("edits") or []
    if not isinstance(edits, list):
        raise EditError("context_management.edits must be a list")
    out = []
    for i, e in enumerate(edits):
        where = f"context_management.edits[{i}]"
        if not isinstance(e, dict):
            raise EditError(f"{where} must be an object")
        t = e.get("type")
        if t == COMPACT:
            instr = e.get("instructions")
            if instr is not None and not isinstance(instr, str):
                raise EditError(f"{where}.instructions must be a string")
            out.append(Compact(
                trigger=_value(e.get("trigger"), f"{where}.trigger",
                               {"input_tokens"}, None),
                pause=bool(e.get("pause_after_compaction", False)),
                instructions=(instr or None) and instr.strip() or None,
                keep=_value(e.get("keep"), f"{where}.keep", {"messages"},
                            None)))
        elif t == CLEAR_TOOLS:
            ex = e.get("exclude_tools") or []
            if not isinstance(ex, list):
                raise EditError(f"{where}.exclude_tools must be a list")
            tr = e.get("trigger") or {"type": "input_tokens",
                                      "value": CLEAR_TOOLS_TRIGGER_DEFAULT}
            out.append(ClearTools(
                trigger_type=(tr.get("type") if isinstance(tr, dict)
                              else "input_tokens"),
                trigger=_value(tr, f"{where}.trigger",
                               {"input_tokens", "tool_uses"},
                               CLEAR_TOOLS_TRIGGER_DEFAULT),
                keep=_value(e.get("keep"), f"{where}.keep", {"tool_uses"}, 3),
                clear_at_least=_value(e.get("clear_at_least"),
                                      f"{where}.clear_at_least",
                                      {"input_tokens"}, 0),
                exclude=set(map(str, ex)),
                clear_inputs=bool(e.get("clear_tool_inputs", False))))
        elif t == CLEAR_THINKING:
            k = e.get("keep", {"type": "thinking_turns", "value": 1})
            out.append(ClearThinking(
                keep="all" if k == "all" else
                _value(k, f"{where}.keep", {"thinking_turns"}, 1)))
        else:
            raise EditError(f"{where}.type {t!r} is not an edit this server "
                            f"performs ({CLEAR_THINKING}, {CLEAR_TOOLS}, "
                            f"{COMPACT})")
    return out


# ------------------------------------------------------------ cut points

def cut_of(m: dict) -> Optional[str]:
    """The summary a message carries, or None: an assistant message's
    `compaction`, or a system message that is exactly WRAP-wrapped."""
    if not isinstance(m, dict):
        return None
    c = m.get("compaction")
    if isinstance(c, str):
        return c
    content = m.get("content")
    if m.get("role") == "system" and isinstance(content, str):
        s = content.strip()
        if s.startswith(WRAP[0]) and s.endswith(WRAP[1]):
            return s[len(WRAP[0]):-len(WRAP[1])].strip("\n")
    return None


def _rest_of_cut(m: dict) -> Optional[dict]:
    """What a cut message says besides its summary, as a message; None
    when nothing."""
    if m.get("role") == "system":
        return None
    rest = {k: v for k, v in m.items() if k != "compaction"}
    if rest.get("content") or rest.get("tool_calls"):
        return rest
    return None


def wrap(summary: str) -> str:
    return f"{WRAP[0]}\n{summary}\n{WRAP[1]}"


def summary_message(summary: str) -> dict:
    return {"role": "user", "content": SUMMARY_HEAD + summary}


# ------------------------------------------------------------ the plan

@dataclass
class Plan:
    """Which messages of a history go: `lead` system messages and the goal
    at `goal` stay, [start, end) is dropped, [end:] is the kept tail."""
    lead: int
    goal: Optional[int]
    start: int
    end: int

    @property
    def dropped(self) -> int:
        return self.end - self.start


def plan(msgs: list, keep: int) -> Optional[Plan]:
    """the harness's policy over OpenAI messages; None when fewer than MIN_DROP
    would go."""
    lead = 0
    while lead < len(msgs) and msgs[lead].get("role") == "system":
        lead += 1
    goal = next((i for i in range(lead, len(msgs))
                 if msgs[i].get("role") == "user"), None)
    start = (goal + 1) if goal is not None else lead
    end = max(start, len(msgs) - max(int(keep), 0))
    # never start the tail on an orphaned tool result: widen it back to
    # the assistant message whose calls they answer
    while start < end < len(msgs) and msgs[end].get("role") == "tool":
        end -= 1
    if end - start < MIN_DROP:
        return None
    return Plan(lead, goal, start, end)


def compacted(msgs: list, p: Plan, summary: str) -> list:
    head = list(msgs[:p.lead])
    if p.goal is not None:
        head.append(msgs[p.goal])
    return head + [summary_message(summary)] + list(msgs[p.end:])


def view(msgs: list, keep: int) -> tuple:
    """(the messages the model sees, the number of cuts folded). Each cut
    rewrites what came before it as it was rewritten when that summary was
    written: the head, the summary, the tail kept then."""
    out, cuts = [], 0
    for m in msgs:
        s = cut_of(m)
        if s is None:
            out.append(m)
            continue
        cuts += 1
        p = plan(out, keep)
        if p is None:
            # too short to have dropped anything: the summary stands in
            # front of the whole of it
            lead = 0
            while lead < len(out) and out[lead].get("role") == "system":
                lead += 1
            out = out[:lead] + [summary_message(s)] + out[lead:]
        else:
            out = compacted(out, p, s)
        rest = _rest_of_cut(m)
        if rest is not None:
            out.append(rest)
    return out, cuts


# ------------------------------------------------------------ tool calls

@dataclass
class ToolUse:
    n: int                     # T<n> in the prompt and the findings
    id: str
    name: str
    args: str
    result: Optional[str]      # None: no result in the span


def _args(fn: dict, width: int = 160) -> str:
    raw = fn.get("arguments") or ""
    try:
        raw = json.dumps(json.loads(raw), separators=(",", ":"))
    except (TypeError, ValueError):
        pass
    return raw if len(raw) <= width else raw[:width - 1] + "…"


def _text(c) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c
                       if isinstance(p, dict) and p.get("type") == "text")
    return ""


def tool_uses(msgs: list) -> List[ToolUse]:
    """Every tool call in `msgs`, in order, with its result."""
    results = {m.get("tool_call_id"): _text(m.get("content"))
               for m in msgs if m.get("role") == "tool"}
    out = []
    for m in msgs:
        for c in m.get("tool_calls") or []:
            fn = c.get("function") or {}
            out.append(ToolUse(len(out) + 1, c.get("id") or "",
                               fn.get("name") or "?", _args(fn),
                               results.get(c.get("id"))))
    return out


# ------------------------------------------------------------ the prompt

def budget(dropped_tokens: int, lo: int, hi: int) -> int:
    """A tenth of what is dropped, clamped to [lo, hi]."""
    return int(min(max(dropped_tokens // 10, lo), hi))


# the harness's prompt (scout/coherence/summarizer.py _build_prompt), extended
# for a coding agent: its four headings, plus files touched, the current and
# next step, and open errors; tool results distilled, not dropped.
SUMMARY_PROMPT = """\
The conversation above is being compacted to bound its context. Your \
summary replaces the older messages: whoever continues this work will see \
the first user message, your summary, and the most recent messages \
verbatim -- nothing else.

Summarize the conversation in concise bullet points under these headings:
## Goal
## Decisions made
## Information gathered (key tool results)
## Files and identifiers touched
## Current step
## Next step
## Open errors and questions

Keep exact file paths, line numbers, identifiers, commands and error \
messages. Keep within {budget} tokens (~{chars} characters). Do not call \
tools. Return ONLY the summary."""

FINDINGS_PROMPT = """\

Then add a line "## Tool findings" and, for each tool call listed below, \
one line "T<n>: <finding>" saying in one sentence what that call \
established -- the answer to what it was for, not its raw output (a search \
for where X is defined: "T3: X is defined at src/foo.py:120"). If it \
established nothing, write "T<n>: nothing relevant".

{calls}"""


def prompt(budget_tokens: int, uses: List[ToolUse],
           instructions: Optional[str] = None) -> str:
    """The user turn that asks for the summary, appended to the
    conversation as it stands (so the history is a prefix-cache hit).
    `instructions` replace the summary prompt, as the API documents; the
    tool findings are still asked for."""
    head = instructions or SUMMARY_PROMPT.format(
        budget=budget_tokens, chars=budget_tokens * 4)
    if not uses:
        return head
    calls = "\n".join(f"T{u.n}: {u.name} {u.args}" for u in uses)
    return head + FINDINGS_PROMPT.format(calls=calls)


_THINK = re.compile(r"<think>.*?</think>", re.S)
_FIND_HEAD = re.compile(r"^\s*#*\s*\**\s*Tool findings\b.*$", re.I | re.M)
_FIND = re.compile(r"^\s*[-*]?\s*\**T(\d+)\**\s*[:.)\-]\s*(.+?)\s*$", re.M)


def parse_output(text: str, n_uses: int) -> tuple:
    """(summary, {n: finding}) from the model's answer. A finding line
    anywhere after the heading counts; one for no listed call is
    ignored."""
    text = _THINK.sub("", text or "").strip()
    m = _FIND_HEAD.search(text)
    summary, tail = (text[:m.start()], text[m.end():]) if m else (text, "")
    found = {}
    for f in _FIND.finditer(tail):
        n = int(f.group(1))
        if 1 <= n <= n_uses and f.group(2).strip():
            found.setdefault(n, f.group(2).strip())
    return summary.strip(), found


def render(summary: str, uses: List[ToolUse], found: dict,
           distill: bool = True) -> tuple:
    """(the compaction's text, distilled, cleared): the summary, then one
    line per dropped tool call -- its finding, or a clear where
    distillation failed (or is off)."""
    lines, distilled, cleared = [], 0, 0
    for u in uses:
        f = found.get(u.n) if distill else None
        if f:
            distilled += 1
            lines.append(f"- {u.name} {u.args} -> {f}")
        else:
            cleared += 1
            lines.append(f"- {u.name} {u.args} -> {CLEARED}")
    text = summary.strip()
    if lines:
        text += ("\n\n## Tool findings (earlier tool calls)\n"
                 + "\n".join(lines))
    return text, distilled, cleared


def fallback_summary(n_dropped: int) -> str:
    """the harness's backstop marker: the span went, and no summary could be
    made."""
    return (f"{n_dropped} earlier messages were removed to bound the "
            f"context; no summary could be made of them.")


# ------------------------------------------------------------ clearing

def clear_thinking(msgs: list, e: ClearThinking) -> tuple:
    """(messages, applied edit): reasoning dropped from all but the last
    `keep` assistant turns that carry it."""
    if e.keep == "all":
        return msgs, None
    idx = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"
           and (m.get("reasoning_content") or m.get("reasoning"))]
    drop = idx[:max(len(idx) - int(e.keep), 0)]
    if not drop:
        return msgs, None
    out = list(msgs)
    for i in drop:
        out[i] = {k: v for k, v in out[i].items()
                  if k not in ("reasoning_content", "reasoning")}
    return out, {"type": CLEAR_THINKING, "cleared_thinking_turns": len(drop),
                 "cleared_input_tokens": 0}


def clear_tool_uses(msgs: list, e: ClearTools, tokens: int,
                    ntok: Callable[[str], int]) -> tuple:
    """(messages, applied edit): past the trigger, the oldest tool results
    beyond the last `keep` tool uses become a placeholder (the call stays,
    so the pair is intact). Nothing happens unless at least
    `clear_at_least` tokens would go."""
    uses = [(i, j, c) for i, m in enumerate(msgs)
            for j, c in enumerate(m.get("tool_calls") or [])]
    fired = (len(uses) >= e.trigger if e.trigger_type == "tool_uses"
             else tokens >= e.trigger)
    if not fired or len(uses) <= e.keep:
        return msgs, None
    old = uses[:len(uses) - e.keep]
    ids = {c.get("id"): (i, j) for i, j, c in old
           if (c.get("function") or {}).get("name") not in e.exclude}
    out, cleared, saved = list(msgs), 0, 0
    for k, m in enumerate(out):
        if m.get("role") == "tool" and m.get("tool_call_id") in ids:
            text = _text(m.get("content"))
            if text == CLEARED:
                continue
            saved += max(ntok(text) - ntok(CLEARED), 0)
            out[k] = dict(m, content=CLEARED)
            cleared += 1
    if e.clear_inputs:
        for i, j in ids.values():
            calls = [dict(c) for c in out[i]["tool_calls"]]
            fn = dict(calls[j].get("function") or {})
            saved += ntok(fn.get("arguments") or "")
            fn["arguments"] = "{}"
            calls[j]["function"] = fn
            out[i] = dict(out[i], tool_calls=calls)
    if not cleared or saved < e.clear_at_least:
        return msgs, None
    return out, {"type": CLEAR_TOOLS, "cleared_tool_uses": cleared,
                 "cleared_input_tokens": saved}
