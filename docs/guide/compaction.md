# Compaction

A long agent conversation eventually fills the model's context window.
Knurlogic can shorten it for you: older turns become a summary, written by
the model already serving the request, and the recent turns stay word for
word. Your client asks; the server does it.

Nothing is stored on the server. The summary comes back in the response
and the client sends it again with the rest of its history. The history is
never rewritten silently.

## How to ask

Anthropic's `context_management` field (the one Claude Code sends), on
`/v1/messages`. The same object is accepted on `/v1/chat/completions`.

```json
"context_management": {"edits": [
  {"type": "compact_20260112",
   "trigger": {"type": "input_tokens", "value": 150000},
   "pause_after_compaction": false,
   "instructions": null}
]}
```

| edit | what it does | defaults |
|---|---|---|
| `compact_20260112` | summarize older turns once the prompt passes `trigger` | trigger 150k input tokens (the platform's 50k minimum is not enforced). Knurlogic adds `keep: {"type": "messages", "value": N}`, the tail kept verbatim. |
| `clear_tool_uses_20250919` | clear old tool results | trigger 100k input tokens; keep 3 tool uses; also `clear_at_least`, `exclude_tools`, `clear_tool_inputs` |
| `clear_thinking_20251015` | drop old thinking | `keep: "all"` or `{"type": "thinking_turns", "value": N}` |

Custom `instructions` replace the summary prompt.

## What comes back

| | `/v1/messages` | `/v1/chat/completions` |
|---|---|---|
| the summary | a leading `compaction` block | `message.compaction` (`delta.compaction` when streaming) |
| a pause (`pause_after_compaction`) | `stop_reason: "compaction"` | `finish_reason: "compaction"` |
| edits applied | `context_management.applied_edits` | top-level `context_management.applied_edits` |
| usage of the summary pass | `usage.iterations` | `usage.knurlogic.compaction` |
| what to resend | the block; everything before it is ignored | a message with `compaction`, or a system message in `<knurlogic:compaction>` |

## What is kept

- The system message, the first user message (the goal) and the last N
  messages stay.
- The kept tail never starts on a tool result; it is widened back to the
  call that asked for it.
- Nothing happens unless at least two messages would go.
- Each dropped tool result becomes a one-line finding ("X is defined at
  src/foo.py:120") written in the same pass as the summary (**distill**),
  or nothing (**clear**).
- A summary is never longer than what it replaces. If the pass fails or
  would be longer, the span is dropped behind a marker and the edit says
  `fallback: true`; your request still succeeds.

The summary pass continues the cached conversation, so only the summary's
own tokens cost time. The compacted prompt becomes a fresh prompt-cache
entry for your next turn.

## When to ask

Every chat response carries `usage.knurlogic.context = {tokens, window}`:
the prompt against the model's window.

## Server-side settings

Settings -> Knurlogic -> Compaction applies to every model on every
machine, from the next request.

| setting (page name) | variable | default | values |
|---|---|---|---|
| Compact unasked | `KNURLOGIC_COMPACT_AUTO` | off | off, on: compact a request that asked for nothing once it passes the trigger |
| Auto compact | `KNURLOGIC_COMPACT_TRIGGER` | 0.8 | share of the window: 0.5-0.9. Also used by a compact edit with no trigger when the window is below 150k. |
| Keep recent | `KNURLOGIC_COMPACT_KEEP_TURNS` | 6 | messages kept word for word: 2, 4, 6, 8, 12, 16 |
| Dropped tool results | `KNURLOGIC_COMPACT_TOOL_RESULTS` | distill | distill (slower, keeps findings), clear (faster; the model may repeat calls) |

With automatic compaction off, a client that never asks is refused once
its prompt is past the window.

## Limits

- Changing Keep recent between turns shifts the kept tail.
- Custom `instructions` replace the summary prompt, but the findings
  request is still added.
- Not implemented: the on-demand mode (`compact-2026-09-04`, top-level
  `compaction: {type: summarize}`).

Design: [../design/compaction.md](../design/compaction.md).
