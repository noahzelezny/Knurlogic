# Context compaction

Any harness can plug into knurlogic without managing its agents' context by
hand: the harness requests compaction, the server performs it.

## The harness requests, the server performs

- The harness requests it through Anthropic's `context_management` (the field
  Anthropic-compatible coding harnesses already send). The same object is accepted on `/v1/chat/completions` as
  an extension.
- The server runs the compaction with the model already serving the request, as
  a continuation of the cached conversation: the history is a prefix-cache hit
  and only the summary's own tokens cost. The compacted prefix becomes a fresh
  prompt-cache entry.
- Nothing is stored server-side: the summary goes back in the response and the
  client resends it. The server never rewrites history silently -- a stateless
  client keeps resending the full history and the two views would drift.
- Automatic (operator-default) compaction exists and is off by default.

## Distill, don't clear

A tool call usually exists to answer a question ("where is X defined?").
Outside the recent tail, each dropped tool result becomes a one-line finding
(`T<n>: X is defined at src/foo.py:120`), written in the same model pass as the
summary; a clear placeholder is the fallback. The recent tail stays verbatim
(the agent may still be using it).

## Summary prompt

Sections: Goal, Decisions made, Information gathered, Files and identifiers
touched, Current step, Next step, Open errors and questions. The prompt lives as
markdown, not a Python string (`context_management/prompts/compact.md`, and
`findings.md` for the per-call findings).

There is no token budget; the one rule is that a summary is never longer than
what it replaces. The prompt asks for the shortest summary that keeps
everything needed to continue; the pass's `max_tokens` is the dropped span's
tokens plus one finding line (48 tokens) per dropped tool call, and a summary
longer than the dropped text falls back to the marker.

Invariants: the system message, the first user message (the goal) and the last
N messages stay; the kept tail never starts on an orphaned tool result.

## Anthropic API shapes accepted

- Clearing edits (beta `context-management-2025-06-27`):
  `clear_tool_uses_20250919` {trigger {type input_tokens|tool_uses, value},
  default 100k input tokens; keep {type tool_uses, value}, default 3;
  clear_at_least, exclude_tools, clear_tool_inputs}; `clear_thinking_20251015`
  {keep "all" | {type thinking_turns, value}}, listed first. Response
  `context_management.applied_edits` (in the final `message_delta` when
  streaming).
- Compaction (beta `compact-2026-01-12`): edit `compact_20260112` {trigger
  {type input_tokens, value}, default 150k (the platform's 50k minimum is not
  enforced: local windows are smaller), pause_after_compaction, instructions}.
  Response block `{type: compaction, content}`; streamed as content_block_start,
  one compaction_delta, content_block_stop; a pause is stop_reason
  "compaction"; `usage.iterations` per pass. The client resends the block;
  everything before it is ignored.
- Not implemented: the on-demand mode (`compact-2026-09-04`, top-level
  `compaction: {type: summarize}`, signed block; exclusive with
  context_management).

## knurlogic shapes

| | Anthropic `/v1/messages` | OpenAI `/v1/chat/completions` |
|---|---|---|
| summary | leading `compaction` block | `message.compaction` (`delta.compaction` streaming) |
| pause | `stop_reason: compaction` | `finish_reason: compaction` |
| applied edits | `context_management.applied_edits` | top-level `context_management.applied_edits` |
| usage | `usage.iterations` | `usage.knurlogic.compaction` |
| resend | the block is the cut point | a message with `compaction`, or a system message in `<knurlogic:compaction>` |

Every chat response carries `usage.knurlogic.context = {tokens, window}` so a
harness can decide when to ask. There is no advisory header (no harness looks
for one).

## Code

- `context_management/context_edits.py` -- history surgery, no model.
- `context_management/compaction.py` -- prepare / summarize.
- `context_management/prompts/*.md` -- the summary and findings prompts,
  shipped as package data.

`context_management` is a top-level package, model-agnostic, with no mlx and no
HTTP. `interfaces/http/compaction.py` (`CompactingChat.chat`, a mixin of `App`) wires it in. Knobs
`KNURLOGIC_COMPACT_*` in `tuning/groups.py`: AUTO off, TRIGGER 0.8,
KEEP_TURNS 6, TOOL_RESULTS distill; the Settings page has a Compaction tab.
Tests: `tests/test_compaction.py`.

## Limits

- The resend re-derives the kept tail from the keep knob; changing the knob
  between turns shifts it.
- Custom `instructions` replace the summary prompt, but the findings request is
  still appended.
- The superseded long cache entry is not demoted for early eviction.

## Module notes

### knurlogic/context_management/compaction.py

A chat request (OpenAI's shape; `/v1/messages` arrives here translated)
goes through `prepare` before it becomes a Job:

1. every compaction the client resent is folded in (`context_edits.view`),
2. the clearing edits are applied in the order given,
3. a compact edit whose trigger the prompt has passed becomes a `Pending`
   summary pass -- or, with `KNURLOGIC_COMPACT_AUTO` on, a request that
   asked for nothing does too.

`summarize` runs that pass as a continuation of the conversation: the
history as the model already saw it (a prefix-cache hit), plus one user
turn asking for the summary and, in the same pass, a one-line finding per
dropped tool call. Only the summary's own tokens cost anything. The request
then runs on the compacted prompt, whose prefill is the fresh prompt-cache
entry the client's next turn hits.

Nothing is kept server-side: the summary goes back in the response and the
client resends it. Where the pass fails -- an error, an empty answer, one
longer than what it replaces -- the span is dropped behind a backstop
marker and the edit says `fallback: true`; the user's turn never fails for
it.

### knurlogic/context_management/context_edits.py

**What a harness sends.** Anthropic's `context_management.edits`:

```
{"type": "clear_thinking_20251015", "keep": "all" | {"type":
 "thinking_turns", "value": N}}
{"type": "clear_tool_uses_20250919", "trigger": {"type": "input_tokens"
 | "tool_uses", "value": N}, "keep": {"type": "tool_uses", "value": 3},
 "clear_at_least": {"type": "input_tokens", "value": N},
 "exclude_tools": [...], "clear_tool_inputs": false}
{"type": "compact_20260112", "trigger": {"type": "input_tokens",
 "value": 150000}, "pause_after_compaction": false, "instructions": null}
```

The same object is accepted on `/v1/chat/completions`. knurlogic adds one
field to the compact edit, `keep: {"type": "messages", "value": N}` (the
tail kept verbatim; the operator's default otherwise).

**Nothing is stored.** A compaction goes back to the client -- a
`compaction` content block (Anthropic) or `message.compaction` (OpenAI) --
and the client resends it with the rest of its history. On input the server
folds the history over each compaction it finds (`view`): what came before
it becomes the first user message, the summary and the tail that was kept,
exactly as the model saw it when the summary was written, so the rewrite is
the same every turn and its prompt a prefix-cache hit.

**Invariants:** the leading system message and the first user message (the
goal) are kept; the last N messages are kept; the kept tail never starts on
a tool result (it is widened back to the call that asked for it); nothing
is done unless at least two messages would go.
