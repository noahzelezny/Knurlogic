# Context compaction (built 2026-09-28, commit 310d1f3)

Goal (the maintainer): knurlogic as universal as possible -- any harness plugs in. A swarm of
sub-agents should not need its memory managed by hand.

## Decision: the harness asks, the server performs

- The harness asks through Anthropic's `context_management` (the field Claude Code
  already sends). The same object is accepted on `/v1/chat/completions` as an extension.
- The server runs the compaction with the model already serving the request, as a
  continuation of the cached conversation: the history is a prefix-cache hit, only the
  summary's own tokens cost. The compacted prefix becomes a fresh prompt-cache entry.
- Nothing is stored server-side: the summary goes back in the response and the client
  resends it. Silent server-side rewriting is rejected -- a stateless client keeps
  resending the full history and the two views drift.
- Automatic (operator-default) compaction exists but is OFF by default.

## Distill, don't clear (the maintainer)

A tool call usually exists to answer a question ("where is X defined?"). Outside the
recent tail, each dropped tool result becomes a one-line finding (`T<n>: X is defined at
src/foo.py:120`) in the same model pass as the summary; a clear placeholder is the
fallback. The recent tail stays verbatim (the agent may still be using it).

## Summary prompt

the harness's (`scout/coherence/summarizer.py:175-190`) extended for coding agents: Goal,
Decisions made, Information gathered, Files and identifiers touched, Current step, Next
step, Open errors and questions. Budget about 1/10 of the dropped tokens, clamped 1k-8k.
the harness's invariants kept: system message, the first user message (the goal) and the last
N messages stay; the kept tail never starts on an orphaned tool result.

## API shapes (verified against platform.claude.com, 2026-09-28)

- Clearing edits (beta `context-management-2025-06-27`):
  `clear_tool_uses_20250919` {trigger {type input_tokens|tool_uses, value} default 100k
  input tokens, keep {type tool_uses, value} default 3, clear_at_least, exclude_tools,
  clear_tool_inputs}; `clear_thinking_20251015` {keep "all" | {type thinking_turns,
  value}}, listed first. Response `context_management.applied_edits` (final
  `message_delta` when streaming).
- Compaction (beta `compact-2026-01-12`): edit `compact_20260112` {trigger {type
  input_tokens, value} default 150k (docs' 50k minimum not enforced: local windows are
  smaller), pause_after_compaction, instructions}. Response block `{type: compaction,
  content}`; stream: content_block_start, one compaction_delta, content_block_stop;
  pause -> stop_reason "compaction"; `usage.iterations` per pass. The client resends the
  block; everything before it is ignored.
- Not implemented: the newer on-demand mode (`compact-2026-09-04`, top-level
  `compaction: {type: summarize}`, signed block; exclusive with context_management).

## knurlogic shapes

| | Anthropic `/v1/messages` | OpenAI `/v1/chat/completions` |
|---|---|---|
| summary | leading `compaction` block | `message.compaction` (`delta.compaction` streaming) |
| pause | `stop_reason: compaction` | `finish_reason: compaction` |
| applied edits | `context_management.applied_edits` | top-level `context_management.applied_edits` |
| usage | `usage.iterations` | `usage.knurlogic.compaction` |
| resend | the block is the cut point | a message with `compaction`, or a system message in `<knurlogic:compaction>` |

Every chat response carries `usage.knurlogic.context = {tokens, window}` so a harness
can decide when to ask. No custom advisory header (no harness looks for one).

## Code

`interfaces/context_edits.py` (history surgery, no model), `interfaces/compaction.py`
(prepare / summarize), `http/server.py` App.chat wiring, knobs `KNURLOGIC_COMPACT_*`
in `tuning/settings.py` (AUTO off, TRIGGER 0.8, KEEP_TURNS 6, SUMMARY_MIN 1024,
SUMMARY_MAX 8192, TOOL_RESULTS distill), Settings "Compaction" tab. Tests:
`tests/test_compaction.py`.

## Open

- Live check on a real model (tokens before/after, summary quality, cache hit): not run.
- The resend re-derives the kept tail from the keep knob; changing the knob between
  turns shifts it.
- Custom `instructions` replace the summary prompt but the findings request is still
  appended.
- Not done: demoting the old long cache entry for early eviction.
- Dropped by the maintainer: an advisory header, and agent self-advocacy for cache budget (it
  makes the scheduler a referee).
- the harness adopting knurlogic's compaction instead of its own summarizer.
