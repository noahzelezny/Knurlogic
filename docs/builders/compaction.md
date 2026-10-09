# Building on compaction

Server-side context management: a harness asks for its conversation to be
compacted or cleared, and the server does it, so the harness does not
manage its agents' context. The why: [compaction](../design/compaction.md).

## Where the code is

`src/knurlogic/context_management/`, model-agnostic: messages in,
messages out. No mlx, no HTTP.

| file | what |
|---|---|
| `context_edits.py` | history surgery over OpenAI-shaped messages: `parse` (the request's `context_management` to edits: `Compact`, `ClearTools`, `ClearThinking`), `plan` / `compacted` / `view` (fold a resent compaction back in), `clear_tool_uses`, `clear_thinking`, the summary prompt and its parsing (`prompt`, `parse_output`, `render`, `fallback_summary`) |
| `compaction.py` | `prepare` (fold in, apply the clearing edits, decide whether a summary pass is due: `Pending`), `summarize` (run the pass as a continuation), `attach` (put the outcome on the response), `paused`, `settings` (the `KNURLOGIC_COMPACT_*` values) |

Hooks outside it:

- `interfaces/http/server.py`: `App.chat` runs every chat request
  through `C.prepare`, then `C.summarize` when a pass is pending, then the
  engine; `App._warm` prefills a compacted prompt into the prompt cache;
  `count_tokens` counts what the model would see (`E.view`).
- `tuning/groups.py`: `COMPACT_KNOBS`, `compact_settings`,
  `check_compact_knob`.
- `tuning/preferences.py`: `compaction_env` (saved values over the
  environment).
- `interfaces/page/documents.py`: `compaction_document` for the Settings
  panel.

## Rules that keep it correct

- **Nothing is kept server-side.** The compaction travels back to the
  client, which resends it; `view` folds it in on the next request.
- **The summary pass is a continuation** of the conversation, so only the
  summary's own tokens cost anything (the prompt is already cached).
- **A failed pass never fails the turn.** The span is dropped behind a
  backstop marker with `fallback: true`, and the user's turn still runs.
- **Automatic compaction is opt-in** (`KNURLOGIC_COMPACT_AUTO`, off by
  default). Off, a client that does not ask is refused past the window.
- **This package depends only on `tuning/`** (`groups`, `preferences`). The model is reached
  through what the caller passes (`prepare`'s `count`, `summarize`'s
  `generate`).

## Extending

A new edit type: a class and its parsing in `context_edits.parse`, its
surgery in `context_edits.py`, applied in `compaction.prepare` in order.
A new knob goes in `COMPACT_KNOBS` with its default, values and reason.

## Notes

Compaction spans `context_management/` (the logic), `interfaces/http/server.py`
(`App.chat`, `_warm`: the summary pass and warm-up), `tuning/groups.py`
(knobs), `tuning/preferences.py` (saved values) and
`interfaces/page/documents.py` (the panel).

## Tests

`tests/context_management/test_compaction.py`,
`tests/integration/test_knurlogic_wide.py` (saved compaction settings).
