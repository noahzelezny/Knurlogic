# engine/templates provenance

## deepseek_v4.jinja

- source: `encoding/encoding_dsv4.py` in
  [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
  (revision 60d8d70770c6776ff598c94bb586a859a38244f1, fetched 2026-09-28).
  MIT License, Copyright (c) 2023 DeepSeek (the repo's LICENSE; model card
  `license: mit`). DeepSeek ships V4's chat encoding as Python only: the
  official tokenizer_config.json has no `chat_template`.
- ported to Jinja for knurlogic, 2026-09-28: merge_tool_messages,
  sort_tool_results_by_call_order, drop_thinking, render_message (system,
  developer, user, latest_reminder, assistant; DSML tool calls; tasks;
  response_format) rendered in one template.
- replaces: the mlx-community conversion's `chat_template.jinja` (sha256
  718a756a...c6e, `STUBS` in `__init__.py`), which renders no tools, no
  tool calls and no tool results, and reasoning only under
  thinking_mode='thinking'.
- proven: `tests/test_deepseek_v4.py` renders DeepSeek's four golden
  outputs (encoding/tests, vendored with the encoder in
  tests/fixtures_deepseek_v4/) byte for byte, and agent conversations
  (tools, parallel calls with results out of order, no system message,
  no tools, chat and thinking) equal to `encode_messages`.
- deviations, all for serving through transformers' apply_chat_template:
  - `tools` is a kwarg there, not a field on the system message: rendered
    on the first system message, or on an empty one put first.
  - the transition after a final user turn (`<｜Assistant｜><think>` or
    `</think>`) is the generation prompt: only with add_generation_prompt.
  - the mode comes from `thinking_mode`, else `enable_thinking` (mlx-lm
    passes it on every render), else chat. `drop_thinking` as upstream.
  - tool-call arguments arrive decoded (prompt.flatten); a string that is
    not an object renders as `{"arguments": ...}`, as upstream's fallback.
  - content before tool calls loses trailing newlines: the engine streams
    the `\n\n` before `<｜DSML｜tool_calls` as content, which upstream's
    parser treats as part of the block.
  - `reasoning_effort="max"`'s prefix is not rendered (the name would make
    knurlogic's thinking detection read the template as GLM's).

## parse_deepseek_v4 (`__init__.py`)

- after `parse_tool_calls` in the same file (MIT, as above), rewritten as
  two regular expressions over the block the engine's state machine cuts
  out (start `<｜DSML｜tool_calls`, end `</｜DSML｜tool_calls>`).
