# families/deepseek/templates provenance

The manifest's `chat_templates` (families/deepseek/__init__.py) names these
templates, what they replace, what selects the variant and how their
message parts are joined; engine/templates is the generic code reading it.

## deepseek_v4.jinja

- source: `encoding/encoding_dsv4.py` in
  [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
  (revision 60d8d70770c6776ff598c94bb586a859a38244f1, fetched 2026-09-28).
  MIT License, Copyright (c) 2023 DeepSeek (the repo's LICENSE; model card
  `license: mit`). DeepSeek ships V4's chat encoding as Python only: the
  official tokenizer_config.json has no `chat_template`.
- ported to Jinja for knurlogic: merge_tool_messages,
  sort_tool_results_by_call_order, drop_thinking, render_message (system,
  developer, user, latest_reminder, assistant; DSML tool calls; tasks;
  response_format) rendered in one template.
- replaces: the mlx-community conversion's `chat_template.jinja` (sha256
  718a756a...c6e, `stubs` in the manifest), which renders no tools, no
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

## parse_deepseek_v4 (`../chat_template.py`)

- after `parse_tool_calls` in the same file (MIT, as above), rewritten as
  two regular expressions over the block the engine's state machine cuts
  out (start `<｜DSML｜tool_calls`, end `</｜DSML｜tool_calls>`).

## deepseek_v4_vision (`text("deepseek_v4_vision")`)

- source: `encoding/encoding_dsv4.py` of
  deepseek-ai/DeepSeek-V4-Flash-Vision-Exp (MIT, as above; vendored in
  tests/support/fixtures_deepseek_v4_vision/). Against Flash's it differs
  in the reasoning-effort prefixes -- "low" (the default) none, "high"
  Flash's Think Max prompt, "max" a new one -- and in image blocks: a
  message's parts joined with "\n\n", an image as `<｜deepseek_image｜>`.
- the template is deepseek_v4.jinja with `{%- set dsv4_vision = true -%}`
  first; the jinja picks the prefixes by it. The image parts are not the
  template's: engine/vision splices the placeholder, and
  runtime/prompt.flatten joins every message's parts with "\n\n" for this
  template (`part_separator` in the manifest), text-only ones too.
- chosen by the artifact's config.json, not its folder name: model_type
  deepseek_v4 with vision_n_layers > 0 or dspark_block_size > 0 (Flash has
  neither), when it has no template, a stub, or a DSML template; or a
  template that is this one.

## message parts (`part_separator` in the manifest)

- encoding_dsv4.py (Flash's and Vision-Exp's alike, render_message ~305)
  joins a tool_result's list of text parts with "\n\n": Flash's template
  gets `{"tool": "\n\n"}`. Flash's encoder takes no list content on a user
  message (merge_tool_messages copies it as one text block); mlx-lm's ""
  stays there. Vision-Exp's (_process_image_blocks ~740) joins every
  message's blocks with "\n\n": `{"default": "\n\n"}`.
- proven: tests/engine/test_vision_deepseek.py renders the artifact's
  two-image example equal to its encoder, chat and thinking at each
  effort.
- its thinking levels are its own four, the "deepseek_vision_effort"
  dialect (families/deepseek): off / low (the default) / high / max ->
  chat / thinking with reasoning_effort low / high / max.
