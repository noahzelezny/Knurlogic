# families/glm5/templates provenance

## thinking_moves.json

- edits to the maker's template (GLM-5.3-Flash, the artifact's chat_template.jinja; MIT (the model's)), applied when the
  artifact's template is served (engine/templates.cache_safe); nothing of
  the maker's template is shipped here, only the edits' text.
- deviation: the `<|system|>Reasoning Effort: ...` turn, written before the system prompt, is written just before the generation prompt's `<|assistant|>` instead (GLM renders system turns anywhere).
- why: a thinking level is about the reply asked for; text changed before
  the conversation leaves nothing of it for the prompt cache, so a changed
  level was a full prefill. After the edits a level changes only what
  follows the conversation's last answer
  (tests/engine/test_thinking.py).
- measured (GLM-5.3-Flash VQ-2.7bpw, pipeline split over two Macs, 8 problems x 2, 2026-10-18): tokens per answer low 463 -> 559 moved, high 3448 -> 2945, max 4552 -> 4970, so the levels still scale; correct 11/11, 12/11, 14/14 (the same prompt run twice as a control scored 7 and 11: noise at this size).
- an artifact whose template differs from the one these edits were written
  against (an edit's text not found once) keeps its template unedited.
