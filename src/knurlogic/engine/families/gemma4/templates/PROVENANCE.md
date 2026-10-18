# families/gemma4/templates provenance

## thinking_moves.json

- edits to the maker's template (Gemma 4 (e4b, 26B-A4B), the artifact's chat_template.jinja; the model's terms), applied when the
  artifact's template is served (engine/templates.cache_safe); nothing of
  the maker's template is shipped here, only the edits' text.
- deviation: `<|think|>`, written at the top of the first system turn (which thinking alone used to open), is written as a system turn of its own just before the generation prompt's model turn instead.
- why: a thinking level is about the reply asked for; text changed before
  the conversation leaves nothing of it for the prompt cache, so a changed
  level was a full prefill. After the edits a level changes only what
  follows the conversation's last answer
  (tests/engine/test_thinking.py).
- measured (gemma-4-e4b VQ-PLE, one Mac, 8 problems x 2, 2026-10-17): thinking off 1277 tokens per answer, on 2734 with `<|think|>` in the first system turn and 2703 moved; correct 14/16, 15/16, 15/16.
- an artifact whose template differs from the one these edits were written
  against (an edit's text not found once) keeps its template unedited.
