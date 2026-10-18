# families/qwen/templates provenance

## thinking_moves.json

- edits to the maker's template (Qwen3.8 (27B, Flash-Next), the artifact's chat_template.jinja; Apache-2.0 (the model's)), applied when the
  artifact's template is served (engine/templates.cache_safe); nothing of
  the maker's template is shipped here, only the edits' text.
- deviation: `reasoning_instructions`, written into the system turn (or a system turn of its own), opens the last user query (`ns.last_query_index`) instead; Qwen3.8 refuses a system message anywhere but first.
- why: a thinking level is about the reply asked for; text changed before
  the conversation leaves nothing of it for the prompt cache, so a changed
  level was a full prefill. After the edits a level changes only what
  follows the conversation's last answer
  (tests/engine/test_thinking.py).
- measured (Qwen3.8-27B VQ-3.9bpw, one Mac, 8 problems x 2, 2026-10-17): "medium" (no text) 9/16, "low" 9/16 in the system turn and 10/16 moved, "xhigh" 11/16 both; 24 of 80 requests failed in a server out-of-memory (fixed since), so the evidence is thinner than the other families'.
- an artifact whose template differs from the one these edits were written
  against (an edit's text not found once) keeps its template unedited.
