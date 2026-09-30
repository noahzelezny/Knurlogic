# The real-model gates (tools/)

## tools/vision_gate.py

`tools/vision_gate.py` is run by hand, one rung at a time, behind the
load lock and `ready()`. It is never invoked from the test suite: `tests/`
only loads tiny random-weight fixtures, and the real gate needs an actual
artifact, an actual image, and real memory.

Per rung it checks:

- a vision tensor count bound (the tower loaded some weights, and not the
  whole checkpoint again): the script has no live handle, so it reads the
  count `serve`'s stdout prints and checks it is > 0 and < the artifact's
  total tensor count;
- a text-only answer (the server still answers without an image);
- an image answer -- a red square, and OCR-style recall of a rendered "42";
- a five-turn conversation where turns 2-5's
  `usage.prompt_tokens_details.cached_tokens` show the image span was not
  re-prefilled (`prompt - cached` on turn N ~= turn N's own new text), and
  turn 5 still answers a question about the turn-1 image;
- memory back to baseline after `POST /loaded.json {"action": "unload"}`.

It holds the load lock (`machine/loadlock.model_load`) for the duration and
exits `loadlock.EXIT_BUSY` if another load is already in progress, the same
rule `ready()` enforces for every other loader.

