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

## tools/vq_gate.py

`tools/vq_gate.py gate` is run by hand on real rungs, one at a time,
behind the model-load lock. Same artifact weights, same short prompt, two
runtimes: the rung's published bundled `model.py`, and knurlogic's
(the vendored VQ runtime pinned in `engine/vq/PROVENANCE.md` plus the
rung's knobs from `rungs.json`). PASS means logits over the whole prompt
within atol 1e-5 and 40 greedy tokens identical. `--record` then marks the
rung verified in `rungs.json` -- the only thing that lets knurlogic serve
it on its own runtime. The `knobs` record is always read from the shipped
artifact on the Hub, never a local copy, which may have drifted.

**Why each side is its own process.** Both runtimes read their flags once at
import, into module globals, and both size the mlx buffer cache at import.
Two sides in one process would share whatever the first import froze and
whatever memory the first model left; a gate that can pass by sharing state
is not a gate. Each side also runs with every `VQ_*`/`VQLAB_*` variable
removed from its environment, so each reads its own defaults -- the bundle
its baked text, knurlogic its knobs -- which is exactly the claim under
test.
