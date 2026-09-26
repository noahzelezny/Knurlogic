"""engine/ -- what runs a model. The only folder that may import mlx.

A test enforces that boundary: anything outside engine/ that imports mlx
fails the suite, so swapping the engine stays a change to this folder.

  serve/           what is served: load, memory, the live knobs, thinking
                   translation, the vision family and drafting head bound
                   to the loaded model (see serve/__init__.py)
  runtime/         knurlogic's own server's engine half: the model host,
                   the scheduler, the executor, the prompt and request
                   stages (docs/SERVER.md; HTTP is interfaces/http)
  mtp/             multi-token-prediction drafting, sequential and batched.
                   Its front door (`knurlogic.engine.mtp`) is stdlib only, so
                   asking whether an artifact has a head costs no mlx import
  families/        one folder per model family, everything knurlogic knows
                   about it: MANIFEST (architectures, model_type spellings,
                   heads, prefill widths), architecture/ (vendored model
                   code + PROVENANCE, pins, licenses), vision/, heads/.
                   Adding a family = one folder + one line in
                   families/__init__.py
  vision/          images as context, family-agnostic: contracts, the cache
                   key, the image store, the request path
  vq/              knurlogic's own VQ runtime; serves a rung only once
                   tools/vq_gate.py proves it bit-identical to the rung's
                   published model.py (rungs.json)
  arch.py          which architecture a model_type needs, and whether it is
                   present (the maps are built from the family manifests)
  register.py      puts vendored architectures in front of installed ones
  vendor.py        takes an architecture file under version control
  smoke.py         generates a token and proves where the code came from

Depends on nothing else in knurlogic. Importing this package imports no mlx;
only calling into serve/ or mtp's engine-side modules does.
"""
