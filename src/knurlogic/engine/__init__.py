"""engine/ -- what runs a model. The only folder that may import mlx.

A test enforces that boundary: anything outside engine/ that imports mlx
fails the suite, so swapping the engine stays a change to this folder.

  serve/           the one place that calls mlx-lm and mlx-vlm: load, serve,
                   memory, the live knobs, and each change knurlogic makes to
                   mlx-lm's server in its own module (see serve/__init__.py)
  mtp/             multi-token-prediction drafting, sequential and batched.
                   Its front door (`knurlogic.engine.mtp`) is stdlib only, so
                   asking whether an artifact has a head costs no mlx import
  vision/          images as context: contracts, the cache key, the image
                   store, and one package per family (qwen, gemma4, glm5)
  vq/              knurlogic's own VQ runtime; serves a rung only once
                   tools/vq_gate.py proves it bit-identical to the rung's
                   published model.py (rungs.json)
  architectures/   model files vendored from mlx-lm / mlx-vlm, pinned by
                   digest (PROVENANCE.md says which build each came from)
  arch.py          which architecture a model_type needs, and whether it is
                   present -- the one map of how configs spell a family
  register.py      puts vendored architectures in front of installed ones
  override.py      replaces a module inside mlx-lm, mlx-vlm or exo without
  overrides/       forking it; the files it serves live in overrides/
  vendor.py        takes an architecture file under version control
  smoke.py         generates a token and proves where the code came from

Depends on nothing else in knurlogic. Importing this package imports no mlx;
only calling into serve/ or mtp's engine-side modules does.
"""
