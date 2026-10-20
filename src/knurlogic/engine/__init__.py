"""engine/ -- what runs a model. The only folder that may import mlx.

A test enforces that boundary: anything outside engine/ that imports mlx
fails the suite, so swapping the engine stays a change to this folder.

  model/         the served model: load it, its memory, thinking, vision
  runtime/       the server's engine half: model host, scheduler, executor
  split/         one model across ranks: tensor and pipeline
  prompt_cache/  the prompt cache, in memory and on disk
  mtp/           drafting with a multi-token-prediction head
  vision/        images as context: the contracts, store and cache key
  families/      one folder per model family (VQLab imports these)
  templates/     chat templates knurlogic supplies in place of an artifact's
  arrays.py      every mx.array an object holds (arrays_of)
  backend.py     which MLX back end computes (metal, cuda, cpu), the device
  gated_delta.py the gated-delta (GatedDeltaNet) scan on every back end
  kvquant.py     the quantized KV cache (8, 6 or 4 bits)
  kvattn.py      the decode attention kernel over kvquant's 8-bit cache
  matmul_padding.py  per-chip rounding off: identical results on M3 and M4
  embeddings.py  the text embedder: load on demand, embed, drop when idle
  arch.py        which architecture module a model_type needs, verified
  register.py    installs the architecture modules into mlx-lm
  smoke.py       `knurlogic smoke`: generate a token, prove where code came from
  vendor.py      `knurlogic vendor`: take an architecture file under control

Of the rest of knurlogic it imports only machine/ (this box: units, paths,
files), tuning/ (the knob registry and the fit arithmetic) and the
package version, never cluster/ or interfaces/. Importing this package
imports no mlx; only calling into model/ or mtp's engine-side modules does.
Layout: docs/design/engine.md.
"""
