"""Multi-token prediction: find a head, and run one.

Two halves, and only one of them touches an engine. `_artifacts` answers
what an artifact HAS -- a built head, raw graft weights, or only a config
key it inherited. Pure stdlib (a safetensors header is a length prefix and
a JSON blob), so `doctor`, `discover` and the page ask without paying for
mlx; its names are re-exported here.

`binding` (the head bound to the loaded model, DRAFT) imports no mlx
either. Everything else in this package RUNS a head and imports mlx, so
none of it is imported here: `import knurlogic.engine.mtp` stays free;
`from knurlogic.engine.mtp import batch_generator` is where the engine
arrives. mlx-lm has no MTP path, so an artifact's drafting head is weight
nothing else runs. Design: docs/design/drafting.md.

  _artifacts.py       what an artifact has: a built head, graft weights, a key
  binding.py          the head bound to the loaded model (DRAFT)
  registry.py         which head a family drafts with
  batch_generator.py  the batch engine, drafting with an MTP head
  batch_loop.py       one-token drafting over a batch
  block_loop.py       K-token block drafting over a batch (DSpark)
  caches.py           snapshot and rollback for a speculative step
  capture.py          per-family capture of the pre-lm_head activation
  sampling.py         the sampler and exact rejection sampling
  seed.py             seed a head over a prompt in prefill-sized chunks
"""

from ._artifacts import (  # noqa: F401 -- re-exported, the package's API
    BUILT,
    DECLARED,
    GRAFTABLE,
    NONE,
    SIDECAR_GLOB,
    Head,
    Status,
    find_head,
    graft_weights,
    status,
)

__all__ = [
    "BUILT", "DECLARED", "GRAFTABLE", "NONE", "SIDECAR_GLOB",
    "Head", "Status", "find_head", "graft_weights", "status",
]
