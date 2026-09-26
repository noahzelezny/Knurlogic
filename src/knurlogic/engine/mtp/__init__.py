"""Multi-token prediction: find a head, and run one.

TWO HALVES, AND ONLY ONE OF THEM TOUCHES AN ENGINE.

`_artifacts` answers what an artifact HAS -- a built head, raw graft weights,
or a config key it inherited and nothing else. Pure stdlib: a safetensors
header is a length prefix and a JSON blob. `doctor`, `discover` and the page
all use it, and none of them should pay for mlx to ask a question about a
file. Those names are re-exported here.

Everything else in this package RUNS a head, so every one of those modules
imports mlx. They are NOT imported here. `import knurlogic.engine.mtp`
stays free; `from knurlogic.engine.mtp import batch_generator` is where the
engine arrives.

  _artifacts.py       what an artifact HAS: a head, graft weights (stdlib)
  registry.py         model_type -> head spec, built from the family
                      manifests (engine/families/); load_head binds one
  batch_loop.py       requests in one batch (MTPBatch, admit): prefill,
                      seed the head, draft and verify
  batch_generator.py  mlx-lm's BatchGenerator contract over batch_loop,
                      incl. segment checkpoints and the cache report
  caches.py           snapshot and rollback for a speculative step
  capture.py seed.py sampling.py   the pieces those share

Drafting across machines is not here yet: it comes with the cluster
executor (engine/runtime/executor.py, docs/SERVER.md "Cluster readiness").

WHOSE CODE THIS IS. The drafting half was written by Noah in his exo fork and
in vqlab, and upstream exo-explore/exo has none of it -- 0 files under
`engines/mlx/mtp/` on origin/main against 15 in the fork. It lived in two
copies that had drifted apart (caches.py, registry.py and the sequential
loop differed by 30, 63 and 296 lines), which is the argument for one copy here rather than a
third out there. mlx-lm has no MTP path at all, so an artifact's drafting
head is weight nobody else will run.
"""

from ._artifacts import (  # noqa: F401
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
