"""Multi-token prediction: find a head, and run one.

Two halves, and only one of them touches an engine. `_artifacts` answers
what an artifact HAS -- a built head, raw graft weights, or only a config
key it inherited. Pure stdlib (a safetensors header is a length prefix and
a JSON blob), so `doctor`, `discover` and the page ask without paying for
mlx; its names are re-exported here.

Everything else in this package RUNS a head and imports mlx, so none of it
is imported here: `import knurlogic.engine.mtp` stays free;
`from knurlogic.engine.mtp import batch_generator` is where the engine
arrives. mlx-lm has no MTP path, so an artifact's drafting head is weight
nothing else runs. Design: docs/design/drafting.md.
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
