"""Multi-token prediction: find a head, and run one.

TWO HALVES, AND ONLY ONE OF THEM TOUCHES AN ENGINE.

`_artifacts` answers what an artifact HAS -- a built head, raw graft weights,
or a config key it inherited and nothing else. Pure stdlib: a safetensors
header is a length prefix and a JSON blob. `doctor`, `discover` and the page
all use it, and none of them should pay for mlx to ask a question about a
file. Those names are re-exported here.

Everything else in this package RUNS a head, so every one of those modules
imports mlx. They are NOT imported here. `import knurlogic.mtp` stays free;
`from knurlogic.mtp import loop` is where the engine arrives.

WHOSE CODE THIS IS. The drafting half was written by the maintainer in his exo fork and
in vqlab, and upstream exo-explore/exo has none of it -- 0 files under
`engines/mlx/mtp/` on origin/main against 15 in the fork. It lived in two
copies that had drifted apart (caches.py, registry.py and loop.py differed by
30, 63 and 296 lines), which is the argument for one copy here rather than a
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
