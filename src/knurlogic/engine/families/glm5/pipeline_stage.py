"""GLM-5.3's trunk as a pipeline stage (engine/split/pipeline.restage),
named by the manifest's `pipeline` entry. Layers are read through their
stage wrappers (attribute reads see through Recv / Send).
"""
from __future__ import annotations


def restage(model, core, keep: list, start: int, end: int) -> None:
    """glm5_next: the first linear / full-attention layer, frozen from the
    full list at __init__ (fork commit f3ab3a83), re-taken over `keep`."""
    core.ssm_idx = next((i for i, lyr in enumerate(keep)
                         if getattr(lyr, "is_linear", False)), 0)
    core.fa_idx = next((i for i, lyr in enumerate(keep)
                        if not getattr(lyr, "is_linear", True)), 0)
