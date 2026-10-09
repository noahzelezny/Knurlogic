"""Qwen's trunks as a pipeline stage (engine/split/pipeline.restage):
the per-layer indices each trunk froze from the WHOLE layer list at
__init__, re-taken over the stage's `keep` = its layers [start, end).
Named by the manifest's `pipeline` entries. Layers are read through their
stage wrappers (attribute reads see through Recv / Send).
"""
from __future__ import annotations


def restage_qwen3_5(model, core, keep: list, start: int, end: int) -> None:
    """qwen3_5 / qwen3_5_moe: PipelineMixin's contract (its uniform split
    and all_gather not used) and the first linear / full-attention layer."""
    core.start_idx, core.end_idx = 0, None
    core.pipeline_rank, core.pipeline_size = 0, 1
    core.ssm_idx = next((i for i, lyr in enumerate(keep) if lyr.is_linear),
                        None)
    core.fa_idx = next((i for i, lyr in enumerate(keep)
                        if not lyr.is_linear), None)


def restage_qwen4_exp(model, core, keep: list, start: int,
                      end: int) -> None:
    """qwen4_exp: full-model indices in ple_layers and a full-length
    make_cache (fork commit dd946407)."""
    core.ple_layers = [i - start for i in core.ple_layers
                       if start <= i < end]
    whole = model.make_cache

    def make_cache():
        return whole()[start:end]
    model.make_cache = make_cache
