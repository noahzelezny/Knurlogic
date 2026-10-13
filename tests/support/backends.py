"""Marks for tests that hold for a GPU back end of MLX (Metal, CUDA) and not
for its CPU back end (docs/design/os-agnostic.md):

- `gpu_rounding`: an exact or near-exact comparison -- against a torch
  reference, between two paths (compiled and plain, a split and the whole
  model, a drafted and an undrafted run) -- whose numbers were measured on
  a GPU; MLX's CPU back end rounds bfloat16 and FP8 differently.
- `bf16_gather`: a bfloat16 MoE model's expert matmul (`gather_mm`), which
  MLX's CPU back end takes in float32 only.

Both run on the Macs today and on CUDA before 0.2.0 is tagged.
"""
import pytest


def _cpu() -> bool:
    try:
        from knurlogic.engine import backend
    except ImportError:            # no mlx: the test skips on its own
        return False
    return backend.name() == "cpu"


ON_CPU = _cpu()

gpu_rounding = pytest.mark.skipif(
    ON_CPU, reason="numbers measured on a GPU; MLX's CPU back end rounds "
    "bfloat16 / FP8 differently")
bf16_gather = pytest.mark.skipif(
    ON_CPU, reason="MLX's CPU back end has no bfloat16 gather_mm (MoE "
    "experts)")
