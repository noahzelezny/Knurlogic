"""Build the gemma4 vision goldens in the reference interpreter
(mlx-vlm 0.6.17):

    $KNURLOGIC_VLM_PYTHON tests/goldens/build_gemma4.py

gemma4_vision_tower.npz   mlx-vlm's own gemma4 VisionModel, tiny random
                          weights (seed 0), on one tiny image -- holds
                          engine/families/gemma4/vision/vision.py's vendored port to
                          the reference tower.
gemma4_mask_overlay.npz   mlx-vlm's own
                          `Gemma4TextModel._apply_blockwise_bidirectional_overlay`
                          (language.py:466-515) on a seeded causal mask and
                          block-id array -- holds the mask overlay ported
                          into `architectures/gemma4_text.py::_make_masks`
                          to the reference.

Tiny, float32, seed 0; no model files are read.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fixtures_vision as fv  # noqa: E402

import mlx.core as mx  # noqa: E402
from mlx_vlm.models.gemma4.config import VisionConfig  # noqa: E402
from mlx_vlm.models.gemma4.language import Gemma4TextModel  # noqa: E402
from mlx_vlm.models.gemma4.vision import VisionModel  # noqa: E402


def vision_tower():
    mx.random.seed(0)
    cfg = VisionConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, head_dim=16,
        patch_size=16, pooling_kernel_size=3, default_output_length=16,
        position_embedding_size=256, use_clipped_linears=False,
        standardize=False)
    model = VisionModel(cfg)
    weights = dict(fv.__dict__.get("_gemma4_flat_weights", {}))  # unused, kept explicit
    params = model.parameters()
    from mlx.utils import tree_flatten, tree_unflatten
    flat = tree_flatten(params)
    rng = np.random.default_rng(0)
    seeded = [(k, mx.array(rng.standard_normal(v.shape).astype(np.float32)) * 0.02)
              for k, v in flat]
    model.update(tree_unflatten(seeded))
    mx.eval(model.parameters())

    H, W = 48, 32  # 3x2 patches of 16 -> 6 patches -> pool/9 needs >=9; use bigger
    H, W = 48, 48  # 3x3 patches of 16 = 9 patches -> 1 pooled token
    pixel_values = mx.array(rng.standard_normal((1, 3, H, W)).astype(np.float32))
    out = model(pixel_values)
    mx.eval(out)
    fv.save_golden("gemma4_vision_tower", {
        "pixel_values": np.array(pixel_values), "out": np.array(out),
        **{f"w.{k}": np.array(v) for k, v in seeded}},
        {"what": "mlx_vlm.models.gemma4.vision.VisionModel, tiny random "
                 "weights", "seed": 0, "H": H, "W": W})


def mask_overlay():
    mx.random.seed(1)
    B, N = 2, 12
    rng = np.random.default_rng(1)
    # Two images: positions 2-4 (image 0) and 7-9 (image 1); rest is text.
    mm_type = np.zeros((B, N), dtype=np.int64)
    mm_type[:, 2:5] = 1
    mm_type[:, 7:10] = 1
    mm_type_ids = mx.array(mm_type)

    causal = mx.arange(N)[:, None] >= mx.arange(N)[None, :]
    causal = mx.broadcast_to(causal, (B, N, N))

    import types

    class _Shim:
        pass

    shim = _Shim()
    shim._block_sequence_ids_for_mask = types.MethodType(
        Gemma4TextModel._block_sequence_ids_for_mask, shim)
    block_ids = shim._block_sequence_ids_for_mask(mm_type_ids)
    overlaid = Gemma4TextModel._apply_blockwise_bidirectional_overlay(
        shim, mx.expand_dims(causal, 1), mm_type_ids)
    mx.eval(block_ids, overlaid)
    fv.save_golden("gemma4_mask_overlay", {
        "mm_type_ids": mm_type, "causal": np.array(causal),
        "block_ids": np.array(block_ids), "overlaid": np.array(overlaid)},
        {"what": "mlx_vlm.models.gemma4.language.Gemma4TextModel."
                 "_block_sequence_ids_for_mask / "
                 "_apply_blockwise_bidirectional_overlay", "seed": 1})


if __name__ == "__main__":
    vision_tower()
    mask_overlay()
    print("wrote gemma4_vision_tower.npz, gemma4_mask_overlay.npz")
