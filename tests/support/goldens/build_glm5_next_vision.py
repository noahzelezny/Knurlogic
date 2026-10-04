"""Build the GLM-5.3-Flash (glm5_next) VISION reference golden: HF
transformers' own Glm5NextImageProcessor and Glm5NextVisionModel (the
maker's published reference) run in float32 on a tiny random tower with
the real vision_config's structure (patch 14, merge 2, temporal patch 2,
silu, swiglu_limit 10) and small dims.

    python tests/support/goldens/build_glm5_next_vision.py   # needs torch + transformers 5.16.1

The image is 168x112 (w x h), four flat quadrants of distinct colours, so
any patch mis-order shows up. 168x112 is already a multiple of
patch*merge and above min_image_tokens, so the reference neither resizes
nor pads it (8x12 patch grid, 24 merged tokens).
glm5_next_vision.npz holds: the config (json), the image, the reference's
pixel_values / image_grid_thw / pooler_output, and the tower weights in
the checkpoint's names (vision_model.* minus the prefix). Seed 0.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "glm5_next_vision.npz"

VISION = dict(
    model_type="glm5_next_vision", depth=2, hidden_size=32, num_heads=2,
    intermediate_size=64, out_hidden_size=48, projection_intermediate_size=64,
    patch_size=14, spatial_merge_size=2, temporal_patch_size=2,
    in_channels=3, image_size=448, hidden_act="silu", swiglu_limit=10.0,
    attention_bias=True, attention_dropout=0.0, rms_norm_eps=1e-5,
    initializer_range=0.02,
)


def image():
    h, w = 112, 168
    a = np.zeros((h, w, 3), np.uint8)
    a[: h // 2, : w // 2] = (255, 0, 0)
    a[: h // 2, w // 2:] = (0, 255, 0)
    a[h // 2:, : w // 2] = (0, 0, 255)
    a[h // 2:, w // 2:] = (255, 255, 0)
    return a


def main():
    import torch
    from PIL import Image
    from transformers.models.glm5_next.configuration_glm5_next import (
        Glm5NextVisionConfig,
    )
    from transformers.models.glm5_next.image_processing_glm5_next import (
        Glm5NextImageProcessor,
    )
    from transformers.models.glm5_next.modeling_glm5_next import (
        Glm5NextVisionModel,
    )

    torch.manual_seed(0)
    img = image()
    proc = Glm5NextImageProcessor()
    out = proc(images=Image.fromarray(img), return_tensors="pt")
    cfg = Glm5NextVisionConfig(**VISION)
    model = Glm5NextVisionModel(cfg).float().eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.2)
        ref = model(out["pixel_values"].float(), out["image_grid_thw"])
    weights = {f"w::{k}": v.detach().numpy().astype(np.float32)
               for k, v in model.state_dict().items()}
    np.savez_compressed(
        OUT, config=json.dumps(VISION), image=img,
        pixel_values=out["pixel_values"].numpy().astype(np.float32),
        image_grid_thw=out["image_grid_thw"].numpy(),
        pooler_output=ref.pooler_output.numpy().astype(np.float32),
        **weights)
    print(OUT, out["pixel_values"].shape, out["image_grid_thw"].tolist(),
          ref.pooler_output.shape)


if __name__ == "__main__":
    main()
