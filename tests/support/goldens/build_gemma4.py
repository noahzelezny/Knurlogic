"""Build the gemma4 vision goldens from the maker's reference: HF
transformers 5.16.1 `models/gemma4/` (Google's port) -- the vision tower
(`Gemma4VisionModel`), the projection into the text model
(`Gemma4MultimodalEmbedder`), the image processor's resize
(`image_processing_gemma4.get_aspect_ratio_preserving_size`) and the
image-block mask functions (`masking_utils`), under torch on the CPU:

    $TORCH_PYTHON tests/support/goldens/build_gemma4.py

gemma4_vision_tower.npz
    tower/<name>/...   two tiny random towers + embedders, float32:
                       e4b-style (clipped linears, no standardize) and
                       26B-style (standardize, a 24-wide head split 12/12
                       across the two rope axes), one 24x36 image each,
                       fed as the image processor patchifies it
                       (`convert_image_to_patches`) and padded with
                       (-1, -1) positions the way a batch is -- the
                       padding is masked and stripped, so the output is
                       the image's own soft tokens.
    pool/<dtype>/...   the pooler's tail in float16 and bfloat16 on
                       activations large enough that `* sqrt(hidden)`
                       leaves the float16 range: HF's real
                       `Gemma4VisionModel.forward` with the patch
                       embedder and encoder stubbed to hand it the given
                       hidden states, so its pooler, its float32 scaling
                       and standardize, and its final cast are what run.
    resize             (w, h, resized w, resized h, soft tokens) per size
                       at the released processor settings (patch 16,
                       pool 3, max_soft_tokens 280); -1s where HF raises.
gemma4_mask_overlay.npz
    HF's own mask functions (`masking_utils.causal_mask_function`,
    `blockwise_overlay`, `or_masks`; block ids from
    `modeling_gemma4.get_block_sequence_ids_for_mask`) evaluated on a
    12-token, two-image prompt: what `create_masks_for_vision_model`
    builds for a sliding layer whose window is wider than the prompt.

Seeded (torch 0 / numpy 0), tiny; no model files are read. Each npz's
`__meta__` records the transformers/torch versions, interpreter and date.
"""
import datetime
import json
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fixtures_vision as fv  # noqa: E402

#: the released processor settings (processor_config.json, e4b and 26B)
PATCH, POOL, MAX_SOFT = 16, 3, 280
RESIZE_SIZES = [
    (896, 896), (448, 448), (1000, 300), (300, 1000), (20, 2000),
    (2000, 20), (64, 64), (1920, 1080), (5, 5), (1, 1), (3000, 3000),
    (1234, 567), (100, 101), (4096, 100), (100, 4096), (17, 2999),
    (640, 480), (1024, 768), (1366, 768), (2560, 1440), (500, 500),
    (1, 5000), (97, 41), (41, 97)]


def _meta(what):
    import torch
    import transformers
    return {"what": what, "transformers": transformers.__version__,
            "torch": torch.__version__, "python": platform.python_version(),
            "reference_python": sys.executable,
            "built": datetime.date.today().isoformat(),
            "command": "$TORCH_PYTHON tests/support/goldens/build_gemma4.py"}


def _vcfg(hidden, heads, head_dim, standardize, clipped):
    return dict(hidden_size=hidden, intermediate_size=2 * hidden,
                num_hidden_layers=2, num_attention_heads=heads,
                num_key_value_heads=heads, head_dim=head_dim, patch_size=4,
                pooling_kernel_size=3, position_embedding_size=64,
                standardize=standardize, use_clipped_linears=clipped,
                rms_norm_eps=1e-6,
                rope_parameters={"rope_theta": 100.0, "rope_type": "default"})


def tower(name, standardize, clipped, head_dim, hidden, heads, text_hidden,
          rec):
    import torch
    from transformers.models.gemma4.configuration_gemma4 import (
        Gemma4TextConfig, Gemma4VisionConfig)
    from transformers.models.gemma4.image_processing_gemma4 import \
        convert_image_to_patches
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4MultimodalEmbedder, Gemma4VisionModel)

    vd = _vcfg(hidden, heads, head_dim, standardize, clipped)
    vc = Gemma4VisionConfig(**vd)
    vc._attn_implementation = "eager"
    tc = Gemma4TextConfig(hidden_size=text_hidden, num_hidden_layers=1,
                          num_attention_heads=2, num_key_value_heads=1,
                          head_dim=8, vocab_size=32)
    m = Gemma4VisionModel(vc).eval()
    e = Gemma4MultimodalEmbedder(vc, tc).eval()
    with torch.no_grad():
        for p in list(m.parameters()) + list(e.parameters()):
            p.copy_(torch.randn_like(p) * 0.5)
        if standardize:
            m.std_bias.copy_(torch.randn(hidden) * 0.3)
            m.std_scale.copy_(torch.rand(hidden) + 0.5)
        if clipped:
            for mod in m.modules():
                if getattr(mod, "use_clipped_linears", False) and \
                        hasattr(mod, "input_min"):
                    mod.input_min.fill_(-1.5)
                    mod.input_max.fill_(1.5)
                    mod.output_min.fill_(-4.0)
                    mod.output_max.fill_(4.0)
    # 24x36 px -> 6x9 patches of 4 = 54 -> 6 soft tokens (pool 3)
    H, W, p = 24, 36, 4
    img = torch.rand(3, H, W)
    patches = convert_image_to_patches(img, p)
    ph, pw = H // p, W // p
    grid = torch.stack(torch.meshgrid(torch.arange(pw), torch.arange(ph),
                                      indexing="xy"), -1).reshape(-1, 2)
    pad = 9 * 4
    pv = torch.nn.functional.pad(patches, (0, 0, 0, pad))
    pos = torch.nn.functional.pad(grid, (0, 0, 0, pad), value=-1)
    with torch.no_grad():
        out = m(pixel_values=pv[None],
                pixel_position_ids=pos[None]).last_hidden_state
        proj = e(out)
    pre = f"tower/{name}/"
    rec[pre + "img"] = img.numpy()
    rec[pre + "tower"] = out.numpy()
    rec[pre + "proj"] = proj.numpy()
    rec[pre + "vcfg"] = np.array(json.dumps(
        dict(vd, default_output_length=(ph * pw + pad) // 9)))
    rec[pre + "text_hidden"] = np.array(text_hidden)
    for k, v in m.state_dict().items():
        rec[pre + f"w/vision_tower.{k}"] = v.numpy()
    for k, v in e.state_dict().items():
        rec[pre + f"w/embed_vision.{k}"] = v.numpy()


class _Given:
    """A stub module handing back a fixed tensor."""

    def __init__(self, value):
        self.value = value


def pool(dtype_name, rec):
    """HF's Gemma4VisionModel.forward run on given hidden states: the patch
    embedder and the encoder are stubbed, everything after them (pooler,
    float32 sqrt(hidden) scaling, strip, standardize, the cast back) is
    HF's own code."""
    import torch
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from transformers.models.gemma4.configuration_gemma4 import \
        Gemma4VisionConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel

    dtype = getattr(torch, dtype_name)
    hidden, ph, pw = 768, 9, 9
    vc = Gemma4VisionConfig(hidden_size=hidden, intermediate_size=16,
                            num_hidden_layers=1, num_attention_heads=1,
                            num_key_value_heads=1, head_dim=8,
                            pooling_kernel_size=3, standardize=True)
    m = Gemma4VisionModel(vc).eval()
    rng = np.random.default_rng(0)
    # large activations: 3x3 averages around 2200 (up to ~2900), times
    # sqrt(768) = 27.7, pass float16's 65504 -- the reason HF scales in
    # float32 -- and std_bias (~61000, itself representable) brings them
    # back into range
    h = (2200.0 + 600.0 * rng.standard_normal((1, ph * pw, hidden)))
    h = torch.tensor(h, dtype=torch.float32).to(dtype)
    std_bias = torch.tensor(2200.0 * 27.7 + 300.0 * rng.standard_normal(
        hidden), dtype=torch.float32).to(dtype)
    std_scale = torch.tensor(0.5 + rng.random(hidden),
                             dtype=torch.float32).to(dtype)

    class Embedder(torch.nn.Module):
        weight = torch.nn.Parameter(torch.zeros(1, dtype=dtype))

        def forward(self, pv, pos, padding):
            return h

    class Encoder(torch.nn.Module):
        def forward(self, inputs_embeds, attention_mask, pixel_position_ids,
                    **kw):
            return BaseModelOutputWithPast(last_hidden_state=inputs_embeds)

    m.patch_embedder = Embedder()
    m.encoder = Encoder()
    with torch.no_grad():
        m.std_bias = torch.nn.Parameter(std_bias, requires_grad=False)
        m.std_scale = torch.nn.Parameter(std_scale, requires_grad=False)
        grid = torch.stack(torch.meshgrid(torch.arange(pw), torch.arange(ph),
                                          indexing="xy"), -1).reshape(-1, 2)
        pv = torch.zeros(1, ph * pw, 3 * 16 * 16)
        out = m(pixel_values=pv, pixel_position_ids=grid[None]
                ).last_hidden_state
    assert out.dtype == dtype
    pre = f"pool/{dtype_name}/"
    rec[pre + "hidden"] = h.float().numpy()
    rec[pre + "std_bias"] = std_bias.float().numpy()
    rec[pre + "std_scale"] = std_scale.float().numpy()
    rec[pre + "out"] = out.float().numpy()
    rec[pre + "grid"] = np.array([ph, pw])


def resize():
    from transformers.models.gemma4.image_processing_gemma4 import \
        get_aspect_ratio_preserving_size
    rows = []
    for w, h in RESIZE_SIZES:
        try:
            th, tw = get_aspect_ratio_preserving_size(
                h, w, PATCH, MAX_SOFT * POOL * POOL, POOL)
            rows.append((w, h, tw, th, (th // PATCH) * (tw // PATCH)
                         // (POOL * POOL)))
        except ValueError:
            rows.append((w, h, -1, -1, -1))
    return np.array(rows)


def vision_tower():
    import torch
    torch.manual_seed(0)
    rec = {}
    tower("e4b", standardize=False, clipped=True, head_dim=16, hidden=32,
          heads=2, text_hidden=24, rec=rec)
    tower("g26", standardize=True, clipped=False, head_dim=24, hidden=48,
          heads=2, text_hidden=24, rec=rec)
    pool("float16", rec)
    pool("bfloat16", rec)
    rec["resize"] = resize()
    fv.save_golden("gemma4_vision_tower", rec, _meta(
        "transformers Gemma4VisionModel + Gemma4MultimodalEmbedder (eager, "
        "float32), Gemma4VisionModel's pooler tail in float16/bfloat16, "
        "get_aspect_ratio_preserving_size"))


def mask_overlay():
    import torch
    from transformers.masking_utils import (blockwise_overlay,
                                            causal_mask_function, or_masks)
    from transformers.models.gemma4.modeling_gemma4 import \
        get_block_sequence_ids_for_mask
    B, N = 2, 12
    # two images: positions 2-4 (image 0) and 7-9 (image 1); rest is text
    mm_type = np.zeros((B, N), dtype=np.int64)
    mm_type[:, 2:5] = 1
    mm_type[:, 7:10] = 1
    block_ids = get_block_sequence_ids_for_mask(torch.tensor(mm_type),
                                                device="cpu")
    fn = or_masks(causal_mask_function, blockwise_overlay(block_ids))
    b = torch.arange(B)[:, None, None, None]
    q = torch.arange(N)[None, None, :, None]
    k = torch.arange(N)[None, None, None, :]
    overlaid = fn(b, torch.zeros(1, dtype=torch.long)[None, :, None, None],
                  q, k).expand(B, 1, N, N)
    causal = causal_mask_function(0, 0, q[0, 0], k[0, 0]).expand(N, N)
    fv.save_golden("gemma4_mask_overlay", {
        "mm_type_ids": mm_type, "block_ids": block_ids.numpy(),
        "causal": np.broadcast_to(causal.numpy(), (B, N, N)),
        "overlaid": overlaid.numpy()},
        _meta("transformers masking_utils causal_mask_function OR "
              "blockwise_overlay(get_block_sequence_ids_for_mask)"))


if __name__ == "__main__":
    vision_tower()
    mask_overlay()
    print("wrote gemma4_vision_tower.npz, gemma4_mask_overlay.npz")
