"""The Qwen reference golden: what Qwen's own reference code (HF
transformers' modeling_qwen3_5 / modeling_qwen3_5_moe / modeling_qwen4_exp,
transformers 5.16.1) computes for a tiny random text model, so
tests/engine/test_qwen_reference.py can hold knurlogic's vendored trunks to
it on the same weights.

    $TORCH_PYTHON tests/support/goldens/build_qwen_reference.py

($TORCH_PYTHON: an interpreter with torch and transformers >= 5.16 -- the
first with qwen4_exp.) Writes qwen_reference.npz: per family, the config,
the logits of a prefill and of each teacher-forced decode step through
the reference's DynamicCache, float32 on the CPU, eager attention.

No weights are stored: `init_weights` makes them from numpy alone, keyed
by the reference's parameter names, and the test re-makes the same numbers.
The names and shapes are stored, so the test needs no torch.

What the tiny configs make run (each is a path the real models take):
- the deltanet's q/k l2norm near zero: layer 0's in_proj_qkv is scaled by
  SCALE, so sum(q^2) is ~1e-4 and the l2norm's eps (1e-6 on the SUM,
  FLA's) is visible in the logits;
- qwen4_exp's QSA sparse path (indexer_budget 8 < the 21-token prefill) and
  its block rope (4 index heads: with 2, a relu-zero tie between blocks
  made the reference's top-k pick arbitrary at one row); its PLE n-gram hash with an EOS inside the prompt (the
  segment reset); `seed` left out of the config, as every released
  config.json does, so the hash multipliers are the config default's;
- partial rotary with interleaved MRoPE sections (text positions).
Also (VARIANTS) configs that leave keys out, take layer_types off the
interval or set norm_topk_prob false; each trunk again in bfloat16 and,
one module at a time, its bf16 deltanet / norms / MLP or MoE; the vision
tower; the image processor on a non-square image scaled up and down; and
get_rope_index on a two-image prompt. meta["build"] records what built it.
Only stdlib + numpy at module level (the test imports this file).
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "qwen_reference.npz"

FAMILIES = ("qwen3_5", "qwen3_5_moe", "qwen4_exp")
EOS = 2
#: a 21-token prefill with an EOS inside it, then 4 teacher-forced steps
PROMPT = [5, 17, 42, 9, 33, 60, 11, 48, 27, EOS, 7, 55, 21, 36, 4, 13, 29,
          50, 8, 61, 19]
DECODE = [23, 3, 44, 58]
SCALE = {"model.layers.0.linear_attn.in_proj_qkv.weight": 1e-3}

_COMMON = dict(
    vocab_size=64, hidden_size=32, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2, head_dim=32,
    rms_norm_eps=1e-6, max_position_embeddings=4096,
    linear_num_value_heads=4, linear_num_key_heads=2,
    linear_key_head_dim=32, linear_value_head_dim=16,
    linear_conv_kernel_dim=4, full_attention_interval=2,
    layer_types=["linear_attention", "full_attention"] * 2,
    tie_word_embeddings=False, attention_bias=False, hidden_act="silu",
    eos_token_id=EOS, bos_token_id=EOS, pad_token_id=None,
    rope_parameters=dict(rope_type="default", rope_theta=10000.0,
                         partial_rotary_factor=0.5, mrope_interleaved=True,
                         mrope_section=[3, 3, 2]))

CONFIGS = {
    "qwen3_5": dict(_COMMON, model_type="qwen3_5_text",
                    intermediate_size=64),
    "qwen3_5_moe": dict(_COMMON, model_type="qwen3_5_moe_text",
                        num_experts=4, num_experts_per_tok=2,
                        moe_intermediate_size=16,
                        shared_expert_intermediate_size=16),
    "qwen4_exp": dict(_COMMON, model_type="qwen4_exp_text",
                      layer_types=["linear_attention"] * 3
                      + ["full_attention"],
                      full_attention_interval=4,
                      num_experts=4, num_experts_per_tok=2,
                      moe_intermediate_size=16,
                      shared_expert_intermediate_size=16,
                      output_gate_type="sigmoid", hc_count=2, hc_lowrank=8,
                      indexer_n_heads=4, indexer_kv_heads=1,
                      indexer_head_dim=32, indexer_budget=8,
                      indexer_compress_ratio=2, ngram_size=3,
                      heads_per_ngram=2, ngram_vocab_size_base=50,
                      make_ngram_vocab_size_divisible_by=8,
                      split_ngram_parts=4, ple_embed_dim=32,
                      ple_layer_ids=[2], ple_conv_kernel_size=4),
}


def _drop(cfg: dict, *keys, rope=()) -> dict:
    cfg = {k: v for k, v in cfg.items() if k not in keys}
    if rope:
        cfg["rope_parameters"] = {k: v for k, v in cfg["rope_parameters"].items()
                                  if k not in rope}
    return cfg


#: config variants, each a path a config.json can take that the base
#: configs do not: (family, config). Held in float32 like the base ones.
VARIANTS = {
    # the reference router always renormalizes the top-k
    # (modeling_qwen3_5_moe Qwen3_5MoeTopKRouter): the key is ignored
    "qwen3_5_moe/norm_topk_false": (
        "qwen3_5_moe", dict(CONFIGS["qwen3_5_moe"], norm_topk_prob=False)),
    # keys left out: head_dim (256), rope_theta (10000.0) and
    # partial_rotary_factor (0.25) take the reference config's defaults
    "qwen3_5/defaults": (
        "qwen3_5", _drop(CONFIGS["qwen3_5"], "head_dim",
                         rope=("rope_theta", "partial_rotary_factor"))),
    "qwen3_5_moe/defaults": (
        "qwen3_5_moe", _drop(CONFIGS["qwen3_5_moe"], "head_dim",
                             rope=("rope_theta", "partial_rotary_factor"))),
    # layer_types that full_attention_interval (left out: 4) does not give
    "qwen3_5/layer_types": (
        "qwen3_5", dict(_drop(CONFIGS["qwen3_5"], "full_attention_interval"),
                        layer_types=["full_attention", "linear_attention",
                                     "linear_attention", "full_attention"])),
    "qwen3_5_moe/layer_types": (
        "qwen3_5_moe", dict(
            _drop(CONFIGS["qwen3_5_moe"], "full_attention_interval"),
            layer_types=["full_attention", "linear_attention",
                         "linear_attention", "full_attention"])),
    # the reference's own layer-type name, off the interval
    "qwen4_exp/layer_types": (
        "qwen4_exp", dict(
            _drop(CONFIGS["qwen4_exp"], "full_attention_interval"),
            layer_types=["linear_attention", "linear_attention",
                         "qwen_sparse_attention", "linear_attention"])),
    # output_gate_type, ple_embed_dim, split_ngram_parts, rope_theta and
    # partial_rotary_factor left out; norm_topk_prob false (honoured here)
    "qwen4_exp/defaults": (
        "qwen4_exp", dict(_drop(CONFIGS["qwen4_exp"], "output_gate_type",
                                "ple_embed_dim", "split_ngram_parts",
                                rope=("rope_theta", "partial_rotary_factor")),
                          norm_topk_prob=False)),
}

#: the families run again in bfloat16 on both sides (weights rounded to
#: bfloat16 first), against the float32 run of the same reference
BF16 = FAMILIES

#: the vision tower, tiny (both towers are the same code, held each)
VISION = dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=4,
              in_channels=3, patch_size=4, spatial_merge_size=2,
              temporal_patch_size=2, out_hidden_size=16,
              num_position_embeddings=16, hidden_act="gelu_pytorch_tanh")
VISION_GRID = [1, 6, 8]
#: the released preprocessor_config.json's kwargs (Qwen3.5 / 3.6 / Flash-Next)
PROC = dict(patch_size=16, temporal_patch_size=2, merge_size=2,
            image_mean=[.5] * 3, image_std=[.5] * 3,
            size={"shortest_edge": 65536, "longest_edge": 16777216})
#: a non-square image; "up": below min_pixels (65536) so smart_resize
#: scales it up; "down": max_pixels 16384 scales it down
IMAGE_HW = (203, 311)
PROC_SIZES = {"up": PROC["size"],
              "down": {"shortest_edge": 4096, "longest_edge": 16384}}
#: a two-image prompt: text, image (6x8 patches), text, image (4x4), text
IMG, VS, VE = 50, 51, 52
ROPE_IDS = ([1, 2, 3, VS] + [IMG] * 12 + [VE, 4, 5, VS] + [IMG] * 4
            + [VE, 6, 7, 8])
ROPE_GRIDS = [[1, 6, 8], [1, 4, 4]]


#: buffers the modules build for themselves
BUFFERS = ("layer_multipliers", "ngram_heads_vocab_sizes",
           "ngram_heads_offsets")


def init_weights(shapes: dict, scale: dict = None) -> dict:
    """Deterministic float32 weights from numpy alone, one seeded stream
    per reference parameter name. Matrices 1/sqrt(fan_in); zero-centred
    norms (the reference's `1 + weight`) 0.1 N; the deltanet's gated norm
    (a plain `weight`) 1 + 0.1 N; A_log log U(1, 16); other vectors 0.1 N."""
    out = {}
    for name in sorted(shapes):
        if any(b in name for b in BUFFERS):
            continue
        shape = tuple(shapes[name])
        rng = np.random.default_rng([7, zlib.crc32(name.encode())])
        if name.endswith("A_log"):
            a = np.log(rng.uniform(1, 16, shape))
        elif len(shape) >= 2:
            a = rng.standard_normal(shape) / np.sqrt(np.prod(shape[1:]))
        elif name.endswith("linear_attn.norm.weight"):
            a = 1.0 + 0.1 * rng.standard_normal(shape)
        else:
            a = 0.1 * rng.standard_normal(shape)
        a = a * (SCALE if scale is None else scale).get(name, 1.0)
        out[name] = a.astype(np.float32)
    return out


def bf16(a: np.ndarray) -> np.ndarray:
    """Round float32 to bfloat16 (nearest even), kept as float32: the
    weights both sides load for the bfloat16 run."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.view(np.float32)


def vision_weights(shapes: dict) -> dict:
    """The tiny tower's weights, numpy alone, seeded by name."""
    out = {}
    for name in sorted(shapes):
        shape = tuple(shapes[name])
        rng = np.random.default_rng([11, zlib.crc32(name.encode())])
        a = rng.standard_normal(shape)
        if len(shape) >= 2:
            a = a / np.sqrt(np.prod(shape[1:]))
        elif name.endswith(("norm1.weight", "norm2.weight", "norm.weight")):
            a = 1.0 + 0.1 * a
        else:
            a = 0.1 * a
        out[name] = a.astype(np.float32)
    return out


def vision_inputs():
    """(pixel patches for the tower, the uint8 image for the processor)."""
    rng = np.random.default_rng(1)
    t, h, w = VISION_GRID
    px = rng.standard_normal(
        (t * h * w, VISION["in_channels"] * VISION["temporal_patch_size"]
         * VISION["patch_size"] ** 2)).astype(np.float32)
    img = rng.integers(0, 256, (*IMAGE_HW, 3), dtype=np.uint8)
    return px, img


def to_levels(px) -> np.ndarray:
    """pixel_values as the uint8 levels they are ((u / 255 - 0.5) / 0.5 of
    a resized uint8 image): a quarter of the float32 bytes, exactly."""
    px = np.asarray(px, np.float32)
    u = np.rint((px + 1.0) * 127.5).astype(np.uint8)
    assert np.abs(from_levels(u) - px).max() < 1e-6, "not uint8 levels"
    return u


def from_levels(u: np.ndarray) -> np.ndarray:
    return ((u.astype(np.float32) / 255.0 - 0.5) / 0.5).astype(np.float32)


def _vision_reference(arrays: dict, meta: dict):
    import torch
    from PIL import Image
    from transformers.models.qwen3_5 import configuration_qwen3_5 as c35
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m35
    from transformers.models.qwen4_exp import configuration_qwen4_exp as c4
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m4
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import \
        Qwen2VLImageProcessorPil
    px, img = vision_inputs()
    meta["vision/config"] = VISION
    for fam, cfg_cls, cls in (
            ("qwen3_5", c35.Qwen3_5VisionConfig, m35.Qwen3_5VisionModel),
            ("qwen4_exp", c4.Qwen4ExpVisionConfig, m4.Qwen4ExpVisionModel)):
        cfg = cfg_cls(**VISION)
        cfg._attn_implementation = "eager"
        tower = cls(cfg).float().eval()
        shapes = {k: tuple(v.shape) for k, v in tower.state_dict().items()}
        w = vision_weights(shapes)
        tower.load_state_dict({k: torch.from_numpy(v) for k, v in w.items()})
        with torch.no_grad():
            r = tower(torch.from_numpy(px), torch.tensor([VISION_GRID]))
        arrays[f"vision/{fam}/feats"] = r.pooler_output.numpy()
        meta[f"vision/{fam}/shapes"] = {k: list(v) for k, v in shapes.items()}
    # the image processor: transformers' PIL backend (the reference numpy
    # port knurlogic vendors) and, if torchvision is there, its default one
    kw = {k: v for k, v in PROC.items() if k != "size"}
    pil = Image.fromarray(img)
    meta["proc/sizes"] = PROC_SIZES
    for tag, size in PROC_SIZES.items():
        r = Qwen2VLImageProcessorPil(**kw, size=size)(
            images=pil, return_tensors="np")
        arrays[f"proc/{tag}/px"] = to_levels(r["pixel_values"])
        arrays[f"proc/{tag}/grid"] = np.asarray(r["image_grid_thw"])
        try:
            from transformers.models.qwen2_vl.image_processing_qwen2_vl \
                import Qwen2VLImageProcessor
            r = Qwen2VLImageProcessor(**kw, size=size)(
                images=pil, return_tensors="np")
            arrays[f"proc/{tag}/tv_px"] = to_levels(r["pixel_values"])
        except Exception as e:  # no torchvision: the PIL backend only
            print("torchvision image processor unavailable:", e)
    # get_rope_index on a two-image prompt, both families
    tc = {k: v for k, v in CONFIGS["qwen3_5"].items() if k != "model_type"}
    tc4 = {k: v for k, v in CONFIGS["qwen4_exp"].items()
           if k != "model_type"}
    types = [1 if t == IMG else 0 for t in ROPE_IDS]
    for fam, cfg_cls, cls, t in (
            ("qwen3_5", c35.Qwen3_5Config, m35.Qwen3_5Model, tc),
            ("qwen4_exp", c4.Qwen4ExpConfig, m4.Qwen4ExpModel, tc4)):
        cfg = cfg_cls(text_config=t, vision_config=VISION,
                      image_token_id=IMG, vision_start_token_id=VS,
                      vision_end_token_id=VE)
        cfg._attn_implementation = "eager"
        model = cls(cfg).eval()
        pos, delta = model.get_rope_index(
            torch.tensor([ROPE_IDS]), torch.tensor([types]),
            image_grid_thw=torch.tensor(ROPE_GRIDS))
        arrays[f"rope/{fam}/pos"] = pos.numpy()[:, 0]
        arrays[f"rope/{fam}/delta"] = delta.numpy().reshape(-1)


#: bfloat16 components, one module at a time on the same input, where a
#: whole-model bf16 run is too noisy to show a rounding-order difference.
#: dt_bias is offset so `a + dt_bias` sits where bf16 spacing (2^-5 at
#: 4..8) is coarse next to `a` (~0.1): the reference adds in float32.
COMPONENT_DT_BIAS = 6.0
#: (module path under model.layers, input width); the zero-centred norms
#: (`1 + weight` in float32 in the reference) are qwen3_5's input norm and
#: qwen4_exp's grouped hyper-connection norm and q_norm
_C35 = (("0.linear_attn", 32), ("0.input_layernorm", 32), ("0.mlp", 32))
COMPONENTS = {
    "qwen3_5": _C35, "qwen3_5_moe": _C35,
    "qwen4_exp": (("0.linear_attn", 32), ("0.mlp", 32),
                  ("0.attn_hyper_connection.hc_norm", 64),
                  ("3.self_attn.q_norm", 32)),
}


def component_input(width: int) -> np.ndarray:
    rng = np.random.default_rng([5, width])
    return bf16(rng.standard_normal((1, len(PROMPT), width)).astype(
        np.float32))


def component_weights(shapes: dict) -> dict:
    w = {k: bf16(v) for k, v in init_weights(shapes, scale={}).items()}
    for k in w:
        if k.endswith("dt_bias"):
            w[k] = bf16(w[k] + COMPONENT_DT_BIAS)
    return w


def _components(family: str) -> dict:
    """Layer 0's deltanet, input norm and MLP/MoE of the bf16 reference,
    each on component_input."""
    import torch
    torch.set_grad_enabled(False)
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as m35m
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m35
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m4
    from transformers.models.qwen3_5 import configuration_qwen3_5 as c35
    from transformers.models.qwen3_5_moe import \
        configuration_qwen3_5_moe as c35m
    from transformers.models.qwen4_exp import configuration_qwen4_exp as c4
    cls = {"qwen3_5": (c35.Qwen3_5TextConfig, m35.Qwen3_5ForCausalLM),
           "qwen3_5_moe": (c35m.Qwen3_5MoeTextConfig,
                           m35m.Qwen3_5MoeForCausalLM),
           "qwen4_exp": (c4.Qwen4ExpTextConfig, m4.Qwen4ExpForCausalLM)}
    cfg_cls, model_cls = cls[family]
    config = cfg_cls(**{k: v for k, v in CONFIGS[family].items()
                        if k != "model_type"})
    config._attn_implementation = "eager"
    model = model_cls(config).float().eval()
    shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    w = component_weights(shapes)
    model.load_state_dict({k: torch.from_numpy(v) for k, v in w.items()},
                          strict=False)
    model = model.to(torch.bfloat16)
    out = {}
    for c, width in COMPONENTS[family]:
        mod = model.model.layers
        for p in c.split("."):
            mod = mod[int(p)] if p.isdigit() else getattr(mod, p)
        y = mod(torch.from_numpy(component_input(width)).bfloat16())
        y = y[0] if isinstance(y, tuple) else y
        out[c] = y.float().numpy()
    return out


def bf16_weights_of(shapes: dict) -> dict:
    """The bf16 run's weights: init_weights rounded to bfloat16, without
    SCALE. SCALE puts layer 0's q/k near zero to show the l2norm eps in
    float32; in bf16 the reference's own l2norm there (four bf16 roundings,
    CPU's two-step rsqrt) is so ill-conditioned it is most of the
    reference's bf16-vs-f32 difference, so it would hide everything else."""
    return {k: bf16(v) for k, v in init_weights(shapes, scale={}).items()}


def _reference(family: str, cfg: dict = None, dtype: str = "float32",
               bf16_weights: bool = False):
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m35
    from transformers.models.qwen3_5 import configuration_qwen3_5 as c35
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as m35m
    from transformers.models.qwen3_5_moe import \
        configuration_qwen3_5_moe as c35m
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m4
    from transformers.models.qwen4_exp import configuration_qwen4_exp as c4
    cls = {"qwen3_5": (c35.Qwen3_5TextConfig, m35.Qwen3_5ForCausalLM),
           "qwen3_5_moe": (c35m.Qwen3_5MoeTextConfig,
                           m35m.Qwen3_5MoeForCausalLM),
           "qwen4_exp": (c4.Qwen4ExpTextConfig, m4.Qwen4ExpForCausalLM)}
    cfg_cls, model_cls = cls[family]
    cfg = CONFIGS[family] if cfg is None else cfg
    cfg = {k: v for k, v in cfg.items() if k != "model_type"}
    config = cfg_cls(**cfg)
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    model = model_cls(config).float().eval()
    sd = model.state_dict()
    shapes = {k: tuple(v.shape) for k, v in sd.items()}
    if bf16_weights:
        w = bf16_weights_of(shapes)
    else:
        w = init_weights(shapes)
    model.load_state_dict({k: torch.from_numpy(v) for k, v in w.items()},
                          strict=False)
    if dtype == "bfloat16":
        # as from_pretrained(dtype=torch.bfloat16): every parameter and
        # floating buffer in bfloat16 (the reference keeps none in float32)
        model = model.to(torch.bfloat16)
    from transformers import DynamicCache
    cache = DynamicCache(config=config)
    out = []
    with torch.no_grad():
        r = model(input_ids=torch.tensor([PROMPT]), past_key_values=cache,
                  use_cache=True)
        out.append(r.logits[0].float().numpy())
        for t in DECODE:
            r = model(input_ids=torch.tensor([[t]]),
                      past_key_values=r.past_key_values, use_cache=True)
            out.append(r.logits[0].float().numpy())
    names = sorted(w)
    return (np.concatenate(out, 0).astype(np.float32), names,
            [shapes[n] for n in names])


def main():
    import datetime
    import platform

    import PIL
    import torch
    import transformers
    arrays = {}
    meta = {"transformers": transformers.__version__, "prompt": PROMPT,
            "decode": DECODE, "scale": SCALE,
            "build": {"date": datetime.date.today().isoformat(),
                      "script": "tests/support/goldens/build_qwen_reference.py",
                      "transformers": transformers.__version__,
                      "torch": torch.__version__, "numpy": np.__version__,
                      "pillow": PIL.__version__,
                      "python": platform.python_version(),
                      "platform": platform.platform(),
                      "device": "cpu", "attention": "eager"}}
    for fam in FAMILIES:
        logits, names, shapes = _reference(fam)
        arrays[f"{fam}/logits"] = logits
        meta[f"{fam}/config"] = CONFIGS[fam]
        meta[f"{fam}/shapes"] = dict(zip(names, [list(s) for s in shapes]))
        print(fam, logits.shape, float(np.abs(logits).max()))
    for name, (fam, cfg) in VARIANTS.items():
        logits, names, shapes = _reference(fam, cfg)
        arrays[f"{name}/logits"] = logits
        meta[f"{name}/config"] = cfg
        meta[f"{name}/shapes"] = dict(zip(names, [list(s) for s in shapes]))
        print(name, float(np.abs(logits).max()))
    for fam in BF16:
        # the reference in bfloat16, and in float32 on the same
        # (bf16-rounded) weights: what bf16 arithmetic alone moves
        logits, _, _ = _reference(fam, dtype="bfloat16", bf16_weights=True)
        f32, _, _ = _reference(fam, bf16_weights=True)
        arrays[f"{fam}/bf16/logits"] = logits
        arrays[f"{fam}/bf16/f32_logits"] = f32
        print(fam, "bf16 vs float32 reference, rms",
              float(np.sqrt(((logits - f32) ** 2).mean())))
        for c, y in _components(fam).items():
            arrays[f"{fam}/bf16/{c}"] = y
    _vision_reference(arrays, meta)
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez_compressed(OUT, **arrays)
    print(OUT, OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    sys.exit(main())
