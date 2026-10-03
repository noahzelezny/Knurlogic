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

#: buffers the modules build for themselves
BUFFERS = ("layer_multipliers", "ngram_heads_vocab_sizes",
           "ngram_heads_offsets")


def init_weights(shapes: dict) -> dict:
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
        a = a * SCALE.get(name, 1.0)
        out[name] = a.astype(np.float32)
    return out


def _reference(family: str):
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
    cfg = {k: v for k, v in CONFIGS[family].items() if k != "model_type"}
    config = cfg_cls(**cfg)
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    model = model_cls(config).float().eval()
    sd = model.state_dict()
    shapes = {k: tuple(v.shape) for k, v in sd.items()}
    w = init_weights(shapes)
    model.load_state_dict({k: torch.from_numpy(v) for k, v in w.items()},
                          strict=False)
    from transformers import DynamicCache
    cache = DynamicCache(config=config)
    out = []
    with torch.no_grad():
        r = model(input_ids=torch.tensor([PROMPT]), past_key_values=cache,
                  use_cache=True)
        out.append(r.logits[0].numpy())
        for t in DECODE:
            r = model(input_ids=torch.tensor([[t]]),
                      past_key_values=r.past_key_values, use_cache=True)
            out.append(r.logits[0].numpy())
    names = sorted(w)
    return (np.concatenate(out, 0).astype(np.float32), names,
            [shapes[n] for n in names])


def main():
    import transformers
    arrays = {}
    meta = {"transformers": transformers.__version__, "prompt": PROMPT,
            "decode": DECODE, "scale": SCALE}
    for fam in FAMILIES:
        logits, names, shapes = _reference(fam)
        arrays[f"{fam}/logits"] = logits
        meta[f"{fam}/config"] = CONFIGS[fam]
        meta[f"{fam}/shapes"] = dict(zip(names, [list(s) for s in shapes]))
        print(fam, logits.shape, float(np.abs(logits).max()))
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez_compressed(OUT, **arrays)
    print(OUT, OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    sys.exit(main())
