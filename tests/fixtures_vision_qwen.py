"""Tiny Qwen vision fixtures (P1): one config per family, the shared scaler
from fixtures_vision, and the few structural choices that make the tiny
model exercise what the real one does.

WHY head_dim 128 AND partial_rotary_factor 0.5. The released trunks all
have head_dim 256 x 0.25: 64 rotary dims, 32 frequencies, and the MRoPE
section [11, 11, 10] sums to exactly 32. The shared scaler's head_dim 32
would leave 4 frequencies and the h/w sections would never reach their real
tails; the interleave is what G3/G4 exist to hold, so the tiny model keeps
the real ROTARY layout (64 dims, 32 frequencies). Not head_dim 256 itself:
measured 2026-09-23, MLX's float32 steel attention kernel at head dim 256
needs 53,760 B of threadgroup memory (limit 32,768) and refuses a 165-token
prefill; 128 x 0.5 gives the same 64 rotary dims.

WHY indexer_budget 32 (qwen4_exp). The QSA indexer only sparsifies past its
budget (2048 real); a tiny prompt never gets there. 32 makes the sparse path,
and its block rope, run on a 100-token prompt.

WHY seed 1234 (qwen4_exp). The n-gram hash multipliers are rebuilt from
`seed`; mlx-vlm defaults it to 1234, knurlogic's TextArgs to 0, and the
released config.json carries no seed. Measured 2026-09-23: Flash-Next
2.1's checkpoint `layers.1.ple.ple_embedding.layer_multipliers` is
[23703573157769, 20109073645365, 8052911324071] -- exactly seed 1234's --
while knurlogic hashes with its private seed-0 copy. That is a TEXT-path
defect outside P1 (reported to the integrator, not fixed here: P1 may not
move the text path); the tiny config names the seed so both sides agree and
G4 measures MRoPE, not this.

Imports under both interpreters (the golden builder runs in the mlx-vlm one):
stdlib + numpy at module level.
"""
from __future__ import annotations

import copy
from typing import Any, Dict

import fixtures_vision as FV

FAMILIES = ("qwen3_5", "qwen3_5_moe", "qwen4_exp")

_COMMON = dict(hidden_size=128, num_hidden_layers=4, num_attention_heads=4,
               num_key_value_heads=2, head_dim=128, rms_norm_eps=1e-6,
               vocab_size=FV.TINY_VOCAB, max_position_embeddings=262144,
               linear_num_value_heads=4, linear_num_key_heads=2,
               linear_key_head_dim=32, linear_value_head_dim=32,
               linear_conv_kernel_dim=4, full_attention_interval=2,
               tie_word_embeddings=False, attention_bias=False,
               eos_token_id=2,
               rope_parameters=dict(mrope_interleaved=True,
                                    mrope_section=[11, 11, 10],
                                    partial_rotary_factor=0.5,
                                    rope_theta=10000000, type="default"))

TEXT: Dict[str, Dict[str, Any]] = {
    "qwen3_5": dict(_COMMON, model_type="qwen3_5_text",
                    intermediate_size=256),
    "qwen3_5_moe": dict(_COMMON, model_type="qwen3_5_moe_text",
                        intermediate_size=256, num_experts=4,
                        num_experts_per_tok=2, moe_intermediate_size=64,
                        shared_expert_intermediate_size=64,
                        decoder_sparse_step=1, norm_topk_prob=True),
    "qwen4_exp": dict(_COMMON, model_type="qwen4_exp_text",
                      full_attention_interval=4,
                      num_experts=4, num_experts_per_tok=2,
                      moe_intermediate_size=64,
                      shared_expert_intermediate_size=64,
                      output_gate_type="sigmoid", hc_count=4, hc_lowrank=16,
                      indexer_n_heads=4, indexer_kv_heads=1,
                      indexer_head_dim=128, indexer_budget=32,
                      indexer_compress_ratio=4, ngram_size=3,
                      heads_per_ngram=8, ngram_vocab_size_base=1000,
                      make_ngram_vocab_size_divisible_by=128,
                      split_ngram_parts=4, ple_embed_dim=128,
                      ple_layer_ids=[2], ple_conv_kernel_size=4,
                      bos_token_id=2, seed=1234),
}

#: The released processor (preprocessor_config.json of every Qwen rung read
#: on 2026-09-23: shortest_edge 65536, longest_edge 16777216, patch 16,
#: temporal 2, merge 2, mean/std 0.5).
PREPROCESSOR = {"size": {"longest_edge": 16777216, "shortest_edge": 65536},
                "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
                "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]}


def config(family: str) -> Dict[str, Any]:
    """The tiny full config (vision + text + remapped ids) for `family`."""
    cfg = FV.tiny_config(family, text_config=copy.deepcopy(TEXT[family]))
    tc = cfg["text_config"]
    # the scaler's text sizes, then P1's structural overrides back on top
    for k, v in TEXT[family].items():
        if k in ("head_dim", "num_hidden_layers", "full_attention_interval",
                 "num_experts", "num_experts_per_tok", "indexer_head_dim",
                 "ple_embed_dim", "hc_lowrank", "partial_rotary_factor"):
            tc[k] = v
    cfg["eos_token_id"] = tc["eos_token_id"]
    cfg["vision_config"]["num_position_embeddings"] = 64   # 8x8 grid
    return cfg


#: Non-trainable buffers the modules compute for themselves (int tables,
#: hash multipliers): never overwritten by init_weights.
_BUFFERS = ("layer_multipliers", "ngram_heads_vocab_sizes",
            "ngram_heads_offsets")


def init_weights(shapes: Dict[str, tuple], seed: int = 0):
    """Deterministic float32 weights from numpy alone, keyed by mlx-vlm's
    parameter names -- so the golden builder (mlx-vlm interpreter) and the
    test (knurlogic, no mlx-vlm) put the SAME numbers in two different
    module trees without shipping megabytes of weights in the golden.

    Scale: 1/sqrt(fan_in) for matrices (logits stay sharp enough that
    greedy tokens are not near-ties), norms 1 + 0.1 N, other vectors 0.02 N.
    Names are walked sorted, each from its own seeded stream, so adding a
    parameter never shifts another's values."""
    import zlib
    import numpy as np
    out = {}
    for name in sorted(shapes):
        if any(b in name for b in _BUFFERS):
            continue
        shape = tuple(shapes[name])
        rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
        if len(shape) >= 2:
            fan_in = int(np.prod(shape[1:]))
            a = rng.standard_normal(shape) / np.sqrt(fan_in)
        elif "norm" in name:
            a = 1.0 + 0.1 * rng.standard_normal(shape)
        else:
            a = 0.02 * rng.standard_normal(shape)
        out[name] = a.astype(np.float32)
    return out


def trunk_name(family: str, name: str) -> str:
    """mlx-vlm parameter name -> knurlogic trunk parameter name (read off
    both trees for the tiny configs: qwen3_5 / qwen3_5_moe are identical,
    qwen4_exp drops the language_model level and names shards shard_N)."""
    if family != "qwen4_exp":
        return name
    if name.startswith("language_model.model."):
        name = "model." + name[len("language_model.model."):]
    elif name.startswith("language_model."):
        name = name[len("language_model."):]
    import re
    return re.sub(r"\.ngram_embedding\.shards\.(\d+)\.",
                  r".ngram_embedding.shard_\1.", name)


def ids(family: str) -> Dict[str, int]:
    return FV.tiny_ids(family)


def prompt(family: str, n_image_tokens: int, pre=(3, 17, 45, 9, 101),
           post=(7, 88, 12, 250, 33, 64)):
    """[text, vision_start, image x n, vision_end, text] -- the layout the
    Qwen template produces around one image."""
    t = ids(family)
    return (list(pre) + [t["vision_start_token_id"]]
            + [t["image_token_id"]] * n_image_tokens
            + [t["vision_end_token_id"]] + list(post))
