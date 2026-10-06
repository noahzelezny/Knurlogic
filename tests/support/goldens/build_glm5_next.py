"""Build the GLM-5.3-Flash (glm5_next) reference golden: a tiny random
text model run through HF transformers' own Glm5NextTextModel (the maker's
published reference; zai-org's GLM-5 repo ships no model code) in float32,
with its weights exported in the checkpoint's names so the test loads them
through knurlogic's vendored sanitize.

    <venv with torch + transformers 5.16.1>/bin/python \
        tests/support/goldens/build_glm5_next.py

Reference: transformers 5.16.1, models/glm5_next/modeling_glm5_next.py.
glm5_next.npz holds: the config, the weights (float16 values, read as
float32 on both sides so they are exact), a 21-token prompt and 4 decode
tokens, `meta` (what built it), and the reference's logits for the prefill
and each decode step
through its DynamicCache. index_topk (8) is below the sequence, so the
DSA indexer really selects (pools of 4, tail appended); weights are scaled
so some SwiGLU inputs leave +-swiglu_limit. Seed 0; no model files read.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "glm5_next.npz"

CONFIG = dict(
    model_type="glm5_next_text",
    vocab_size=128, hidden_size=64, intermediate_size=64,
    moe_intermediate_size=16, num_hidden_layers=4,
    num_attention_heads=2, num_key_value_heads=2,
    n_shared_experts=1, n_routed_experts=4, routed_scaling_factor=2.5,
    kv_lora_rank=32, q_lora_rank=32, qk_rope_head_dim=0,
    v_head_dim=16, qk_nope_head_dim=16, qk_head_dim=16, head_dim=0,
    n_group=1, topk_group=1, num_experts_per_tok=2, norm_topk_prob=True,
    hidden_act="silu", max_position_embeddings=4096, rms_norm_eps=1e-5,
    index_topk=8, index_head_dim=16, index_n_heads=8, index_kpool=4,
    index_kpool_compress=True, index_kpool_always_select_tail=True,
    indexer_rope_interleave=True,
    layer_types=["linear_attention", "deepseek_sparse_attention",
                 "linear_attention", "deepseek_sparse_attention"],
    indexer_types=["full"] * 4,
    mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
    first_k_dense_replace=1,
    linear_attn_config={"num_heads": 2, "head_dim": 32,
                        "gate_lower_bound": -5.0, "short_conv_kernel_size": 4,
                        "kda_layers": [0, 2], "full_attn_layers": [1, 3]},
    swiglu_limit=10.0, hc_mult=4, hc_eps=1e-6, hc_sinkhorn_iters=20,
    mhc=True, mla_use_nope=True, moe_router_dtype="float32",
    num_nextn_predict_layers=0, scoring_func="sigmoid", topk_method="noaux_tc",
    tie_word_embeddings=False, attention_bias=False, pad_token_id=0,
)
PROMPT = [(7 * i + 3) % 128 for i in range(21)]
DECODE = [5, 77, 31, 100]


def checkpoint_names(sd, cfg):
    """HF module state -> the checkpoint's tensor names (experts per
    expert, everything under model.)."""
    out = {}
    inter = cfg["moe_intermediate_size"]
    for k, v in sd.items():
        if k.endswith("mlp.experts.gate_up_proj"):
            p = k[: -len("gate_up_proj")]
            for e in range(v.shape[0]):
                out[f"model.{p}{e}.gate_proj.weight"] = v[e, :inter]
                out[f"model.{p}{e}.up_proj.weight"] = v[e, inter:]
        elif k.endswith("mlp.experts.down_proj"):
            p = k[: -len("down_proj")]
            for e in range(v.shape[0]):
                out[f"model.{p}{e}.down_proj.weight"] = v[e]
        elif k == "lm_head.weight":
            out[k] = v
        else:
            out["model." + k] = v
    return out


def draw(name, p, r):
    """One weight's random value from its unit normal draw `r`."""
    if name.endswith("norm.weight") or "layernorm" in name:
        val = 1.0 + 0.1 * r
    elif name.endswith("k_norm.bias"):
        val = 0.1 * r
    elif name.endswith(("A_log",)):
        val = 0.3 * r
    elif name.endswith("dt_bias"):
        val = 0.5 * r
    elif name.endswith(("hc.base",)) or name.endswith("_hc.base"):
        val = 0.5 * r
    elif name.endswith("_hc.scale"):
        val = 1.0 + 0.2 * r
    elif name.endswith("_hc.fn"):
        val = 0.1 * r
    elif "e_score_correction_bias" in name:
        val = 0.1 * r
    elif "embed_tokens" in name:
        val = r
    elif "mlp" in name and ("gate_up" in name or "gate_proj" in name
                            or "up_proj" in name):
        val = 5.0 * r / p.shape[-1] ** 0.5   # some |x| > 10
    elif "index_kpool_compress_ape" in name:
        val = 0.5 * r
    else:
        val = r / p.shape[-1] ** 0.5
    return val


def _meta():
    """What built the golden: the reference's versions and paths."""
    import datetime

    import torch
    import transformers
    return {"torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__, "dtype": "float32",
            "experts_implementation": "eager",
            "attn_implementation": "eager", "device": "cpu",
            "built": datetime.date.today().isoformat()}


def main():
    import torch
    from transformers.cache_utils import DynamicCache
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel

    torch.manual_seed(0)
    cfg = Glm5NextTextConfig(**{k: v for k, v in CONFIG.items()
                               if k != "model_type"})
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"
    model = Glm5NextTextModel(cfg).float().eval()
    lm_head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, p in list(model.named_parameters()) + [
                ("lm_head.weight", lm_head.weight)]:
            r = torch.randn(p.shape, generator=g)
            val = draw(name, p, r)
            p.copy_(val.to(torch.float16).float())

        sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        sd["lm_head.weight"] = lm_head.weight.detach().clone()

        cache = DynamicCache(config=cfg)
        ids = torch.tensor([PROMPT])
        h = model(input_ids=ids, past_key_values=cache, use_cache=True)
        logits = [lm_head(h.last_hidden_state)[0].numpy()]
        for t in DECODE:
            h = model(input_ids=torch.tensor([[t]]), past_key_values=cache,
                      use_cache=True)
            logits.append(lm_head(h.last_hidden_state)[0].numpy())

    w = checkpoint_names(sd, CONFIG)
    arrays = {"w/" + k: v.numpy().astype(np.float16) for k, v in w.items()}
    np.savez_compressed(
        OUT, config=np.array(json.dumps(CONFIG)),
        meta=np.array(json.dumps(_meta())),
        prompt=np.array(PROMPT, np.int32), decode=np.array(DECODE, np.int32),
        prefill_logits=logits[0].astype(np.float32),
        decode_logits=np.stack([step[-1] for step in logits[1:]]).astype(np.float32),
        **arrays)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
