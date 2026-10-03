"""The gemma4 text goldens: what the maker's reference (HF transformers
`models/gemma4/modeling_gemma4.py`, Google's own port; transformers
5.16.1) computes for two tiny random Gemma 4 text models, under torch in
float32 on the CPU.

    $TORCH_PYTHON tests/support/goldens/build_gemma4_text.py

Writes gemma4_text.npz. Two configs, chosen to cover every numeric path
the released rungs take:

  dense  e-style (e2b/e4b): per-layer embeddings, KV sharing (the last two
         layers reuse the last sliding and the last full layer's K/V),
         double-wide MLP on the shared layers, sliding/full mix with
         proportional partial RoPE and a bigger head on the full layers,
         final logit softcap 30.
  moe    26B-A4B-style: MoE block beside the dense MLP (router with
         scale and per-expert scale), K=V on the full layers with their own
         kv-head count, no PLE, no sharing; use_bidirectional_attention
         "vision", and a second prefill with an image block (BLOCK) under
         the reference's own mask builder (`create_masks_for_vision_model`:
         sliding layers AND(window, OR(causal, block)), full layers causal)
         -> `<cfg>/prefill_block`.

Every weight is seeded random -- norms, layer scalars and per-expert
scales off 1 too, so a missing gain or a (1+w) shows. Each model sees an
11-token prefill (past the 6-token sliding window) then 5 decode tokens
one at a time through the reference's own DynamicCache. The npz holds the
weights under the checkpoint's own names (`<cfg>/w/<name>`), the config
(`<cfg>/config`, json), and the logits (`<cfg>/prefill`, `<cfg>/decode`).
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "gemma4_text.npz"

PROMPT = [3, 17, 42, 5, 9, 60, 33, 2, 11, 48, 27]
DECODE = [7, 55, 21, 36, 4]
#: an image block over prompt positions 2..8 (7 tokens: longer than the
#: 6-token window, so the window still cuts inside the block)
BLOCK = [-1, -1, 0, 0, 0, 0, 0, 0, 0, -1, -1]

_COMMON = dict(
    vocab_size=64, hidden_size=32, intermediate_size=48,
    num_attention_heads=2, num_key_value_heads=1, head_dim=16,
    global_head_dim=32, rms_norm_eps=1e-6, sliding_window=6,
    max_position_embeddings=256, final_logit_softcapping=30.0,
    hidden_activation="gelu_pytorch_tanh", tie_word_embeddings=True,
    rope_parameters={
        "full_attention": {"partial_rotary_factor": 0.25,
                           "rope_theta": 1000000.0,
                           "rope_type": "proportional"},
        "sliding_attention": {"rope_theta": 10000.0,
                              "rope_type": "default"}},
    pad_token_id=0, bos_token_id=2, eos_token_id=1)

CONFIGS = {
    "dense": dict(
        _COMMON, model_type="gemma4_text", num_hidden_layers=6,
        layer_types=["sliding_attention", "sliding_attention",
                     "full_attention", "sliding_attention",
                     "sliding_attention", "full_attention"],
        num_kv_shared_layers=2, use_double_wide_mlp=True,
        hidden_size_per_layer_input=8, vocab_size_per_layer_input=64,
        attention_k_eq_v=False, num_global_key_value_heads=None,
        enable_moe_block=False),
    "moe": dict(
        _COMMON, model_type="gemma4_text", num_hidden_layers=4,
        num_key_value_heads=2,
        layer_types=["sliding_attention", "full_attention",
                     "sliding_attention", "full_attention"],
        num_kv_shared_layers=0, use_double_wide_mlp=False,
        hidden_size_per_layer_input=0, vocab_size_per_layer_input=64,
        attention_k_eq_v=True, num_global_key_value_heads=1,
        enable_moe_block=True, num_experts=4, top_k_experts=2,
        moe_intermediate_size=16, use_bidirectional_attention="vision"),
}


def _run(name, cfg_dict, seed):
    import torch
    from transformers import DynamicCache
    from transformers.models.gemma4.configuration_gemma4 import \
        Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

    torch.manual_seed(seed)
    cfg = Gemma4TextConfig(**{k: v for k, v in cfg_dict.items()
                              if k != "model_type"})
    cfg._attn_implementation = "eager"
    model = Gemma4ForCausalLM(cfg).float().eval()
    g = torch.Generator().manual_seed(seed)
    weights = {}
    with torch.no_grad():
        named = dict(model.named_parameters())
        named.update({k: v for k, v in model.named_buffers()
                      if k.endswith("layer_scalar")})
        for k, p in named.items():
            if k.endswith(("norm.weight", "layer_scalar", "per_expert_scale",
                           "router.scale")):
                v = 1.0 + 0.3 * torch.randn(p.shape, generator=g)
            else:
                v = 0.15 * torch.randn(p.shape, generator=g)
            p.copy_(v)
            weights[k] = v.numpy().astype(np.float32)
        ids = torch.tensor([PROMPT])
        cache = DynamicCache(config=cfg)
        out = model(input_ids=ids, past_key_values=cache, use_cache=True)
        prefill = out.logits[0].numpy()
        decode = []
        for t in DECODE:
            out = model(input_ids=torch.tensor([[t]]),
                        past_key_values=out.past_key_values, use_cache=True)
            decode.append(out.logits[0, -1].numpy())
        if cfg_dict.get("use_bidirectional_attention") == "vision":
            from transformers.models.gemma4.modeling_gemma4 import \
                create_masks_for_vision_model
            emb = model.model.embed_tokens(ids)
            masks = create_masks_for_vision_model(
                cfg, emb, None, None, torch.arange(len(PROMPT))[None],
                torch.tensor([BLOCK]))
            rec_block = model(input_ids=ids, attention_mask=masks).logits[0]
        # the same tokens in one pass: the cache path must agree with it
        full = model(input_ids=torch.tensor([PROMPT + DECODE])).logits[0]
        assert np.allclose(full[len(PROMPT):].numpy(), np.stack(decode),
                           atol=1e-4), "reference cache disagrees with itself"
    rec = {f"{name}/w/{k}": v for k, v in weights.items()}
    rec[f"{name}/config"] = np.array(json.dumps(cfg_dict))
    rec[f"{name}/prefill"] = prefill.astype(np.float32)
    rec[f"{name}/decode"] = np.stack(decode).astype(np.float32)
    if cfg_dict.get("use_bidirectional_attention") == "vision":
        rec[f"{name}/prefill_block"] = rec_block.numpy().astype(np.float32)
    return rec


def main():
    rec = {}
    for i, (name, c) in enumerate(CONFIGS.items()):
        rec.update(_run(name, c, seed=i))
    rec["prompt"] = np.array(PROMPT)
    rec["decode_ids"] = np.array(DECODE)
    rec["block"] = np.array(BLOCK)
    np.savez_compressed(OUT, **rec)
    print(OUT, OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
