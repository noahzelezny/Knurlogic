"""The gemma4 text goldens: what the maker's reference (HF transformers
`models/gemma4/modeling_gemma4.py` and `configuration_gemma4.py`, Google's
own port; transformers 5.16.1) computes for tiny random Gemma 4 text
models, under torch on the CPU.

    $TORCH_PYTHON tests/support/goldens/build_gemma4_text.py

Writes gemma4_text.npz (its `__meta__` says what built it). Configs:

  dense     e-style (e2b/e4b): per-layer embeddings, KV sharing (the last
            two layers reuse the last sliding and the last full layer's
            K/V), double-wide MLP on the shared layers, sliding/full mix
            with proportional partial RoPE and a bigger head on the full
            layers, final logit softcap 30.
  moe       26B-A4B-style: MoE block beside the dense MLP (router with
            scale and per-expert scale), K=V on the full layers with their
            own kv-head count, no PLE, no sharing;
            use_bidirectional_attention "vision", and a second prefill with
            an image block (BLOCK) under the reference's own mask builder
            (`create_masks_for_vision_model`: sliding layers AND(window,
            OR(causal, block)), full layers causal) -> `moe/prefill_block`.
  defaults  only the sizes a tiny model needs (vocab, hidden, intermediate,
            layer count, window); every other key -- layer_types, heads,
            kv heads, head dims, softcap, KV sharing, double-wide MLP,
            rope_parameters, PLE width -- is left out, so the logits are
            what HF's Gemma4TextConfig defaults give: 5:1 sliding pattern
            with the last layer forced full, no final softcap, 8 heads / 4
            kv heads, head 256 / global head 512, no KV sharing.
  all       use_bidirectional_attention "all": non-causal on every layer,
            the sliding window halved (+1) by the config, the sliding
            layers attending |q - k| <= window both ways. Besides the
            prefill and the cached decode: the same prompt prefilled in
            two chunks through the cache -> `all/chunked` (the second
            chunk's logits), which is what the reference's cache does when
            a prompt arrives in pieces.

  dense_bf16, moe_bf16
            dense's and moe's configs on other weights (q/k norm gain 0.3,
            see BF16), in float32 and then in bfloat16 under both of HF's
            attention implementations -> `<cfg>/bf16_eager/*`,
            `<cfg>/bf16_sdpa/*`; their spread is what the parity test's
            bf16 tolerance is justified against.

`resolve` records how HF's configuration_gemma4 resolves a set of text
configs (some with keys left out, the released artifacts' own from
tests/support/artifacts, and the gemma4 wrapper's) -- the fields the model
is built from. `ops/*` are HF's Gemma4RMSNorm and gelu_pytorch_tanh * up on
bfloat16 inputs, as bf16 bit patterns.

Every weight is seeded random -- norms, layer scalars and per-expert
scales off 1 too, so a missing gain or a (1+w) shows. Each model sees an
11-token prefill (past the window) then 5 decode tokens one at a time
through the reference's own DynamicCache. The npz holds the weights under
the checkpoint's own names (`<cfg>/w/<name>`; REGENERATED configs hold
their shapes and seed instead, see np_weights), the config (`<cfg>/config`,
json, exactly the keys handed to Gemma4TextConfig), and the logits
(`<cfg>/prefill`, `<cfg>/decode`).
"""
import datetime
import json
import platform
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "gemma4_text.npz"
ARTIFACTS = Path(__file__).parents[1] / "artifacts"

PROMPT = [3, 17, 42, 5, 9, 60, 33, 2, 11, 48, 27]
DECODE = [7, 55, 21, 36, 4]
#: an image block over prompt positions 2..8 (7 tokens: longer than the
#: 6-token window, so the window still cuts inside the block)
BLOCK = [-1, -1, 0, 0, 0, 0, 0, 0, 0, -1, -1]
#: "all": the prompt's first chunk when it is prefilled in two pieces
CHUNK = 4

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
    # 8 layers: the 5:1 default pattern gives s s s s s f s s, and HF
    # forces the last one full -> s s s s s f s f
    "defaults": dict(
        model_type="gemma4_text", vocab_size=64,
        vocab_size_per_layer_input=64, hidden_size=32, intermediate_size=48,
        num_hidden_layers=8, sliding_window=6),
    "dense_bf16": None,     # filled below: dense's / moe's config
    "moe_bf16": None,
    # window 8 -> 8 // 2 + 1 = 5 after HF's config
    "all": dict(
        _COMMON, model_type="gemma4_text", num_hidden_layers=4,
        sliding_window=8,
        layer_types=["sliding_attention", "full_attention",
                     "sliding_attention", "full_attention"],
        num_kv_shared_layers=0, use_double_wide_mlp=False,
        hidden_size_per_layer_input=0, vocab_size_per_layer_input=64,
        attention_k_eq_v=False, enable_moe_block=False,
        use_bidirectional_attention="all"),
}

CONFIGS["dense_bf16"] = CONFIGS["dense"]
CONFIGS["moe_bf16"] = CONFIGS["moe"]

#: text configs whose resolution (HF's __post_init__) is recorded
RESOLVE = {
    "empty": {},
    "seven_layers": {"num_hidden_layers": 7},
    "last_sliding": {"num_hidden_layers": 6,
                     "layer_types": ["sliding_attention"] * 6},
    "all": {"use_bidirectional_attention": "all"},
    "all_1024": {"use_bidirectional_attention": "all",
                 "sliding_window": 1024},
    "k_eq_v": {"attention_k_eq_v": True, "num_global_key_value_heads": 2},
    "global_kv_without_k_eq_v": {"num_global_key_value_heads": 2},
}

#: the fields the model is built from
FIELDS = ("hidden_size", "intermediate_size", "num_hidden_layers",
          "num_attention_heads", "num_key_value_heads", "head_dim",
          "sliding_window", "layer_types", "final_logit_softcapping",
          "use_double_wide_mlp", "num_kv_shared_layers", "vocab_size",
          "vocab_size_per_layer_input", "hidden_size_per_layer_input",
          "attention_k_eq_v", "enable_moe_block", "num_experts",
          "top_k_experts", "moe_intermediate_size", "rope_parameters",
          "rms_norm_eps", "tie_word_embeddings", "max_position_embeddings",
          "use_bidirectional_attention", "hidden_activation",
          "attention_bias")


def _resolved(tc):
    # head_dim / num_key_value_heads are per-layer in HF: the global value
    # is what the sliding layers take, the per-layer lists below the rest
    tc.allow_global_per_layer_attribute_access = True
    out = {f: getattr(tc, f) for f in FIELDS}
    n = tc.num_hidden_layers
    out["layer_head_dim"] = [tc.per_layer_config[i].head_dim
                             for i in range(n)]
    out["layer_kv_heads"] = [tc.per_layer_config[i].num_key_value_heads
                             for i in range(n)]
    return out


def resolve():
    from transformers.models.gemma4.configuration_gemma4 import (
        Gemma4Config, Gemma4TextConfig)
    rec = {}
    for name, d in RESOLVE.items():
        rec[name] = {"input": d, "text": True,
                     "hf": _resolved(Gemma4TextConfig(**d))}
    # the gemma4 wrapper: text_config resolves the same way, and a
    # top-level vocab_size is not the text model's
    wrapped = {"model_type": "gemma4", "vocab_size": 262144,
               "text_config": {"vocab_size": 100, "num_hidden_layers": 3}}
    rec["wrapper"] = {"input": wrapped, "text": False, "hf": _resolved(
        Gemma4Config(**{k: v for k, v in wrapped.items()
                        if k != "model_type"}).text_config)}
    rec["wrapper_empty"] = {"input": {"model_type": "gemma4"}, "text": False,
                            "hf": _resolved(Gemma4Config().text_config)}
    for d in sorted(ARTIFACTS.glob("gemma-4-*")):
        cfg = json.loads((d / "config.json").read_text())
        tc = Gemma4Config(text_config=cfg["text_config"]).text_config
        rec[f"artifact:{d.name}"] = {"input": cfg["text_config"],
                                     "text": True, "hf": _resolved(tc)}
    return rec


def _gain(name):
    return name.endswith(("norm.weight", "layer_scalar", "per_expert_scale",
                          "router.scale"))


def np_weights(shapes, seed, qk_gain=1.0):
    """The weights of the configs that store only their shapes
    (REGENERATED): numpy's PCG64 in name order, so the parity test makes
    the same arrays without the npz carrying them (the defaults config's
    256/512-wide heads would be megabytes).

    `qk_gain`: the q/k norms' mean gain. The attention scale is 1.0, so with
    256-wide heads unit q/k norms give scores in the hundreds and a softmax
    so sharp that fp32 summation order alone moves the logits by ~1e-3 (the
    reference's own cache vs its one-pass forward); a smaller gain keeps the
    comparison about the arithmetic."""
    rng = np.random.default_rng(seed)
    out = {}
    for k in sorted(shapes):
        z = rng.standard_normal(shapes[k]).astype(np.float32)
        if k.endswith(("q_norm.weight", "k_norm.weight")):
            out[k] = (qk_gain * (1.0 + 0.3 * z)).astype(np.float32)
        elif _gain(k):
            out[k] = (1.0 + 0.3 * z).astype(np.float32)
        else:
            out[k] = (0.15 * z).astype(np.float32)
    return out


#: configs whose npz carries shapes and a seed, not weights (np_weights)
REGENERATED = {"defaults": 0.3, "all": 1.0, "dense_bf16": 0.3,
               "moe_bf16": 0.3}
#: the bf16 runs. Random tiny models in bf16 are chaotic -- attention
#: scale 1.0 with unit q/k norms gives sharp softmaxes, and one flipped
#: rounding moves the logits by tenths -- so these use the dense and moe
#: configs with q/k norm gain 0.3, which keeps HF's own two attention
#: implementations within ~0.1 of each other in bf16
BF16 = ("dense_bf16", "moe_bf16")


def _named(model):
    named = dict(model.named_parameters())
    named.update({k: v for k, v in model.named_buffers()
                  if k.endswith("layer_scalar")})
    return named


def _seed_weights(model, seed):
    import torch
    g = torch.Generator().manual_seed(seed)
    weights = {}
    with torch.no_grad():
        for k, p in _named(model).items():
            if _gain(k):
                v = 1.0 + 0.3 * torch.randn(p.shape, generator=g)
            else:
                v = 0.15 * torch.randn(p.shape, generator=g)
            p.copy_(v)
            weights[k] = v.numpy().astype(np.float32)
    return weights


def _np_seed_weights(model, seed, qk_gain):
    import torch
    named = _named(model)
    weights = np_weights({k: tuple(p.shape) for k, p in named.items()},
                         seed, qk_gain)
    with torch.no_grad():
        for k, p in named.items():
            p.copy_(torch.from_numpy(weights[k]))
    return weights


def _prefill_decode(model, cfg):
    import torch
    from transformers import DynamicCache
    with torch.no_grad():
        ids = torch.tensor([PROMPT])
        out = model(input_ids=ids, past_key_values=DynamicCache(config=cfg),
                    use_cache=True)
        prefill = out.logits[0].float().numpy()
        decode = []
        for t in DECODE:
            out = model(input_ids=torch.tensor([[t]]),
                        past_key_values=out.past_key_values, use_cache=True)
            decode.append(out.logits[0, -1].float().numpy())
    return prefill, np.stack(decode)


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
    if name in REGENERATED:
        weights = _np_seed_weights(model, seed, REGENERATED[name])
        rec = {f"{name}/shapes": np.array(json.dumps(
            {k: list(v.shape) for k, v in weights.items()})),
               f"{name}/seed": np.array(seed),
               f"{name}/qk_gain": np.array(REGENERATED[name])}
    else:
        weights = _seed_weights(model, seed)
        rec = {f"{name}/w/{k}": v for k, v in weights.items()}
    prefill, decode = _prefill_decode(model, cfg)
    rec[f"{name}/config"] = np.array(json.dumps(cfg_dict))
    rec[f"{name}/prefill"] = prefill.astype(np.float32)
    rec[f"{name}/decode"] = decode.astype(np.float32)
    with torch.no_grad():
        ids = torch.tensor([PROMPT])
        if cfg_dict.get("use_bidirectional_attention") == "vision" \
                and name not in BF16:
            from transformers.models.gemma4.modeling_gemma4 import \
                create_masks_for_vision_model
            emb = model.model.embed_tokens(ids)
            masks = create_masks_for_vision_model(
                cfg, emb, None, None, torch.arange(len(PROMPT))[None],
                torch.tensor([BLOCK]))
            rec[f"{name}/prefill_block"] = model(
                input_ids=ids, attention_mask=masks).logits[0].numpy()
        if cfg_dict.get("use_bidirectional_attention") == "all":
            # not causal: the one-pass logits of PROMPT + DECODE are NOT
            # the cached decode's; record what the cache does in pieces
            cache = DynamicCache(config=cfg)
            model(input_ids=ids[:, :CHUNK], past_key_values=cache,
                  use_cache=True)
            out = model(input_ids=ids[:, CHUNK:], past_key_values=cache,
                        use_cache=True)
            rec[f"{name}/chunked"] = out.logits[0].numpy()
        else:
            # the same tokens in one pass: the cache path must agree
            full = model(input_ids=torch.tensor([PROMPT + DECODE])).logits[0]
            assert np.allclose(full[len(PROMPT):].numpy(), decode,
                               atol=1e-4), f"{name}: reference cache disagrees"
    if name in BF16:
        # the same weights in bfloat16, under both of HF's attention
        # implementations: eager rounds the softmax probabilities to bf16
        # before P @ V, sdpa does not (nor does ours)
        for impl in ("eager", "sdpa"):
            cfg._attn_implementation = impl
            m = Gemma4ForCausalLM(cfg).float().eval()
            _np_seed_weights(m, seed, REGENERATED[name])
            bp, bd = _prefill_decode(m.to(torch.bfloat16), cfg)
            rec[f"{name}/bf16_{impl}/prefill"] = bp.astype(np.float32)
            rec[f"{name}/bf16_{impl}/decode"] = bd.astype(np.float32)
    return rec


def _bf16_bits(t):
    """A bfloat16 torch tensor as its uint16 bit patterns (exact, small)."""
    import torch
    return t.contiguous().view(torch.int16).numpy().view(np.uint16)


def ops():
    """HF's Gemma4RMSNorm (with and without scale) and the MLP's
    gelu_pytorch_tanh(gate) * up on bfloat16 inputs: the rounding points
    the bf16 model has. Stored as bf16 bit patterns."""
    import torch
    from transformers.activations import ACT2FN
    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm
    g = torch.Generator().manual_seed(0)
    n, d = 16, 1024
    x = (torch.randn(n, d, generator=g) * 3).to(torch.bfloat16)
    w = (1 + 0.3 * torch.randn(d, generator=g)).to(torch.bfloat16)
    gate = (torch.randn(n, d, generator=g) * 2).to(torch.bfloat16)
    up = (torch.randn(n, d, generator=g) * 2).to(torch.bfloat16)
    norm = Gemma4RMSNorm(d, eps=1e-6).to(torch.bfloat16)
    bare = Gemma4RMSNorm(d, eps=1e-6, with_scale=False)
    with torch.no_grad():
        norm.weight.copy_(w)
        out = {"x": x, "w": w, "gate": gate, "up": up, "norm": norm(x),
               "norm_noscale": bare(x),
               "act": ACT2FN["gelu_pytorch_tanh"](gate) * up}
    return {f"ops/{k}": _bf16_bits(v) for k, v in out.items()}


def main():
    import torch
    import transformers
    rec = {}
    for i, (name, c) in enumerate(CONFIGS.items()):
        rec.update(_run(name, c, seed=i))
    rec["prompt"] = np.array(PROMPT)
    rec["decode_ids"] = np.array(DECODE)
    rec["block"] = np.array(BLOCK)
    rec["chunk"] = np.array(CHUNK)
    rec["resolve"] = np.array(json.dumps(resolve()))
    rec.update(ops())
    rec["__meta__"] = np.array(json.dumps({
        "what": "HF transformers Gemma4ForCausalLM / Gemma4TextConfig / "
                "Gemma4Config, eager attention, CPU",
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "reference_python": sys.executable,
        "seeds": {n: i for i, n in enumerate(CONFIGS)},
        "built": datetime.date.today().isoformat(),
        "command": "$TORCH_PYTHON tests/support/goldens/build_gemma4_text.py",
    }, sort_keys=True))
    np.savez_compressed(OUT, **rec)
    print(OUT, OUT.stat().st_size, "bytes")
    for n in BF16:
        e, s = (np.concatenate([rec[f"{n}/bf16_{i}/prefill"],
                                rec[f"{n}/bf16_{i}/decode"]])
                for i in ("eager", "sdpa"))
        f = np.concatenate([rec[f"{n}/prefill"], rec[f"{n}/decode"]])
        print(n, "bf16 eager vs fp32", np.abs(e - f).max(),
              "eager vs sdpa", np.abs(e - s).max())


if __name__ == "__main__":
    main()
