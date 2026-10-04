# Vendored architecture modules

Each entry is a claim that this file is the arithmetic the
artifacts were validated against -- not merely that it imports.

## glm5_next (package) -- mlx-vlm 0.6.17

- from: `mlx_vlm/models/glm5_next` of an installed mlx-vlm 0.6.17
  (language.py sha256 f1c66fecf998..., byte-identical across the
  installs checked)
- closure: every module it imports, transitively, copied VERBATIM into
  `glm5_next/_mlx_vlm/` with mlx-vlm's own tree shape (kv_quant, turboquant,
  models/{activations, base, cache, gated_delta, mla, mlp, rope_utils,
  switch_layers, deepseek_v32/{config,language}, deepseek_v4/hyper_connection}),
  so their relative imports are unchanged. The only edit: glm5_next's own
  `from ..X import` lines point at `knurlogic.engine.families.glm5.architecture.glm5_next._mlx_vlm.models.X`.
- why 0.6.17, not 0.7.1: the released GLM-5.3-Flash rungs were built and
  scored on 0.6.17. Its linear-attention modules are `forget_gate`,
  `b_proj`, `g_a_proj`; 0.7.1 renamed and restructured them (gated_delta
  144 changed lines, switch_layers 199, mla 37, hyper_connection 56) and
  its sanitize does not map the old names, so the rungs do not load on
  0.7.1 at all. The artifact is the authority, not the newest upstream.
- verified: GLM-5.3-Flash 2.7 bit-identical to its published model.py
  (tools/vq_gate.py; M4 Max, mlx 0.31.2); tools/vision_gate.py PASS
  through `knurlogic serve` with no mlx-vlm installed.
- mlx-lm loads it through `engine/vq/runtime.model_classes`, which builds
  the nested configs (mlx-vlm's update_module_configs, done the same way)
  and returns logits from the language model.

### Edits 1-5, a reference audit (GLM-5.3-Flash)

Held to the maker's reference, HF transformers 5.16.1
`models/glm5_next/modeling_glm5_next.py` (zai-org's GLM-5 repo ships no
model code; the release is BF16, so there is no FP8/QAT rounding to
simulate). A tiny random float32 model run through the reference's own
classes (tests/support/goldens/build_glm5_next.py; 21-token prefill, 4
decode steps through each side's cache, index_topk 8 so the indexer
selects) gave max |logit diff| 3.20 prefill / 2.07 decode before these
edits and 6e-6 / 4e-6 after (tests/engine/test_glm5_next_reference.py).
All edits are in `language.py`; the `_mlx_vlm` closure stays verbatim.

1. **SwiGLU clamp at swiglu_limit** (`_clamped_swiglu`, `Glm5NextMLP`,
   `Glm5NextMoE`). The reference clamps gate above at 10 and up to +-10 in
   the dense MLP, the shared expert (Glm5NextTextMLP, modeling 101-104)
   and the routed experts (Glm5NextTextExperts._apply_gate, 137-142); the
   port used mlx-vlm's unclamped DeepseekMLP / SwitchGLU everywhere. Moves
   any token whose MLP activations leave +-10; the whole golden diff.
2. **Router logits in float32** (`Glm5NextMoEGate`): the reference
   computes `F.linear(x.float(), weight.float())` (moe_router_dtype
   float32, modeling 160); the port matmul'd in the model dtype (bf16)
   then took the sigmoid in float32. bf16 only; can flip a near-tied
   expert choice.
3. **q_a / kv_a RMSNorm eps** = rms_norm_eps (1e-5) as the reference
   (modeling 1103, 1116); the port hard-coded 1e-6 from DeepSeek-V3.2.
   Every token, small.
4. **Indexer k_norm LayerNorm eps 1e-6** (modeling 763); the port used
   mlx's default 1e-5. Every indexer key, small; can change a selection.
5. **Indexer scoring in float32** (`Glm5NextIndexer`): the reference
   takes q @ pool_keys, the relu and the head weights in float32
   (modeling 825-830) and the k-pool softmax in float32, cast to the keys'
   dtype (962-966); the port ran them in bf16. bf16 only; can flip a
   near-tied pool selection.

Checked and equal: NoPE MLA scale (qk_head_dim^-0.5, 256), absorbed and
expanded MLA, sparse mask from topk + tail, k-pool start at the first
valid key, mHC (input RMSNorm at rms_norm_eps, sigmoid+eps / 2*sigmoid,
softmax+eps then 20 Sinkhorn iterations, hc_eps 1e-6), final unweighted
stream mean then norm, KDA (safe gate lower bound -5, l2norm eps 1e-6,
Dk^-0.5 query scale, beta sigmoid, float32 state, gated RMSNorm in
float32), sigmoid routing with e_score_correction_bias, n_group 1,
norm_topk_prob, routed_scaling_factor 2.5, untied lm_head. Not ported
(config never uses it): `indexer_types == "shared"` cross-layer top-k
reuse -- every GLM-5.3-Flash layer is "full". Padding-only difference
left as is: the reference zeroes padded rows of the linear-attention
input before every projection (apply_mask_to_padding_states); the port
masks the q/k/v stream only.

### Edits 6+, a second reference audit (GLM-5.3-Flash, bf16 and memory)

6. **The MLA latent is stored once** (`Glm5NextSparseAttention`,
   `_LatentCache`). K = V = the normed latent; the port passed it as both
   keys and values, so every cache (bf16 KVCache, 8-bit QuantKVCache, the
   batch caches merged from them) held two copies. The values are now a
   zero-width array, as the DSA indexer's cache already does, so trim,
   state, merge, extract and kvquant work unchanged on one copy. The
   reference (DynamicCache) stores the latent once too. Per token on
   GLM-5.3-Flash (11 MLA layers): 28182 -> 16918 bytes bf16, 17622 ->
   11638 at 8 bits (tests/engine/test_glm5_cache_bytes.py; tuning's
   kv_bytes_per_token now equals what the cache stores). No logit change.
