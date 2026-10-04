# Vendored architecture modules

Each entry is a claim that this file is the arithmetic the
artifacts were validated against -- not merely that it imports.

## qwen4_exp.py

- taken: 2026-09-18
- from: `<a venv>`
- interpreter: `<a venv>`
- mlx-lm: 0.32.0
- sha256: `15df2080b3197db26067e1e4e23c2ce152841a8f0f60ff1fb2b2976e93dbb4b9`
- note: the environment qwen4_exp VQ artifacts are fit and scored in; carries PipelineMixin and the predicate-arity shim

## qwen3_5.py

- taken: 2026-09-18
- from: `<a venv>`
- interpreter: `<a venv>`
- mlx-lm: 0.32.0
- sha256: `14c4898a03567998e825cb1817942001871e979b9e0cefd3b4383cbbb61eddf3`
- note: the mlx-lm 0.32.0 environment is taken as authoritative: it is where the VQ artifacts are fit and scored, and it is the SUPERSET -- its qwen3_5 carries PipelineMixin, the other environment's copy does not

## qwen3_5_moe.py

- taken: 2026-09-18
- from: `<a venv>`
- interpreter: `<a venv>`
- mlx-lm: 0.32.0
- sha256: `ef9e8e1f6a5c097b29587c8330e8eb9c9cbdc52fbb4597fbc2362606c1996619`
- note: the mlx-lm 0.32.0 environment is taken as authoritative: it is where the VQ artifacts are fit and scored, and it is the SUPERSET -- its qwen3_5 carries PipelineMixin, the other environment's copy does not

## Vendored edits (audit against Qwen's reference, 2026-10-03)

The sha256 lines above are the files as taken. Each file now also carries
the vision (MRoPE) and long-context edits pins.json notes, and the numbered
edits below, each marked `knurlogic vendored edit N` in the source; pins.json
holds the current sha256. The reference is HF transformers 5.16.1's
modeling_qwen3_5.py / modeling_qwen3_5_moe.py / modeling_qwen4_exp.py; the
evidence is tests/engine/test_qwen_reference.py on
tests/support/goldens/qwen_reference.npz (build_qwen_reference.py: tiny
random configs, float32, 21-token prefill + 4 cached decode steps). Max abs
logit diff to the reference, before -> after edits 1-3: qwen3_5 3.6e-01 ->
5.9e-06, qwen3_5_moe 5.2e-01 -> 1.3e-05, qwen4_exp 2.19 -> 1.6e-06.

### Edit 1, a fix (qwen3_5.py, qwen4_exp.py; so also qwen3_5_moe)

1. **The deltanet's q/k l2norm eps** (`GatedDeltaNet.__call__`). The
   reference normalizes q and k with FLA's l2norm, `x * rsqrt(sum(x^2) +
   1e-6)` (modeling_qwen3_5.py:243-247 / 263-264, modeling_qwen4_exp.py
   259-262); the taken files used `mx.fast.rms_norm(x, None, 1e-6)`, which
   adds the eps to mean(x^2) -- an eps of dk * 1e-6 (128x the reference's
   at dk 128) on the sum. Now `rms_norm(x, None, 1e-6 / dk)`, as mlx-lm
   0.32.0's own `gated_delta.normalize_qk`. Every token, but tiny unless a
   head's q or k is near zero: on the golden's plain weights ~3e-5 of logit
   (qwen3_5 2.7e-05 -> 8.6e-06); with layer 0's in_proj_qkv scaled 1e-3
   (sum(q^2) ~1e-4) 0.36 / 0.52 of logit.

### Edit 2, a fix (qwen4_exp.py)

2. **The n-gram hash seed** (`TextArgs.seed`). The PLE n-gram embedding
   hashes token ids with multipliers built from `seed`; the reference
   config defaults it to 1234 (configuration_qwen4_exp.py:156) and the
   released checkpoints' `ple_embedding.layer_multipliers` are 1234's (e.g.
   Flash-Next 2.1's [23703573157769, 20109073645365, 8052911324071]). No
   released config.json names a seed, so the taken file's default 0 hashed
   every token into the wrong n-gram rows: the PLE layer's embedding was
   wrong on every token (2.2 of logit on the golden). Default now 1234.

### Edit 3, latent config defaults (qwen4_exp.py)

3. **`output_gate_type` unset means `hidden_act`, and `norm_topk_prob`.**
   The reference's deltanet gate is `output_gate_type or hidden_act`
   (modeling_qwen4_exp.py:437-439, silu); the taken file defaulted to
   sigmoid. The reference router renormalizes the top-k only when
   `norm_topk_prob` (:898-917, default True); the taken file always did.
   The released configs set `output_gate_type: sigmoid` and leave
   `norm_topk_prob` at True, so neither changes a released model (measured
   on a config with neither set and norm_topk_prob false: 1.47 -> 8e-7).

### Edit 4, a guard (qwen4_exp.py)

4. **The checkpoint's own n-gram multipliers win.** `sanitize` puts a
   checkpoint's int64 `ple_embedding.layer_multipliers` into the module's
   `_mults`, the values hashed with; the seed rebuild (edit 2) is only the
   fallback for a checkpoint without them or with a non-integer copy. A wrong or missing seed in a config cannot give
   wrong n-gram rows again (all released and VQ checkpoints store them as
   I64). Test: test_qwen_reference.py::test_qwen4_exp_uses_the_checkpoints_own_ngram_multipliers.

## Vendored edits 5-8 (second audit, 2026-10-03)

The golden (qwen_reference.npz, same builder) now also holds config
variants, a bfloat16 run of each trunk and of single bf16 components, the
vision tower, the image processor and rope_index; its `meta["build"]`
records the date, transformers 5.16.1, torch 2.14.1, numpy 2.5.3, Pillow
12.3.0, Python 3.13.7, CPU, eager attention. Every test named below failed
on the files before these edits and passes after.

### Edit 5, config defaults (qwen4_exp.py)

5. **A key a config leaves out means what it means to the reference.**
   `TextArgs`'s defaults were the released Flash-Next's (hidden 2560, 48
   layers, 24 heads, 48 value heads, expert sizes 640, indexer 4/1/128/
   2048/4, split_ngram_parts 128, ple_embed_dim 2560, ple_layer_ids [2],
   eos 248044, rope_theta 1e7, partial_rotary_factor 0.25); they are now
   Qwen4ExpTextConfig's (configuration_qwen4_exp.py:82-160; rope_theta
   10000 and partial 1.0 from modeling_rope_utils). As the reference's
   `__post_init__` / `validate_architecture`: ple_layer_ids None is []
   (sorted, deduplicated), ple_embed_dim None is hidden_size, layer_types
   come from the config and the reference's own name
   `qwen_sparse_attention` reads as this file's `full_attention` (the
   taken file built a linear cache for it); a config with attention layers
   and no indexer_* (the reference has no default and cannot build its
   QSA layer) or with PLE and no eos_token_id is refused, as is an unknown
   layer type or output gate. Released configs set every one of these, so
   no released model changes. Tests: test_a_config_variant_...
   [qwen4_exp/defaults] (did not load: 128 n-gram shards for 512 -> 8e-7),
   [qwen4_exp/layer_types] (crashed on the linear cache -> 1.6e-6),
   test_qwen4_exp_refuses_*, test_qwen4_exp_without_ple_layer_ids_has_no_ple_layer.

### Edit 6, config defaults and layer_types (qwen3_5.py; so also qwen3_5_moe)

6. **Absent keys and layer_types.** The taken `TextModelArgs` defaults were
   neither reference's (vocab 151936, head_dim hidden/heads, rope_theta
   100000, 64 value heads, key dim 192, ...); `Model` now fills a key the
   config leaves out from Qwen3_5TextConfig's or Qwen3_5MoeTextConfig's
   defaults (they differ, so per family: `REFERENCE_DEFAULTS`), rope_theta
   10000.0 and a top-level rope_theta / partial_rotary_factor fill a
   rope_parameters without them, as the reference's standardization. The
   layer kinds were `(idx + 1) % full_attention_interval`; the reference
   reads only `layer_types`, so DecoderLayer and the masks' first-layer
   indices (and heads/qwen35.py's MTP block) now read `layer_types`,
   derived from the interval only when absent; legacy names remapped as
   the reference's. Released configs agree with the interval, so no
   released model changes. Tests: test_a_config_variant_...
   [qwen3_5{,_moe}/defaults] (3.3 / 5.5 -> 2e-5), [qwen3_5{,_moe}/layer_types]
   (did not load: attention weights on a layer built linear -> 3e-5).

### Edit 7, the router (qwen3_5.py, Qwen3.5-MoE)

7. **The top-k is always renormalized.** Qwen3_5MoeTopKRouter
   (modeling_qwen3_5_moe.py:763-779) has no norm_topk_prob and always
   divides by the top-k sum; mlx-lm's Qwen3-Next block honoured the key,
   so a config saying false gave un-renormalized weights here only.
   Qwen4-Exp's router does honour it (modeling_qwen4_exp.py:898-917, edit
   3, kept). Test: [qwen3_5_moe/norm_topk_false] (4.8e-2 -> 1e-5).

### Edit 8, bfloat16 rounding order (qwen3_5.py, qwen4_exp.py)

8. **Round where the reference rounds.** Found by one-module bf16 goldens
   (test_a_bfloat16_component_rounds_as_the_reference) and measured on the
   whole trunk:
   - the deltanet's decay: `a + dt_bias` was added in bf16 before
     softplus; the reference adds `a.float() + dt_bias`
     (modeling_qwen3_5.py:502). `gated_delta` (a local copy of mlx-lm's
     gated_delta_update dispatch) casts first;
   - torch computes a bf16 sigmoid / silu in float32 and rounds once;
     MLX's bf16 kernels round inside (silu is x * sigmoid(x)), a bf16 step
     off on ~1/3 of the elements. `_sigmoid` / `_silu` take the
     reference's order at every site: the deltanet's beta and conv silu,
     the attention output gate, the MLPs and the experts (`_SwiGLU`), the
     shared-expert gate, qwen4_exp's hyper-connections and PLE gate;
   - the zero-centred norms: the reference norms in float32 times
     `1 + weight.float()` and rounds once. qwen4_exp added `1.0 + weight`
     in bf16; qwen3_5's sanitize folded `+ 1` into a bf16 weight. sanitize
     now keeps the folded weight float32 (cast_predicate keeps it so in a
     conversion) and `RMSNorm` rounds once; an artifact converted before
     this edit carries the bf16 fold and runs as before;
   - the routers: softmax, top-k and renormalization in float32 on the
     logits in x's dtype, the weights then rounded to it (mlx-lm's
     Qwen3-Next picked the top-k among bf16-rounded probabilities, where a
     tie can pick another expert; qwen4_exp took float32 logits of a
     float32 copy of x).
   Checked and left: RMSNormGated already rounds as the reference (MLX's
   rms_norm rounds the normalized x before the weight). Not matched, on
   purpose: the deltanet's q/k l2norm, which the reference's torch fallback
   rounds to bf16 four times (and CPU's bf16 rsqrt in two steps) while
   FLA's kernel, what it runs on CUDA, stays float32; knurlogic's rms_norm
   rounds once. Fed the reference's normalized q/k, knurlogic's
   recurrence gives its core output bit for bit. Result, bf16 components:
   norms and MLP/MoE blocks bit for bit (were 1 step off on 36-58% of
   elements), the deltanet 1.17e-2 on outputs up to 1.9 (was 1.95e-2 to
   2.34e-2). Whole trunk in bf16 (no SCALE), rms logit diff to the
   reference's bf16 over its own bf16-vs-f32 rms: 0.45 / 0.43 / 0.41
   (qwen3_5 / _moe / qwen4_exp), was 1.38 / 1.34 / 1.32; max 0.15 / 0.33 /
   0.07, was 0.79 / 1.27 / 0.18. Float32 is unchanged (the casts are
   no-ops).
