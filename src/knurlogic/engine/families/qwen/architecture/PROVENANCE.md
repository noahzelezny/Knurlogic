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
