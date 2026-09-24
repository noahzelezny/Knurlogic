# Vendored architecture modules

Each entry is a claim that this file is the arithmetic the
artifacts were validated against -- not merely that it imports.

## glm5_next (package) -- mlx-vlm 0.6.17, re-vendored 2026-09-23

- from: `/opt/anaconda3/envs/exo/lib/python3.13/site-packages/mlx_vlm/models/glm5_next`
  (mlx-vlm 0.6.17; language.py sha256 f1c66fecf998..., byte-identical to
  the copy in `~/mlx-vlm` and in the external drive `glm5vlm` venv)
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
  0.7.1 at all. The 2026-09-18 switch to stock 0.7.1 ("upstream was
  ahead") was undone for that reason: the artifact is the authority.
- verified: GLM-5.3-Flash 2.7 G-VQ bit-identical to its published model.py
  (M4, mlx 0.31.2); tools/vision_gate.py PASS through `knurlogic serve`
  with no mlx-vlm installed.
- mlx-lm loads it through `engine/vq/runtime.model_classes`, which builds
  the nested configs (mlx-vlm's update_module_configs, done the same way)
  and returns logits from the language model.
