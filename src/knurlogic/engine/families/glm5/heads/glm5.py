"""GLM-5.3 (glm5_next) MTP drafting head.

Vendored from vqlab `src/vqlab/mtp_head_glm5.py` @ c492d1d (since moved to
`src/vqlab/mtp/mtp_head_glm5.py`, unchanged). Copied rather than imported;
fixes go upstream first, then re-vendor. Deviations from that source: the
MoE half is built with `Glm5NextMoE`, not `DeepseekV32MoE`, so the head
carries the trunk's SwiGLU clamp and float32 router (glm5_next edits 1-2);
the sidecar is opened through engine/model/weight_files (every artifact
file is); the docstrings and comments are rewritten. The `vqlab_mtp`
metadata key is kept as-is, so sidecars built by `vqlab mtp-pack` load
here unchanged.

This module only LOADS the packed q6 sidecar (`mtp-head-q6.safetensors`,
889 tensors). The head is NOT a trunk `Glm5NextDecoderLayer`: it has no
hyper-connection weights (a plain-residual DeepSeek-style block assembled
here from the attention and MoE sub-modules) and its OWN final norm
(`shared_head.norm`) before the trunk's shared lm_head. `model` is the
mlx_vlm glm5_next LanguageModel (pass a VLM wrapper's `.language_model`).
Design: docs/design/drafting.md (GLM head).
"""
from __future__ import annotations

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

from knurlogic.engine.model.trunk import text_model
from knurlogic.engine.mtp.head_base import GraftHead, fill, prefixed, save_sidecar

_GLUE = ("eh_proj", "enorm", "hnorm", "shared_head")


class Glm5Head(GraftHead):
    """One drafting head bound to a loaded glm5_next trunk."""

    def __init__(self, model, arch):
        lang = text_model(model)
        core = lang.model
        cfg = core.config
        self.model = lang
        self.core = core
        self.arch = arch
        self.cfg = cfg
        self.D = cfg.hidden_size
        self.eps = cfg.rms_norm_eps
        self.tie = cfg.tie_word_embeddings
        self.lm_head = None if self.tie else lang.lm_head

        # The block's halves, straight from the arch module, built as the
        # trunk's MoE layers are: Glm5NextMoE carries the SwiGLU clamp and
        # the float32 router (glm5_next edits 1-2); the head's weights
        # (layer 45) have the same names and shapes as every trunk MoE layer.
        self.self_attn = arch.Glm5NextSparseAttention(cfg)
        self.mlp = arch.Glm5NextMoE(cfg)
        self.input_layernorm = nn.RMSNorm(self.D, eps=self.eps)
        self.post_attention_layernorm = nn.RMSNorm(self.D, eps=self.eps)

        # Glue + the head's own final norm (shared_head.norm upstream).
        self.enorm = nn.RMSNorm(self.D, eps=self.eps)
        self.hnorm = nn.RMSNorm(self.D, eps=self.eps)
        self.eh_proj = nn.Linear(2 * self.D, self.D, bias=False)
        self.final_norm = nn.RMSNorm(self.D, eps=self.eps)

    # ------------------------------------------------------------- caches
    def make_draft_cache(self):
        """The fa-layer cache shape the sparse attention expects:
        CacheList(main-KV, indexer-KV), mlx_vlm classes throughout."""
        return self.arch.CacheList(self.arch.KVCache(), self.arch.KVCache())

    # -------------------------------------------------------------- build
    def _modules(self):
        return {
            "self_attn": self.self_attn,
            "mlp": self.mlp,
            "input_layernorm": self.input_layernorm,
            "post_attention_layernorm": self.post_attention_layernorm,
            "enorm": self.enorm,
            "hnorm": self.hnorm,
            "eh_proj": self.eh_proj,
            "final_norm": self.final_norm,
        }

    def load_graft(self, g, prefix="layers.45."):
        """Fill from the bf16 graft (keys as mtp-extract leaves them).

        Applies the same two transforms the trunk's sanitize applies to
        every layer — stacking the per-expert weights into switch_mlp and
        splitting kv_b_proj into embed_q / unembed_out — then loads the
        rest by name.
        """
        cfg = self.cfg
        w = dict(prefixed(g, prefix))
        if not w:
            raise ValueError(f"no graft keys under prefix {prefix!r}")

        # 1. Expert stack -> switch_mlp (mirrors DSV32Model.sanitize).
        for m in ("gate_proj", "up_proj", "down_proj"):
            key0 = f"mlp.experts.0.{m}.weight"
            if key0 in w:
                stacked = mx.stack([
                    w.pop(f"mlp.experts.{e}.{m}.weight")
                    for e in range(cfg.n_routed_experts)
                ])
                w[f"mlp.switch_mlp.{m}.weight"] = stacked

        # 2. kv_b_proj split -> embed_q / unembed_out (bf16 path of the
        #    trunk sanitize; the graft is unquantized by construction).
        v = w.pop("self_attn.kv_b_proj.weight")
        head_dim = cfg.qk_nope_head_dim + cfg.v_head_dim
        v = v.reshape(cfg.num_attention_heads, head_dim, -1)
        w["self_attn.embed_q.weight"] = mx.contiguous(
            v[:, : cfg.qk_nope_head_dim, :].swapaxes(-1, -2))
        w["self_attn.unembed_out.weight"] = mx.contiguous(
            v[:, cfg.qk_nope_head_dim :, :])

        # 3. Rename the head's own final norm to this module's slot.
        w["final_norm.weight"] = w.pop("shared_head.norm.weight")

        # 4. Load by name, refusing silent drops in either direction.
        mods = self._modules()
        slots = set().union(*(self.slots(mod, f"{name}.")
                              for name, mod in mods.items()))
        missing = self.check_slots(slots, w)
        if missing:
            raise ValueError(
                f"{len(missing)} head parameters got no graft "
                f"tensor, e.g. {missing[:4]}")
        for name, mod in mods.items():
            fill(mod, w, name + ".")
        mx.eval(self._params())
        return self

    # The quantize recipe must match the published heads: the expert
    # stack (switch_mlp and the shared experts, ~96% of the weight) at
    # `expert_bits`, everything else at `bits`; the router gate stays bf16
    # (it has no `to_quantized`) and the DSA indexer is `bits`/gs32
    # (mtp-head-q6: wk/wq_b/weights_proj at 6-bit, 32-wide groups). Any
    # other recipe changes the packed shapes and is a new head format.
    @staticmethod
    def expert_match(path):
        return ".switch_mlp." in path or path.endswith(
            ("gate_proj", "up_proj", "down_proj"))

    def _quantized(self):
        return (self.self_attn, self.mlp)

    def save(self, path, bits, group_size=32, expert_bits=None):
        flat = {}
        for name, mod in self._modules().items():
            for k, v in tree_flatten(mod.parameters()):
                flat[f"{name}.{k}"] = v
        return save_sidecar(path, flat, {
            "family": "glm5_next", "bits": bits,
            "group_size": group_size, "expert_bits": expert_bits})

    # --------------------------------------------------------------- draft
    def _trunk(self, h_row, nxt_id, cache=None):
        """(trunk hidden at t, token t+1) -> the head's output activation.

        `h_row` is the capture at the trunk's final `norm` INPUT — the
        mean-collapsed (B, T, D) hidden, which is what hnorm expects
        (eh_proj is 2D -> D). The block is plain-residual at (B, T, D);
        the mask is built the way the trunk builds its fa mask.
        """
        e = self.enorm(self.core.embed_tokens(nxt_id))
        hs = self.hnorm(h_row)
        x = self.eh_proj(mx.concatenate([e, hs], axis=-1))
        mask = self.arch.create_attention_mask(
            x, cache[0] if cache is not None else None, return_array=True)
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        x = x + r
        x = x + self.mlp(self.post_attention_layernorm(x))
        return self.final_norm(x)
