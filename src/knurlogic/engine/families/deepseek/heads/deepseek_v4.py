"""The deepseek_v4 multi-token-prediction head (DeepSeek-V4-Flash): bind a
packed sidecar to a loaded trunk and draft with it.

Wiring, per the official inference/model.py (MTPBlock) -- DeepSeek-V3's
MTP with V4's hyper-connections:

    h      the trunk's [B, T, hc, D] streams going INTO its hc_head
           (all hc streams, before the head collapse and the final norm)
    e      enorm(embed(x_{t+1}))                      shared embedding
    x      e_proj(e)[:, :, None] + h_proj(hnorm(h))   hnorm per stream
    x      ONE trunk block of the layers' own class, at layer id
           num_hidden_layers (compress_ratios[43] == 0 on Flash: a plain
           128-token window, no compressor, score-routed -- not hashed)
    out    lm_head(norm(hc_head(x)))                  shared lm_head, the
           head's own hc_head (sigmoid collapse) and norm

THE SIDECAR (what `vqlab` packs, `mtp-head-*.safetensors` beside the
trunk, outside its index). Every key is under `mtp.0.`:

    the block in the trunk's POST-sanitize layer layout:
      attn.{attn_sink, kv_norm.weight, q_norm.weight, wqkv_a.*, wq_b.*,
            wo_a.*, wo_b.*}, attn_norm.weight, ffn_norm.weight,
      ffn.gate.{weight, e_score_correction_bias},
      ffn.shared_experts.{gate,up,down}_proj.*,
      ffn.switch_mlp.{gate,up,down}_proj.*,
      hc_attn.{fn, base, scale}, hc_ffn.{fn, base, scale}
    the MTP-only tensors under their official names:
      e_proj.*, h_proj.*, enorm.weight, hnorm.weight, norm.weight,
      hc_head_fn, hc_head_base, hc_head_scale

Any linear may be quantized; the recipe is read off the tensors, so no
metadata is required: `<m>.scales` beside `<m>.weight` quantizes `<m>`,
with `<m>.biases` affine, without them mxfp4 (4-bit) or mxfp8 (8-bit);
bits and group size follow from the packed shapes.

Routed experts may instead be VQ: `<m>.codes` / `<m>.codebook` /
`<m>.vq_scales` in place of `<m>.weight` / `<m>.scales`, the format the
trunk's own VQ experts use. They run on the trunk's bundled runtime's
VQSwitchLinear (the class the artifact's model.py defines, found through
the loaded model's class); knurlogic carries no VQ runtime of its own.
Codebook [K, dim]; group = in / vq_scales' last axis; codes are packed
uint32 words of ceil(log2 K)-bit fields (ceil(in/dim/32) * bits wide per
row) or, when a row is in/dim wide, one code per entry.
"""
from __future__ import annotations

import dataclasses

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from knurlogic.engine.split.pipeline import unwrap

SIDECAR_NAME = "mtp-head-mxfp4.safetensors"
PREFIX = "mtp.0."
#: MTP-only tensors: official name -> this module's parameter path
_OWN = {"hc_head_fn": "hc_head.fn", "hc_head_base": "hc_head.base",
        "hc_head_scale": "hc_head.scale"}
_TOP = ("e_proj", "h_proj", "enorm", "hnorm", "norm", "hc_head")


def _to_module(k: str) -> str:
    """A sidecar key (prefix stripped) -> its path in `_Head`."""
    if k in _OWN:
        return _OWN[k]
    if k.split(".")[0] in _TOP:
        return k
    return "block." + k


def _to_sidecar(k: str) -> str:
    for src, dst in _OWN.items():
        if k == dst:
            return src
    return k[len("block."):] if k.startswith("block.") else k


class _Head(nn.Module):
    def __init__(self, block, hc_head, D, eps, norm_cls):
        super().__init__()
        self.block = block
        self.e_proj = nn.Linear(D, D, bias=False)
        self.h_proj = nn.Linear(D, D, bias=False)
        # the trunk's RMSNorm: the reference's, one rounding (edit 26)
        self.enorm = norm_cls(D, eps=eps)
        self.hnorm = norm_cls(D, eps=eps)
        self.norm = norm_cls(D, eps=eps)
        self.hc_head = hc_head


def _recipe(w: dict, path: str, mod) -> dict | None:
    """The to_quantized kwargs a packed `path` was written with, or None."""
    s = w.get(f"{path}.scales")
    if s is None or not hasattr(mod, "to_quantized"):
        return None
    in_dims = int(mod.weight.shape[-1])
    bits = int(w[f"{path}.weight"].shape[-1]) * 32 // in_dims
    gs = in_dims // int(s.shape[-1])
    if f"{path}.biases" in w:
        mode = "affine"
    else:
        mode = {4: "mxfp4", 8: "mxfp8"}.get(bits)
        if mode is None:
            raise ValueError(f"{path}: {bits}-bit without biases is neither "
                             f"affine nor mxfp4/mxfp8")
    return {"group_size": gs, "bits": bits, "mode": mode}


def _vq_class(model):
    """VQSwitchLinear from the bundled runtime `model` was loaded with:
    the module globals its class's own methods run in (mlx-lm executes a
    model_file without registering it in sys.modules)."""
    for cls in type(model).__mro__:
        for v in vars(cls).values():
            g = getattr(v, "__globals__", None)
            if g is not None and "VQSwitchLinear" in g:
                return g["VQSwitchLinear"]
    raise ValueError(
        "deepseek_v4 head: its routed experts are VQ (.codes/.vq_scales) "
        "but the trunk was not loaded through a bundled VQ runtime "
        "(no VQSwitchLinear beside its Model class)")


def _vq_module(w: dict, path: str, mod, vq_cls):
    """A VQSwitchLinear shaped for the packed `path`, or None when `path`
    is not VQ. Its shapes come from the dense skeleton `mod` (experts,
    out, in) and the codebook (K, dim), so a sidecar tensor of any other
    shape fails bind's shape check."""
    codes, vs = w.get(f"{path}.codes"), w.get(f"{path}.vq_scales")
    if codes is None and vs is None:
        return None
    cb = w.get(f"{path}.codebook")
    if codes is None or vs is None or cb is None:
        raise ValueError(f"{path}: VQ needs .codes, .codebook and "
                         f".vq_scales together")
    shape = getattr(getattr(mod, "weight", None), "shape", ())
    if len(shape) != 3:
        raise ValueError(f"{path}: VQ tensors on a module that is not a "
                         f"switch (expert) linear")
    E, OUT, IN = (int(n) for n in shape)
    K, dim = (int(n) for n in cb.shape)
    ngrp = int(vs.shape[-1])
    if ngrp <= 0 or IN % ngrp or IN % dim:
        raise ValueError(f"{path}: in={IN} does not divide into "
                         f"{ngrp} scale groups of dim-{dim} codes")
    nsub = IN // dim
    bits = (K - 1).bit_length()
    width = int(codes.shape[-1])
    if width == (nsub + 31) // 32 * bits:
        pb, ct = bits, mx.uint32
    elif width == nsub:
        pb, ct = 0, (mx.uint8 if K <= 256 else mx.uint16)
    else:
        raise ValueError(f"{path}: codes rows are {width} wide; in={IN} at "
                         f"dim {dim} is {(nsub + 31) // 32 * bits} packed "
                         f"{bits}-bit words or {nsub} codes")
    return vq_cls(mx.zeros((E, OUT, width), dtype=ct),
                  mx.zeros((K, dim), dtype=mx.float16),
                  mx.zeros((E, OUT, ngrp), dtype=mx.float16),
                  group_size=IN // ngrp, pack_bits=pb,
                  in_features=IN if pb else None)


class MTPHead:
    """One drafting head bound to a loaded deepseek_v4 trunk."""

    def __init__(self, model, arch):
        core = model.model
        args = core.args
        self.model = model
        self.core = core
        self.arch = arch
        self.args = args
        D = args.hidden_size
        # A dense skeleton: the block class quantizes its experts to mxfp4
        # at construction when the trunk carries no quantization config;
        # the sidecar's own tensors say how it is packed (from_sidecar).
        skel = dataclasses.replace(
            args, quantization=args.quantization or {"skeleton": True})
        # Built by its GLOBAL index (the official MTP layer id), of the
        # layers' own class -- never a pipeline stage's Recv/Send wrapper.
        block = type(unwrap(core.layers[0]))(skel, args.num_hidden_layers)
        hc_head = arch.HyperHead(D, args.hc_mult, args.rms_norm_eps,
                                 args.hc_eps)
        self.m = _Head(block, hc_head, D, args.rms_norm_eps, arch.RMSNorm)

    def make_draft_cache(self):
        return self.arch.DeepseekV4Cache(self.args.sliding_window)

    # ------------------------------------------------------------- sidecar
    def bind(self, w: dict) -> dict:
        """Shape this head to sidecar tensors and check it matches them
        exactly: every name, every shape. `w` maps sidecar keys (with the
        `mtp.0.` prefix) to anything with a `.shape` -- arrays, or a
        header's shapes, so a file can be checked without loading it.
        Returns the same values under this module's parameter paths."""
        bad = [k for k in w if not k.startswith(PREFIX)]
        if bad:
            raise ValueError(f"deepseek_v4 head: {len(bad)} tensors outside "
                             f"{PREFIX!r}, e.g. {bad[:3]}")
        mw = {_to_module(k[len(PREFIX):]): v for k, v in w.items()}

        vq = {path: mod for path, mod in self.m.named_modules()
              if any(f"{path}.{t}" in mw
                     for t in ("codes", "codebook", "vq_scales"))}
        if vq:
            cls = _vq_class(self.model)
            vq = {p: _vq_module(mw, p, m, cls) for p, m in vq.items()}
            self.m.update_modules(tree_unflatten(list(vq.items())))

        def pred(path, mod):
            return _recipe(mw, path, mod) or False

        nn.quantize(self.m, class_predicate=pred)
        slots = dict(tree_flatten(self.m.parameters()))
        missing = sorted(set(slots) - set(mw))
        extra = sorted(set(mw) - set(slots))
        if missing or extra:
            raise ValueError(
                f"deepseek_v4 head does not match the sidecar: missing "
                f"{missing[:4]} ({len(missing)}), unexpected {extra[:4]} "
                f"({len(extra)})")
        wrong = [(k, tuple(v.shape), tuple(slots[k].shape))
                 for k, v in mw.items()
                 if tuple(v.shape) != tuple(slots[k].shape)]
        if wrong:
            raise ValueError(f"deepseek_v4 head: shape mismatch {wrong[:4]}")
        return mw

    def load_weights(self, w: dict) -> MTPHead:
        """Fill from sidecar tensors (keys with the `mtp.0.` prefix)."""
        mw = self.bind(w)
        self.m.load_weights(list(mw.items()), strict=True)
        mx.eval(self.m.parameters())
        return self

    def tensors(self) -> dict:
        """This head's parameters under their sidecar names."""
        return {PREFIX + _to_sidecar(k): v
                for k, v in tree_flatten(self.m.parameters())}

    def save(self, path) -> dict:
        flat = self.tensors()
        mx.save_safetensors(str(path), flat, metadata={"format": "mlx"})
        return flat

    @classmethod
    def from_sidecar(cls, model, arch, path):
        return cls(model, arch).load_weights(mx.load(str(path)))

    # --------------------------------------------------------------- draft
    def _trunk(self, h_row, nxt_id, cache=None):
        """(trunk streams at t [B, T, hc, D], token t+1) -> the head's
        normed output. T > 1 is real (two committed positions per
        speculative step, and the prompt seed); the block builds its own
        window mask, as the trunk's layers do."""
        m = self.m
        e = m.enorm(self.core.embed_tokens(nxt_id))
        # knurlogic edit 21: e_proj and h_proj are FP8 linears; their
        # inputs through act_quant, as the reference's MTPBlock
        act = self.arch.fp8_act
        x = (m.e_proj(act(e))[:, :, None, :]
             + m.h_proj(act(m.hnorm(h_row))))
        x = m.block(x.astype(h_row.dtype), cache, nxt_id)
        return m.norm(m.hc_head(x))

    def draft_logits(self, h_row, nxt_id, cache=None):
        """(trunk streams at t, token t+1) -> logits for token t+2, float32
        as the reference's head (trunk edit 24)."""
        return self.arch.head_logits(self.model.lm_head,
                                     self._trunk(h_row, nxt_id, cache))

    def advance(self, h_row, nxt_id, cache):
        """Fill the head's cache for these positions without the lm_head
        projection (the prompt seed never reads those logits)."""
        self._trunk(h_row, nxt_id, cache)
        return cache
