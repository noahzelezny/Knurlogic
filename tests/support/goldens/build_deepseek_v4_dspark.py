"""The DSpark goldens: a tiny DeepSeek-V4 with DSpark stages, as an HF
checkpoint in the official names and formats, and what DeepSeek's own
reference code (deepseek-ai/DeepSeek-V4-Flash-Vision-Exp `inference/`,
MIT) computes from it under torch.

    # 1. the tiny HF checkpoint, and from it the MLX artifact: the trunk
    #    through the vendored sanitize, the sidecar through dspark_pack
    PYTHONPATH=src ~/knurlogic-venv/bin/python \\
        tests/support/goldens/build_deepseek_v4_dspark.py weights
    # 2. the golden, from the HF checkpoint, under torch
    REF=".../deepseek-ai--DeepSeek-V4-Flash-Vision-Exp"
    $TORCH_PYTHON tests/support/goldens/build_deepseek_v4_dspark.py \\
        golden "$REF"

The HF checkpoint (deepseek_v4_dspark_hf/) carries what the real one does:
FP8 linears with 128x128 E8M0 block scales in the DSpark stages, FP4
routed experts with per-32 E8M0 scales everywhere, bf16 norms and Markov
head, fp32 hyper-connection and gate tensors. Every compress ratio is 0
(the reference's compressor needs CUDA kernels; DSpark's own stages are
ratio 0 anyway). The reference runs float32 on the CPU with its three
CUDA kernels written out in torch (`sparse_attn`, `hc_split_sinkhorn`,
`act_quant`, the last as the identity: the trunk does not simulate FP8
on the kv either).

Writes deepseek_v4_dspark.npz: the trunk's logits (prefill, then each
decode step), the main hidden state, and forward_spec's draft ids,
logits and confidence at each decode step (temperature 0).
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
TINY = HERE / "deepseek_v4_dspark_tiny"
HF = HERE / "deepseek_v4_dspark_hf"
OUT = HERE / "deepseek_v4_dspark.npz"

V, D, L, E, I = 64, 64, 4, 4, 32
K, NOISE, R, STAGES, TARGETS = 5, 63, 8, 3, [1, 2, 3]
CONFIG = dict(
    model_type="deepseek_v4", vocab_size=V, hidden_size=D,
    num_hidden_layers=L, num_attention_heads=4, num_key_value_heads=1,
    q_lora_rank=32, o_lora_rank=16, o_groups=2, head_dim=32,
    qk_rope_head_dim=16, sliding_window=8, compress_ratios=[0] * L,
    index_n_heads=8, index_head_dim=16, index_topk=2,
    moe_intermediate_size=I, n_routed_experts=E, n_shared_experts=1,
    num_experts_per_tok=2, num_hash_layers=1, hc_mult=4,
    hc_sinkhorn_iters=3, max_position_embeddings=256,
    num_nextn_predict_layers=STAGES, dspark_block_size=K,
    dspark_noise_token_id=NOISE, dspark_target_layer_ids=TARGETS,
    dspark_markov_rank=R, tie_word_embeddings=False, eos_token_id=1,
    bos_token_id=0)

#: an 11-token prefill (past the 8-token window) and 5 decode tokens
PROMPT = [3, 17, 42, 5, 9, 60, 33, 2, 11, 48, 27]
DECODE = [7, 55, 21, 36, 4]


# ------------------------------------------------------- the HF checkpoint
def _tensors(rng) -> dict:
    """name -> (safetensors dtype, numpy array of its bytes / values)."""
    import mlx.core as mx
    H, hd, ql, ol, G = 4, 32, 32, 16, 2
    mix = (2 + 4) * 4
    t: dict = {}

    def f32(name, shape, s=0.15):
        t[name] = ("F32", (s * rng.standard_normal(shape)).astype(np.float32))

    def bf16(name, shape, s=0.15, one=False):
        a = (1 + 0.1 * rng.standard_normal(shape) if one
             else s * rng.standard_normal(shape)).astype(np.float32)
        u = a.view(np.uint32)
        t[name] = ("BF16", ((u + 0x7FFF + ((u >> 16) & 1)) >> 16)
                   .astype(np.uint16))

    def fp8(name, shape):
        # e4m3 values around 1.2 times one 2^-3 scale per 128x128 block
        a = (1.2 * rng.standard_normal(shape)).astype(np.float32)
        t[name] = ("F8_E4M3", np.array(mx.to_fp8(mx.array(a))))
        sb = (-(-shape[0] // 128), -(-shape[1] // 128))
        t[name[:-len("weight")] + "scale"] = (
            "U8", np.full(sb, 124, dtype=np.uint8))

    def experts(pre):
        for e in range(E):
            for w, (o, i) in (("w1", (I, D)), ("w2", (D, I)), ("w3", (I, D))):
                k = f"{pre}.ffn.experts.{e}.{w}"
                t[k + ".weight"] = ("I8", rng.integers(
                    0, 256, size=(o, i // 2), dtype=np.uint8))
                t[k + ".scale"] = ("U8", rng.integers(
                    118, 124, size=(o, i // 32)).astype(np.uint8))

    def block(pre, lin, hash_layer):
        lin(f"{pre}.attn.wq_a.weight", (ql, D))
        lin(f"{pre}.attn.wkv.weight", (hd, D))
        lin(f"{pre}.attn.wq_b.weight", (H * hd, ql))
        lin(f"{pre}.attn.wo_a.weight", (G * ol, H * hd // G))
        lin(f"{pre}.attn.wo_b.weight", (D, G * ol))
        f32(f"{pre}.attn.attn_sink", (H,))
        bf16(f"{pre}.attn.q_norm.weight", (ql,), one=True)
        bf16(f"{pre}.attn.kv_norm.weight", (hd,), one=True)
        bf16(f"{pre}.attn_norm.weight", (D,), one=True)
        bf16(f"{pre}.ffn_norm.weight", (D,), one=True)
        bf16(f"{pre}.ffn.gate.weight", (E, D))
        if hash_layer:
            # two distinct experts per token, as a real table has (the
            # reference's `y[idx] += ...` would drop a repeated one)
            t[f"{pre}.ffn.gate.tid2eid"] = ("I32", np.stack(
                [rng.permutation(E)[:2] for _ in range(V)]).astype(np.int32))
        else:
            f32(f"{pre}.ffn.gate.bias", (E,), 0.05)
        for w, shape in (("w1", (I, D)), ("w2", (D, I)), ("w3", (I, D))):
            lin(f"{pre}.ffn.shared_experts.{w}.weight", shape)
        experts(pre)
        for hc in ("attn", "ffn"):
            f32(f"{pre}.hc_{hc}_fn", (mix, 4 * D))
            f32(f"{pre}.hc_{hc}_base", (mix,))
            f32(f"{pre}.hc_{hc}_scale", (3,), 0.5)

    def plain(name, shape):
        f32(name, shape)

    f32("embed.weight", (V, D), 1.0)
    for i in range(L):
        block(f"layers.{i}", plain, i < 1)
    bf16("norm.weight", (D,), one=True)
    f32("head.weight", (V, D))
    f32("hc_head_fn", (4, 4 * D))
    f32("hc_head_base", (4,))
    f32("hc_head_scale", (1,), 0.5)
    for s in range(STAGES):
        pre = f"mtp.{s}"
        block(pre, fp8, False)
        if s == 0:
            fp8(f"{pre}.main_proj.weight", (D, len(TARGETS) * D))
            bf16(f"{pre}.main_norm.weight", (D,), one=True)
        if s == STAGES - 1:
            bf16(f"{pre}.norm.weight", (D,), one=True)
            f32(f"{pre}.hc_head_fn", (4, 4 * D))
            f32(f"{pre}.hc_head_base", (4,))
            f32(f"{pre}.hc_head_scale", (1,), 0.5)
            bf16(f"{pre}.markov_head.markov_w1.weight", (V, R), 1.0)
            bf16(f"{pre}.markov_head.markov_w2.weight", (V, R))
            bf16(f"{pre}.confidence_head.proj.weight", (1, D + R))
    return t


def _save(path: Path, t: dict) -> None:
    hdr, off, blobs = {}, 0, []
    for k in sorted(t):
        dt, a = t[k]
        b = np.ascontiguousarray(a).tobytes()
        hdr[k] = {"dtype": dt, "shape": list(a.shape),
                  "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    blob = json.dumps(hdr).encode()
    blob += b" " * (-len(blob) % 8)
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(blob)) + blob + b"".join(blobs))


def weights() -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten
    sys.path.insert(0, str(HERE.parents[2] / "src"))
    from knurlogic.engine import register
    from knurlogic.engine.families.deepseek.heads import dspark_pack
    register.register("deepseek_v4")
    import mlx_lm.models.deepseek_v4 as M

    t = _tensors(np.random.default_rng(0))
    HF.mkdir(parents=True, exist_ok=True)
    _save(HF / "model.safetensors", t)
    (HF / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {k: "model.safetensors" for k in sorted(t)}},
        indent=1))

    TINY.mkdir(parents=True, exist_ok=True)
    for f in TINY.glob("*.safetensors"):
        f.unlink()
    hf = mx.load(str(HF / "model.safetensors"))
    trunk = {k: v for k, v in hf.items() if not k.startswith("mtp.")}
    model = M.Model(M.ModelArgs.from_dict(CONFIG))
    w = model.sanitize(trunk)
    model.load_weights(list(w.items()), strict=True)
    mx.save_safetensors(str(TINY / "model.safetensors"),
                        dict(tree_flatten(model.parameters())),
                        metadata={"format": "mlx"})
    (TINY / "config.json").write_text(json.dumps(CONFIG, indent=1))
    dspark_pack.pack(HF, TINY)


# ------------------------------------------------------- the torch golden
def _kernels():
    """The reference's CUDA kernels (inference/kernel.py), in torch."""
    import types

    import torch

    def act_quant(x, *a, **k):
        return x        # the trunk does not simulate FP8 either

    def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
        b, m, h, d = q.shape
        idx = topk_idxs.long()
        valid = idx >= 0
        g = kv[torch.arange(b)[:, None, None], idx.clamp(min=0)]  # b m k d
        s = torch.einsum("bmhd,bmkd->bmhk", q.float(), g.float())
        s = s * softmax_scale
        s = s.masked_fill(~valid[:, :, None, :], float("-inf"))
        mx_ = torch.maximum(s.amax(-1), attn_sink.float()[None, None])
        p = torch.exp(s - mx_[..., None])
        den = p.sum(-1) + torch.exp(attn_sink.float()[None, None] - mx_)
        o = torch.einsum("bmhk,bmkd->bmhd", p, g.float()) / den[..., None]
        return o.to(q.dtype)

    def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4,
                          sinkhorn_iters=20, eps=1e-6):
        hc = hc_mult
        pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
        post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1]
                                 + hc_base[hc:2 * hc])
        comb = (mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]
                ).unflatten(-1, (hc, hc))
        comb = comb.softmax(-1) + eps
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
        for _ in range(sinkhorn_iters - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + eps)
            comb = comb / (comb.sum(-2, keepdim=True) + eps)
        return pre, post, comb

    def refused(*a, **k):
        raise RuntimeError("a quantized gemm: the golden runs dequantized")

    return types.SimpleNamespace(
        act_quant=act_quant, fp4_act_quant=refused, fp8_gemm=refused,
        fp4_gemm=refused, sparse_attn=sparse_attn,
        hc_split_sinkhorn=hc_split_sinkhorn)


_FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6],
                dtype=np.float32)


def _dequant(st: dict) -> dict:
    """The checkpoint as float32 torch tensors in the reference's names:
    FP8 times its block scales, FP4 times its per-32 scales."""
    import torch
    out = {}
    for k, v in st.items():
        if k.endswith(".scale"):
            continue
        sk = k[:-len("weight")] + "scale"
        if v.dtype == torch.float8_e4m3fn:
            s = torch.exp2(st[sk].float() - 127)
            s = s.repeat_interleave(128, 0).repeat_interleave(128, 1)
            out[k] = v.float() * s[:v.shape[0], :v.shape[1]]
        elif v.dtype == torch.int8 and ".experts." in k:
            b = v.view(torch.uint8).numpy()
            vals = np.stack([_FP4[b & 0xF], _FP4[b >> 4]], -1).reshape(
                b.shape[0], -1)
            s = np.exp2(st[sk].numpy().astype(np.float32) - 127)
            out[k] = torch.from_numpy(vals * np.repeat(s, 32, axis=1))
        elif v.dtype in (torch.int32, torch.int64):
            out[k] = v
        else:
            out[k] = v.float()
    return out


def golden(ref: str) -> None:
    import types

    import torch
    from safetensors.torch import load_file

    inf = Path(ref) / "inference"
    sys.dont_write_bytecode = True      # nothing is written beside it
    sys.modules["kernel"] = _kernels()
    sys.modules["vision"] = types.SimpleNamespace(ViT=None, Aligner=None)
    sys.path.insert(0, str(inf))
    import model as M

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)
    c = CONFIG
    args = M.ModelArgs(
        max_batch_size=1, max_seq_len=64, temperature=0, dtype="bf16",
        scale_fmt=None, expert_dtype=None, scale_dtype="fp32",
        vocab_size=V, dim=D, moe_inter_dim=I, n_layers=L,
        n_hash_layers=c["num_hash_layers"], n_mtp_layers=STAGES,
        n_heads=c["num_attention_heads"], n_routed_experts=E,
        n_shared_experts=1, n_activated_experts=c["num_experts_per_tok"],
        score_func="sqrtsoftplus", route_scale=1.5, swiglu_limit=10.0,
        q_lora_rank=c["q_lora_rank"], head_dim=c["head_dim"],
        rope_head_dim=c["qk_rope_head_dim"], norm_eps=1e-6,
        o_groups=c["o_groups"], o_lora_rank=c["o_lora_rank"],
        window_size=c["sliding_window"],
        compress_ratios=tuple([0] * (L + STAGES)), rope_theta=10000.0,
        hc_mult=4, hc_sinkhorn_iters=c["hc_sinkhorn_iters"], hc_eps=1e-6,
        dspark_block_size=K, dspark_noise_token_id=NOISE,
        dspark_target_layer_ids=tuple(TARGETS), dspark_markov_rank=R)
    model = M.Transformer(args).float()
    w = _dequant(load_file(str(HF / "model.safetensors")))
    for s in range(STAGES):        # the shared table and head, as tied
        w[f"mtp.{s}.embed.weight"] = w["embed.weight"]
        w[f"mtp.{s}.head.weight"] = w["head.weight"]
    model.load_state_dict(w, strict=True)

    x = torch.tensor([PROMPT + DECODE])
    P = len(PROMPT)
    out: dict = {k: [] for k in ("logits", "main_hidden", "draft_ids",
                                 "draft_logits", "confidence")}
    ids, logits, mh = model(x[:, :P], 0)
    model.forward_spec(ids, mh, 0)
    out["logits"].append(logits[0])
    out["prefill_main_hidden"] = mh[0]
    for i in range(P, P + len(DECODE)):
        ids, logits, mh = model(x[:, i:i + 1], i)
        dids, dlog, conf = model.forward_spec(ids, mh, i)
        out["logits"].append(logits[0])
        out["main_hidden"].append(mh[0, -1])
        out["draft_ids"].append(dids[0])
        out["draft_logits"].append(dlog[0])
        out["confidence"].append(conf[0])
    arrays = {k: (torch.stack(v) if isinstance(v, list) else v)
              .float().numpy() if k != "draft_ids"
              else torch.stack(v).numpy().astype(np.int32)
              for k, v in out.items()}
    arrays["torch"] = np.array(torch.__version__)
    np.savez_compressed(OUT, **arrays)
    print({k: v.shape for k, v in arrays.items()})


if __name__ == "__main__":
    if sys.argv[1] == "weights":
        weights()
    else:
        golden(sys.argv[2])
