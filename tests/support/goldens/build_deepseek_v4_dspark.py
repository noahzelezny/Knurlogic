"""The DSpark goldens: a tiny DeepSeek-V4 with DSpark stages, as an HF
checkpoint in the official names and formats, and what DeepSeek's own
reference code (deepseek-ai/DeepSeek-V4-Flash-Vision-Exp `inference/`,
MIT) computes from it under torch.

    # 1. the tiny HF checkpoint, and from it the MLX artifact: the trunk
    #    through the vendored sanitize, the sidecar through dspark_pack
    PYTHONPATH=src ~/venv/bin/python \\
        tests/support/goldens/build_deepseek_v4_dspark.py weights
    # 2. the golden, from the HF checkpoint, under torch
    REF=".../deepseek-ai--DeepSeek-V4-Flash-Vision-Exp"
    $TORCH_PYTHON tests/support/goldens/build_deepseek_v4_dspark.py \\
        golden "$REF"

The HF checkpoint (deepseek_v4_dspark_hf/) carries what the real one does:
FP8 linears with 128x128 E8M0 block scales in the DSpark stages, FP4
routed experts with per-32 E8M0 scales everywhere, bf16 norms and Markov
head, fp32 hyper-connection and gate tensors. The trunk's layers are
compress ratios 0 / 4 / 128 / 4 (the ratio-4 ones with their indexer,
YaRN on the compressed layers as Flash's), the DSpark stages ratio 0 as
in the real model; head_dim 80 leaves 64 non-rope dims and the indexer
has 128 dims (the real model's), whole FP8 / FP4 blocks, so the
reference's low-precision simulation runs everywhere it does on the real
model. The indexer has 16 heads: with 4, its FP4-rounded scores tie at
the top-k boundary for some queries, where either choice is the
reference's and the two runs part ways. The prompt is one whose run has
no such tie, and no FP8 rounding that a float32 ulp of summation order
flips. The reference runs
float32 on the CPU with its CUDA kernels written out in torch (`kernels`:
`sparse_attn`, `hc_split_sinkhorn`, `act_quant` and `fp4_act_quant` with
inplace=True, and fast_hadamard_transform's `hadamard_transform`);
model.py's rotate_activation is the same call without its bf16 assert.

Writes deepseek_v4_dspark.npz: the trunk's logits (prefill, then each
decode step), the main hidden state, forward_spec's draft ids, logits
and confidence at each decode step (temperature 0), and `qat_*`: the
torch kernels' outputs on fixed inputs, which the MLX simulation must
reproduce bit for bit.
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
HD, RD, IH, IHD = 80, 16, 16, 128
RATIOS = [0, 4, 128, 4]
CONFIG = dict(
    model_type="deepseek_v4", vocab_size=V, hidden_size=D,
    num_hidden_layers=L, num_attention_heads=4, num_key_value_heads=1,
    q_lora_rank=32, o_lora_rank=16, o_groups=2, head_dim=HD,
    qk_rope_head_dim=RD, sliding_window=8, compress_ratios=RATIOS,
    compress_rope_theta=160000.0,
    rope_scaling={"type": "yarn", "factor": 4,
                  "original_max_position_embeddings": 64,
                  "beta_fast": 32, "beta_slow": 1},
    index_n_heads=IH, index_head_dim=IHD, index_topk=8,
    moe_intermediate_size=I, n_routed_experts=E, n_shared_experts=1,
    num_experts_per_tok=2, num_hash_layers=1, hc_mult=4,
    hc_sinkhorn_iters=3, max_position_embeddings=256,
    num_nextn_predict_layers=STAGES, dspark_block_size=K,
    dspark_noise_token_id=NOISE, dspark_target_layer_ids=TARGETS,
    dspark_markov_rank=R, tie_word_embeddings=False, eos_token_id=1,
    bos_token_id=0)

#: a 126-token prefill (past the 8-token window; 31 ratio-4 rows, of
#: which the indexer keeps 8) and 5 decode tokens (the first ratio-128 row
#: is pooled by the second)
PROMPT = [3, 17, 42, 5, 9, 60, 33, 2, 11, 48, 27] + [
    int(t) for t in np.random.default_rng(1).integers(2, 62, 115)]
DECODE = [7, 55, 21, 36, 4]


# ------------------------------------------------------- the HF checkpoint
def _tensors(rng) -> dict:
    """name -> (safetensors dtype, numpy array of its bytes / values)."""
    import mlx.core as mx
    H, hd, ql, ol, G = 4, HD, 32, 16, 2
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

    def compressor(pre, ratio, d):
        coff = 1 + (ratio == 4)
        f32(f"{pre}.ape", (ratio, coff * d))
        f32(f"{pre}.wkv.weight", (coff * d, D))
        f32(f"{pre}.wgate.weight", (coff * d, D))
        bf16(f"{pre}.norm.weight", (d,), one=True)

    def block(pre, lin, hash_layer, ratio=0):
        if ratio:
            compressor(f"{pre}.attn.compressor", ratio, hd)
        if ratio == 4:
            lin(f"{pre}.attn.indexer.wq_b.weight", (IH * IHD, ql))
            bf16(f"{pre}.attn.indexer.weights_proj.weight", (IH, D))
            compressor(f"{pre}.attn.indexer.compressor", ratio, IHD)
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
        block(f"layers.{i}", plain, i < 1, RATIOS[i])
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
def _pow2_ceil(t):
    """kernel.py's fast_round_scale on t = amax * max_inv (float32):
    fast_pow2(fast_log2_ceil(t)), the same bit manipulation."""
    import torch
    b = t.view(torch.int32)
    e = ((b >> 23) & 0xFF) - 127 + ((b & ((1 << 23) - 1)) != 0).int()
    return ((e + 127) << 23).view(torch.float32)


def _e2m1(v):
    """float32 in [-6, 6] -> e2m1, round to nearest, ties to the even
    mantissa (the CUDA conversion T.Cast(FP4, ...) lowers to)."""
    import torch
    a = v.abs()
    grid = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6])
    lo = (torch.searchsorted(grid, a.contiguous(), right=True) - 1).clamp(max=6)
    below, above = grid[lo], grid[(lo + 1).clamp(max=7)]
    up = (a - below > above - a) | ((a - below == above - a) & (lo % 2 == 1))
    q = torch.where(up & (a > below), above, below)
    return torch.where(v < 0, -q, q)


def kernels():
    """The reference's CUDA kernels (inference/kernel.py; fast_hadamard_
    transform for rotate_activation), in torch: each computes what its
    kernel does per element, in the kernel's float32."""
    import types

    import torch

    def act_quant(x, block_size=128, scale_fmt=None,
                  scale_dtype=torch.float32, inplace=False):
        # act_quant_kernel: amax per block floored at 1e-4; s = amax / 448
        # or (scale_fmt set: round_scale) its power-of-two ceiling;
        # clamp(x / s) -> e4m3 -> * s, back in x's dtype
        assert inplace, "the golden runs every GEMM dequantized"
        N = x.size(-1)
        assert N % block_size == 0
        xf = x.float().unflatten(-1, (N // block_size, block_size))
        amax = xf.abs().amax(-1, keepdim=True).clamp(min=1e-4)
        inv = torch.tensor(1 / 448.0, dtype=torch.float32)
        s = _pow2_ceil(amax * inv) if scale_fmt is not None else amax * inv
        y = (xf / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * s
        x.copy_(y.flatten(-2).to(x.dtype))
        return x

    def fp4_act_quant(x, block_size=32, inplace=False):
        # fp4_quant_kernel: amax floored at 6 * 2**-126, s its power-of-
        # two ceiling over 6; clamp(x / s, +-6) -> e2m1 -> * s
        assert inplace
        N = x.size(-1)
        assert N % block_size == 0
        xf = x.float().unflatten(-1, (N // block_size, block_size))
        amax = xf.abs().amax(-1, keepdim=True).clamp(min=6 * 2.0 ** -126)
        s = _pow2_ceil(amax * torch.tensor(1 / 6.0, dtype=torch.float32))
        y = _e2m1((xf / s).clamp(-6.0, 6.0)) * s
        x.copy_(y.flatten(-2).to(x.dtype))
        return x

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

    def hadamard_transform(x, scale=1.0):
        # the Sylvester transform as radix-2 butterflies, lowest stride
        # first, in float32, scaled at the end
        d = x.size(-1)
        y, h = x.float().reshape(-1, d), 1
        while h < d:
            y = y.view(-1, d // (2 * h), 2, h)
            y = torch.stack([y[:, :, 0] + y[:, :, 1],
                             y[:, :, 0] - y[:, :, 1]], 2).view(-1, d)
            h *= 2
        y = y * torch.tensor(scale, dtype=torch.float32)
        return y.view(x.shape).to(x.dtype)

    def rotate_activation(x):
        return hadamard_transform(x, scale=x.size(-1) ** -0.5)

    return types.SimpleNamespace(
        act_quant=act_quant, fp4_act_quant=fp4_act_quant, fp8_gemm=refused,
        fp4_gemm=refused, sparse_attn=sparse_attn,
        hc_split_sinkhorn=hc_split_sinkhorn,
        hadamard_transform=hadamard_transform,
        rotate_activation=rotate_activation)


def qat_cases() -> dict:
    """Inputs for the kernels' unit golden: random activations over a wide
    range of magnitudes, blocks that are all zero or below the amax
    floors, values on e4m3 / e2m1 rounding ties, and bf16-valued rows
    (the dtype the real model runs)."""
    rng = np.random.default_rng(11)
    x = (rng.standard_normal((64, 256)) * np.exp2(
        rng.integers(-20, 12, (64, 1)))).astype(np.float32)
    x[0] = 0
    x[1] = 1e-6 * rng.standard_normal(256)
    x[2, :64] = 3e-5
    x[3] = np.tile(np.array([.25, .75, 1.25, 1.75, 2.5, 3.5, 5, 6],
                            np.float32), 32) * (rng.integers(0, 2, 256) * 2 - 1)
    x[4] = np.tile(np.array([448, 1.0625, 1.1875, 0.0009765625, 240, 250,
                             -1.0625, 3], np.float32), 32)
    u = x.view(np.uint32)
    xb = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16 << 16).view(np.float32)
    return {"x": x, "xb": xb}


def qat_golden(cases: dict) -> dict:
    import torch
    k = kernels()
    out = {}
    for name, a in cases.items():
        dt = torch.bfloat16 if name == "xb" else torch.float32
        t = torch.from_numpy(a).to(dt)
        out[f"qat_{name}"] = a
        out[f"qat_{name}_fp8"] = k.act_quant(
            t.clone(), 64, "ue8m0", inplace=True).float().numpy()
        out[f"qat_{name}_fp4"] = k.fp4_act_quant(
            t.clone(), 32, inplace=True).float().numpy()
        for d in (64, 128):
            out[f"qat_{name}_rot{d}"] = k.rotate_activation(
                t.reshape(-1, d)).float().numpy().reshape(a.shape)
        out[f"qat_{name}_rotfp4"] = k.fp4_act_quant(k.rotate_activation(
            t.reshape(-1, 128)), 32, inplace=True).float().numpy().reshape(
                a.shape)
    return out


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
    k = kernels()
    sys.modules["kernel"] = k
    sys.modules["fast_hadamard_transform"] = k
    sys.modules["vision"] = types.SimpleNamespace(ViT=None, Aligner=None)
    sys.path.insert(0, str(inf))
    import model as M
    # the reference asserts bf16 here (its CUDA kernel's dtype); the
    # golden runs float32
    M.rotate_activation = k.rotate_activation

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)
    c = CONFIG
    args = M.ModelArgs(
        max_batch_size=1, max_seq_len=256, temperature=0, dtype="bf16",
        scale_fmt="ue8m0", expert_dtype=None, scale_dtype="fp32",
        vocab_size=V, dim=D, moe_inter_dim=I, n_layers=L,
        n_hash_layers=c["num_hash_layers"], n_mtp_layers=STAGES,
        n_heads=c["num_attention_heads"], n_routed_experts=E,
        n_shared_experts=1, n_activated_experts=c["num_experts_per_tok"],
        score_func="sqrtsoftplus", route_scale=1.5, swiglu_limit=10.0,
        q_lora_rank=c["q_lora_rank"], head_dim=c["head_dim"],
        rope_head_dim=c["qk_rope_head_dim"], norm_eps=1e-6,
        o_groups=c["o_groups"], o_lora_rank=c["o_lora_rank"],
        window_size=c["sliding_window"],
        compress_ratios=tuple(RATIOS + [0] * STAGES), rope_theta=10000.0,
        compress_rope_theta=160000.0, original_seq_len=64, rope_factor=4,
        beta_fast=32, beta_slow=1, index_n_heads=IH, index_head_dim=IHD,
        index_topk=c["index_topk"],
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
    arrays.update(qat_golden(qat_cases()))
    np.savez_compressed(OUT, **arrays)
    print({k: v.shape for k, v in arrays.items()})


if __name__ == "__main__":
    if sys.argv[1] == "weights":
        weights()
    else:
        golden(sys.argv[2])
