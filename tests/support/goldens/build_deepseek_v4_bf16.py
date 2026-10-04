"""The bf16 goldens: what DeepSeek's reference code (deepseek-ai/
DeepSeek-V4-Flash-Vision-Exp `inference/model.py` and `kernel.py`, MIT;
Flash's model.py is the same code for every piece here) computes in
bfloat16 -- the dtype it serves in -- for the trunk's pieces whose
arithmetic bf16 shows and float32 hides:

    norm_*      RMSNorm (attn_norm / ffn_norm / q_norm / kv_norm / the
                compressors' norm / the final norm / the heads' norms) on
                bf16 rows of 4096, 1024, 512 and 128
    qnorm       the per-head q norm, `q *= rsqrt(q.square().mean(-1) +
                eps)`, bf16 arithmetic (`qnorm_y_gpu`: with an accurate
                rsqrt, as the reference's GPU computes it)
    gate_*      Gate.forward (float32 scores) at Flash's shapes (256
                experts of 4096, top 6, sqrtsoftplus, route scale 1.5): a
                score layer, a score layer with bias_vl, a hash layer with
                bias_vl
    head_*      ParallelHead's F.linear(x.float(), weight) on 1, 6 and 20
                bf16 rows
    comp_*      Compressor.forward on a bf16 prefill, then one token at a
                time: ratio 4 (overlap, 512 dims), ratio 128 (512 dims)
                and the indexer's (ratio 4, 128 dims, rotate)
    e2e_*       the DSpark golden's run (build_deepseek_v4_dspark: a
                tiny Vision-Exp with FP8 linears, FP4 experts, compressed
                layers and three DSpark stages; 126-token prefill, 5
                decode steps with forward_spec) on its checkpoint with
                every tensor the real one stores bf16 stored bf16
                (hf_tensors), the reference in bf16
    hc_*        hc_split_sinkhorn (kernel.py) and hc_pre's collapse on
                fixed float32 mixes and bf16 streams (`hc_big_*`: eps 0.25,
                3 iterations; `*_f64`: the port in float64)

Each piece runs on the CPU (the golden) and again on the GPU (torch
MPS; the FP8 / FP4 casts MPS lacks hop to the CPU): `spread_<name>` is
the reference's own CPU-vs-GPU spread on each output (elements that
differ, the largest difference, the largest in bf16 ulps), which the
tests' tolerances are read against.

    # the end-to-end checkpoint (FP8 values need mlx): anywhere, it is
    # not kept -- the test writes the same one and checks its sha256
    PYTHONPATH=src ~/venv/bin/python \\
        tests/support/goldens/build_deepseek_v4_bf16.py weights "$HF"
    REF=".../deepseek-ai--DeepSeek-V4-Flash-Vision-Exp"
    $TORCH_PYTHON tests/support/goldens/build_deepseek_v4_bf16.py \\
        golden "$REF" "$HF"

The kernels kernel.py writes for CUDA come from
build_deepseek_v4_dspark.kernels (torch ports, each in the kernel's float32).
The inputs are not stored: `inputs()` makes them (numpy only, so the
tests make the same ones; `inputs_sha` checks they did). bf16 outputs are
stored as their float32 values (exact).
"""
import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "deepseek_v4_bf16.npz"
#: the device the reference runs on (its CPU-hopped GEMMs return there)
_DEV = ["cpu"]

D = 4096
EPS = 1e-6
#: Flash's gate
N_EXP, TOPK, ROUTE_SCALE, VOCAB = 256, 6, 1.5, 1000
#: the compressors' cases: (name, ratio, head_dim, rotate, prefill, decode)
COMP = [("comp4", 4, 512, False, 37, 11),
        ("comp128", 128, 512, False, 200, 60),
        ("compidx", 4, 128, True, 37, 11)]
RD = 64
#: the compressed layers' rope: compress_rope_theta, YaRN factor 4 past 64
ROPE = dict(base=160000.0, factor=4, original_seq_len=64, beta_fast=32,
            beta_slow=1)
MAX_SEQ = 512
HEAD_ROWS = (1, 6, 20)
HEAD_V = 1024
SINK_ROWS, SINK_D, SINK_ITERS = 256, 512, 20
BIG_EPS, BIG_EPS_ITERS = 0.25, 3


def _bf16(a):
    """float32 -> the nearest bf16 values (ties to even), as float32."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    u = a.view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16 << 16).astype(
        np.uint32).view(np.float32)


def _hidden(rng, shape):
    """bf16 rows shaped like a normed hidden state: unit-ish per element,
    a per-row scale over a few binades."""
    s = np.exp2(rng.integers(-3, 4, shape[:-1] + (1,))).astype(np.float32)
    return _bf16(rng.standard_normal(shape).astype(np.float32) * s
                 * (1 + 0.3 * rng.standard_normal(shape[-1])).astype(
                     np.float32))


def _reference(ref: str):
    """model.py with kernel.py's CUDA kernels as torch ports; the FP8 /
    FP4 casts torch has no MPS kernel for run on the CPU."""
    import torch
    import build_deepseek_v4_dspark as G
    k = G.kernels()

    def hop(fn):
        # in place: on a CPU copy, written back; not in place (linear()'s
        # FP8 / FP4 GEMM input): left on the CPU, where the GEMM runs
        def run(x, *a, **kw):
            if x.device.type == "cpu":
                return fn(x, *a, **kw)
            with torch.device("cpu"):
                c = x.detach().cpu()
                r = fn(c, *a, **kw)
            if kw.get("inplace") or (a and a[-1] is True):
                x.copy_(c.to(x.device))
                return x
            return r
        return run

    def gemm_hop(fn):
        # the FP8 / FP4 weights stay on the CPU (MPS has no float8): the
        # GEMM runs there on the dequantized operands and its output goes
        # to the run's device
        def run(a, a_s, b, b_s, *rest, **kw):
            with torch.device("cpu"):
                y = fn(a.cpu(), a_s.cpu(), b, b_s, *rest, **kw)
            return y.to(_DEV[0])
        return run

    k.act_quant = hop(k.act_quant)
    k.fp4_act_quant = hop(k.fp4_act_quant)
    k.fp8_gemm = gemm_hop(k.fp8_gemm)
    k.fp4_gemm = gemm_hop(k.fp4_gemm)
    inf = Path(ref) / "inference"
    sys.dont_write_bytecode = True
    sys.modules["kernel"] = k
    sys.modules["fast_hadamard_transform"] = k
    sys.modules["vision"] = types.SimpleNamespace(ViT=None, Aligner=None)
    sys.path.insert(0, str(inf))
    import model as M
    M.rotate_activation = k.rotate_activation
    # what Transformer.__init__ sets for a config with scale_fmt ue8m0
    M.scale_fmt = "ue8m0"
    M.scale_dtype = torch.float32
    return M, k


def _t(a, dev, dt=None):
    import torch
    t = torch.from_numpy(np.ascontiguousarray(a))
    return (t.to(dt) if dt is not None else t).to(dev)


def _np(t):
    import torch
    if t.dtype in (torch.int32, torch.int64):
        return t.cpu().numpy()
    return t.detach().float().cpu().numpy()


def norms(M, inp, dev):
    import torch
    out = {}
    for d in (4096, 1024, 512, 128):
        m = M.RMSNorm(d, EPS).to(dev)
        m.weight.data = _t(inp[f"norm{d}_w"], dev)
        out[f"norm{d}_y"] = _np(m(_t(inp[f"norm{d}_x"], dev, torch.bfloat16)))
    q = _t(inp["qnorm_x"], dev, torch.bfloat16)
    q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + EPS)
    out["qnorm_y"] = _np(q)
    if dev == "cpu":
        # torch's CPU rsqrt on bf16 is an estimate on arm64 (rsqrt(0.0659)
        # = 3.906 where the bf16 nearest the true 3.895 is 3.891); CUDA's
        # rsqrtf, MPS's and MLX's are accurate. The same ops with the
        # rsqrt rounded from float64 -- what the reference computes on
        # its GPU (the MPS run is this bit for bit, `qnorm_gpu_check`)
        q = _t(inp["qnorm_x"], dev, torch.bfloat16)
        e = q.square().mean(-1, keepdim=True) + EPS
        out["qnorm_y_gpu"] = _np(q * (1 / e.double().sqrt()).bfloat16())
    return out


def norm_inputs(rng):
    inp = {}
    for d in (4096, 1024, 512, 128):
        inp[f"norm{d}_x"] = _hidden(rng, (32, d))
        inp[f"norm{d}_w"] = _bf16(1 + 0.3 * rng.standard_normal(d))
    inp["qnorm_x"] = _hidden(rng, (8, 64, 512))
    return inp


def gate_inputs(rng):
    inp = {"gate_x": _hidden(rng, (256, D)),
           "gate_w": _bf16(0.02 * rng.standard_normal((N_EXP, D))),
           "gate_bias": (0.02 * rng.standard_normal(N_EXP)).astype(
               np.float32),
           "gate_bias_vl": (0.02 * rng.standard_normal(N_EXP)).astype(
               np.float32),
           "gate_tid2eid": np.stack([rng.permutation(N_EXP)[:TOPK]
                                     for _ in range(VOCAB)]).astype(np.int32)}
    ids = rng.integers(0, VOCAB, 256)
    img = rng.random(256) < 0.3
    inp["gate_ids"] = np.where(img, VOCAB + rng.integers(0, 5, 256),
                               ids).astype(np.int64)
    return inp


def gates(M, inp, dev):
    import torch
    out = {}
    for name, layer, vl in (("gate", 1, 0), ("gate_vl", 1, 1),
                            ("gate_hash", 0, 1)):
        args = M.ModelArgs(dim=D, n_routed_experts=N_EXP,
                           n_activated_experts=TOPK,
                           score_func="sqrtsoftplus", route_scale=ROUTE_SCALE,
                           n_hash_layers=1, vocab_size=VOCAB,
                           vision_n_layers=vl)
        g = M.Gate(layer, args).to(dev)
        g.weight.data = _t(inp["gate_w"], dev, torch.bfloat16)
        if g.bias is not None:
            g.bias.data = _t(inp["gate_bias"], dev)
        if g.bias_vl is not None:
            g.bias_vl.data = _t(inp["gate_bias_vl"], dev)
        if g.hash:
            g.tid2eid.data = _t(inp["gate_tid2eid"], dev)
        ids = inp["gate_ids"] if vl else inp["gate_ids"] % VOCAB
        w, i = g(_t(inp["gate_x"], dev, torch.bfloat16), _t(ids, dev))
        out[f"{name}_weights"] = _np(w)
        out[f"{name}_inds"] = _np(i).astype(np.int32)
    return out


def head_inputs(rng):
    inp = {"head_w": _bf16(0.02 * rng.standard_normal((HEAD_V, D)))}
    for r in HEAD_ROWS:
        inp[f"head_x{r}"] = _hidden(rng, (r, D))
    return inp


def heads(M, inp, dev):
    import torch
    out = {}
    h = M.ParallelHead(HEAD_V, D).to(dev)
    h.weight.data = _t(inp["head_w"], dev)
    for r in HEAD_ROWS:
        x = _t(inp[f"head_x{r}"], dev, torch.bfloat16)[None]
        out[f"head_y{r}"] = _np(h(x, full_logits=True)[0])
    return out


def comp_inputs(rng):
    inp = {}
    for name, ratio, d, _rot, P, T in COMP:
        coff = 1 + (ratio == 4)
        inp[f"{name}_x"] = _hidden(rng, (1, P + T, D))
        inp[f"{name}_wkv"] = _bf16(0.02 * rng.standard_normal((coff * d, D)))
        inp[f"{name}_wgate"] = _bf16(
            0.02 * rng.standard_normal((coff * d, D)))
        inp[f"{name}_ape"] = (0.5 * rng.standard_normal(
            (ratio, coff * d))).astype(np.float32)
        inp[f"{name}_norm"] = _bf16(1 + 0.3 * rng.standard_normal(d))
    return inp


def comps(M, inp, dev):
    import torch
    out = {}
    for name, ratio, d, rot, P, T in COMP:
        args = M.ModelArgs(dim=D, rope_head_dim=RD, norm_eps=EPS,
                           max_batch_size=1, max_seq_len=MAX_SEQ)
        c = M.Compressor(args, ratio, d, rot)
        c.wkv.weight.data = _t(inp[f"{name}_wkv"], "cpu")
        c.wgate.weight.data = _t(inp[f"{name}_wgate"], "cpu")
        c.ape.data = _t(inp[f"{name}_ape"], "cpu")
        c.norm.weight.data = _t(inp[f"{name}_norm"], "cpu")
        c = c.to(dev)
        c.kv_cache = torch.zeros(1, MAX_SEQ // ratio, d,
                                 dtype=torch.bfloat16, device=dev)
        c.freqs_cis = M.precompute_freqs_cis(
            RD, MAX_SEQ, ROPE["original_seq_len"], ROPE["base"],
            ROPE["factor"], ROPE["beta_fast"], ROPE["beta_slow"]).to(dev)
        x = _t(inp[f"{name}_x"], dev, torch.bfloat16)
        c(x[:, :P], 0)
        for i in range(P, P + T):
            c(x[:, i:i + 1], i)
        out[f"{name}_rows"] = _np(c.kv_cache[0, :(P + T) // ratio])
    return out


def hc_inputs(rng):
    n, mix = SINK_ROWS, (2 + 4) * 4
    return {"hc_mixes": (rng.standard_normal((n, mix))
                         * np.exp2(rng.integers(-2, 3, (n, 1)))).astype(
                             np.float32),
            "hc_scale": np.array([0.7, 0.9, 2.5], np.float32),
            "hc_base": (0.5 * rng.standard_normal(mix)).astype(np.float32),
            "hc_x": _hidden(rng, (n, 4, SINK_D))}


def hcs(M, k, inp, dev):
    import torch
    mixes = _t(inp["hc_mixes"], dev)[None]
    pre, post, comb = k.hc_split_sinkhorn(
        mixes, _t(inp["hc_scale"], dev), _t(inp["hc_base"], dev), 4,
        SINK_ITERS, EPS)
    x = _t(inp["hc_x"], dev, torch.bfloat16)[None]
    # Block.hc_pre's collapse: float32 streams, summed, back in bf16
    y = torch.sum(pre.unsqueeze(-1) * x.float(), dim=2).to(torch.bfloat16)
    out = {"hc_pre": _np(pre[0]), "hc_post": _np(post[0]),
           "hc_comb": _np(comb[0]), "hc_y": _np(y[0])}
    # an eps large enough to see where it enters (the real one's effect
    # after 20 iterations is ~1e-11 of comb, under float32's noise)
    _, post, comb = k.hc_split_sinkhorn(
        mixes, _t(inp["hc_scale"], dev), _t(inp["hc_base"], dev), 4,
        BIG_EPS_ITERS, BIG_EPS)
    out.update(hc_big_post=_np(post[0]), hc_big_comb=_np(comb[0]))
    if dev == "cpu":
        # the same port in float64: how far float32 itself is from exact
        _, post, comb = k.hc_split_sinkhorn(
            mixes.double(), _t(inp["hc_scale"], dev).double(),
            _t(inp["hc_base"], dev).double(), 4, SINK_ITERS, EPS)
        out.update(hc_post_f64=post[0].numpy(), hc_comb_f64=comb[0].numpy())
    return out


# ------------------------------------------------------------- end to end
#: what the real checkpoint stores bf16 that the DSpark golden's tiny one
#: stores float32: the embedding, the head, the compressors' wkv / wgate
_BF16_IN_REAL = ("embed.weight", "head.weight", ".compressor.wkv.weight",
                 ".compressor.wgate.weight")


def hf_tensors() -> dict:
    """The DSpark golden's tiny checkpoint (build_deepseek_v4_dspark:
    FP8 linears, FP4 experts, ratios 0 / 4 / 128 / 4, three DSpark stages)
    with every tensor the real checkpoint stores bf16 stored bf16 (needs
    mlx, for the FP8 values)."""
    import build_deepseek_v4_dspark as G
    t = G._tensors(np.random.default_rng(0))
    for name in list(t):
        if name.endswith(_BF16_IN_REAL):
            dt, a = t[name]
            assert dt == "F32", name
            u = np.ascontiguousarray(a, np.float32).view(np.uint32)
            t[name] = ("BF16", ((u + 0x7FFF + ((u >> 16) & 1)) >> 16)
                       .astype(np.uint16))
    return t


def write_hf(path: Path) -> str:
    """The checkpoint in `path` (model.safetensors and its index) -> its
    sha256."""
    import hashlib
    import json

    import build_deepseek_v4_dspark as G
    path.mkdir(parents=True, exist_ok=True)
    t = hf_tensors()
    G._save(path / "model.safetensors", t)
    (path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {k: "model.safetensors" for k in sorted(t)}},
        indent=1))
    return hashlib.sha256((path / "model.safetensors").read_bytes()
                          ).hexdigest()


def _to(model, dev):
    """Every parameter and buffer to `dev` but the FP8 / FP4 weights and
    their E8M0 scales (MPS has no float8: their GEMMs run on the CPU)."""
    import torch
    low = (torch.float8_e4m3fn, torch.float4_e2m1fn_x2, torch.float8_e8m0fnu)
    for mod in model.modules():
        for n, p in list(mod._parameters.items()):
            if p is not None and p.dtype not in low:
                mod._parameters[n] = torch.nn.Parameter(
                    p.data.to(dev), requires_grad=False)
        for n, b in list(mod._buffers.items()):
            if b is not None and b.dtype not in low:
                mod._buffers[n] = b.to(dev)
    return model


def e2e(M, hf: Path, dev: str) -> dict:
    """The DSpark golden's run (build_deepseek_v4_dspark.golden: a
    126-token prefill, then 5 decode steps with forward_spec at
    temperature 0) with the reference in bf16, its default dtype, on
    `dev`."""
    import torch
    from safetensors.torch import load_file

    import build_deepseek_v4_dspark as G
    for f in (M.get_window_topk_idxs, M.get_compress_topk_idxs,
              M.get_dspark_topk_idxs, M.precompute_freqs_cis):
        f.cache_clear()
    c = G.CONFIG
    args = M.ModelArgs(
        max_batch_size=1, max_seq_len=256, temperature=0, dtype="fp8",
        scale_fmt="ue8m0", expert_dtype="fp4", scale_dtype="fp32",
        vocab_size=G.V, dim=G.D, moe_inter_dim=G.I, n_layers=G.L,
        n_hash_layers=c["num_hash_layers"], n_mtp_layers=G.STAGES,
        n_heads=c["num_attention_heads"], n_routed_experts=G.E,
        n_shared_experts=1, n_activated_experts=c["num_experts_per_tok"],
        score_func="sqrtsoftplus", route_scale=1.5, swiglu_limit=10.0,
        q_lora_rank=c["q_lora_rank"], head_dim=c["head_dim"],
        rope_head_dim=c["qk_rope_head_dim"], norm_eps=1e-6,
        o_groups=c["o_groups"], o_lora_rank=c["o_lora_rank"],
        window_size=c["sliding_window"],
        compress_ratios=tuple(G.RATIOS + [0] * G.STAGES), rope_theta=10000.0,
        compress_rope_theta=160000.0, original_seq_len=64, rope_factor=4,
        beta_fast=32, beta_slow=1, index_n_heads=G.IH, index_head_dim=G.IHD,
        index_topk=c["index_topk"],
        hc_mult=4, hc_sinkhorn_iters=c["hc_sinkhorn_iters"], hc_eps=1e-6,
        dspark_block_size=G.K, dspark_noise_token_id=G.NOISE,
        dspark_target_layer_ids=tuple(G.TARGETS), dspark_markov_rank=G.R)
    with torch.device("cpu"):
        model = M.Transformer(args)
    st = load_file(str(hf / "model.safetensors"))
    deq = G._dequant(st)
    for s in range(G.STAGES):
        deq[f"mtp.{s}.embed.weight"] = deq["embed.weight"]
        deq[f"mtp.{s}.head.weight"] = deq["head.weight"]
    low = (torch.float8_e4m3fn, torch.float4_e2m1fn_x2, torch.float8_e8m0fnu)
    # each tensor in the dtype the reference declares it (bf16 values
    # into its float32 parameters exactly)
    w = {k: st[k].view(p.dtype) if p.dtype in low else deq[k].to(p.dtype)
         for k, p in model.state_dict().items()}
    model.load_state_dict(w, strict=True)
    _to(model, dev)
    _DEV[0] = dev
    x = torch.tensor([G.PROMPT + G.DECODE], device=dev)
    P = len(G.PROMPT)
    out: dict = {k: [] for k in ("logits", "main_hidden", "draft_ids",
                                 "draft_logits", "confidence")}
    with torch.device(dev), torch.inference_mode():
        ids, logits, mh = model(x[:, :P], 0)
        model.forward_spec(ids, mh, 0)
        out["logits"].append(logits[0])
        out["prefill_main_hidden"] = mh[0]
        for i in range(P, P + len(G.DECODE)):
            ids, logits, mh = model(x[:, i:i + 1], i)
            dids, dlog, conf = model.forward_spec(ids, mh, i)
            out["logits"].append(logits[0])
            out["main_hidden"].append(mh[0, -1])
            out["draft_ids"].append(dids[0])
            out["draft_logits"].append(dlog[0])
            out["confidence"].append(conf[0])
    _DEV[0] = "cpu"
    return {"e2e_" + k: (_np(torch.stack(v)) if isinstance(v, list)
                         else _np(v)) for k, v in out.items()}


def inputs() -> dict:
    """Every piece's inputs (numpy only: the tests make the same)."""
    rng = np.random.default_rng(23)
    inp = {}
    for f in (norm_inputs, gate_inputs, head_inputs, comp_inputs,
              hc_inputs):
        inp.update(f(rng))
    return inp


def inputs_sha(inp: dict) -> str:
    import hashlib
    h = hashlib.sha256()
    for k in sorted(inp):
        h.update(k.encode())
        h.update(np.ascontiguousarray(inp[k]).tobytes())
    return h.hexdigest()


def ulps(a, b):
    """|a - b| in bf16 ulps of b (a, b: bf16 values as float32)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    e = np.floor(np.log2(np.maximum(np.abs(b), 2.0 ** -126)))
    return np.abs(a - b) / np.exp2(e - 7)


def spread(a, b) -> np.ndarray:
    """[elements that differ, max |a - b|, max bf16 ulps, max |a - b| /
    |b|, mean |a - b|]; for indices, [rows whose sets differ, 0, 0, 0,
    0]."""
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind == "i":
        diff = (np.sort(a, -1) != np.sort(b, -1)).any(-1)
        return np.array([diff.sum(), 0, 0, 0, 0], np.float64)
    d = np.abs(a.astype(np.float64) - b)
    rel = d / np.maximum(np.abs(b.astype(np.float64)), 2.0 ** -126)
    return np.array([(d > 0).sum(), d.max(), ulps(a, b).max(), rel.max(),
                     d.mean()], np.float64)


def main(ref: str, hf: Path) -> None:
    import hashlib
    import platform

    import torch
    M, k = _reference(ref)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(0)
    inp = inputs()

    def run(dev):
        got = {}
        _DEV[0] = dev
        with torch.inference_mode():
            got.update(norms(M, inp, dev))
            got.update(gates(M, inp, dev))
            got.update(heads(M, inp, dev))
            got.update(comps(M, inp, dev))
            got.update(hcs(M, k, inp, dev))
        got.update(e2e(M, hf, dev))
        _DEV[0] = "cpu"
        return got

    arrays = run("cpu")
    gpu = torch.backends.mps.is_available()
    if gpu:
        for key, v in run("mps").items():
            arrays["spread_" + key] = spread(v, arrays[key])
            print(key, arrays["spread_" + key])
            if key == "qnorm_y":
                arrays["qnorm_gpu_check"] = spread(v, arrays["qnorm_y_gpu"])
                print("qnorm_y_gpu", arrays["qnorm_gpu_check"])
    # the CPU's q norm is its rsqrt estimate's (kept: its spread above)
    arrays.pop("qnorm_y")
    arrays["inputs_sha"] = np.array(inputs_sha(inp))
    arrays["e2e_hf_sha"] = np.array(hashlib.sha256(
        (hf / "model.safetensors").read_bytes()).hexdigest())
    arrays["__meta__"] = np.array(
        f"torch {torch.__version__} (CPU{' and MPS' if gpu else ''}, "
        f"{platform.machine()}); numpy {np.__version__}; reference "
        f"{Path(ref).name}/inference model.py (RMSNorm, Attention's q norm, "
        f"Gate, ParallelHead, Compressor, precompute_freqs_cis; "
        f"Transformer.forward / forward_spec end to end) with kernel.py as "
        f"build_deepseek_v4_dspark.kernels ports; default dtype bfloat16; "
        f"script tests/support/goldens/build_deepseek_v4_bf16.py; inputs "
        f"default_rng(23); end to end on hf_tensors() (its sha256 "
        f"e2e_hf_sha)")
    np.savez_compressed(OUT, **arrays)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    if sys.argv[1] == "weights":
        print(write_hf(Path(sys.argv[2])))
    else:
        main(sys.argv[2], Path(sys.argv[3]))
