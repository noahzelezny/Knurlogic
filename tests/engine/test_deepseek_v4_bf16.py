"""DeepSeek-V4's trunk in bfloat16 -- the dtype it serves in -- held to
DeepSeek's reference code run in bfloat16 (tests/support/goldens/
build_deepseek_v4_bf16.py: model.py / kernel.py under torch on the CPU,
with the reference's own CPU-vs-GPU spread from a second run on MPS).
Every other DeepSeek golden runs float32, where none of these show: the
gate's scores, the head's logits and the compressors' pooling are float32
in the reference; its norms round once; its per-head q norm is bf16
arithmetic. Each runs the GPU path (the fused Metal kernels) on bf16
inputs."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests" / "support" / "goldens"))

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

import build_deepseek_v4_bf16 as B  # noqa: E402

GOLD = dict(np.load(B.OUT))
INP = B.inputs()
BF = mx.bfloat16


@pytest.fixture(scope="module")
def A():
    from knurlogic.engine import register
    register.register("deepseek_v4")
    import mlx_lm.models.deepseek_v4 as M
    return M


def _bf(name):
    return mx.array(INP[name]).astype(BF)


def _f(a):
    return np.array(a.astype(mx.float32))


def _report(name, got, want):
    """(elements that differ, max bf16 ulps) of got against the golden."""
    d = np.abs(got.astype(np.float64) - want)
    n, u = int((d > 0).sum()), float(B.ulps(got, want).max())
    print(f"{name}: {n} of {want.size} differ, max {u:.0f} ulp "
          f"(reference CPU vs GPU: {GOLD.get('spread_' + name, '-')})")
    return n, u


def test_the_inputs_are_the_goldens():
    assert B.inputs_sha(INP) == str(GOLD["inputs_sha"])


# ------------------------------------------------------------------ norms
def test_every_weighted_norm_rounds_once_as_the_references(A):
    """model.py's RMSNorm: (w.float() * x * rsqrt(mean(x^2) + eps)) to
    bf16, one rounding (mx.fast.rms_norm rounds x * rsqrt first, then the
    product). The block's fused norm + act_quant (attn_norm, ffn_norm),
    the attention's q_norm / kv_norm, the final norm, and the module every
    other weighted norm is (compressors', heads'). The reference's CPU and
    GPU runs agree bit for bit; ours may differ from it only where a
    float32 mean summed in another order rounds the other way."""
    eps = B.EPS
    bad = {}
    for d in (4096, 1024, 512, 128):
        x, w = _bf(f"norm{d}_x"), _bf(f"norm{d}_w")
        want = GOLD[f"norm{d}_y"]
        y, yq = A.rms_norm_act(x, w, eps)
        bad[f"norm{d}_y fused"] = _report(f"norm{d}_y", _f(y), want)
        assert (_f(yq) == _f(A.fp8_act(mx.array(want).astype(BF)))).all() \
            or bad[f"norm{d}_y fused"][0]
        m = A.DeepseekV4Model(A.ModelArgs(
            hidden_size=d, num_hidden_layers=0, vocab_size=8)).norm
        m.weight = w
        bad[f"norm{d}_y final"] = _report(f"norm{d}_y", _f(m(x)), want)
    # q_norm (rounded for wq_b) and kv_norm off one wqkv_a output
    qkv = mx.concatenate([_bf("norm1024_x"), _bf("norm512_x")], axis=-1)
    qr, kv = A._attn_qkv_split_norm(qkv, _bf("norm1024_w"),
                                    _bf("norm512_w"), 1024, eps)
    bad["kv_norm"] = _report("norm512_y", _f(kv), GOLD["norm512_y"])
    want_qr = _f(A.fp8_act(mx.array(GOLD["norm1024_y"]).astype(BF)))
    bad["q_norm"] = (int((_f(qr) != want_qr).sum()), 0.0)
    for k, (n, u) in bad.items():
        # at most 1 element in 1000, by one ulp
        size = GOLD["norm512_y" if k in ("kv_norm", "q_norm")
                    else k.split()[0]].size
        assert n <= size // 1000 and u <= 1, (k, n, u)


def test_no_norm_with_a_weight_rounds_twice(A):
    """Every weighted norm the trunk, the MTP head and the DSpark head
    build is the single-rounding module (the reference's RMSNorm), never
    mlx's nn.RMSNorm."""
    from knurlogic.engine.families.deepseek.heads import (
        deepseek_v4 as mtp, deepseek_v4_dspark as dspark)
    args = A.ModelArgs(hidden_size=128, num_hidden_layers=2,
                       num_attention_heads=2, q_lora_rank=128,
                       o_lora_rank=64, o_groups=2, head_dim=80,
                       qk_rope_head_dim=16, compress_ratios=[4, 128],
                       index_n_heads=2, index_head_dim=128,
                       moe_intermediate_size=128, n_routed_experts=4,
                       num_experts_per_tok=2, num_hash_layers=1,
                       vocab_size=64, quantization={"skeleton": True},
                       dspark_block_size=3, num_nextn_predict_layers=2,
                       dspark_target_layer_ids=[0, 1], dspark_markov_rank=8)
    model = A.Model(args)
    found = [p for p, m in model.named_modules() if type(m) is nn.RMSNorm]
    head = mtp.MTPHead(model, A)
    found += ["mtp." + p for p, m in head.m.named_modules()
              if type(m) is nn.RMSNorm]
    ds = dspark.DSparkHead(model, A)
    found += ["dspark." + p for p, m in ds.m.named_modules()
              if type(m) is nn.RMSNorm]
    assert found == []
    n = sum(type(m) is A.RMSNorm for m in (
        [m for _, m in model.named_modules()]
        + [m for _, m in head.m.named_modules()]
        + [m for _, m in ds.m.named_modules()]))
    # trunk: 2 x (attn, ffn, q, kv) + 2 compressors + 1 indexer
    # compressor + final; MTP: its block's 4 + enorm, hnorm, norm; DSpark:
    # 2 stages x 4, main_norm, norm
    assert n == 2 * 4 + 3 + 1 + 4 + 3 + 2 * 4 + 2


def test_the_per_head_q_norm_is_the_references_bf16_arithmetic(A):
    """`q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + eps)` on bf16
    q: each op rounded to bf16, the mean accumulated in float32, the
    rsqrt accurate -- the reference's GPU run (its MPS run is exactly
    this; its CPU run differs on 38% of the elements, by up to 3 ulps,
    because torch's CPU bf16 rsqrt is an estimate on arm64). Ours did it
    in float32 with one rounding: 43% of the elements an ulp or two off."""
    x = _bf("qnorm_x")                              # [8 rows, 64 heads, 512]
    flat = x.reshape(1, 8, 64 * 512)
    got = _f(A._attn_q_proj_norm(flat, 64, 512, B.EPS)[0].transpose(1, 0, 2))
    n, u = _report("qnorm_y", got, GOLD["qnorm_y_gpu"])
    assert n <= GOLD["qnorm_y_gpu"].size // 10000 and u <= 1, (n, u)


# ------------------------------------------------------------------ gate
@pytest.mark.parametrize("name,layer,vl", [("gate", 1, 0), ("gate_vl", 1, 1),
                                           ("gate_hash", 0, 1)])
def test_the_gate_routes_on_float32_scores(A, name, layer, vl):
    """Gate.forward's `linear(x.float(), weight.float())` at Flash's shapes
    (256 experts of 4096, top 6) on 256 bf16 rows: the same experts for
    every row (the reference's CPU and GPU runs agree), the weights within
    float32 summation noise (2e-6; the reference's own CPU-GPU spread is
    6e-8 on the score layers, 9e-5 on the hash layer's)."""
    args = A.ModelArgs(hidden_size=B.D, n_routed_experts=B.N_EXP,
                       num_experts_per_tok=B.TOPK, num_hash_layers=1,
                       vocab_size=B.VOCAB, vision_n_layers=vl,
                       routed_scaling_factor=B.ROUTE_SCALE)
    g = A.MoEGate(args, layer)
    g.weight = _bf("gate_w")
    if hasattr(g, "e_score_correction_bias"):
        g.e_score_correction_bias = mx.array(INP["gate_bias"])
    if vl:
        g.bias_vl = mx.array(INP["gate_bias_vl"])
    if layer == 0:
        g.tid2eid = mx.array(INP["gate_tid2eid"])
    ids = INP["gate_ids"] if vl else INP["gate_ids"] % B.VOCAB
    ids = mx.array(ids.astype(np.int32))[None]
    x = _bf("gate_x")[None]
    inds, w = g(x, ids, (ids >= B.VOCAB) if vl else None)
    inds, w = np.array(inds[0]), np.array(w[0])
    assert w.dtype == np.float32
    want_i, want_w = GOLD[f"{name}_inds"], GOLD[f"{name}_weights"]
    oi, ow = np.argsort(inds, -1), np.argsort(want_i, -1)
    inds, w = np.take_along_axis(inds, oi, -1), np.take_along_axis(w, oi, -1)
    want_i = np.take_along_axis(want_i, ow, -1)
    want_w = np.take_along_axis(want_w, ow, -1)
    rows = int((inds != want_i).any(-1).sum())
    same = (inds == want_i).all(-1)
    err = float(np.abs(w - want_w)[same].max())
    print(f"{name}: {rows} of 256 rows route elsewhere, weights max err "
          f"{err:.2e}")
    assert rows == 0
    assert err < 2e-6


# ------------------------------------------------------------------ head
def _head_model(A, w):
    args = A.ModelArgs(hidden_size=B.D, num_hidden_layers=0,
                       vocab_size=B.HEAD_V)
    model = A.Model(args)
    model.lm_head.weight = w
    return model


def test_the_head_is_float32_logits(A):
    """ParallelHead: `F.linear(x.float(), weight)`, the bf16 weight read in
    float32 and float32 logits, on 1, 6 and 20 rows (decode, a verify, a
    prefill's rows) through Model.__call__. Against the reference's CPU
    logits and the exact (float64) product of the same bf16 operands: the
    reference's own float32 sums are up to 6.1e-5 from exact (its CPU and
    GPU runs 5.3e-5 apart); ours must be no farther from either than the
    reference's spread (a bf16 logit is ~1e-2 off)."""
    model = _head_model(A, _bf("head_w"))
    tol = max(GOLD[f"spread_head_y{r}"][1] for r in B.HEAD_ROWS)
    w64 = INP["head_w"].astype(np.float64)
    for r in B.HEAD_ROWS:
        x = _bf(f"head_x{r}")[None]
        model.model = lambda *a, _x=x, **k: _x
        got = model(mx.zeros((1, r), dtype=mx.int32))
        assert got.dtype == mx.float32, r
        got = np.array(got[0])
        exact = INP[f"head_x{r}"].astype(np.float64) @ w64.T
        err = float(np.abs(got - exact).max())
        ref = float(np.abs(GOLD[f"head_y{r}"] - exact).max())
        to_ref = float(np.abs(got - GOLD[f"head_y{r}"]).max())
        print(f"head rows {r}: ours {err:.2e} from exact, the reference's "
              f"{ref:.2e}; ours to the reference's {to_ref:.2e} "
              f"(tolerance {tol:.2e})")
        assert err <= tol and to_ref <= 1.25 * tol, (r, err, to_ref)


@pytest.mark.parametrize("N", [256, 2048, 10000])
def test_the_float32_linear_is_float32_and_the_same_in_any_batch(A, N):
    """f32_linear (the gate's 256 outputs and the compressors' 2048 on its
    split-K kernel, a head-sized 10000 on its per-output kernel, past 64
    rows the op path): within float32 summation of the exact product of
    the bf16 operands, and each row's result the same bits whether it is
    computed alone or in a batch of 1-80 rows (a verify forward and the
    plain steps it replaces must agree)."""
    rng = np.random.default_rng(N)
    w = mx.array(B._bf16(0.02 * rng.standard_normal((N, B.D)))).astype(BF)
    x = mx.array(B._hidden(rng, (80, B.D))).astype(BF)
    exact = (np.array(x.astype(mx.float32)).astype(np.float64)
             @ np.array(w.astype(mx.float32)).astype(np.float64).T)
    alone = np.concatenate([np.array(A.f32_linear(x[i:i + 1], w))
                            for i in range(80)])
    scale = np.abs(exact).max()
    assert np.abs(alone - exact).max() < 4e-6 * scale
    for rows in (6, 16, 20, 64):
        got = np.array(A.f32_linear(x[:rows], w))
        assert (got == alone[:rows]).all(), rows
    big = np.array(A.f32_linear(x, w))     # 80 rows: the op path
    assert np.abs(big - exact).max() < 4e-6 * scale


@pytest.mark.parametrize("mode,gs,bits", [("affine", 64, 8),
                                          ("mxfp4", 32, 4)])
def test_a_quantized_head_is_float32_over_its_dequantized_weight(
        A, mode, gs, bits):
    """An MLX-quantized head (third-party artifacts: mlx-community's and
    VQ's affine 8-bit; or mxfp4): float32 logits from quantized_matmul on
    float32 activations -- the dequantized weight in float32 (not rounded
    to bf16). mxfp4 is float32-exact; MLX's affine kernel is 1.5e-3 of the
    logits' scale off the exact product (measured, any bits, float32
    scales too), under the 8-bit quantization's own error (the reference
    has no quantized head)."""
    lin = nn.Linear(B.D, 512, bias=False)
    lin.weight = _bf("head_w")[:512]
    q = lin.to_quantized(group_size=gs, bits=bits, mode=mode)
    x = _bf("head_x6")[None]
    got = A.head_logits(q, x)
    assert got.dtype == mx.float32
    deq = mx.dequantize(q.weight, q.scales, q.biases, group_size=gs,
                        bits=bits, mode=mode, dtype=mx.float32)
    want = (np.array(x.astype(mx.float32)[0]).astype(np.float64)
            @ np.array(deq).astype(np.float64).T)
    tol = 2e-3 if mode == "affine" else 1e-6
    assert np.abs(np.array(got[0]) - want).max() < tol * np.abs(want).max()


def test_the_mtp_heads_logits_are_float32(A):
    """The MTP head's draft logits through the same float32 head."""
    from knurlogic.engine.families.deepseek.heads import deepseek_v4 as mtp
    model = _head_model(A, _bf("head_w"))
    head = object.__new__(mtp.MTPHead)
    head.model, head.arch = model, A
    x = _bf("head_x6")[None]
    head._trunk = lambda *a, **k: x
    got = head.draft_logits(None, None)
    assert got.dtype == mx.float32
    exact = (INP["head_x6"].astype(np.float64)
             @ INP["head_w"].astype(np.float64).T)
    err = float(np.abs(np.array(got[0]) - exact).max())
    assert err <= GOLD["spread_head_y6"][1], err


# ------------------------------------------------------------ compressor
@pytest.mark.parametrize("case", B.COMP, ids=[c[0] for c in B.COMP])
def test_the_compressor_pools_in_float32(A, case):
    """Compressor.forward keeps float32 from wkv / wgate through the ape,
    the softmax and the weighted sum and rounds once (then its norm, rope
    and FP8 / FP4 rounding): a bf16 prefill (windows and a tail), then one
    token at a time (the decode emit kernel), against the reference's rows
    (its CPU and GPU runs agree bit for bit). Ours may part only where a
    float32 sum in another order rounds the other way: at most 1 element
    in 1000 by more than an ulp."""
    name, ratio, d, rot, P, T = case
    rope = A.DeepseekV4RoPE(B.RD, B.ROPE["base"], {
        "type": "yarn", "factor": B.ROPE["factor"],
        "original_max_position_embeddings": B.ROPE["original_seq_len"],
        "beta_fast": B.ROPE["beta_fast"], "beta_slow": B.ROPE["beta_slow"]})
    c = A.Compressor(dim=B.D, compress_ratio=ratio, head_dim=d,
                     rope_head_dim=B.RD, rms_norm_eps=B.EPS, rope=rope,
                     rotate=rot)
    c.wkv_gate.weight = mx.concatenate(
        [_bf(f"{name}_wkv"), _bf(f"{name}_wgate")], axis=0)
    c.ape = mx.array(INP[f"{name}_ape"])
    c.norm.weight = _bf(f"{name}_norm")
    cache = A.DeepseekV4Cache(128)
    x = _bf(f"{name}_x")
    c(x[:, :P], cache, 0)
    for i in range(P, P + T):
        c(x[:, i:i + 1], cache, i)
    got = _f(cache.get_branch(A._K_COMP).pool[0])
    want = GOLD[f"{name}_rows"]
    assert got.shape == want.shape
    n, u = _report(f"{name}_rows", got, want)
    far = int((B.ulps(got, want) > 1).sum())
    assert far <= want.size // 1000, (n, far, u)


# --------------------------------------------------------------- sinkhorn
def _sinkhorn(A, eps, iters):
    return A.hc_sinkhorn_collapse(
        mx.array(INP["hc_mixes"])[None], mx.array(INP["hc_scale"]),
        mx.array(INP["hc_base"]), _bf("hc_x")[None], 4, iters,
        mx.array([eps], dtype=mx.float32))


def _rel(got, want):
    return float((np.abs(np.array(got[0]) - want) / np.abs(want)).max())


def test_the_fused_sinkhorn_takes_eps_where_the_reference_does(A):
    """kernel.py's hc_split_sinkhorn: `comb = comb.softmax(-1) + eps`,
    i.e. exp / row_sum + eps -- no eps in the softmax's denominator --
    then `comb / (sum + eps)` along columns and rows. The fused sinkhorn +
    collapse kernel at an eps large enough to see where it enters (0.25,
    3 iterations; the real 1e-6 moves comb ~1e-11 after 20 iterations,
    under float32's noise): comb within the reference's own CPU-GPU
    spread (relative 3.4e-7, doubled). (post has no eps in it; the test
    below holds it to exact.)"""
    _, _, comb = _sinkhorn(A, B.BIG_EPS, B.BIG_EPS_ITERS)
    e_comb = _rel(comb, GOLD["hc_big_comb"])
    print(f"eps {B.BIG_EPS}: comb max relative err {e_comb:.2e}")
    assert e_comb <= 2 * GOLD["spread_hc_big_comb"][3], e_comb


def test_the_fused_sinkhorn_and_collapse_at_the_real_eps(A):
    """The real eps and 20 iterations on 256 rows of fixed float32 mixes:
    post and comb no farther from exact (the reference port in float64)
    than the reference's own float32 run is (its exp's argument rounding
    puts comb's smallest entries ~3e-6 of themselves off), and hc_pre's
    float32 collapse to bf16 parting from the reference's by an ulp at
    most where they do (2 of 131072 in its CPU-GPU spread)."""
    y, post, comb = _sinkhorn(A, B.EPS, B.SINK_ITERS)
    for name, got in (("post", post), ("comb", comb)):
        exact = GOLD[f"hc_{name}_f64"]
        ours, ref = _rel(got, exact), _rel(mx.array(GOLD[f"hc_{name}"])[None],
                                           exact)
        print(f"{name}: ours {ours:.2e} from exact, the reference's {ref:.2e}")
        assert ours <= 1.25 * ref, (name, ours, ref)
    n, u = _report("hc_y", _f(y[0]), GOLD["hc_y"])
    assert n <= 8 and u <= 1


# ------------------------------------------------------------- attn_sink
def test_mlx_attention_takes_no_float32_sinks_with_bf16_queries(A):
    """The reference's sparse_attn adds exp(attn_sink - max) with
    attn_sink float32; MLX's fused attention refuses float32 sinks beside
    bf16 queries, so the trunk passes them rounded to bf16 (PROVENANCE
    edit 28 records the measured effect). If MLX ever takes them, pass
    them as they are."""
    q = mx.zeros((1, 2, 1, 64), dtype=BF)
    k = mx.zeros((1, 1, 8, 64), dtype=BF)
    with pytest.raises(ValueError):
        mx.eval(mx.fast.scaled_dot_product_attention(
            q, k, k, scale=0.125, sinks=mx.zeros((2,), dtype=mx.float32)))


# ------------------------------------------------------------- end to end
@pytest.fixture(scope="module")
def tiny16(A, tmp_path_factory):
    """The DSpark golden's tiny checkpoint with every tensor the real one
    stores bf16 stored bf16 (the builder's hf_tensors; checked by its
    sha256), made into an MLX artifact and DSpark sidecar the way the
    DSpark golden's are."""
    import json

    from mlx.utils import tree_flatten

    import build_deepseek_v4_dspark as G
    from knurlogic.engine.families.deepseek.heads import dspark_pack
    root = tmp_path_factory.mktemp("bf16")
    hf, tiny = root / "hf", root / "tiny"
    assert B.write_hf(hf) == str(GOLD["e2e_hf_sha"])
    tiny.mkdir()
    w = mx.load(str(hf / "model.safetensors"))
    model = A.Model(A.ModelArgs.from_dict(G.CONFIG))
    model.load_weights(list(model.sanitize(
        {k: v for k, v in w.items() if not k.startswith("mtp.")}).items()),
        strict=True)
    mx.save_safetensors(str(tiny / "model.safetensors"),
                        dict(tree_flatten(model.parameters())),
                        metadata={"format": "mlx"})
    (tiny / "config.json").write_text(json.dumps(G.CONFIG, indent=1))
    dspark_pack.pack(hf, tiny)
    return tiny


def _trace16(path):
    """build_deepseek_v4_dspark's run (prefill, 5 forced decode steps,
    the head's draft at each) through the MLX load path, in bf16."""
    from contextlib import ExitStack

    from mlx_lm.utils import load_model

    import build_deepseek_v4_dspark as G
    from knurlogic.engine.mtp import registry
    from knurlogic.engine.mtp.capture import capture_input
    from knurlogic.interfaces import loading
    from knurlogic.machine.artifact import Artifact
    assert loading.register(Artifact.load(str(path))) == []
    model, _ = load_model(path)
    assert model.model.embed_tokens.weight.dtype == BF
    head, _ = registry.load_head(model, model_path=path)
    out = {k: [] for k in ("logits", "main_hidden", "draft_ids",
                           "draft_logits", "confidence")}
    with ExitStack() as st:
        gets = [st.enter_context(capture_input(model.model, p))
                for p in head.capture_paths()]
        cache, hc = model.make_cache(), head.make_draft_cache()
        lg = model(mx.array([G.PROMPT]), cache=cache)
        mh = head.main_hidden([g() for g in gets])
        assert mh.dtype == BF
        out["prefill_main_hidden"] = mh[0]
        head.advance(mh, hc)
        out["logits"].append(lg[0, -1])
        for t in G.DECODE:
            lg = model(mx.array([[t]]), cache=cache)
            mh = head.main_hidden([g() for g in gets])
            head.advance(mh, hc)
            ids, dl, conf = head.draft(mx.argmax(lg[:, -1], axis=-1), hc)
            for k, v in (("logits", lg[0, -1]), ("main_hidden", mh[0, -1]),
                         ("draft_ids", ids[0]), ("draft_logits", dl[0]),
                         ("confidence", conf[0])):
                out[k].append(v)
    return {k: np.array((mx.stack(v) if isinstance(v, list) else v)
                        .astype(mx.float32)) for k, v in out.items()}


def test_the_bf16_forward_end_to_end_is_within_the_references_spread(
        tiny16):
    """The DSpark golden's run with the model in bf16 (its checkpoint's
    bf16 tensors bf16, as the real one's), through the GPU path -- the
    fused norm, gate, sinkhorn + collapse, compressor emit and float32
    head kernels -- against the reference's bf16 run on the CPU. On this
    tiny random model a float32 ulp at an FP8 boundary moves the logits
    by 0.1-0.7, and the reference's own CPU and GPU (MPS) runs part that
    way: logits up to 1.3 apart, draft logits 7.1, 4 of 5 draft blocks.
    So this bounds gross departures only (the pieces are held tightly
    above): ours must be about as far from the CPU run as its GPU run is
    (max and mean of every output within 1.5x)."""
    got = _trace16(tiny16)
    rows = []
    for k in ("logits", "prefill_main_hidden", "main_hidden",
              "draft_logits", "confidence"):
        want = GOLD["e2e_" + k]
        d = np.abs(got[k].astype(np.float64) - want)
        ref = GOLD["spread_e2e_" + k]
        rows.append((k, d.max(), d.mean(), ref[1], ref[4]))
        print(f"{k}: max {d.max():.3g} mean {d.mean():.3g} (reference CPU "
              f"vs GPU: max {ref[1]:.3g} mean {ref[4]:.3g})")
    ids = int((got["draft_ids"].astype(np.int32)
               != GOLD["e2e_draft_ids"]).any(-1).sum())
    print(f"draft blocks that differ: {ids} of 5 (reference: "
          f"{int(GOLD['spread_e2e_draft_ids'][0])})")
    for k, worst, mean, rmax, rmean in rows:
        assert worst <= 1.5 * rmax and mean <= 1.5 * rmean, (k, worst,
                                                              mean)
