"""The vendored gemma4_text (engine/families/gemma4/architecture) held to
the maker's reference: HF transformers' modeling_gemma4.py and
configuration_gemma4.py run on the same tiny random weights under torch
(tests/support/goldens/build_gemma4_text.py; the npz's `__meta__` says
which transformers/torch built it).

Configs: the e-style dense model (per-layer embeddings, KV sharing,
double-wide MLP, sliding/full mix, proportional partial RoPE), the
26B-A4B-style MoE (router, per-expert scale, K=V full layers), one that
leaves out every key HF defaults (`defaults`), and
use_bidirectional_attention "all". Each runs an 11-token prefill past the
window, then 5 decode steps through the model's own make_cache -- the
rotating sliding cache wraps. dense and moe also run in bfloat16 (other
weights, `<cfg>_bf16`) against HF run in bfloat16, and the bf16 rounding
points of the norm and the MLP activation are held op by op."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests/support/goldens"))

mx = pytest.importorskip("mlx.core")

from build_gemma4_text import np_weights  # noqa: E402

GOLD = np.load(ROOT / "tests/support/goldens/gemma4_text.npz")
#: float32 both sides; the reference and MLX differ only in summation order
ATOL = 2e-4


def _arch():
    from knurlogic.engine import register
    # override: the vendored file, even if something imported mlx-lm's own
    register.register("gemma4_text", override=True)
    register.register("gemma4", override=True)
    from mlx_lm.models import gemma4_text as arch
    assert "families/gemma4/architecture" in arch.__file__
    return arch


def _weights(name):
    if f"{name}/shapes" in GOLD.files:
        shapes = json.loads(str(GOLD[f"{name}/shapes"]))
        return np_weights({k: tuple(v) for k, v in shapes.items()},
                          int(GOLD[f"{name}/seed"]),
                          float(GOLD[f"{name}/qk_gain"]))
    pre = f"{name}/w/"
    return {k[len(pre):]: GOLD[k] for k in GOLD.files if k.startswith(pre)}


def _model(name, dtype=None):
    from mlx.utils import tree_flatten

    arch = _arch()
    cfg = json.loads(str(GOLD[f"{name}/config"]))
    model = arch.Model(arch.ModelArgs.from_dict(cfg))
    weights = {k: mx.array(v) for k, v in _weights(name).items()}
    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()), strict=True)
    assert len(weights) == len(tree_flatten(model.parameters()))
    if dtype is not None:
        model.set_dtype(dtype)
    mx.eval(model.parameters())
    return model


def _run(model):
    cache = model.make_cache()
    prefill = model(mx.array(GOLD["prompt"])[None], cache=cache)[0]
    decode = [model(mx.array([[t]]), cache=cache)[0, -1]
              for t in GOLD["decode_ids"].tolist()]
    return (np.array(prefill.astype(mx.float32)),
            np.stack([np.array(d.astype(mx.float32)) for d in decode]))


@pytest.mark.parametrize("name", ["dense", "moe", "defaults", "all",
                                  "dense_bf16", "moe_bf16"])
def test_gemma4_text_matches_the_reference(name):
    prefill, decode = _run(_model(name))
    np.testing.assert_allclose(prefill, GOLD[f"{name}/prefill"],
                               atol=ATOL, rtol=0)
    for i in range(len(decode)):
        np.testing.assert_allclose(decode[i], GOLD[f"{name}/decode"][i],
                                   atol=ATOL, rtol=0, err_msg=f"step {i}")


def test_bidirectional_all_prefilled_in_pieces_matches_the_reference():
    """"all" is not causal, so a prompt prefilled in two chunks is not the
    one-pass prefill: the first chunk never sees the second, and the second
    sees only what the sliding cache kept of the first. Ours keeps what
    the reference's cache keeps (window - 1 earlier tokens)."""
    model = _model("all")
    prompt = mx.array(GOLD["prompt"])[None]
    k = int(GOLD["chunk"])
    cache = model.make_cache()
    model(prompt[:, :k], cache=cache)
    got = model(prompt[:, k:], cache=cache)[0]
    want = GOLD["all/chunked"]
    assert np.abs(want - GOLD["all/prefill"][k:]).max() > 0.01  # chunks bite
    np.testing.assert_allclose(np.array(got), want, atol=ATOL, rtol=0)


#: bfloat16 both sides, on the dense and moe configs with q/k norm gain 0.3
#: (build_gemma4_text.py BF16). Two correct bf16 implementations of the
#: same model need not agree to the ulp: HF's own eager and sdpa attention
#: (eager rounds the softmax probabilities to bf16 before P @ V; sdpa, like
#: our fused attention, does not) disagree by `spread` on these weights,
#: 0.10 dense / 0.37 moe. Ours also rounds RoPE once (fp32 inside the
#: kernel) where HF rounds three times, and normalizes in one fused kernel
#: (see test_text_rmsnorm_is_within_one_bf16_ulp_of_the_reference). It must
#: sit within twice that spread of HF's eager bf16 logits -- the same
#: weights in float32 must match to ATOL, and a real precision bug (a step
#: computed in the wrong dtype, a dropped scale) moves these logits by
#: whole units.
BF16_SPREAD_MULT = 2.0


@pytest.mark.parametrize("name", ["dense_bf16", "moe_bf16"])
def test_gemma4_text_bf16_matches_the_reference_in_bf16(name):
    prefill, decode = _run(_model(name, mx.bfloat16))
    got = np.concatenate([prefill, decode])
    eager, sdpa = (np.concatenate([GOLD[f"{name}/bf16_{i}/prefill"],
                                   GOLD[f"{name}/bf16_{i}/decode"]])
                   for i in ("eager", "sdpa"))
    spread = np.abs(eager - sdpa).max()
    diff = np.abs(got - eager).max()
    assert diff <= BF16_SPREAD_MULT * spread, (name, diff, spread)


def _bf16(key):
    """A golden's bf16 bit patterns as an mx.bfloat16 array."""
    bits = GOLD[key].astype(np.uint32) << 16
    return mx.array(bits.view(np.float32)).astype(mx.bfloat16)


def _ulps(got, want):
    """Per element, how many bf16 steps apart (both are bf16 values)."""
    a = np.array(got.astype(mx.float32)).view(np.int32) >> 16
    b = np.array(want.astype(mx.float32)).view(np.int32) >> 16
    return np.abs(a.astype(np.int64) - b.astype(np.int64))


def test_the_mlp_activation_rounds_where_the_reference_does():
    """gelu_pytorch_tanh(gate) * up in bf16: torch computes the gelu in
    float32 and rounds once; mlx's gelu_approx on a bf16 array rounds after
    every step (42% of these elements a step or more off). knurlogic edit
    5 computes it in float32 (fused by mx.compile, free); what is left is
    the two tanh implementations: a step or two on under 1%, and near
    gate = -5, where mlx's float32 tanh reaches -1 and torch's does not,
    0 against ~1e-6."""
    arch = _arch()
    want = _bf16("ops/act")
    got = arch.geglu(_bf16("ops/gate"), _bf16("ops/up"))
    ulps = _ulps(got, want)
    tiny = np.abs(np.array((got - want).astype(mx.float32))) < 1e-5
    assert (ulps > 0).mean() < 0.01, (ulps > 0).mean()
    assert ((ulps <= 2) | tiny).all(), ulps[~tiny].max()


def test_text_rmsnorm_is_within_one_bf16_ulp_of_the_reference():
    """HF's Gemma4RMSNorm computes (x * rsqrt(mean(x^2) + eps)) * w in
    float32 and rounds once; mx.fast.rms_norm rounds x * rsqrt to bf16,
    then multiplies by w and rounds again: a quarter of the elements land
    one bf16 step from the reference, none further. Kept: both exact forms
    measured slower at decode (architecture/PROVENANCE.md, "RMSNorm
    rounding"). The unscaled norm (v_norm, the router's) rounds once and
    matches."""
    import mlx.nn as nn
    norm = nn.RMSNorm(1024, eps=1e-6)
    norm.weight = _bf16("ops/w")
    ulps = _ulps(norm(_bf16("ops/x")), _bf16("ops/norm"))
    assert ulps.max() <= 1, ulps.max()
    bare = mx.fast.rms_norm(_bf16("ops/x"), None, 1e-6)
    assert _ulps(bare, _bf16("ops/norm_noscale")).max() == 0


def _resolved(arch, rec):
    """What our ModelArgs / Model build from a config, in the golden's
    field names. The model is built lazily (MLX allocates nothing until
    an array is evaluated), so a full-size default model is cheap."""
    if rec["text"]:
        args = arch.ModelArgs.from_dict(dict(rec["input"],
                                             model_type="gemma4_text"))
        model = arch.Model(args)
    else:
        from mlx_lm.models import gemma4
        wrapper = gemma4.Model(gemma4.ModelArgs.from_dict(rec["input"]))
        model = wrapper.language_model
        args = model.args
    out = {f: getattr(args, f) for f in rec["hf"]
           if f not in ("layer_head_dim", "layer_kv_heads")}
    out["final_logit_softcapping"] = model.final_logit_softcapping
    out["layer_head_dim"] = [layer.self_attn.head_dim for layer in model.layers]
    out["layer_kv_heads"] = [layer.self_attn.n_kv_heads for layer in model.layers]
    return out


RESOLVE = json.loads(str(GOLD["resolve"]))


@pytest.mark.parametrize("name", sorted(RESOLVE))
def test_config_resolves_as_the_reference_does(name):
    """Keys left out take HF's Gemma4TextConfig defaults; layer_types gets
    HF's 5:1 pattern and its last layer forced full; "all" halves the
    window; the released artifacts' own text configs resolve the same."""
    arch = _arch()
    rec = RESOLVE[name]
    got = _resolved(arch, rec)
    for f, v in rec["hf"].items():
        assert got[f] == v, (name, f, got[f], v)


def test_a_config_it_cannot_run_is_refused():
    """HF reads hidden_activation and attention_bias; ours runs
    gelu_pytorch_tanh without biases, so anything else is refused at load
    rather than run as the wrong model."""
    arch = _arch()
    with pytest.raises(ValueError, match="hidden_activation"):
        arch.ModelArgs.from_dict({"model_type": "gemma4_text",
                                  "hidden_activation": "silu"})
    with pytest.raises(ValueError, match="attention_bias"):
        arch.ModelArgs.from_dict({"model_type": "gemma4_text",
                                  "attention_bias": True})


def test_an_image_block_masks_as_the_reference_does():
    """use_bidirectional_attention "vision" (26B-A4B, 31B): the image block
    attends both ways on the sliding layers, cut by the window; the full
    layers stay causal (vendored edit 1)."""
    model = _model("moe")
    want = GOLD["moe/prefill_block"]
    assert np.abs(want - GOLD["moe/prefill"]).max() > 0.1  # the block bites
    block = mx.array(GOLD["block"], dtype=mx.int32)[None]
    got = model(mx.array(GOLD["prompt"])[None], cache=model.make_cache(),
                mm_mask=block)[0]
    np.testing.assert_allclose(np.array(got), want, atol=ATOL, rtol=0)
    # chunked as the serve path chunks: two text tokens already cached,
    # the image block starting the next chunk
    cache = model.make_cache()
    prompt = mx.array(GOLD["prompt"])[None]
    model(prompt[:, :2], cache=cache)
    got = model(prompt[:, 2:], cache=cache, mm_mask=block[:, 2:])[0]
    np.testing.assert_allclose(np.array(got), want[2:], atol=ATOL, rtol=0)


def test_an_e_model_stays_causal_over_an_image():
    """use_bidirectional_attention None (e2b/e4b): the reference builds a
    plain causal mask with an image in the prompt; mm_mask changes
    nothing."""
    model = _model("dense")
    block = mx.array(GOLD["block"], dtype=mx.int32)[None]
    got = model(mx.array(GOLD["prompt"])[None], cache=model.make_cache(),
                mm_mask=block)[0]
    np.testing.assert_allclose(np.array(got), GOLD["dense/prefill"],
                               atol=ATOL, rtol=0)
