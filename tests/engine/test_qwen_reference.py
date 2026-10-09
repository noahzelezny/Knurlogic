"""knurlogic's vendored Qwen trunks compute what Qwen's reference computes.

tests/support/goldens/qwen_reference.npz holds the logits of HF
transformers' own Qwen3.5 / Qwen3.5-MoE / Qwen4-Exp text models (float32,
CPU, eager) on a tiny random config, for a 21-token prefill and 4 decode
steps through the reference cache (build_qwen_reference.py says what the
config makes run). Here the same weights, re-made from numpy by name, go
through knurlogic's sanitize into its MLX trunk, float32, through its own
cache, and the logits must agree. The same golden holds config variants,
bfloat16 runs (trunk and single modules), the vision tower, the image
processor and rope_index; and the Qwen tool-call dialect is parsed through
knurlogic's path.
"""
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "support" / "goldens"))

mx = pytest.importorskip("mlx.core")

import build_qwen_reference as G  # noqa: E402

#: float32 on both sides, different kernels and summation orders: measured
#: max |diff| ~1e-6 on logits of magnitude ~4
TOL = 1e-4


def _golden():
    z = np.load(G.OUT)
    return z, json.loads(bytes(z["meta"]).decode())


def _model(family, meta, key=None, dtype=mx.float32, init=G.init_weights):
    from knurlogic.engine import register
    register.register(family, override=True)
    arch = importlib.import_module(f"mlx_lm.models.{family}")
    key = key or family
    cfg = dict(meta[f"{key}/config"])
    top = cfg["model_type"].removesuffix("_text")
    model = arch.Model(arch.ModelArgs.from_dict(
        {"model_type": top, "text_config": cfg}))
    model.set_dtype(dtype)
    w = init({k: tuple(v) for k, v in meta[f"{key}/shapes"].items()})
    if dtype == mx.bfloat16:
        w = {k: G.bf16(v) for k, v in w.items()}
    if family == "qwen4_exp":
        w = _shard_ngram(w, cfg)
    w = model.sanitize({k: mx.array(v).astype(dtype) for k, v in w.items()})
    have = set(dict(__import__("mlx.utils").utils.tree_flatten(
        model.parameters())))
    assert set(w) <= have, sorted(set(w) - have)
    # only the n-gram buffers (rebuilt from the config) are not loaded
    assert all(any(b in k for b in G.BUFFERS) for k in have - set(w)), \
        sorted(have - set(w))
    model.load_weights(list(w.items()), strict=False)
    mx.eval(model.parameters())
    return model


def _shard_ngram(w, cfg):
    """The reference keeps one n-gram table; the released checkpoints (and
    knurlogic) split it into split_ngram_parts row shards."""
    out = {}
    n = cfg.get("split_ngram_parts", 512)  # the reference config's default
    for k, v in w.items():
        if not k.endswith("ngram_embedding.weight"):
            out[k] = v
            continue
        rows = math.ceil(v.shape[0] / n)
        v = np.pad(v, [(0, rows * n - v.shape[0]), (0, 0)])
        for i in range(n):
            out[k.replace(".weight", f".shard_{i}.weight")] = \
                v[i * rows:(i + 1) * rows]
    return out


def _run(model, meta):
    cache = model.make_cache()
    out = [model(mx.array([meta["prompt"]]), cache=cache)[0]]
    for t in meta["decode"]:
        out.append(model(mx.array([[t]]), cache=cache)[0])
    return np.array(mx.concatenate(out, axis=0).astype(mx.float32))


@pytest.mark.parametrize("family", G.FAMILIES)
def test_the_trunk_computes_the_reference_logits(family):
    z, meta = _golden()
    ref = z[f"{family}/logits"]
    got = _run(_model(family, meta), meta)
    assert got.shape == ref.shape
    diff = float(np.abs(got - ref).max())
    assert diff < TOL, f"{family}: max |logit diff| {diff:.3e} vs reference"


def test_qwen4_exp_uses_the_checkpoints_own_ngram_multipliers():
    """A checkpoint's stored int64 layer_multipliers are the ones hashed with,
    not a rebuild from the config's seed (vendored edit 4): a wrong seed in
    a config can no longer give wrong n-gram rows."""
    _, meta = _golden()
    model = _model("qwen4_exp", meta)
    k, ple = next((k, m) for k, m in model.named_modules()
                  if k.endswith("ple_embedding"))
    stored = mx.array([3, 5, 7][:ple._mults.shape[0]], dtype=mx.int64)
    model.sanitize({f"{k}.layer_multipliers": stored})
    assert ple._mults.tolist() == stored.tolist()
    # a cast (non-integer) copy is not trusted: the rebuild stays
    before = ple._mults.tolist()
    model.sanitize({f"{k}.layer_multipliers": stored.astype(mx.float32)})
    assert ple._mults.tolist() == before


# --- config variants: keys left out, layer_types, the router ---------------

@pytest.mark.parametrize("name", sorted(G.VARIANTS))
def test_a_config_variant_computes_the_reference_logits(name):
    """Each variant is a config.json shape the base configs do not take:
    keys left out (they mean the reference config's defaults), layer_types
    off the full_attention_interval pattern, norm_topk_prob false (ignored
    by Qwen3.5-MoE's router, honoured by Qwen4-Exp's)."""
    z, meta = _golden()
    family, _ = G.VARIANTS[name]
    got = _run(_model(family, meta, key=name), meta)
    diff = float(np.abs(got - z[f"{name}/logits"]).max())
    assert diff < TOL, f"{name}: max |logit diff| {diff:.3e} vs reference"


def _q4_args(**over):
    from knurlogic.engine import register
    register.register("qwen4_exp", override=True)
    arch = importlib.import_module("mlx_lm.models.qwen4_exp")
    _, meta = _golden()
    cfg = dict(meta["qwen4_exp/config"], **over)
    cfg = {k: v for k, v in cfg.items() if v is not None}
    return arch.ModelArgs.from_dict({"model_type": "qwen4_exp",
                                     "text_config": cfg})


@pytest.mark.parametrize("key", ["indexer_n_heads", "indexer_budget"])
def test_qwen4_exp_refuses_a_config_without_the_qsa_indexer(key):
    """The reference has no default for indexer_*: it cannot build a QSA
    layer without them, so neither does knurlogic (it used to fill in
    Flash-Next's values)."""
    with pytest.raises(ValueError, match="indexer"):
        _q4_args(**{key: None})


def test_qwen4_exp_refuses_ple_without_an_eos_token():
    with pytest.raises(ValueError, match="eos_token_id"):
        _q4_args(eos_token_id=None)


def test_qwen4_exp_without_ple_layer_ids_has_no_ple_layer():
    """ple_layer_ids left out is [] (no PLE), as the reference; it used to
    default to [2]."""
    a = _q4_args(ple_layer_ids=None, ple_embed_dim=None).text
    assert a.ple_layer_ids == [] and a.ple_embed_dim == a.hidden_size


# --- bfloat16 ----------------------------------------------------------------

#: bfloat16 on both sides, the same bf16-rounded weights. They cannot agree
#: bit for bit: the deltanet's q/k l2norm rounds differently (see
#: PROVENANCE.md, edit 8), as do attention's softmax and every matmul's
#: accumulation order, and a 4-layer model carries each bf16 step on. The
#: yardstick is what bf16 arithmetic alone does to the reference: its bf16
#: logits against its own float32 logits on the same weights (`noise`).
#: knurlogic's bf16 logits must sit closer to the reference's bf16 logits
#: than BF16_FRACTION of that, in RMS over all 25 x 64 logits (the max is
#: one row's top-k tie or so, too noisy to hold), and under the noise's
#: max. Measured 2026-10-03: 0.41-0.45 of the noise after edit 8,
#: 1.32-1.38 before it.
BF16_FRACTION = 0.6


def _rms(d):
    return float(np.sqrt(np.mean(np.square(d))))


@pytest.mark.parametrize("family", G.BF16)
def test_the_trunk_in_bfloat16_computes_the_reference_bfloat16_logits(family):
    z, meta = _golden()
    ref = z[f"{family}/bf16/logits"]
    noise = ref - z[f"{family}/bf16/f32_logits"]
    got = _run(_model(family, meta, dtype=mx.bfloat16,
                      init=G.bf16_weights_of), meta)
    rms, max_ = _rms(got - ref), float(np.abs(got - ref).max())
    assert rms < BF16_FRACTION * _rms(noise), (
        f"{family}: bf16 rms logit diff {rms:.3e} vs the reference's bf16; "
        f"its own bf16-vs-f32 rms {_rms(noise):.3e}")
    assert max_ < float(np.abs(noise).max()), (
        f"{family}: bf16 max logit diff {max_:.3e}, the reference's own "
        f"bf16-vs-f32 max {float(np.abs(noise).max()):.3e}")


#: one module, same bf16 weights and input on both sides. The norms and the
#: MLP / MoE block (router, experts, shared expert) must round exactly as
#: the reference. The deltanet differs only in its q/k l2norm: the
#: reference's torch fallback rounds it to bf16 four times (sum of squares,
#: + eps, sqrt then 1/x: CPU's bf16 rsqrt is two steps), knurlogic's
#: rms_norm once in float32 (FLA's kernel, what the reference runs on CUDA,
#: keeps it in float32); fed the reference's own normalized q/k, knurlogic's
#: recurrence gives its core output bit for bit. That leaves 1.17e-2 on
#: outputs up to 1.9 (under one bf16 step there, 2^-6 = 1.56e-2); before
#: vendored edit 8 (bf16 a + dt_bias, beta and the conv's silu rounded
#: inside MLX's kernels) it was 1.95e-2 to 2.34e-2.
COMPONENT_TOL = {"linear_attn": 1.5e-2}


@pytest.mark.parametrize("family,path", [
    (f, p) for f in G.BF16 for p, _ in G.COMPONENTS[f]])
def test_a_bfloat16_component_rounds_as_the_reference(family, path):
    z, meta = _golden()
    model = _model(family, meta, dtype=mx.bfloat16,
                   init=G.component_weights)
    mod = model.layers
    for p in path.split("."):
        mod = mod[int(p)] if p.isdigit() else getattr(mod, p)
    width = dict(G.COMPONENTS[family])[path]
    x = mx.array(G.component_input(width)).astype(mx.bfloat16)
    y = mod(x) if "linear_attn" not in path else mod(x, None, None)
    got = np.array(y.astype(mx.float32))
    ref = z[f"{family}/bf16/{path}"]
    kind = path.split(".")[-1]
    diff = float(np.abs(got - ref).max())
    if kind in COMPONENT_TOL:
        assert diff < COMPONENT_TOL[kind], f"{family} {path}: {diff:.3e}"
    else:   # a norm: the reference's rounding, bit for bit
        assert np.array_equal(got, ref), \
            f"{family} {path}: {(got != ref).mean():.0%} differ, max {diff:.3e}"


# --- vision: tower, image processor, rope_index ------------------------------

@pytest.mark.parametrize("family", ["qwen3_5", "qwen4_exp"])
def test_the_vision_tower_computes_the_reference_features(family):
    from knurlogic.engine.families.qwen.vision.vision import VisionConfig, VisionModel
    z, meta = _golden()
    tower = VisionModel(VisionConfig.from_dict(
        dict(meta["vision/config"], model_type=family)))
    tower.set_dtype(mx.float32)
    w = G.vision_weights(meta[f"vision/{family}/shapes"])
    w = tower.sanitize({k: mx.array(v) for k, v in w.items()})
    tower.load_weights(list(w.items()), strict=True)
    tower.eval()
    px, _ = G.vision_inputs()
    feats, _ = tower(mx.array(px), mx.array([G.VISION_GRID]))
    ref = z[f"vision/{family}/feats"]
    assert feats.shape == ref.shape
    diff = float(np.abs(np.array(feats) - ref).max())
    assert diff < TOL, f"{family} tower: max |diff| {diff:.3e}"


@pytest.mark.parametrize("tag", sorted(G.PROC_SIZES))
def test_the_image_processor_computes_the_reference_pixels(tag):
    """A non-square image (203x311) scaled up past min_pixels and down to
    max_pixels: the grid and every pixel value equal transformers' PIL
    backend (the numpy port knurlogic vendors). Its torchvision backend
    resizes with a different bicubic: there a pixel may be one uint8 step
    (2/255 after the 0.5/0.5 normalize) off."""
    from PIL import Image

    from knurlogic.engine.families.qwen.vision.processing import ImageProcessor
    z, meta = _golden()
    size = meta["proc/sizes"][tag]
    kw = {k: v for k, v in G.PROC.items() if k != "size"}
    proc = ImageProcessor(**kw, min_pixels=size["shortest_edge"],
                          max_pixels=size["longest_edge"])
    _, img = G.vision_inputs()
    pv, grid = proc(Image.fromarray(img))
    assert list(grid) == z[f"proc/{tag}/grid"][0].tolist()
    ref = G.from_levels(z[f"proc/{tag}/px"])
    assert pv.shape == ref.shape
    assert float(np.abs(pv - ref).max()) < 1e-6
    if f"proc/{tag}/tv_px" in z.files:
        tv = G.from_levels(z[f"proc/{tag}/tv_px"])
        assert float(np.abs(pv - tv).max()) <= 2 / 255 + 1e-6


@pytest.mark.parametrize("family", ["qwen3_5", "qwen4_exp"])
def test_rope_index_gives_the_reference_positions_on_two_images(family):
    from knurlogic.engine.families.qwen.vision.rope_index import rope_index
    z, _ = _golden()
    pos, delta = rope_index(G.ROPE_IDS, [tuple(g) for g in G.ROPE_GRIDS],
                            G.IMG, G.VS, G.VISION["spatial_merge_size"])
    assert np.array_equal(np.asarray(pos), z[f"rope/{family}/pos"])
    assert int(delta) == int(z[f"rope/{family}/delta"][0])


# --- the Qwen tool-call dialect through knurlogic's path ---------------------

def test_a_qwen_xml_tool_call_parses_through_knurlogics_path():
    """Qwen3.5/3.6 templates ask for <function=NAME><parameter=P>...: the
    parser knurlogic serves with is the one inferred from the shipped
    template (serve.tool_support, mlx-lm's rule), run by Request._parse_tool
    into OpenAI tool_calls with typed arguments."""
    from knurlogic.engine import model as engine
    from knurlogic.engine.runtime.request import Request
    tpl = (ROOT / "tests/support/goldens/qwen3_6_chat_template.jinja"
           ).read_text()
    name = engine.tool_support(tpl)["parser"]
    assert name == "qwen3_coder"
    parser = importlib.import_module(f"mlx_lm.tool_parsers.{name}")
    tools = [{"type": "function", "function": {
        "name": "get_weather", "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}, "days": {"type": "integer"},
            "units": {"type": "object"}}}}}]
    text = ("<function=get_weather>\n<parameter=city>\nSan Francisco, CA\n"
            "</parameter>\n<parameter=days>\n3\n</parameter>\n"
            "<parameter=units>\n{\"temp\": \"C\"}\n</parameter>\n</function>")
    req = Request.__new__(Request)
    req.tool_parser, req.tools, req._tool_idx = (
        parser.parse_tool_call, tools, 0)
    calls = req._parse_tool(text)
    assert len(calls) == 1 and calls[0]["type"] == "function"
    fn = calls[0]["function"]
    assert fn["name"] == "get_weather"
    assert json.loads(fn["arguments"]) == {
        "city": "San Francisco, CA", "days": 3, "units": {"temp": "C"}}
