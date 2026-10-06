"""DeepSeek-V4-Flash-Vision-Exp images (families/deepseek/vision and the
trunk's knurlogic edits 14-15), held to DeepSeek's own reference code.

The goldens (tests/support/goldens/deepseek_v4_vision.npz) are made by the
artifact's `inference/` code under torch
(tests/support/goldens/build_deepseek_v4_vision.py): its Gate with
bias_vl, its image-span window, its image processor and, on the real
`vision.*` / `aligner.*` weights, its ViT + aligner. The tower test reads
those weights from the artifact (set KNURLOGIC_TEST_DEEPSEEK_V4_VISION,
default the external drive copy) and skips without it; everything else runs
anywhere. Then a tiny random DeepSeek-V4 with vision end to end, the key
and the prompt.
"""
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests" / "support" / "goldens"))

import build_deepseek_v4_vision as G  # noqa: E402

REF = Path(os.environ.get(
    "KNURLOGIC_TEST_DEEPSEEK_V4_VISION",
    "/Volumes/Models/Teacher Models/"
    "deepseek-ai--DeepSeek-V4-Flash-Vision-Exp"))
GOLD = dict(np.load(G.OUT))


def _sha(a) -> np.ndarray:
    import hashlib
    return np.frombuffer(hashlib.sha256(
        np.ascontiguousarray(a, dtype=np.float32).tobytes()).digest(),
        dtype=np.uint8)


# ------------------------------------------------------------ no mlx

def test_the_processor_is_the_references():
    """Grid, patches (bit for bit), the block's types wherever it starts,
    and perm, on eight aspect ratios: square, both orientations, a small
    image (upscaled to min_pixels), a large one (safe_resize) and two past
    the 8:1 width ratio."""
    from knurlogic.engine.families.deepseek.vision import processor as P
    args = P.VisionArgs.from_config(
        {"vision_min_pixels": 147456, "vision_max_n_token": 384})
    for i, (w, h) in enumerate(G.SIZES):
        patches, nvh, nvw, nlh, nlw = P.load_image(G.synthetic(w, h, i), args)
        assert [w, h, nvh, nvw, nlh, nlw] == GOLD[f"proc{i}_grid"].tolist()
        assert (_sha(patches) == GOLD[f"proc{i}_patches_sha"]).all(), (w, h)
        for s in range(4):
            types, perm = P.build_image_block(nlh, nlw, 101 + s)
            assert types.tolist() == GOLD[f"proc{i}_types{s}"].tolist()
        assert perm.tolist() == GOLD[f"proc{i}_perm"].tolist()
        # the block itself does not move with its position; the pads do
        core, _ = P.image_block(nlh, nlw)
        assert (101 + P.compress_pad(101)) % 4 == 3
        assert types[-core.size:].tolist() == core.tolist()


def test_a_deepseek_v4_config_with_vision_layers_has_vision():
    from knurlogic.engine.vision import registry
    assert registry.has_vision_config({"vision_n_layers": 32})
    assert not registry.has_vision_config({"vision_n_layers": 0})
    assert not registry.has_vision_config({"model_type": "deepseek_v4"})
    assert registry.FAMILIES["deepseek_v4"] == \
        "knurlogic.engine.families.deepseek.vision:build"


def test_a_messages_parts_are_joined_as_deepseeks_encoder_joins_them():
    from knurlogic.engine.runtime.prompt import flatten
    from knurlogic.engine.vision.request import with_placeholders
    img = {"type": "image_url", "image_url": {"url": "data:,x"}}
    ms = [{"role": "user", "content": [{"type": "text", "text": "a"}, img,
                                       {"type": "text", "text": "b"}]},
          {"role": "user", "content": [img, {"type": "text", "text": "c"}]},
          {"role": "user", "content": [{"type": "text", "text": "d"},
                                       {"type": "text", "text": "e"}]}]
    out = flatten(with_placeholders(ms, ["<I>", "<I>"]), sep="\n\n")
    assert [m["content"] for m in out] == ["a\n\n<I>\n\nb", "<I>\n\nc",
                                           "d\n\ne"]
    out = flatten(with_placeholders(ms, ["<I>", "<I>"]))
    assert [m["content"] for m in out] == ["a<I>b", "<I>c", "de"]


# ------------------------------------------------------------ the prompt

def _vision_encoder():
    spec = importlib.util.spec_from_file_location(
        "encoding_dsv4_vision",
        ROOT / "tests/support/fixtures_deepseek_v4_vision/encoding_dsv4.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("mode,effort", [
    ("chat", None), ("thinking", None), ("thinking", "high"),
    ("thinking", "max")])
def test_the_vision_template_is_the_vision_exp_encoder(mode, effort):
    """The artifact's own example (two images in one user turn) and a
    system prompt, rendered the way knurlogic renders an image request:
    the placeholders spliced in by with_placeholders with the family's
    separator, flattened, then the template's Vision-Exp variant."""
    cu = pytest.importorskip("transformers.utils.chat_template_utils")
    from knurlogic.engine import templates
    from knurlogic.engine.families.deepseek.vision import (
        IMAGE_PLACEHOLDER,
    )
    from knurlogic.engine.runtime.prompt import flatten
    from knurlogic.engine.vision.request import with_placeholders
    official = _vision_encoder()
    case = json.loads((ROOT / "tests/support/fixtures_deepseek_v4_vision/"
                       "example_vl_harmony.json").read_text())[0]
    msgs = [{"role": "system", "content": "Be brief."}] + case["messages"]
    want, media = official.encode_messages(
        copy.deepcopy(msgs), thinking_mode=mode, reasoning_effort=effort,
        return_multi_modal_data=True)
    assert len(media["images"]) == 2
    flat = flatten(with_placeholders(copy.deepcopy(msgs),
                                     [IMAGE_PLACEHOLDER] * 2), sep="\n\n")
    kw = {"reasoning_effort": effort} if effort else {}
    out, _ = cu.render_jinja_template(
        conversations=[flat], chat_template=templates.text(
            "deepseek_v4_vision"),
        add_generation_prompt=True, bos_token=official.bos_token,
        thinking_mode=mode, **kw)
    assert out[0] == want


def _artifact(root, folder, **config):
    d = root / folder
    d.mkdir()
    (d / "config.json").write_text(json.dumps(config))
    return str(d)


_VISION_EXP = dict(model_type="deepseek_v4", vision_n_layers=32,
                   dspark_block_size=5)


def test_a_deepseek_v4_vision_artifact_gets_the_vision_template(tmp_path):
    from knurlogic.engine import templates
    name = _artifact(tmp_path, "deepseek-ai--DeepSeek-V4-Flash-Vision-Exp-mlx",
                     **_VISION_EXP)
    assert templates.family_for(None, name) == "deepseek_v4_vision"
    flash = _artifact(tmp_path, "DeepSeek-V4-Flash", model_type="deepseek_v4")
    assert templates.family_for(None, flash) == "deepseek_v4"
    vt = templates.text("deepseek_v4_vision")
    assert templates.served_family(vt) == "deepseek_v4_vision"
    assert templates.served_family(templates.text("deepseek_v4")) == \
        "deepseek_v4"
    from knurlogic.engine.serve import thinking
    assert thinking.detect(vt)[0] == "deepseek_vision_effort"
    assert [n["name"] for n in thinking.levels(vt)["native"]] == \
        ["off", "low", "high", "max"]
    assert thinking.levels(vt)["default"] == "low"
    assert thinking.detect(templates.text("deepseek_v4"))[0] == \
        "deepseek_effort"


def test_the_template_is_chosen_by_the_config_not_the_folder_name(tmp_path):
    """A Vision-Exp artifact in a folder without "vision" in its name gets
    the Vision-Exp template and its four effort levels (the folder name
    chose it before: such a folder got Flash's template and Flash's three
    levels); a text-only conversion that keeps only the DSpark fields is
    still Vision-Exp's encoder; a Flash artifact in a folder named
    "vision" stays Flash."""
    from types import SimpleNamespace

    from knurlogic.engine import templates
    from knurlogic.engine.serve import thinking
    plain = _artifact(tmp_path, "DeepSeek-V4-Flash-mlx-4bit", **_VISION_EXP)
    assert templates.family_for(None, plain) == "deepseek_v4_vision"
    tok = SimpleNamespace(chat_template=None, name_or_path=plain)
    assert templates.install(tok) == "deepseek_v4_vision"
    assert [n["name"] for n in thinking.levels(tok.chat_template)["native"]] \
        == ["off", "low", "high", "max"]
    # a copy of knurlogic's Flash template (a DSML marker) in that folder
    assert templates.family_for(templates.text("deepseek_v4"), plain) == \
        "deepseek_v4_vision"
    dspark = _artifact(tmp_path, "ds-text-only", model_type="deepseek_v4",
                       vision_n_layers=0, dspark_block_size=5)
    assert templates.family_for(None, dspark) == "deepseek_v4_vision"
    flash = _artifact(tmp_path, "my-vision-models", model_type="deepseek_v4")
    assert templates.family_for(None, flash) == "deepseek_v4"
    assert templates.family_for(None, str(tmp_path / "missing")) is None


def test_text_parts_are_joined_as_the_vision_encoder_joins_them():
    """Vision-Exp's encoder joins every list of content parts with "\n\n"
    (a text-only one too); Flash's template keeps mlx-lm's ""."""
    from types import SimpleNamespace

    from knurlogic.engine import templates
    from knurlogic.engine.runtime.prompt import flatten, part_separator
    cu = pytest.importorskip("transformers.utils.chat_template_utils")
    vis = SimpleNamespace(chat_template=templates.text("deepseek_v4_vision"),
                          name_or_path="/m/DeepSeek-V4-Flash-Vision-Exp")
    flash = SimpleNamespace(chat_template=templates.text("deepseek_v4"),
                            name_or_path="/m/DeepSeek-V4-Flash")
    assert part_separator(vis) == "\n\n"
    assert part_separator(flash) == ""
    msgs = [{"role": "user", "content": [{"type": "text", "text": "one"},
                                         {"type": "text", "text": "two"}]}]
    want = _vision_encoder().encode_messages(copy.deepcopy(msgs),
                                             thinking_mode="chat")
    out, _ = cu.render_jinja_template(
        conversations=[flatten(msgs, sep=part_separator(vis))],
        chat_template=vis.chat_template, add_generation_prompt=True,
        bos_token=_vision_encoder().bos_token, thinking_mode="chat")
    assert out[0] == want


# ------------------------------------------------------------ mlx

mx = pytest.importorskip("mlx.core")


def _arch():
    from knurlogic.engine import register
    register.register("deepseek_v4")
    import mlx_lm.models.deepseek_v4 as M
    return M


def _gate_args(M):
    return M.ModelArgs.from_dict(dict(
        model_type="deepseek_v4", vocab_size=G.V, hidden_size=16,
        n_routed_experts=8, num_experts_per_tok=2, num_hash_layers=1,
        scoring_func="sqrtsoftplus", routed_scaling_factor=1.5,
        norm_topk_prob=True, vision_n_layers=2, compress_ratios=[0, 0],
        num_hidden_layers=2))


@pytest.mark.parametrize("name,layer", [("hash", 0), ("score", 1)])
def test_bias_vl_routing_is_the_references(name, layer):
    """The reference Gate on 12 tokens, 4 of them image tokens: the same
    experts (as sets: the reference sorts its top-k, argpartition does not)
    with the same weights. Hash layers: tid2eid for text, top-k of
    scores + bias_vl for an image token; score layers: bias for text,
    bias_vl for images."""
    M = _arch()
    g = M.MoEGate(_gate_args(M), layer)
    g.weight = mx.array(GOLD[f"route_{name}_weight"])
    g.e_score_correction_bias = mx.array(GOLD[f"route_{name}_bias"])
    g.bias_vl = mx.array(GOLD[f"route_{name}_bias_vl"])
    if layer == 0:
        g.tid2eid = mx.array(GOLD["route_hash_tid2eid"])
    ids = mx.array(GOLD["route_ids"])[None]
    inds, w = g(mx.array(GOLD["route_x"])[None], ids, ids >= G.V)
    inds, w = np.array(inds[0]), np.array(w[0].astype(mx.float32))
    want_i = GOLD[f"route_{name}_indices"]
    want_w = GOLD[f"route_{name}_weights"]
    o, wo = np.argsort(inds, axis=-1), np.argsort(want_i, axis=-1)
    assert (np.take_along_axis(inds, o, -1)
            == np.take_along_axis(want_i, wo, -1)).all()
    assert np.allclose(np.take_along_axis(w, o, -1),
                       np.take_along_axis(want_w, wo, -1), atol=1e-5)
    # and bias_vl is what moved the image tokens: text rows as without it
    txt = GOLD["route_ids"] < G.V
    plain, _ = g(mx.array(GOLD["route_x"])[None],
                 mx.array(np.where(txt, GOLD["route_ids"], 0))[None],
                 mx.zeros(ids.shape, dtype=mx.bool_))
    assert (np.sort(np.array(plain[0])[txt], -1)
            == np.sort(inds[txt], -1)).all()


def test_the_image_span_window_is_the_references():
    """`image_visible` and the window mask against get_image_visible and
    get_window_topk_idxs_visible on a prompt with two images (one longer
    than the window, one longer than max_image_tokens), whole and as two
    chunks that keep each image whole."""
    M = _arch()
    ids = GOLD["mask_ids"]
    S = ids.size
    win = int(GOLD["mask_window"])
    mi = int(GOLD["mask_max_image_tokens"])
    left, right = M.image_visible(mx.array(ids)[None], G.V, mi)
    assert np.array(left[0]).tolist() == GOLD["mask_left"].tolist()
    assert np.array(right[0]).tolist() == GOLD["mask_right"].tolist()
    got = M._build_window_mask_visible(1, S, 0, win, S, left, right, mi)
    assert (np.array(got[0, 0]) == GOLD["mask_visible"]).all()
    z = mx.zeros_like(left)
    plain = M._build_window_mask_visible(1, S, 0, win, S, z, z, mi)
    assert (np.array(plain[0, 0]) == GOLD["mask_plain"]).all()
    assert (np.array(plain[0, 0]) == np.array(
        M._build_window_mask(1, S, 0, win, S)[0, 0])).all()
    for c in (5, 19):   # before the first image; between the two
        sub = mx.array(ids[c:])[None]
        lf, rt = M.image_visible(sub, G.V, mi)
        wl = (S - c) + min(c, win - 1)   # the rotating window's keys
        m = M._build_window_mask_visible(1, S - c, c, win, wl, lf, rt, mi)
        want = GOLD["mask_visible"][c:, S - wl:]
        assert (np.array(m[0, 0]) == want).all(), c
        off = M._build_window_mask_visible(1, S - c, mx.array([c]), win, wl,
                                           lf, rt, mi)
        assert (np.array(off[0, 0]) == want).all(), c


@pytest.mark.skipif(not (REF / "model.safetensors.index.json").is_file(),
                    reason="the DeepSeek-V4-Flash-Vision-Exp artifact is not "
                           "mounted (set KNURLOGIC_TEST_DEEPSEEK_V4_VISION)")
def test_the_tower_on_the_real_weights_is_the_references():
    """The family's own load (one tensor at a time, vision.* / aligner.* and
    the four image rows only) and preprocess, then the ViT + aligner in
    float32 against the reference's float32 output (stored float16), and in
    bfloat16 -- what serves -- within bf16's error. Then the block encode
    builds: the learned rows, and the aligner's rows in perm order."""
    from PIL import Image

    from knurlogic.engine.families.deepseek.vision import (
        DeepseekVisionFamily,
        processor,
    )
    cfg = json.loads((REF / "config.json").read_text())
    fam = DeepseekVisionFamily(cfg, 129264)
    n = fam.load_weights(str(REF))
    assert n == 263 + 4
    img = Image.open(REF / "inference/examples/images/carrots.jpeg") \
        .convert("RGB").resize(G.TOWER_SIZE)
    import io
    img = Image.open(io.BytesIO(G._png(img)))
    pixels, ref = fam.preprocess(img, "0" * 64)
    nvh, nvw = pixels["n_vit"]
    assert [nvh, nvw, *pixels["n_llm"]] == GOLD["tower_grid"].tolist()
    assert (_sha(pixels["patches"]) == GOLD["tower_patches_sha"]).all()
    want = GOLD["tower_out"].astype(np.float32)
    scale = np.abs(want).max()

    fam.tower.set_dtype(mx.float32)
    got = np.array(fam.tower(mx.array(pixels["patches"]), nvh, nvw))
    err32 = np.abs(got - want).max()
    # float16 storage alone is ~scale * 2**-11
    assert err32 < scale * 2e-3, (err32, scale)

    fam.tower.set_dtype(mx.bfloat16)
    enc = fam.encode(pixels, ref)
    out = np.array(fam.tower(mx.array(pixels["patches"]).astype(mx.bfloat16),
                             nvh, nvw).astype(mx.float32))
    # bf16 moves a few rows far from float32 (row cosine ~0.4 at worst on
    # this image) -- the reference's own bf16 run does the same; ours must
    # stay as close to float32 as the reference's bf16 does
    cos = (out * want).sum(-1) / (np.linalg.norm(out, axis=-1)
                                  * np.linalg.norm(want, axis=-1))
    ref16 = GOLD["tower_bf16_cos"]
    assert cos.mean() > ref16.mean() - 0.005, (cos.mean(), ref16.mean())
    assert np.median(cos) > np.median(ref16) - 1e-3
    print(f"\ntower max |mlx - torch|: float32 {err32:.3e} (max |ref| "
          f"{scale:.3f}); bfloat16 row cosine to float32 mean "
          f"{cos.mean():.4f} min {cos.min():.4f} (torch bfloat16: mean "
          f"{ref16.mean():.4f} min {ref16.min():.4f})")

    types, perm = processor.image_block(*pixels["n_llm"])
    feats = np.array(enc.feats.astype(mx.float32))
    assert feats.shape == (ref.n_tokens, cfg["hidden_size"])
    rows = np.array(fam.rows.astype(mx.float32))
    for t in (0, 1, 3, 4):
        assert (feats[types == t] == rows[t]).all()
    assert (feats[types == processor.IMAGE] == out[perm]).all()


# ------------------------------------------------------------ tiny end to end

TINY = dict(
    model_type="deepseek_v4", vocab_size=64, hidden_size=64,
    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=1,
    q_lora_rank=32, o_lora_rank=16, o_groups=2, head_dim=32,
    qk_rope_head_dim=16, sliding_window=8, compress_ratios=[4, 8, 4, 0],
    index_n_heads=8, index_head_dim=16, index_topk=64,
    moe_intermediate_size=32, n_routed_experts=4, n_shared_experts=1,
    num_experts_per_tok=2, num_hash_layers=1, hc_mult=4,
    hc_sinkhorn_iters=3, max_position_embeddings=1024,
    num_nextn_predict_layers=0, rms_norm_eps=1e-20,
    vision_n_layers=2, vision_dim=32, vision_n_heads=2,
    vision_inter_dim=48, vision_patch_size=14, vision_rope_theta=10000.0,
    vision_downsample_ratio=3, vision_max_n_token=40,
    vision_min_pixels=42 * 42 * 4, vision_max_wh_ratio=8)
IMG_ID = 60     # the placeholder's id in the tiny vocab


def _random(module, rng):
    import mlx.nn as nn  # noqa: F401
    from mlx.utils import tree_flatten, tree_unflatten
    w = []
    for k, v in tree_flatten(module.parameters()):
        if k.endswith("tid2eid"):
            a = rng.integers(0, TINY["n_routed_experts"], size=v.shape)
            w.append((k, mx.array(a.astype(np.int32))))
        elif "switch_mlp" in k and v.dtype == mx.uint8:
            w.append((k, mx.array(rng.integers(118, 124, size=v.shape)
                                  .astype(np.uint8))))
        elif "switch_mlp" in k:
            w.append((k, mx.array(rng.integers(0, 2 ** 32, size=v.shape,
                                               dtype=np.uint64)
                                  .astype(np.uint32))))
        elif k.endswith("norm.weight") or k.endswith("norm1.weight") \
                or k.endswith("norm2.weight"):
            w.append((k, mx.array((1 + 0.1 * rng.standard_normal(v.shape))
                                  .astype(np.float32))))
        else:
            w.append((k, mx.array((0.15 * rng.standard_normal(v.shape))
                                  .astype(np.float32))))
    module.update(tree_unflatten(w))


@pytest.fixture(scope="module")
def tiny():
    from knurlogic.engine.families.deepseek.vision import DeepseekVisionFamily
    M = _arch()
    rng = np.random.default_rng(0)
    model = M.Model(M.ModelArgs.from_dict(TINY))
    _random(model, rng)
    fam = DeepseekVisionFamily(TINY, IMG_ID)
    _random(fam._build_tower(), rng)
    core = model.model
    fam.set_rows({k: getattr(core, k) for k in
                  ("image_start", "image_pad", "image_newline", "image_end")})
    mx.eval(model.parameters(), fam.tower.parameters(), fam.rows)
    return M, model, fam


def _prompt(fam, images):
    """Two images in a text prompt -> (key, features, the reference's ids):
    the serve path's expansion (expand_pads, expand, frame_key) beside the
    reference's prepare_vl_inputs (build_image_block at len(tokens))."""
    from knurlogic.engine.families.deepseek.vision import processor as P
    from knurlogic.engine.vision import key as K
    text = [[3, 17, 42, 5, 9], [11, 48], [27, 7, 30]]
    ids = text[0] + [IMG_ID] + text[1] + [IMG_ID] + text[2]
    refs, store = [], {}
    for i, img in enumerate(images):
        pixels, ref = fam.preprocess(img, f"{i:064x}")
        store[(ref.sha, ref.proc_hash)] = fam.encode(pixels, ref)
        refs.append(ref)
    key = K.expand(K.expand_pads(ids, refs, IMG_ID), refs, IMG_ID)
    key, segs = fam.frame_key(key, [key])
    assert segs == [key]
    toks = []
    for t, ref in zip(text, refs + [None]):
        toks += t
        if ref is not None:
            types, _ = P.build_image_block(*ref.grid_thw[1:], len(toks))
            toks += (TINY["vocab_size"] + types).tolist()
    return key, (lambda sha, ph: store[(sha, ph)]), toks


def test_a_tiny_vision_model_end_to_end(tiny):
    """A tiny random DeepSeek-V4 with vision (2-layer tower): two images
    through preprocess, encode, the key and embed, then the trunk.

    * the ids embed hands the trunk are the reference's prepare_vl_inputs;
    * the trunk given `vl_ids` beside the placeholder ids computes what it
      computes from those ids alone (ids >= vocab_size in `inputs`, the
      embeddings still the family's);
    * prefilled in chunks that keep each image whole (the family's
      chunk_boundaries), it ends where the one-call prefill ends;
    * bias_vl moves the image tokens: zeroing it changes the logits."""
    from PIL import Image

    from knurlogic.engine.vision import key as K
    M, model, fam = tiny
    imgs = [Image.fromarray(np.asarray(G.synthetic(w, h, i))) for i, (w, h)
            in enumerate([(120, 90), (60, 200)])]
    key, feats, want = _prompt(fam, imgs)
    got = fam.embed(model, key, 0, feats)
    vl = got["vl_ids"]
    assert np.array(vl[0]).tolist() == want
    spans = fam.chunk_boundaries(key)
    assert len(spans) == 2
    for s, e in spans:
        assert want[s] == 64 and want[e - 1] == 68 and s % 4 == 3
    ids = mx.array(K.to_ids(key, IMG_ID))[None]
    emb = got["input_embeddings"]

    full = model(ids, cache=model.make_cache(), input_embeddings=emb,
                 vl_ids=vl)
    alone = model(mx.array(want)[None], cache=model.make_cache(),
                  input_embeddings=emb)
    assert np.allclose(np.array(full), np.array(alone), atol=1e-5)

    cache = model.make_cache()
    edges = [0, spans[0][0] - 1, spans[1][0], len(key)]
    for a, b in zip(edges, edges[1:]):
        assert not any(s < a < e for s, e in spans)
        last = model(ids[:, a:b], cache=cache,
                     input_embeddings=emb[:, a:b], vl_ids=vl[:, a:b])
    assert np.abs(np.array(last[0, -1]) - np.array(full[0, -1])).max() < 1e-3

    # the image-span window reaches the attention: with every span seen as
    # plain text (left = right = 0) a token before the first image's end
    # computes differently, and the text before the first image does not
    real = M.image_visible
    try:
        M.image_visible = lambda i, v, m: (mx.zeros(i.shape, mx.int32),) * 2
        causal = model(ids, cache=model.make_cache(), input_embeddings=emb,
                       vl_ids=vl)
    finally:
        M.image_visible = real
    d = np.abs(np.array(causal[0]) - np.array(full[0])).max(-1)
    assert d[:spans[0][0]].max() < 1e-5
    assert d[spans[0][0]:spans[0][1] - 1].max() > 1e-3

    # the image rows the trunk embeds on its own: the four learned rows
    own = model.model.embed(mx.array(want)[None])
    t = np.array(want) - 64
    for typ in (0, 1, 3, 4):
        sel = t == typ
        assert np.allclose(np.array(own[0])[sel], np.array(emb[0])[sel])

    keep = [lyr.ffn.gate.bias_vl for lyr in model.layers]
    try:
        for lyr in model.layers:
            lyr.ffn.gate.bias_vl = 50 * mx.arange(4, dtype=mx.float32)
        moved = model(ids, cache=model.make_cache(), input_embeddings=emb,
                      vl_ids=vl)
        assert np.abs(np.array(moved) - np.array(full)).max() > 1e-3
        # text alone never reads bias_vl
        txt = mx.array([[3, 17, 42, 5, 9, 11, 48]])
        a = model(txt, cache=model.make_cache())
        for lyr, b in zip(model.layers, keep):
            lyr.ffn.gate.bias_vl = b
        b = model(txt, cache=model.make_cache())
        assert np.allclose(np.array(a), np.array(b))
    finally:
        for lyr, b in zip(model.layers, keep):
            lyr.ffn.gate.bias_vl = b


def test_the_trunk_keeps_bias_vl_and_the_image_rows(tiny):
    """sanitize on the HF names: bias_vl keeps its name, every layer's
    bias (hash layers too) becomes e_score_correction_bias, the image rows
    move under model., vision.* / aligner.* / mtp.* are dropped -- and the
    result loads strict."""
    from mlx.utils import tree_flatten
    M, model, _ = tiny
    hf = {}
    for k, v in tree_flatten(model.parameters()):
        k = k.replace("model.layers.", "layers.")
        k = k.replace("ffn.gate.e_score_correction_bias", "ffn.gate.bias")
        k = {"model.image_start": "image_start", "model.image_end":
             "image_end", "model.image_newline": "image_newline",
             "model.image_pad": "image_pad"}.get(k, k)
        hf[k] = v
    assert "layers.0.ffn.gate.bias" in hf and "layers.0.ffn.gate.bias_vl" in hf
    hf["vision.norm.weight"] = mx.ones((32,))
    hf["aligner.w1.bias"] = mx.ones((64,))
    hf["mtp.0.norm.weight"] = mx.ones((64,))
    out = model.sanitize(dict(hf))
    assert "model.layers.0.ffn.gate.bias_vl" in out
    assert "model.layers.0.ffn.gate.e_score_correction_bias" in out
    assert "model.image_newline" in out
    assert not any(k.startswith(("vision.", "aligner.", "mtp."))
                   for k in out)
    fresh = M.Model(M.ModelArgs.from_dict(TINY))
    fresh.load_weights(list(out.items()), strict=True)


def test_the_norm_eps_is_the_configs(tiny):
    """Vision-Exp's rms_norm_eps is 1e-20: every norm DeepSeek's reference
    builds from norm_eps reads it."""
    M, model, _ = tiny
    eps = TINY["rms_norm_eps"]
    for lyr in model.layers:
        assert lyr.attn_norm.eps == lyr.ffn_norm.eps == eps
        assert lyr.attn.eps == lyr.attn.q_norm.eps == lyr.attn.kv_norm.eps \
            == eps
        assert lyr.hc_attn.norm_eps == lyr.hc_ffn.norm_eps == eps
        if lyr.attn.compress_ratio:
            assert lyr.attn.compressor.norm.eps == eps
        if getattr(lyr.attn, "indexer", None) is not None:
            assert lyr.attn.indexer.compressor.norm.eps == eps
    assert model.model.norm.eps == model.model.hc_head.norm_eps == eps


def test_frame_key_pads_each_image_to_its_alignment(tiny):
    """frame_key's pads are the reference's: IMAGE_START at a position == 3
    mod 4, the pads inside the image's own segment."""
    from knurlogic.engine.vision import ImageRef
    from knurlogic.engine.vision import key as K
    _, _, fam = tiny
    for lead in range(6):
        r = ImageRef(sha="a" * 64, proc_hash=fam.spec.proc_hash, n_tokens=5,
                     grid_thw=(1, 1, 1))
        segs = [[1] * lead, [IMG_ID, 2]]
        key, sk = K.expand_segments(segs, [r], IMG_ID)
        key, sk = fam.frame_key(key, sk)
        s = K.image_spans(key)[0].start
        assert s % 4 == 3 and len(key) == lead + (3 - lead % 4) + 5 + 1
        assert sk[0] == [1] * lead
        assert sk[1][:s - lead] == [64 + 1] * (s - lead)


def test_vision_tag_needs_a_tower_in_the_artifact(tmp_path):
    """DeepSeek-V4-Flash and its Vision-Exp share model_type deepseek_v4,
    and a text-only conversion of Vision-Exp keeps vision_n_layers: only
    the artifact whose index names the tower's tensors is a vision one."""
    import json

    from knurlogic.engine.vision import registry
    vcfg = {"model_type": "deepseek_v4", "vision_n_layers": 32}
    rigs = {"flash": ({"model_type": "deepseek_v4"}, ["model.norm.weight"]),
            "teacher": (vcfg, ["model.norm.weight"]),
            "vision": (vcfg, ["model.norm.weight", "vision.norm.weight",
                              "model.layers.0.ffn.gate.bias_vl",
                              "model.image_start", "model.image_end",
                              "model.image_newline", "model.image_pad"])}
    for name, (cfg, keys) in rigs.items():
        d = tmp_path / name
        d.mkdir()
        (d / "config.json").write_text(json.dumps(cfg))
        (d / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {k: "model.safetensors" for k in keys}}))
    assert registry.registered("deepseek_v4")
    assert not registry.registered("deepseek_v4", tmp_path / "flash")
    assert not registry.registered("deepseek_v4", tmp_path / "teacher")
    assert registry.registered("deepseek_v4", tmp_path / "vision")
    assert registry.build("deepseek_v4", str(tmp_path / "teacher"), None) \
        is None



# ------------------------------------- a conversion without vision weights

def _checkpoint(root, name, *, trunk_vision, extra=()):
    """A tiny Vision-Exp checkpoint on disk: config.json says
    vision_n_layers=2 always; the weights are the trunk built with
    (`trunk_vision`) or without its vision tensors, plus `extra` names."""
    from mlx.utils import tree_flatten
    M = _arch()
    cfg = dict(TINY, vision_n_layers=TINY["vision_n_layers"]
               if trunk_vision else 0)
    model = M.Model(M.ModelArgs.from_dict(cfg))
    _random(model, np.random.default_rng(1))
    w = dict(tree_flatten(model.parameters()))
    for k in extra:
        w[k] = mx.ones((8,))
    d = root / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(TINY))
    mx.save_safetensors(str(d / "model.safetensors"), w)
    return d


def _load(monkeypatch, path):
    """engine/serve/load.load_unlocked with mlx-lm's load_model under it
    (the tiny checkpoint has no tokenizer)."""
    import mlx_lm.utils as U

    from knurlogic.engine import templates
    from knurlogic.engine.serve.load import load_unlocked

    def load(p, model_config=None, **kw):
        return U.load_model(Path(p), model_config=model_config)[0], object()
    monkeypatch.setattr(U, "load", load)
    monkeypatch.setattr(templates, "install", lambda tok: None)
    return load_unlocked(str(path))[0]


def test_a_vision_config_with_no_vision_weights_loads_text_only(
        tmp_path, monkeypatch):
    """vision_n_layers in config.json but no tower, no bias_vl, no image
    rows and no hash-layer bias in the weights (the -mlx conversion of
    Vision-Exp): it loads as the text model and refuses images."""
    from knurlogic.engine.serve import state, vision
    from knurlogic.engine.vision import registry
    d = _checkpoint(tmp_path, "text", trunk_vision=False)
    v = registry.vision_weights(TINY, d)
    assert v["state"] == "text_only"
    assert v["text_config"] == {"vision_n_layers": 0}
    model = _load(monkeypatch, d)
    assert model.args.vision_n_layers == 0
    assert not hasattr(model.layers[0].ffn.gate, "bias_vl")
    assert registry.unavailable_why("deepseek_v4", d) == \
        registry.NO_VISION_WEIGHTS
    assert not registry.registered("deepseek_v4", d)

    class Provider:
        pass
    p = Provider()
    p.model = model
    try:
        assert vision.bind(str(d), p) is None
        assert vision.vision_status() == {
            "on": False, "error": registry.NO_VISION_WEIGHTS}
        why = vision.no_vision_why()
        assert "reads text only" in why and registry.NO_VISION_WEIGHTS in why
    finally:
        vision.clear()
        state.VISION.update(error="")


def test_a_conversion_without_the_tower_but_with_the_trunks_vision_tensors_loads_text_only(  # noqa: E501
        tmp_path, monkeypatch):
    """Every vision-only trunk tensor kept (bias_vl, the image rows, the
    hash layers' bias) but no tower: built as the config says, so those
    load (text never reads them), and images are refused. Before the
    text-only change such a conversion loaded as text; refusing it as
    'partial' would have broken it."""
    from knurlogic.engine.vision import registry
    d = _checkpoint(tmp_path, "no-tower", trunk_vision=True)
    v = registry.vision_weights(TINY, d)
    assert v["state"] == "text_only" and v["text_config"] == {}
    model = _load(monkeypatch, d)
    assert model.args.vision_n_layers == TINY["vision_n_layers"]
    assert hasattr(model.layers[0].ffn.gate, "bias_vl")
    assert registry.unavailable_why("deepseek_v4", d) == \
        registry.NO_VISION_WEIGHTS
    assert not registry.registered("deepseek_v4", d)


def test_a_conversion_with_part_of_its_vision_weights_is_refused(
        tmp_path, monkeypatch):
    """The tower without the trunk's vision tensors is neither model:
    refused, saying what is missing."""
    from knurlogic.engine.vision import registry
    no_trunk = _checkpoint(tmp_path, "no-trunk", trunk_vision=False,
                           extra=["vision.norm.weight"])
    for d, missing in ((no_trunk, "gate.bias_vl"),):
        v = registry.vision_weights(TINY, d)
        assert v["state"] == "partial" and missing in v["why"]
        with pytest.raises(RuntimeError, match="only part of its vision"):
            _load(monkeypatch, d)
        assert registry.unavailable_why("deepseek_v4", d) == v["why"]


def test_the_full_conversion_still_has_vision(tmp_path, monkeypatch):
    from knurlogic.engine.vision import registry
    d = _checkpoint(tmp_path, "full", trunk_vision=True,
                    extra=["vision.norm.weight", "aligner.w1.bias"])
    assert registry.vision_weights(TINY, d)["state"] == "full"
    model = _load(monkeypatch, d)
    assert model.args.vision_n_layers == TINY["vision_n_layers"]
    assert hasattr(model.layers[0].ffn.gate, "bias_vl")
    assert hasattr(model.model, "image_pad")
    assert registry.unavailable_why("deepseek_v4", d) == ""
    assert registry.registered("deepseek_v4", d)
