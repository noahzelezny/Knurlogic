"""gemma4 vision (e4b, 26b): gates G1-G5 plus the chunk-snap identity
check, held to HF transformers' own Gemma 4 vision code. Runs without torch:
the goldens are pre-built .npz (`tests/support/goldens/build_gemma4.py`,
run in an interpreter with torch and transformers).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

mx = pytest.importorskip("mlx.core")

import fixtures_vision as fv  # noqa: E402
import fixtures_vision_gemma4 as g4fv  # noqa: E402

from knurlogic.engine.families.gemma4.vision import Gemma4Vision  # noqa: E402
from knurlogic.engine.vision import ImageRef  # noqa: E402
from knurlogic.engine.vision.key import expand  # noqa: E402


def _tiny_family() -> Gemma4Vision:
    cfg = g4fv.tiny_gemma4_config()
    return g4fv.build_gemma4_family(cfg)


# --- G1: load_weights loads a STANDALONE tower, tensor count matches -------

def test_g1_load_weights_standalone_tower(tmp_path):
    fam = _tiny_family()
    n_written, _ = g4fv.write_tiny_tower_safetensors(
        tmp_path / "model.safetensors", fam)

    fresh = _tiny_family()
    before = mx.array(fresh.vision_tower.encoder.layers[0].self_attn.q_proj
                      .linear.weight)
    n_loaded = fresh.load_weights(str(tmp_path))
    after = fresh.vision_tower.encoder.layers[0].self_attn.q_proj.linear.weight

    assert n_loaded == n_written
    assert not mx.allclose(before, after).item()  # the fixture is not zeros


def test_g1_load_weights_can_fail(tmp_path):
    """The gate can fail: point it at an empty directory (no tensors)."""
    fam = _tiny_family()
    (tmp_path / "config.json").write_text("{}")
    n = fam.load_weights(str(tmp_path))
    assert n == 0  # RED if load_weights silently claimed tensors it never read


def _write_quantized_embed(tmp_path):
    """An artifact whose embed_vision projection is stored 8-bit affine and
    whose tower is plain -- the layout gemma e4b VQ ships."""
    import json

    import mlx.nn as nn
    from mlx.utils import tree_flatten
    fam = _tiny_family()
    gs = 32
    nn.quantize(fam.embed_vision, group_size=gs, bits=8)
    w = {f"vision_tower.{k}": v for k, v in
         tree_flatten(fam.vision_tower.parameters())}
    w.update({f"embed_vision.{k}": v for k, v in
              tree_flatten(fam.embed_vision.parameters())})
    mx.save_safetensors(str(tmp_path / "model.safetensors"), w)
    (tmp_path / "config.json").write_text(json.dumps(
        {"quantization": {"group_size": gs, "bits": 8, "mode": "affine"}}))
    return fam


def test_g1_loads_a_quantized_embedder(tmp_path):
    src = _write_quantized_embed(tmp_path)
    fresh = _tiny_family()
    fresh.load_weights(str(tmp_path))
    proj = fresh.embed_vision.embedding_projection
    assert hasattr(proj, "scales")
    assert mx.array_equal(proj.scales,
                          src.embed_vision.embedding_projection.scales).item()


def test_g1_quantized_embedder_can_fail(tmp_path, monkeypatch):
    """Without the quantize step the same artifact refuses to load -- the
    e4b failure this guards ("no parameter named scales")."""
    import knurlogic.engine.families.gemma4.vision as g4
    _write_quantized_embed(tmp_path)
    monkeypatch.setattr(g4, "quantize_like", lambda *a, **k: 0)
    with pytest.raises(ValueError, match="scales"):
        _tiny_family().load_weights(str(tmp_path))


def test_placeholder_is_framed_with_the_artifacts_own_tokens(tmp_path):
    """The placeholder must be the strings the artifact's tokenizer maps to
    boi / image / eoi, or the prompt carries no image token at all (the
    e4b gate failure: 1 image, 0 placeholders)."""
    import json

    from knurlogic.engine.families.gemma4.vision import build
    t = fv.tiny_ids("gemma4")
    (tmp_path / "tokenizer.json").write_text(json.dumps({"added_tokens": [
        {"id": t["boi_token_id"], "content": "<B>"},
        {"id": t["image_token_id"], "content": "<P>"},
        {"id": t["eoi_token_id"], "content": "<E>"}]}))
    cfg = fv.tiny_config("gemma4")
    fam = build(str(tmp_path), None, cfg)
    ref = ImageRef(sha="x", proc_hash=fam.spec.proc_hash, n_tokens=9)
    assert fam.placeholder_text(ref) == "<B><P><E>"


# --- G2: encode() matches the reference tower, on the SAME weights --------
# The golden is HF transformers' own Gemma4VisionModel +
# Gemma4MultimodalEmbedder (tests/support/goldens/build_gemma4.py), fed the
# image as HF's processor patchifies it.

def _hf_family(name):
    """A Gemma4Vision built from the golden's tower config, carrying HF's
    weights for that tower and embedder."""
    import json

    from mlx.utils import tree_flatten, tree_unflatten
    arrays, _ = fv.load_golden("gemma4_vision_tower")
    pre = f"tower/{name}/"
    vcfg = json.loads(str(arrays[pre + "vcfg"]))
    fam = Gemma4Vision(vcfg, int(arrays[pre + "text_hidden"]), 258880,
                       255999, 258882)
    w = {k[len(pre + "w/"):]: mx.array(v) for k, v in arrays.items()
         if k.startswith(pre + "w/")}
    tw = {k[len("vision_tower."):]: v for k, v in w.items()
          if k.startswith("vision_tower.")}
    ew = {k[len("embed_vision."):]: v for k, v in w.items()
          if k.startswith("embed_vision.")}
    have = {k for k, _ in tree_flatten(fam.vision_tower.parameters())}
    assert have == set(tw), f"tower param tree differs from HF's: {have ^ set(tw)}"
    fam.vision_tower.update(tree_unflatten(list(tw.items())))
    fam.embed_vision.update(tree_unflatten(list(ew.items())))
    mx.eval(fam.vision_tower.parameters(), fam.embed_vision.parameters())
    return fam, arrays, pre


@pytest.mark.parametrize("name", ["e4b", "g26"])
def test_g2_encode_matches_reference_tower_golden(name):
    """e4b-style (clipped linears) and 26B-style (standardize; a 24-wide
    head, 12 per rope axis) towers, float32, against HF."""
    fam, arrays, pre = _hf_family(name)
    tower = fam.vision_tower(mx.array(arrays[pre + "img"])[None])
    proj = fam.embed_vision(tower)
    np.testing.assert_allclose(np.array(tower[0]), arrays[pre + "tower"],
                               atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.array(proj[0]), arrays[pre + "proj"],
                               atol=1e-4, rtol=1e-4)


def test_g2_encode_can_fail():
    """Break the tower (pooling kernel lied about) and confirm the
    reference comparison goes red."""
    fam, arrays, pre = _hf_family("e4b")
    fam.vision_tower.pooling_kernel_size = 1  # break: reference used 3
    out = fam.vision_tower(mx.array(arrays[pre + "img"])[None])
    assert out.shape[1:] != arrays[pre + "tower"].shape or not np.allclose(
        np.array(out[0]), arrays[pre + "tower"], atol=1e-4)


def _pooled(dtype_name):
    """Our VisionModel's tail (pool, sqrt(hidden), strip, standardize,
    cast) on the golden's hidden states: the patch embedder and encoder
    are stubbed to hand them over, as the golden stubs HF's."""
    from knurlogic.engine.families.gemma4.vision.config import VisionConfig
    from knurlogic.engine.families.gemma4.vision.vision import VisionModel
    arrays, _ = fv.load_golden("gemma4_vision_tower")
    pre = f"pool/{dtype_name}/"
    dtype = getattr(mx, dtype_name)
    hidden = mx.array(arrays[pre + "hidden"]).astype(dtype)
    ph, pw = arrays[pre + "grid"].tolist()
    vc = VisionConfig(hidden_size=hidden.shape[-1], intermediate_size=16,
                      num_hidden_layers=0, num_attention_heads=1,
                      num_key_value_heads=1, head_dim=8, patch_size=16,
                      pooling_kernel_size=3, standardize=True)
    tower = VisionModel(vc)
    tower.std_bias = mx.array(arrays[pre + "std_bias"]).astype(dtype)
    tower.std_scale = mx.array(arrays[pre + "std_scale"]).astype(dtype)
    tower.patch_embedder = lambda pv, pos, pad: hidden
    tower.encoder = lambda h, pos, mask: h
    out = tower(mx.zeros((1, 3, ph * 16, pw * 16), dtype=dtype))
    return out[0], arrays[pre + "out"]


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_the_pooler_scales_and_standardizes_in_float32(dtype_name):
    """HF scales the pooled features by sqrt(hidden) in float32 and
    standardizes in float32 before casting back (modeling_gemma4.py
    Gemma4VisionPooler.forward, Gemma4VisionModel.forward). On these
    activations float16 overflows if the scale is done in float16, and
    bf16 rounds sqrt(768) to 27.75 (knurlogic edit 6)."""
    got, want = _pooled(dtype_name)
    assert got.dtype == getattr(mx, dtype_name)
    got = np.array(got.astype(mx.float32))
    assert np.isfinite(got).all()
    # Within a step of the working dtype of HF's output, but for the few
    # (<1%) whose 3x3 average, summed in float32 in another order, rounds
    # to the neighbouring value before the scale -- std_bias then cancels
    # most of the magnitude and leaves that step large.
    off = np.abs(got - want) > 2 ** -7 * np.abs(want)
    assert off.mean() < 0.01, off.mean()


def _ops():
    gold = np.load(ROOT / "tests/support/goldens/gemma4_text.npz")

    def bf16(key):
        bits = gold[key].astype(np.uint32) << 16
        return mx.array(bits.view(np.float32)).astype(mx.bfloat16)
    return bf16


def test_the_encoder_norm_rounds_where_the_reference_does():
    """HF's Gemma4RMSNorm in bf16: float32 inside, one rounding (knurlogic
    edit 7; mx.fast.rms_norm on bf16 left a quarter of these a step
    off)."""
    from knurlogic.engine.families.gemma4.vision.vision import RMSNorm
    bf16 = _ops()
    norm = RMSNorm(1024, eps=1e-6)
    norm.weight = bf16("ops/w")
    got = np.array(norm(bf16("ops/x")).astype(mx.float32))
    want = np.array(bf16("ops/norm").astype(mx.float32))
    assert (got != want).mean() < 1e-3, (got != want).mean()


def test_the_encoder_mlp_activation_rounds_where_the_reference_does():
    """gelu_pytorch_tanh(gate) * up in bf16, as torch rounds it (knurlogic
    edit 7): within two bf16 steps everywhere but the saturated tail,
    where mlx's tanh reaches -1 first (0 against ~1e-6)."""
    from knurlogic.engine.families.gemma4.vision.vision import gelu_mul
    bf16 = _ops()
    got = gelu_mul(bf16("ops/gate"), bf16("ops/up")).astype(mx.float32)
    want = bf16("ops/act").astype(mx.float32)
    diff = np.abs(np.array(got - want))
    step = np.abs(np.array(want)) * 2 ** -7
    assert (diff > 0).mean() < 0.01
    assert ((diff <= 2 * step) | (diff < 1e-5)).all()


# --- G3: chunk_boundaries is exactly the image spans ------------------------

def test_g3_chunk_boundaries_is_every_image_span():
    fam = _tiny_family()
    ref1 = ImageRef(sha="a" * 8, proc_hash=fam.spec.proc_hash, n_tokens=3)
    ref2 = ImageRef(sha="b" * 8, proc_hash=fam.spec.proc_hash, n_tokens=2)
    ids = [1, 2, fam.image_token_id, fam.image_token_id, fam.image_token_id,
          3, fam.image_token_id, fam.image_token_id, 4]
    key = expand(ids, [ref1, ref2], fam.image_token_id)
    assert fam.chunk_boundaries(key) == [(2, 5), (6, 8)]


def test_g3_chunk_boundaries_can_fail():
    fam = _tiny_family()
    key = [1, 2, 3]  # no image tokens
    boundaries = fam.chunk_boundaries(key)
    assert boundaries == []
    # break: an implementation that always returned one span would go red
    assert not (boundaries == [(0, 3)])


# --- G4: the bidirectional overlay matches the reference -------------------

def test_g4_mask_overlay_matches_reference():
    arrays, _ = fv.load_golden("gemma4_mask_overlay")
    block_ids = mx.array(arrays["block_ids"])

    from knurlogic.engine import register
    register.register("gemma4_text")
    from mlx_lm.models import gemma4_text as arch

    tc = g4fv.tiny_gemma4_config()["text_config"]
    tc["num_hidden_layers"] = 1
    tc["num_kv_shared_layers"] = 0
    # the maker's placement (architecture PROVENANCE.md, vendored edit 1):
    # sliding layers only, on a "vision" config; a window wider than the
    # golden leaves the overlay's own logic to compare
    B, N = block_ids.shape
    tc["layer_types"] = ["sliding_attention", "full_attention"]
    tc["num_hidden_layers"] = 2
    tc["sliding_window"] = N + 1
    tc["use_bidirectional_attention"] = "vision"
    model = arch.Gemma4TextModel(
        arch.ModelArgs.from_dict(dict(tc, model_type="gemma4_text")))

    h = mx.zeros((B, N, tc["hidden_size"]))
    masks = model._make_masks(h, [None, None], mm_mask=block_ids)
    want = arrays["overlaid"][:, 0]  # golden has a head axis of size 1
    np.testing.assert_array_equal(np.array(masks[0]), want.astype(bool))
    assert masks[1] == "causal"  # full layers stay causal


def test_g4_mask_overlay_can_fail():
    """Break the overlay (drop it) and confirm it stops matching the
    reference, then restore by re-running the passing test above."""
    arrays, _ = fv.load_golden("gemma4_mask_overlay")
    causal = arrays["causal"].astype(bool)
    want = arrays["overlaid"][:, 0].astype(bool)
    assert not np.array_equal(causal, want), (
        "the golden's bidirectional overlay must differ from plain causal, "
        "or this gate cannot fail")


# --- G5: embed() end to end -- merge, PLE zeroing, mm_mask, through the ---
# --- real (registered) trunk ------------------------------------------------

def _embed_end_to_end():
    from knurlogic.engine.vision import EncodedImage
    tc = g4fv.tiny_gemma4_config()["text_config"]
    model, arch = g4fv.tiny_text_model(tc)
    fam = _tiny_family()

    ref = ImageRef(sha="x" * 8, proc_hash=fam.spec.proc_hash, n_tokens=3)
    mx.random.seed(0)
    f = mx.random.normal((3, tc["hidden_size"]))
    store = {(ref.sha, ref.proc_hash): EncodedImage(ref=ref, feats=f)}

    ids = [1, 2, fam.image_token_id, fam.image_token_id, fam.image_token_id, 3]
    key = expand(ids, [ref], fam.image_token_id)
    out = fam.embed(model, key, 0, lambda sha, ph: store[(sha, ph)])
    return fam, model, key, out, f


def test_g5_embed_merges_image_rows_and_runs_through_the_trunk():
    fam, model, key, out, f = _embed_end_to_end()
    assert out["input_embeddings"].shape == (1, len(key), model.args.hidden_size)
    # image rows (2, 3, 4) carry the (unscaled) feature; text rows do not.
    embeds = out["input_embeddings"][0]
    text_row = model.model.embed_tokens(mx.array([1]))[0]
    np.testing.assert_allclose(np.array(embeds[0]), np.array(text_row), atol=1e-6)
    # embed() merges the STORED feats as is -- it is encode() that pre-
    # divides by embed_scale before a feature is cached (see the module
    # docstring); this store was seeded directly with `f`, unscaled.
    np.testing.assert_allclose(np.array(embeds[2]), np.array(f[0]), atol=1e-5)

    res = model(mx.array([1]), cache=None,
               input_embeddings=out["input_embeddings"],
               per_layer_inputs=out.get("per_layer_inputs"),
               mm_mask=out.get("mm_mask"))
    mx.eval(res)
    assert res.shape[:2] == (1, len(key))


def test_g5_per_layer_inputs_zero_image_positions():
    fam, model, key, out, _ = _embed_end_to_end()
    from knurlogic.engine.vision.key import is_sentinel, to_ids
    zeroed = mx.array([[0 if is_sentinel(x) else x for x in key]])
    want = model.model._get_per_layer_inputs(zeroed)
    np.testing.assert_allclose(np.array(out["per_layer_inputs"]), np.array(want))
    # and it must differ from NOT zeroing (an image id read as vocabulary):
    unzeroed = mx.array([to_ids(key, fam.image_token_id)])
    wrong = model.model._get_per_layer_inputs(unzeroed)
    assert not np.allclose(np.array(want), np.array(wrong))


def test_g5_embed_can_fail():
    """Break the merge (skip scatter) and confirm image rows are no longer
    the feature rows, then the passing test above shows it restored."""
    fam, model, key, out, f = _embed_end_to_end()
    unmerged_text = model.model.embed_tokens(
        mx.array([[1, 2, fam.image_token_id, fam.image_token_id,
                  fam.image_token_id, 3]]))
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(np.array(unmerged_text[0, 2]), np.array(f[0]),
                                   atol=1e-5)


# --- chunk-snap identity: an image straddling a 16-token chunk -------------

def test_chunk_snap_gives_identical_tokens_across_the_split():
    """A prefill chunker that snaps edges to `chunk_boundaries` must produce
    the SAME token ids whether or not an image happens to straddle a raw
    16-token chunk -- the snap changes where chunks fall, never what a
    position holds."""
    fam = _tiny_family()
    ref = ImageRef(sha="c" * 8, proc_hash=fam.spec.proc_hash, n_tokens=6)
    # 12 text tokens (indices 0-11), then a 6-token image at [12, 18) --
    # straddles the raw chunk edge at 16.
    ids = list(range(1, 13)) + [fam.image_token_id] * 6 + [99]
    key = expand(ids, [ref], fam.image_token_id)
    boundaries = fam.chunk_boundaries(key)
    assert boundaries == [(12, 18)]

    def snap_chunks(n: int, size: int, spans):
        edges = sorted(set(range(0, n, size)) | {n})
        out = []
        for e in edges:
            for s, en in spans:
                if s < e < en:
                    e = en
            out.append(e)
        out = sorted(set(out))
        return list(zip(out[:-1], out[1:]))

    chunks = snap_chunks(len(key), 16, boundaries)
    flat = [x for c in chunks for x in key[c[0]:c[1]]]
    assert flat == key  # nothing lost or reordered by the snap
    # No chunk edge falls strictly inside the image span:
    for s, e in chunks:
        for bs, be in boundaries:
            assert not (bs < e < be) and not (bs < s < be)


def test_resize_and_token_count_match_the_reference_processor():
    """HF transformers 5.16.1 `image_processing_gemma4.py`
    `get_aspect_ratio_preserving_size(height, width, patch_size,
    max_patches, pooling_kernel_size)`, with max_patches = max_soft_tokens
    * pooling_kernel_size**2 (`_preprocess`):

        factor = sqrt(max_patches * patch_size**2 / (height * width))
        side   = pooling_kernel_size * patch_size
        target = floor(factor * height / side) * side,
                 floor(factor * width / side) * side
        both 0           -> ValueError
        height 0 (wide)  -> side, min(floor(width / height) * side,
                                      max_patches // pool**2 * side)
        width 0 (tall)   -> the same, transposed
        over budget      -> ValueError

    The golden is that function's own output at the released processor
    settings (patch 16, pool 3, max_soft_tokens 280); -1s are its
    ValueErrors."""
    from PIL import Image
    arrays, _ = fv.load_golden("gemma4_vision_tower")
    fam = _tiny_family()
    fam.patch_size, fam.pool, fam.max_soft_tokens = 16, 3, 280
    for w, h, tw, th, n in arrays["resize"].tolist():
        if tw < 0:
            with pytest.raises(ValueError):
                fam.target_size(w, h)
            continue
        assert fam.target_size(w, h) == (tw, th), (w, h)
        if w * h <= 4096 * 4096 and max(w, h) <= 5000:
            px, ref = fam.preprocess(Image.new("RGB", (w, h)), "s")
            assert ref.n_tokens == n, (w, h)
            assert px["pixel_values"].shape[-2:] == (th, tw)


def test_the_tower_produces_exactly_the_predicted_count_off_square():
    # The live failure: a count predicted without the resize disagreed with
    # the tower. Any aspect must now agree, before the tower runs.
    from PIL import Image
    fam = _tiny_family()
    for w, h in [(97, 41), (41, 97), (64, 64)]:
        px, ref = fam.preprocess(Image.new("RGB", (w, h), (9, 90, 200)), "s")
        enc = fam.encode(px, ref)
        assert enc.feats.shape[0] == ref.n_tokens
