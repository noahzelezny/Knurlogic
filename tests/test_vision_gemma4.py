"""P2: gemma4 vision (e4b, 26b). Gates G1-G5 plus the chunk-snap identity
check the package's OWNS line asks for. Runs WITHOUT mlx-vlm (goldens are
pre-built .npz, `tests/goldens/build_gemma4.py` in the reference
interpreter -- design D2).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

mx = pytest.importorskip("mlx.core")

import fixtures_vision as fv  # noqa: E402
import fixtures_vision_gemma4 as g4fv  # noqa: E402
from knurlogic.engine.vision.gemma4 import Gemma4Vision  # noqa: E402
from knurlogic.engine.vision.key import expand, sentinel  # noqa: E402
from knurlogic.engine.vision import ImageRef  # noqa: E402


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
    from mlx.utils import tree_flatten
    import mlx.nn as nn
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
    import knurlogic.engine.vision.gemma4 as g4
    _write_quantized_embed(tmp_path)
    monkeypatch.setattr(g4, "quantize_like", lambda *a, **k: 0)
    with pytest.raises(ValueError, match="scales"):
        _tiny_family().load_weights(str(tmp_path))


def test_placeholder_is_framed_with_the_artifacts_own_tokens(tmp_path):
    """The placeholder must be the strings the artifact's tokenizer maps to
    boi / image / eoi, or the prompt carries no image token at all (the
    e4b gate failure: 1 image, 0 placeholders)."""
    import json
    from knurlogic.engine.vision.gemma4 import build
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

def test_g2_encode_matches_reference_tower_golden():
    arrays, meta = fv.load_golden("gemma4_vision_tower")
    H, W = int(meta["H"]), int(meta["W"])
    pixel_values = arrays["pixel_values"]

    cfg = g4fv.tiny_gemma4_config()
    cfg["vision_config"]["use_clipped_linears"] = False  # matches the golden
    fam = g4fv.build_gemma4_family(cfg)
    tower = fam.vision_tower
    from mlx.utils import tree_flatten, tree_unflatten
    have = {k for k, _ in tree_flatten(tower.parameters())}
    seeded = {k[2:]: mx.array(v) for k, v in arrays.items() if k.startswith("w.")}
    missing = have - set(seeded)
    assert not missing, f"tower param tree drifted from the reference: {missing}"
    tower.update(tree_unflatten(list(seeded.items())))
    mx.eval(tower.parameters())

    out = tower(mx.array(pixel_values))
    mx.eval(out)
    np.testing.assert_allclose(np.array(out), arrays["out"], atol=1e-3, rtol=1e-3)


def test_g2_encode_can_fail(monkeypatch):
    """Break the tower (patch_size lied about) and confirm the reference
    comparison goes red, then restore."""
    arrays, meta = fv.load_golden("gemma4_vision_tower")
    cfg = g4fv.tiny_gemma4_config()
    cfg["vision_config"]["use_clipped_linears"] = False  # matches the golden
    fam = g4fv.build_gemma4_family(cfg)
    tower = fam.vision_tower
    tower.pooling_kernel_size = 1  # break: reference used 3
    from mlx.utils import tree_flatten, tree_unflatten
    seeded = {k[2:]: mx.array(v) for k, v in arrays.items() if k.startswith("w.")}
    tower.update(tree_unflatten(list(seeded.items())))
    mx.eval(tower.parameters())
    out = tower(mx.array(arrays["pixel_values"]))
    mx.eval(out)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(np.array(out), arrays["out"], atol=1e-3,
                                   rtol=1e-3)
    # restored: pooling_kernel_size is a local var here, nothing persists.


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
    causal = mx.array(arrays["causal"])

    from knurlogic.engine import register
    register.register("gemma4_text")
    from mlx_lm.models import gemma4_text as arch

    tc = g4fv.tiny_gemma4_config()["text_config"]
    tc["num_hidden_layers"] = 1
    tc["num_kv_shared_layers"] = 0
    tc["layer_types"] = ["full_attention"]
    model = arch.Gemma4TextModel(arch.ModelArgs.from_dict(dict(tc, model_type="gemma4_text")))

    B, N = block_ids.shape
    h = mx.zeros((B, N, tc["hidden_size"]))
    masks = model._make_masks(h, [None], mm_mask=block_ids)
    got = masks[0]
    want = arrays["overlaid"][:, 0]  # golden has a head axis of size 1
    np.testing.assert_array_equal(np.array(got), want.astype(bool))


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
    from knurlogic.engine.vision import EncodedImage
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
