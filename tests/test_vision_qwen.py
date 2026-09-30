"""Qwen vision (P1): the tower, the processor, MRoPE positions and the
trunks' position inputs, held to mlx-vlm 0.6.17 by committed goldens.

  G1   tower output == mlx-vlm's tower (atol 1e-5), weights read through
       load_weights from a sidecar in BOTH namings (HF model.visual.*,
       MLX vision_tower.*)
  G2   grid, token count and pixel_values == mlx-vlm's processor
  G3   positions and rope_delta of a two-image prompt == get_rope_index
  G4   prefill logits and 40 greedy tokens == mlx-vlm's model, through
       Family.embed + positions and the trunk's position_ids / rope_delta
  G5   the text path is main's, to the bit (qwen_g5_text snapshot), and the
       MRoPE path with text positions IS the 1-D path
  G7b  a text-only turn after the image, WARM from turn 1's cache with only
       rope_delta, == mlx-vlm's COLD turn 2 (fails if D4 is missing: the same
       warm turn without the delta is checked to diverge)
  +    per-row rope_delta in a batch == each row alone (mixed image/text rows,
       the qwen4_exp indexer's stored positions through merge)

Goldens: tests/goldens/build_qwen.py (mlx-vlm interpreter) and
build_qwen_g5.py (this interpreter, before the trunk edits). No mlx-vlm is
imported here. Tiny random models, float32, seed 0.

G5 compares with np.array_equal: same MLX (pinned), same machine class. If
it goes red on NEW HARDWARE with the fingerprint check green, rebuild the
snapshot from main's trunk files on that machine -- never from the edited
ones.
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

mx = pytest.importorskip("mlx.core")

import fixtures_vision as fv  # noqa: E402
import fixtures_vision_qwen as fq  # noqa: E402

FAMS = fq.FAMILIES


# --- fixtures ---------------------------------------------------------------------

def _golden(fam):
    return fv.load_golden(f"qwen_{fam}")


def _arch(fam):
    from knurlogic.engine import register
    register.register(*FAMS, override=True)
    return importlib.import_module(f"mlx_lm.models.{fam}")


def _weights(meta):
    return fq.init_weights({k: tuple(v) for k, v in meta["shapes"].items()})


def _trunk(fam, meta):
    arch = _arch(fam)
    model = arch.Model(arch.ModelArgs.from_dict(meta["config"]))
    model.set_dtype(mx.float32)
    w = {fq.trunk_name(fam, k): mx.array(v) for k, v in _weights(meta).items()
         if not k.startswith("vision_tower.")}
    from mlx.utils import tree_flatten
    have = {k for k, v in tree_flatten(model.parameters())
            if v.dtype == mx.float32}
    assert set(w) == have, (sorted(set(w) ^ have))[:5]
    model.load_weights(list(w.items()), strict=False)
    mx.eval(model.parameters())
    return model


def _model_dir(tmp, fam, meta, naming):
    """A tiny rung on disk: config, preprocessor config, tokenizer added
    tokens, and the vision tensors in a sidecar named by the index -- in the
    HF naming (model.visual.*, patch embed in torch layout) or the MLX one."""
    cfg = meta["config"]
    (tmp / "config.json").write_text(json.dumps(cfg))
    (tmp / "preprocessor_config.json").write_text(json.dumps(fq.PREPROCESSOR))
    t = fq.ids(fam)
    added = [{"id": t[k], "content": c, "special": True}
             for k, c in (("vision_start_token_id", "<|vision_start|>"),
                          ("vision_end_token_id", "<|vision_end|>"),
                          ("image_token_id", "<|image_pad|>"),
                          ("video_token_id", "<|video_pad|>"))]
    (tmp / "tokenizer.json").write_text(json.dumps({"added_tokens": added}))
    vis = {}
    for k, v in _weights(meta).items():
        if not k.startswith("vision_tower."):
            continue
        if naming == "hf":
            if k.endswith("patch_embed.proj.weight"):
                v = np.ascontiguousarray(v.transpose(0, 4, 1, 2, 3))
            k = "model.visual." + k[len("vision_tower."):]
        vis[k] = mx.array(v)
    mx.save_safetensors(str(tmp / "model-vision-graft.safetensors"), vis)
    wm = {k: "model-vision-graft.safetensors" for k in vis}
    wm["language_model.model.embed_tokens.weight"] = "model-00001.safetensors"
    (tmp / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": wm}))
    return tmp, len(vis)


def _family(tmp, fam, meta, naming="hf"):
    from knurlogic.engine.vision import registry
    path, n = _model_dir(tmp, fam, meta, naming)
    f = registry.build(fam, str(path), None, meta["config"])
    assert f is not None
    assert f.load_weights(str(path)) == n
    return f


def _image(arrays, i):
    from PIL import Image
    return Image.fromarray(arrays[f"img{i}"], "RGB")


class _Store:
    """The two lookups the store gives embed/positions (store.lookup)."""

    def __init__(self):
        self.enc = {}

    def put(self, enc):
        self.enc[(enc.ref.sha, enc.ref.proc_hash)] = enc

    def features(self, sha, ph):
        return self.enc[(sha, ph)]

    def refs(self, sha, ph):
        return self.enc[(sha, ph)].ref


def _encoded(fam_obj, arrays):
    """Both golden images preprocessed and encoded; (store, refs)."""
    st, refs = _Store(), []
    for i in (1, 2):
        pix, ref = fam_obj.preprocess(_image(arrays, i), f"sha{i}")
        enc = fam_obj.encode(pix, ref)
        st.put(enc)
        refs.append(ref)
    return st, refs


def _key(fam_obj, ids, refs):
    from knurlogic.engine.vision import key as K
    return K.expand([int(x) for x in ids], refs, fam_obj.image_token_id)


def _greedy(model, cache, tok, n, **kw):
    toks, logits = [], []
    for _ in range(n):
        lg = model(mx.array([[tok]]), cache=cache, **kw)[0, -1]
        logits.append(np.array(lg))
        tok = int(mx.argmax(lg).item())
        toks.append(tok)
    return toks, logits


def _prefill(model, fam_obj, key, start, cache, st):
    """What the serve path does for one uncached span: embed over
    key[start:], positions over the WHOLE key, the chunk's slice."""
    from knurlogic.engine.vision import key as K
    emb = fam_obj.embed(model, key, start, st.features)
    pos, delta = fam_obj.positions(key, st.refs)
    ids = mx.array(K.to_ids(key[start:], fam_obj.image_token_id))[None]
    out = model(ids, cache=cache, input_embeddings=emb["input_embeddings"],
                position_ids=pos[:, :, start:])
    return out, delta


# --- G1 / G2 ------------------------------------------------------------------------

@pytest.mark.parametrize("fam", FAMS)
@pytest.mark.parametrize("naming", ["hf", "mlx"])
def test_g1_tower_matches_reference(fam, naming, tmp_path):
    arrays, meta = _golden(fam)
    f = _family(tmp_path, fam, meta, naming)
    for i in (1, 2):
        pix, ref = f.preprocess(_image(arrays, i), "x")
        enc = f.encode(pix, ref)
        got = np.array(enc.feats)
        assert got.shape == arrays[f"feats{i}"].shape
        np.testing.assert_allclose(got, arrays[f"feats{i}"], atol=1e-5)


@pytest.mark.parametrize("fam", FAMS)
def test_g2_grid_count_and_pixels_match_reference(fam, tmp_path):
    arrays, meta = _golden(fam)
    f = _family(tmp_path, fam, meta)
    for i in (1, 2):
        pix, ref = f.preprocess(_image(arrays, i), "x")
        assert list(pix["grid_thw"]) == arrays[f"grid{i}"][0].tolist()
        assert ref.grid_thw == tuple(arrays[f"grid{i}"][0].tolist())
        assert ref.n_tokens == int(np.prod(arrays[f"grid{i}"][0])) // 4
        np.testing.assert_allclose(pix["pixel_values"], arrays[f"pv{i}"],
                                   atol=1e-6)
    # the placeholder is the tokenizer's own special tokens, one pad each
    assert f.placeholder_text(ref) == \
        "<|vision_start|><|image_pad|><|vision_end|>"
    assert f.spec.image_token_id == fq.ids(fam)["image_token_id"]


def test_placeholder_refused_when_the_tokenizer_does_not_know_it(tmp_path):
    """An image token id the tokenizer does not
    carry as a special added token would be spelled out as text -- the build
    must refuse, not serve a model that silently never sees the image."""
    from knurlogic.engine.vision import VisionError, registry
    arrays, meta = _golden("qwen3_5")
    path, _ = _model_dir(tmp_path, "qwen3_5", meta, "mlx")
    (path / "tokenizer.json").write_text(json.dumps({"added_tokens": []}))
    with pytest.raises(VisionError, match="special added token"):
        registry.build("qwen3_5", str(path), None, meta["config"])


def test_load_weights_takes_only_vision_tensors_named_by_the_index(tmp_path):
    """The index names a trunk shard that does not exist: load_weights must
    read only the vision tensors' file (the trunk's sanitize keeps dropping
    them), and a missing vision tensor is an error."""
    from knurlogic.engine.vision import registry
    arrays, meta = _golden("qwen3_5")
    path, n = _model_dir(tmp_path, "qwen3_5", meta, "hf")
    f = registry.build("qwen3_5", str(path), None, meta["config"])
    assert f.load_weights(str(path)) == n
    idx = json.loads((path / "model.safetensors.index.json").read_text())
    k = next(k for k in idx["weight_map"] if k.endswith("merger.norm.weight"))
    del idx["weight_map"][k]
    (path / "model.safetensors.index.json").write_text(json.dumps(idx))
    with pytest.raises(ValueError):
        f.load_weights(str(path))


# --- G3 -----------------------------------------------------------------------------

@pytest.mark.parametrize("fam", FAMS)
def test_g3_positions_match_reference(fam, tmp_path):
    arrays, meta = _golden(fam)
    f = _family(tmp_path, fam, meta)
    st, refs = _encoded(f, arrays)
    key = _key(f, arrays["ids"], refs)
    pos, delta = f.positions(key, st.refs)
    assert np.array_equal(np.array(pos)[:, 0, :], arrays["pos"][:, 0, :])
    assert delta == int(arrays["delta"].reshape(-1)[0])
    # vision_start comes from the config: the class default 248045 would
    # count no image at all and give plain positions
    assert f.vision_start_token_id == fq.ids(fam)["vision_start_token_id"]


def test_g3_positions_are_pure_in_the_key(tmp_path):
    """D4: text appended after the last image keeps every earlier position
    and the same delta; the positions of the appended text are exactly
    arange + delta -- what the trunk's rope_delta input computes."""
    arrays, meta = _golden("qwen3_5")
    f = _family(tmp_path, "qwen3_5", meta)
    st, refs = _encoded(f, arrays)
    key = _key(f, arrays["ids"], refs)
    pos, delta = f.positions(key, st.refs)
    longer = key + [5, 6, 7, 8]
    pos2, delta2 = f.positions(longer, st.refs)
    assert delta2 == delta
    assert np.array_equal(np.array(pos2)[:, :, :len(key)], np.array(pos))
    tail = np.arange(len(key), len(longer)) + delta
    assert np.array_equal(np.array(pos2)[:, 0, len(key):],
                          np.broadcast_to(tail, (3, 4)))
    assert f.positions([1, 2, 3], st.refs) == (None, 0)


# --- G4 / G7b -----------------------------------------------------------------------

@pytest.mark.parametrize("fam", FAMS)
def test_g4_and_g7b_match_reference(fam, tmp_path):
    arrays, meta = _golden(fam)
    f = _family(tmp_path, fam, meta)
    model = _trunk(fam, meta)
    st, refs = _encoded(f, arrays)
    key = _key(f, arrays["ids"], refs)

    # G4: cold prefill through embed + positions, then decode on rope_delta
    cache = model.make_cache()
    out, delta = _prefill(model, f, key, 0, cache, st)
    lg = out[0, -1]
    np.testing.assert_allclose(np.array(lg), arrays["prefill_logits"],
                               atol=1e-4)
    first = int(mx.argmax(lg).item())
    toks, logits = _greedy(model, cache, first, len(arrays["gen"]) - 1,
                           rope_delta=delta)
    assert [first] + toks == arrays["gen"].tolist()
    np.testing.assert_allclose(np.stack(logits), arrays["gen_logits"][1:],
                               atol=1e-4)

    # G7b: turn 2 is text only. WARM: turn 1's cache holds the prompt and all
    # but the last generated token; only the new text is prefilled, with the
    # delta and no image anywhere in the suffix.
    t2 = [int(x) for x in arrays["t2_ids"]]
    t2_key = key + t2[len(key):]
    hit = len(key) + len(toks)            # cached: prompt + gen[:-1]
    assert t2_key[:len(key)] == key
    warm_pos, warm_delta = f.positions(t2_key, st.refs)
    assert warm_delta == delta
    assert np.array_equal(
        np.array(warm_pos)[:, 0, hit:],
        np.broadcast_to(np.arange(hit, len(t2_key)) + delta, (3, len(t2) - hit)))
    import copy
    base_cache = copy.deepcopy(cache)
    suffix = mx.array([t2[hit:]])
    lg = model(suffix, cache=cache, rope_delta=delta)[0, -1]
    np.testing.assert_allclose(np.array(lg), arrays["t2_logits"], atol=1e-4)
    first = int(mx.argmax(lg).item())
    toks2, _ = _greedy(model, cache, first, len(arrays["t2_gen"]) - 1,
                       rope_delta=delta)
    assert [first] + toks2 == arrays["t2_gen"].tolist()

    # the same warm turn WITHOUT the delta diverges:
    # this gate can fail
    lg_bad = model(suffix, cache=base_cache)[0, -1]
    assert not np.allclose(np.array(lg_bad), arrays["t2_logits"], atol=1e-4)


# --- G5 -----------------------------------------------------------------------------

def _g5_run(model, prompt, split, steps):
    cache = model.make_cache()
    out = []
    model(mx.array(prompt[:split])[None], cache=cache)
    b = model(mx.array(prompt[split:])[None], cache=cache)
    out.append(np.array(b[0, -1]))
    tok = int(mx.argmax(b[0, -1]).item())
    toks = [tok]
    for _ in range(steps):
        lg = model(mx.array([[tok]]), cache=cache)[0, -1]
        out.append(np.array(lg))
        tok = int(mx.argmax(lg).item())
        toks.append(tok)
    return np.stack(out), toks


def _seed_model(fam, cfg):
    from mlx.utils import tree_flatten
    arch = _arch(fam)
    mx.random.seed(0)
    model = arch.Model(arch.ModelArgs.from_dict(cfg))
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    fp = np.array([float(mx.abs(v).sum().item())
                   for _, v in tree_flatten(model.parameters())])
    return model, fp


@pytest.mark.parametrize("fam", FAMS)
def test_g5_text_path_is_mains_to_the_bit(fam):
    arrays, meta = fv.load_golden("qwen_g5_text")
    model, fp = _seed_model(fam, meta[f"{fam}/config"])
    assert np.array_equal(fp, arrays[f"{fam}/w_fingerprint"]), \
        "the seed-0 init moved, not the text path: rebuild nothing, look at init"
    logits, toks = _g5_run(model, meta["prompt"], meta["split"], meta["steps"])
    assert toks == arrays[f"{fam}/tokens"].tolist()
    assert np.array_equal(logits, arrays[f"{fam}/logits"])


@pytest.mark.parametrize("fam", FAMS)
def test_g5_mrope_with_text_positions_is_the_1d_path(fam):
    """Critique issue 4: G5 by source inspection is a tautology. This runs a
    text prompt through the MRoPE path (explicit position_ids = arange on
    all three axes, then decode on rope_delta = 0) and holds it to the 1-D
    path -- which proves the interleave degenerates correctly."""
    arrays, meta = fv.load_golden("qwen_g5_text")
    model, _ = _seed_model(fam, meta[f"{fam}/config"])
    prompt, split = meta["prompt"], meta["split"]
    cache = model.make_cache()
    model(mx.array(prompt[:split])[None], cache=cache,
          position_ids=mx.broadcast_to(mx.arange(split)[None, None],
                                       (3, 1, split)))
    n = len(prompt) - split
    b = model(mx.array(prompt[split:])[None], cache=cache,
              position_ids=mx.broadcast_to(
                  mx.arange(split, len(prompt))[None, None], (3, 1, n)))
    got = [np.array(b[0, -1])]
    tok = int(mx.argmax(b[0, -1]).item())
    for _ in range(meta["steps"]):
        lg = model(mx.array([[tok]]), cache=cache, rope_delta=0)[0, -1]
        got.append(np.array(lg))
        tok = int(mx.argmax(lg).item())
    np.testing.assert_allclose(np.stack(got), arrays[f"{fam}/logits"],
                               atol=1e-5)


# --- per-row rope_delta in a batch -------------------------------------------------

@pytest.mark.parametrize("fam", FAMS)
def test_per_row_rope_delta_in_a_batch_equals_each_row_alone(fam, tmp_path):
    """An image row and a text row, each prefilled alone, merged into one
    batch (mlx-lm's own merge, as BatchGenerator does) and decoded with
    rope_delta [delta, 0]: each row's logits == that row decoded alone. For
    qwen4_exp the image row's cache carries indexer positions and the text
    row's does not -- the merge must give it text positions, and the sparse
    indexer (budget 32) runs on both."""
    import copy
    from mlx_lm.generate import _merge_caches
    arrays, meta = _golden(fam)
    f = _family(tmp_path, fam, meta)
    model = _trunk(fam, meta)
    st, refs = _encoded(f, arrays)
    key = _key(f, arrays["ids"], refs)
    text = [int(x) for x in arrays["t2_ids"][-60:]]

    ci = model.make_cache()
    out, delta = _prefill(model, f, key, 0, ci, st)
    ti = int(mx.argmax(out[0, -1]).item())
    ct = model.make_cache()
    out = model(mx.array([text]), cache=ct)
    tt = int(mx.argmax(out[0, -1]).item())

    solo_i, _ = _greedy(model, copy.deepcopy(ci), ti, 6, rope_delta=delta)
    solo_t, _ = _greedy(model, copy.deepcopy(ct), tt, 6)
    ref_i, ref_t = [], []
    c1, c2 = copy.deepcopy(ci), copy.deepcopy(ct)
    a, b = ti, tt
    for _ in range(6):
        la = model(mx.array([[a]]), cache=c1, rope_delta=delta)[0, -1]
        lb = model(mx.array([[b]]), cache=c2)[0, -1]
        ref_i.append(np.array(la)); ref_t.append(np.array(lb))
        a, b = int(mx.argmax(la).item()), int(mx.argmax(lb).item())

    batch = _merge_caches([ci, ct])
    rd = mx.array([delta, 0])
    a, b = ti, tt
    got_i, got_t = [], []
    for step in range(6):
        lg = model(mx.array([[a], [b]]), cache=batch, rope_delta=rd)[:, -1]
        np.testing.assert_allclose(np.array(lg[0]), ref_i[step], atol=1e-4)
        np.testing.assert_allclose(np.array(lg[1]), ref_t[step], atol=1e-4)
        a, b = int(mx.argmax(lg[0]).item()), int(mx.argmax(lg[1]).item())
        got_i.append(a); got_t.append(b)
    assert got_i == solo_i and got_t == solo_t


@pytest.mark.parametrize("fam", ["qwen3_5", "qwen4_exp"])
def test_chunked_prefill_cutting_images_equals_one_shot(fam, tmp_path):
    """Qwen is causal across an image (chunk_boundaries == []), so a prefill
    chunked every 16 tokens -- edges falling inside both images -- must give
    the one-shot golden logits: embed over each chunk's key slice (sentinels
    carry their row k, no global index) and the positions slice for it."""
    from knurlogic.engine.vision import key as K
    arrays, meta = _golden(fam)
    f = _family(tmp_path, fam, meta)
    model = _trunk(fam, meta)
    st, refs = _encoded(f, arrays)
    key = _key(f, arrays["ids"], refs)
    assert f.chunk_boundaries(key) == []
    pos, _ = f.positions(key, st.refs)
    cache = model.make_cache()
    for a in range(0, len(key), 16):
        b = min(a + 16, len(key))
        emb = f.embed(model, key[:b], a, st.features)
        ids = mx.array(K.to_ids(key[a:b], f.image_token_id))[None]
        out = model(ids, cache=cache, input_embeddings=emb["input_embeddings"],
                    position_ids=pos[:, :, a:b])
    np.testing.assert_allclose(np.array(out[0, -1]), arrays["prefill_logits"],
                               atol=1e-4)
