"""The image store (engine/vision/store.py) and image decoding/hashing
(engine/vision/images.py), plus the stub family end to end through both.

Store arrays here are numpy: the store only reads `.nbytes`, which is the
point -- it counts real bytes, whatever holds them.
"""
import base64
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fixtures_vision as fv  # noqa: E402
from knurlogic.engine.vision import (EncodedImage, ImageEvicted, ImageTooLarge,  # noqa: E402
                                     ImageRef, ImageRejected)
from knurlogic.engine.vision import images  # noqa: E402
from knurlogic.engine.vision.store import (DEFAULT_MAX_BYTES,  # noqa: E402
                                           ImageStore, estimate_nbytes)

MB = 1 << 20


def _enc(tag, nbytes, ph="p"):
    ref = ImageRef(tag * 8, ph, 4, None)
    return EncodedImage(ref, np.zeros(nbytes, dtype=np.uint8))


# --- the store ----------------------------------------------------------------------

def test_evicts_by_bytes_oldest_first():
    s = ImageStore(max_bytes=3 * MB)
    for t in "abc":
        assert s.put("m", _enc(t, MB))
    assert s.nbytes == 3 * MB
    s.get("m", "a" * 8, "p")                     # a is now freshest
    s.put("m", _enc("d", MB))
    assert s.get("m", "b" * 8, "p") is None      # b was oldest
    assert s.get("m", "a" * 8, "p") is not None
    assert s.nbytes == 3 * MB and s.stats()["evictions"] == 1


def test_one_big_image_evicts_many_small_ones():
    """Count-bounded (mlx-vlm's 20 entries) would keep all of these; bytes
    are what the host has."""
    s = ImageStore(max_bytes=10 * MB)
    for t in "abcdefghij":
        s.put("m", _enc(t, MB))
    s.put("m", _enc("z", 8 * MB))
    assert len(s) == 3 and s.nbytes == 10 * MB


def test_per_image_hits_when_a_second_image_is_added():
    """exo keyed on the whole image list, so adding image 2 re-encoded
    image 1. Here image 1 is still a hit."""
    s = ImageStore()
    s.put("m", _enc("a", 10))
    turn2 = ["a" * 8, "b" * 8]
    hits = [s.get("m", sha, "p") is not None for sha in turn2]
    assert hits == [True, False]


def test_keyed_by_model_and_proc_hash():
    s = ImageStore()
    s.put("m1", _enc("a", 10, ph="p"))
    assert s.get("m2", "a" * 8, "p") is None
    assert s.get("m1", "a" * 8, "q") is None
    assert s.get("m1", "a" * 8, "p") is not None


def test_refs_survive_feature_eviction():
    """critique issue 7: positions() for a text turn after an image needs
    its grid after the features are gone."""
    s = ImageStore(max_bytes=MB)
    s.put("m", _enc("a", MB))
    s.put("m", _enc("b", MB))
    assert s.get("m", "a" * 8, "p") is None
    assert s.ref("m", "a" * 8, "p").n_tokens == 4
    feats, refs = s.lookup("m")
    assert refs("a" * 8, "p").sha == "a" * 8
    with pytest.raises(ImageEvicted):
        feats("a" * 8, "p")
    s.clear("m")
    assert s.ref("m", "a" * 8, "p") is None and s.nbytes == 0


def test_pins_hold_past_the_bound_then_release():
    s = ImageStore(max_bytes=MB)
    with s.pinned("m", [("a" * 8, "p"), ("b" * 8, "p")]):
        s.put("m", _enc("a", MB))
        s.put("m", _enc("b", MB))
        assert len(s) == 2
        assert s.stats()["over_bound_by_pins"] == MB
        assert s.budget_bytes() == 2 * MB
    assert len(s) == 1 and s.nbytes == MB


def test_an_unpinned_image_bigger_than_the_bound_does_not_stay():
    s = ImageStore(max_bytes=MB)
    assert s.put("m", _enc("a", 2 * MB)) is False
    assert s.nbytes == 0 and s.ref("m", "a" * 8, "p") is not None


def test_shrinking_the_bound_evicts_now():
    s = ImageStore(max_bytes=4 * MB)
    for t in "abcd":
        s.put("m", _enc(t, MB))
    s.max_bytes = 2 * MB
    assert s.nbytes == 2 * MB and s.get("m", "d" * 8, "p") is not None


def test_nbytes_is_read_from_the_arrays():
    ref = ImageRef("x", "p", 2, None)
    e = EncodedImage(ref, np.zeros((2, 8), np.float32),
                     {"deep": np.zeros(3, np.float32)})
    assert e.nbytes == 2 * 8 * 4 + 3 * 4
    assert estimate_nbytes(8000, 4096) == 65_536_000   # the ~65 MB GLM image
    assert DEFAULT_MAX_BYTES == 256 * MB


# --- decoding and hashing -------------------------------------------------------------

def test_hash_is_of_pixels_not_encoding():
    img = fv.tiny_image(31, 17, seed=1)
    a = fv.png_bytes(img, compress_level=0)
    b = fv.png_bytes(img, compress_level=9)
    assert a != b
    sha_a = images.load(a)[1]
    assert images.load(b)[1] == sha_a
    b64 = base64.b64encode(a).decode()
    assert images.load(b64)[1] == sha_a
    assert images.load("data:image/png;base64," + b64)[1] == sha_a
    other = fv.tiny_image(31, 17, seed=2)
    assert images.load(fv.png_bytes(other))[1] != sha_a


def test_mode_and_size_are_in_the_hash():
    from PIL import Image
    a = Image.new("RGB", (4, 2), (1, 2, 3))
    b = Image.new("RGB", (2, 4), (1, 2, 3))
    assert a.tobytes() == b.tobytes()
    assert images.pixel_sha(a) != images.pixel_sha(b)


def test_normalisation_matches_mlx_vlm_load_image():
    """EXIF orientation applied, then RGB -- as mlx-vlm 0.6.17 does, so
    goldens made through its load_image see the same pixels."""
    g, meta = fv.load_golden("p0_load_image")
    assert meta["mlx_vlm"] == fv.REFERENCE_VERSION
    img = images.decode(g["png"].tobytes())
    assert list(img.size) == list(g["size"])
    np.testing.assert_array_equal(np.asarray(img), g["pixels"])


def test_decompression_bomb_is_refused_before_decoding():
    from PIL import Image
    import io
    big = Image.new("1", (10_000, 9_000))          # 90 Mpx > 89,478,485
    buf = io.BytesIO()
    big.save(buf, format="PNG")
    with pytest.raises(ImageTooLarge, match="maximum for its format is 89478485 pixels"):
        images.decode(buf.getvalue())


def test_large_images_are_clamped_before_hashing(monkeypatch):
    monkeypatch.setattr(images, "MAX_DECODE_PIXELS", 100)
    img, sha = images.load(fv.png_bytes(fv.tiny_image(40, 20)))
    assert img.size[0] * img.size[1] <= 100
    assert img.size[0] > img.size[1]               # aspect kept
    assert sha == images.pixel_sha(img)


def test_urls_and_paths_are_not_fetched(tmp_path):
    with pytest.raises(ImageRejected, match="not fetched"):
        images.decode("https://example.com/cat.png")
    p = tmp_path / "x.png"
    p.write_bytes(fv.png_bytes(fv.tiny_image()))
    with pytest.raises(ImageRejected):
        images.decode(str(p))                      # remote clients: no paths
    assert images.decode(str(p), allow_paths=True).size == (32, 24)


def test_garbage_and_oversize_are_rejected(monkeypatch):
    for bad in (b"not an image", "@@@", b"", "data:image/png,rawtext"):
        with pytest.raises(ImageRejected):
            images.decode(bad)
    monkeypatch.setattr(images, "MAX_BYTES", 10)
    with pytest.raises(ImageTooLarge, match="the maximum is"):
        images.decode(fv.png_bytes(fv.tiny_image()))


# --- the stub family, through images, the store and the key ----------------------------

def test_stub_family_end_to_end():
    mx = pytest.importorskip("mlx.core")
    import mlx.nn as nn
    from knurlogic.engine.vision import key as K
    D, PAD = 64, 500
    fam = fv.StubFamily(PAD, D)
    emb = nn.Embedding(fv.TINY_VOCAB, D)
    fam.embed_fn = emb
    store = ImageStore()
    calls = []
    real_tower = fam.tower
    fam.tower = lambda x: (calls.append(1), real_tower(x))[1]  # from outside

    def ensure(src):
        img, sha = images.load(src)
        pixels, ref = fam.preprocess(img, sha)
        if store.get("m", sha, ref.proc_hash) is None:
            store.put("m", fam.encode(pixels, ref))
        return ref

    png = fv.png_bytes(fv.tiny_image(16, 12, seed=0))
    ref = ensure(png)
    assert ref.n_tokens == (16 // 4) * (12 // 4)
    ensure(png)                                    # turn 2: same image
    assert len(calls) == 1, "the tower runs once per image"
    key = K.expand(K.expand_pads([1, PAD, 2], [ref], PAD), [ref], PAD)
    feats, refs = store.lookup("m")
    out = fam.embed(None, key, 0, feats)["input_embeddings"]
    assert out.shape == (1, len(key), D)
    assert mx.array_equal(out[0, 1:1 + ref.n_tokens],
                          store.get("m", ref.sha, ref.proc_hash).feats)
    assert fam.positions(key, refs) == (None, 0)
    assert fam.chunk_boundaries(key) == [(1, 1 + ref.n_tokens)]
    ref2 = ensure(fv.png_bytes(fv.tiny_image(16, 12, seed=9)))
    assert ref2.n_tokens == ref.n_tokens and ref2.sha != ref.sha
    assert not mx.array_equal(store.get("m", ref2.sha, ref2.proc_hash).feats,
                              store.get("m", ref.sha, ref.proc_hash).feats)


def test_tiny_configs_keep_structure_and_fit_the_vocab():
    for fam in fv.FAMILIES:
        c = fv.tiny_config(fam)
        real = fv.REAL[fam]
        assert c["model_type"] == real["model_type"]
        assert c["vision_config"]["patch_size"] == \
            real["vision_config"]["patch_size"]
        ids = [v for k, v in c.items() if k.endswith("_token_id")]
        assert ids and max(ids) < fv.TINY_VOCAB
        assert c["text_config"]["vocab_size"] == fv.TINY_VOCAB
        vc = c["vision_config"]
        assert vc.get("depth", vc.get("num_hidden_layers")) == 2
        if "out_hidden_size" in vc:
            assert vc["out_hidden_size"] == c["text_config"]["hidden_size"]
    assert fv.tiny_config("glm5_next")["vision_config"]["patch_size"] == 14
    q = fv.tiny_config("qwen3_5")
    assert q["vision_start_token_id"] < q["vision_end_token_id"] < \
        q["image_token_id"], "remap keeps the real order"


def test_a_huge_jpeg_is_decoded_reduced_not_refused():
    """A JPEG decodes at 1/8 scale directly, so one far over the PNG limit
    is fine; the same pixel count as a PNG is refused (it must be unpacked
    whole)."""
    import io
    from PIL import Image
    w, h = 10000, 9500                       # 95 Mpx: over BOMB_PIXELS
    b = io.BytesIO()
    Image.new("RGB", (w, h), (200, 30, 30)).save(b, "JPEG", quality=30)
    img = images.decode(b.getvalue())
    assert img.size[0] * img.size[1] <= images.MAX_DECODE_PIXELS
    assert img.getpixel((10, 10))[0] > 150
    p = io.BytesIO()
    Image.new("1", (w, h)).save(p, "PNG")
    with pytest.raises(ImageTooLarge, match="unpacked whole"):
        images.decode(p.getvalue())
