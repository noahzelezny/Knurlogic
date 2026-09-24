"""The cache key (engine/vision/key.py), the registry, and the scatter.

The key tests that matter most run mlx-lm's REAL LRUPromptCache: the design
rests on the trie accepting sentinels (D6), and a mlx-lm that stops
accepting them must go red here, not in a served conversation.
"""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from knurlogic.engine.vision import ImageRef, KeyMismatch  # noqa: E402
from knurlogic.engine.vision import key as K  # noqa: E402
from knurlogic.engine.vision import registry  # noqa: E402

PAD = 99
A = ImageRef("a" * 64, "p1", 3, (1, 2, 6))
B = ImageRef("b" * 64, "p1", 3, (1, 2, 6))      # same size as A
C = ImageRef("c" * 64, "p1", 2, None)


def _ids(*parts):
    out = []
    for p in parts:
        out += [PAD] * p.n_tokens if isinstance(p, ImageRef) else list(p)
    return out


# --- expand / to_ids -----------------------------------------------------------

def test_round_trip_and_length():
    ids = _ids([1, 2], A, [3], C, [4, 5])
    key = K.expand(ids, [A, C], PAD)
    assert len(key) == len(ids), "the key must be exactly as long as the KV"
    assert K.to_ids(key, PAD) == ids
    assert key[2] == ("img", A.sha, "p1", 0) and key[4] == ("img", A.sha,
                                                           "p1", 2)
    assert [K.is_sentinel(x) for x in key].count(True) == 5
    assert K.has_image(key) and not K.has_image([1, 2, 3])


def test_back_to_back_images_split_by_ref_length():
    key = K.expand(_ids([1], A, C, [2]), [A, C], PAD)
    spans = K.image_spans(key)
    assert [(s.start, s.end, s.sha[0], s.k0) for s in spans] == [
        (1, 4, "a", 0), (4, 6, "c", 0)]


def test_same_image_twice_in_a_row_is_two_spans():
    key = K.expand(_ids(A, A), [A, A], PAD)
    assert [(s.start, s.end) for s in K.image_spans(key)] == [(0, 3), (3, 6)]


def test_mismatches_are_loud():
    # a user typed the pad token as text: one extra image token
    with pytest.raises(KeyMismatch):
        K.expand(_ids(A, [PAD]), [A], PAD)
    with pytest.raises(KeyMismatch):
        K.expand(_ids([1]), [A], PAD)              # image with no run
    with pytest.raises(KeyMismatch):
        K.expand([PAD, PAD, 1], [A], PAD)          # run too short
    with pytest.raises(KeyMismatch):
        K.expand_pads([PAD, 1, PAD], [A], PAD)     # two pads, one image
    with pytest.raises(KeyMismatch):
        K.expand_pads([1], [A], PAD)


def test_expand_pads_then_expand():
    template = [7, PAD, 8, PAD, 9]                 # one pad per image
    ids = K.expand_pads(template, [A, C], PAD)
    assert ids == [7, PAD, PAD, PAD, 8, PAD, PAD, 9]
    assert K.to_ids(K.expand(ids, [A, C], PAD), PAD) == ids


def test_segments_expand_with_the_prompt():
    """critique issue 2: mlx-lm's segments feed insert_segments; they must
    be keys too, and add up to the key."""
    segs = [[1, PAD, 2], [3], [PAD, 4]]
    key, seg_keys = K.expand_segments(segs, [A, C], PAD)
    assert sum(map(len, seg_keys)) == len(key)
    assert [x for s in seg_keys for x in s] == key
    assert len(seg_keys[0]) == 2 + A.n_tokens and len(seg_keys[2]) == 1 + 2
    assert seg_keys[2][0] == ("img", C.sha, "p1", 0)


def test_spans_of_a_slice_start_mid_image():
    key = K.expand(_ids([1], A, [2]), [A], PAD)
    s = K.image_spans(key[2:])
    assert [(x.start, x.end, x.k0) for x in s] == [(0, 2, 1)]
    assert K.images_in(key[2:]) == [(A.sha, "p1")]


def test_proc_hash_is_in_the_sentinel():
    """critique issue 1: a processor change with the same token count must
    not hit the old KV."""
    A2 = ImageRef(A.sha, "p2", A.n_tokens, A.grid_thw)
    assert K.expand(_ids(A), [A], PAD) != K.expand(_ids(A), [A2], PAD)


# --- the real prompt cache (G9 at key level) -------------------------------------

class _Entry:
    nbytes = 1

    def is_trimmable(self):
        return False


def _lru():
    cache = pytest.importorskip("mlx_lm.models.cache")
    return cache.LRUPromptCache(max_size=10)


def test_same_size_different_images_diverge_at_the_image():
    lru = _lru()
    turn1 = K.expand(_ids([1, 2], A, [3, 4]), [A], PAD)
    lru.insert_cache("m", turn1, [_Entry()])
    # a different image of the same size, same text around it
    other = K.expand(_ids([1, 2], B, [3, 4, 5]), [B], PAD)
    res = lru._trie.search("m", other)
    assert res.common_prefix == 2, "must diverge at the image's first token"
    _, rest = lru.fetch_nearest_cache("m", other)
    assert len(rest) == len(other), "no entry may be reused past the image"
    # with PLAIN ids the two collide all the way -- what sentinels prevent
    assert K.to_ids(turn1, PAD) == K.to_ids(other, PAD)[:len(turn1)]


def test_same_image_hits_all_the_way_through():
    lru = _lru()
    turn1 = K.expand(_ids([1, 2], A, [3, 4]), [A], PAD)
    lru.insert_cache("m", turn1, [_Entry()])
    turn2 = turn1 + [5, 6]
    cache, rest = lru.fetch_nearest_cache("m", turn2)
    assert cache is not None and rest == [5, 6]


def test_adding_a_second_image_keeps_the_first_prefix():
    lru = _lru()
    turn1 = K.expand(_ids([1], A, [2]), [A], PAD)
    lru.insert_cache("m", turn1, [_Entry()])
    turn2 = K.expand(_ids([1], A, [2, 3], C, [4]), [A, C], PAD)
    _, rest = lru.fetch_nearest_cache("m", turn2)
    assert len(turn2) - len(rest) == len(turn1)


# --- registry ----------------------------------------------------------------------

def test_registry_table_is_the_frozen_one():
    assert registry.FAMILIES == {
        "qwen3_5": "knurlogic.engine.vision.qwen:build",
        "qwen3_5_moe": "knurlogic.engine.vision.qwen:build",
        "qwen4_exp": "knurlogic.engine.vision.qwen:build",
        "gemma4": "knurlogic.engine.vision.gemma4:build",
        "glm5_next": "knurlogic.engine.vision.glm5:build",
    }


def test_missing_family_module_means_no_vision(monkeypatch):
    monkeypatch.setitem(registry.FAMILIES, "fake",
                        "knurlogic.engine.vision.no_such_family:build")
    cfg = {"vision_config": {"x": 1}}
    assert registry.build("fake", "/nonexistent", None, cfg) is None
    assert registry.build("llama", "/nonexistent", None, cfg) is None
    assert not registry.has_family("fake")


def test_a_broken_family_module_is_not_silenced(tmp_path, monkeypatch):
    """An ImportError from INSIDE a present family module is a broken
    build; turning it into 'no vision' would hide it."""
    pkg = tmp_path / "brokenfam"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("import a_dependency_that_is_absent\n"
                                     "def build(*a): return 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(registry.FAMILIES, "fake", "brokenfam:build")
    with pytest.raises(ModuleNotFoundError):
        registry.build("fake", "/x", None, {"vision_config": {"x": 1}})


def test_registry_routes_to_a_present_family(tmp_path, monkeypatch):
    (tmp_path / "okfam.py").write_text(
        "def build(path, text_model, config):\n"
        "    return ('built', path, config['vision_config'])\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(registry.FAMILIES, "fake", "okfam:build")
    assert registry.build("fake", "/p", None, {"vision_config": {"v": 1}}) \
        == ("built", "/p", {"v": 1})
    # no vision_config: a text-only rung of a vision family
    assert registry.build("fake", "/p", None, {}) is None


# --- the front door stays stdlib -------------------------------------------------

def test_front_door_imports_no_engine_and_no_pil():
    """The page and the MCP read VisionSpec / served_vision(); asking must
    not import mlx or PIL (same rule as engine/mtp's front door)."""
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, 'src');"
         "import knurlogic.engine.vision as v;"
         "import knurlogic.engine.vision.key, knurlogic.engine.vision.store,"
         " knurlogic.engine.vision.registry, knurlogic.engine.vision.images;"
         "v.served_vision();"
         "print(sorted(k for k in sys.modules if k.split('.')[0] in "
         "('mlx', 'PIL') or k.startswith('mlx_')))"],
        capture_output=True, text=True, timeout=120,
        cwd=str(Path(__file__).resolve().parents[1]))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", out.stdout


def test_served_vision_round_trip():
    from knurlogic.engine import vision as v
    assert v.served_vision() is None
    spec = v.VisionSpec("stub", 5, 4, None, 16, 256, None, v.proc_hash({}))
    v.set_served_vision(spec)
    try:
        assert v.served_vision() is spec
        assert spec.to_json()["family"] == "stub"
    finally:
        v.set_served_vision(None)


def test_proc_hash_is_canonical():
    from knurlogic.engine.vision import proc_hash
    assert proc_hash({"a": 1, "b": 2}) == proc_hash({"b": 2, "a": 1})
    assert proc_hash({"a": 1}) != proc_hash({"a": 2})
    assert len(proc_hash({})) == 16


# --- scatter ------------------------------------------------------------------------

def test_masked_scatter_matches_the_mlx_vlm_golden():
    mx = pytest.importorskip("mlx.core")
    import numpy as np
    import fixtures_vision as fv
    from knurlogic.engine.vision.scatter import masked_scatter
    g, meta = fv.load_golden("p0_masked_scatter")
    assert meta["mlx_vlm"] == fv.REFERENCE_VERSION
    L, D = g["embeds"].shape[1:]
    mask = mx.broadcast_to(mx.array(g["mask_rows"])[None, :, None], (1, L, D))
    out = masked_scatter(mx.array(g["embeds"]), mask, mx.array(g["source"]))
    np.testing.assert_allclose(np.array(out), g["out"], atol=1e-6, rtol=0)


def test_merge_takes_rows_by_sentinel_across_a_prefix_cut():
    """A slice starting mid-image takes rows k0.. of THAT image -- no global
    feature index."""
    mx = pytest.importorskip("mlx.core")
    import numpy as np
    from knurlogic.engine.vision import EncodedImage
    from knurlogic.engine.vision.scatter import merge
    D = 4
    fa = mx.arange(3 * D, dtype=mx.float32).reshape(3, D) + 100
    fc = mx.arange(2 * D, dtype=mx.float32).reshape(2, D) + 200
    table = {A.sha: EncodedImage(A, fa), C.sha: EncodedImage(C, fc)}
    key = K.expand(_ids([1], A, [2], C), [A, C], PAD)     # len 1+3+1+2 = 7
    start = 2                                             # cut inside A
    sl = key[start:]
    text = mx.zeros((1, len(sl), D))
    out = np.array(merge(text, sl, lambda s, p: table[s]))[0]
    np.testing.assert_array_equal(out[0], np.array(fa[1]))
    np.testing.assert_array_equal(out[1], np.array(fa[2]))
    np.testing.assert_array_equal(out[2], np.zeros(D))    # text token 2
    np.testing.assert_array_equal(out[3:], np.array(fc))
    with pytest.raises(ValueError):
        merge(mx.zeros((1, 3, D)), sl, lambda s, p: table[s])


@pytest.mark.parametrize("model_type,family", [
    ("qwen3_5_text", "qwen3_5"), ("qwen3_5_moe_text", "qwen3_5_moe"),
    ("qwen4_exp_text", "qwen4_exp"), ("gemma4_text", "gemma4"),
    ("glm5_next_text", "glm5_next"), ("qwen3_5", "qwen3_5")])
def test_the_released_rungs_text_spelling_finds_its_vision_family(
        model_type, family):
    """All 20 released rungs report the text config's model_type; looked up
    raw, /models.json said none of them had vision."""
    from knurlogic.engine.vision import registry
    assert registry.family_of(model_type) == family
    assert registry.has_family(model_type)
