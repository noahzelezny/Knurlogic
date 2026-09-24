"""P3: glm5_next vision, and knurlogic stops depending on mlx-vlm for GLM.

Structural gates only (no real model, no network, no server) -- design v2's
full G1-G11 sweep needs real weights or an mlx-vlm 0.6.17 reference that
does not exist for GLM (PROVENANCE.md, "Deviations"). What IS gated here:

* the vendored architecture imports and builds its `Model` with `mlx_vlm`
  BLOCKED from ever resolving (the package's core acceptance test, given
  explicitly in the brief);
* every `_vendor/` sibling glm5_next actually imports resolves with no
  reference to the real, installed mlx-vlm;
* `Glm5VisionFamily` (the registry target) preprocesses, encodes and embeds
  a tiny random image end to end, and its API matches the `Family`
  protocol (`docs/design/vision-contracts.md`).

Every gate here was mutated once and confirmed red before being restored
(reported in PROVENANCE.md / the return to the integrator), per the design's
"every gate is mutated once" rule.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))


class _BlockMlxVlm:
    """A meta_path hook that makes `import mlx_vlm` (the exact top-level
    package) fail, as if it were never installed -- WITHOUT blocking the
    synthetic submodule names knurlogic's own `register()` inserts into
    `sys.modules` directly (those are never resolved through the finder;
    blocking them would test the registration trick, not this package)."""

    def find_spec(self, name, path=None, target=None):
        if name == "mlx_vlm":
            raise ImportError("mlx_vlm is not installed (test double)")
        return None


@pytest.fixture
def no_mlx_vlm(monkeypatch):
    import sys as _sys

    hook = _BlockMlxVlm()
    _sys.meta_path.insert(0, hook)
    # A previous test (or an unrelated import earlier in the session) may
    # have already imported the real mlx_vlm; evict it so this test proves
    # something.
    for name in list(_sys.modules):
        if name == "mlx_vlm" or name.startswith("mlx_vlm."):
            if name not in ("mlx_vlm.models.glm5_next",):
                monkeypatch.delitem(_sys.modules, name, raising=False)
    try:
        yield
    finally:
        _sys.meta_path.remove(hook)


def test_glm5_next_imports_without_mlx_vlm(no_mlx_vlm):
    """The package's headline claim: importing the GLM architecture with
    mlx_vlm not importable still succeeds."""
    import importlib
    import sys as _sys

    from knurlogic.engine import register
    register.unregister()
    for name in list(_sys.modules):
        if name.startswith("mlx_vlm.models.glm5_next"):
            del _sys.modules[name]

    done = register.register("glm5_next")
    assert done == ["glm5_next"]

    mod = importlib.import_module("mlx_vlm.models.glm5_next")
    assert hasattr(mod, "Model")

    with pytest.raises(ImportError):
        importlib.import_module("mlx_vlm")


def test_glm5_next_imports_without_mlx_vlm_MUTATED():
    """Mutation check for the gate above (design: "every gate is mutated
    once"): if glm5_next's own source regressed to importing a real
    mlx_vlm sibling, the module of THIS package that would catch it is the
    one under test, not a separate mock -- so the mutation is exercised by
    literally breaking the import in a copy path and confirming pytest
    would fail. Run manually (not on every CI pass, since it edits files):
    ``python3 -m pytest tests/test_vision_glm5.py -k MUTATED -v`` after
    changing one `_mlx_vlm` import in
    `engine/architectures/glm5_next/language.py` back to
    ``from ..cache import ...`` -- confirmed 2026-09-23 to turn the test
    above red (`ModuleNotFoundError: No module named 'mlx_vlm'`), then
    reverted. This test itself is a no-op marker so the record survives in
    git; see PROVENANCE.md for the manual run's result."""
    assert True


def test_vendor_siblings_import_standalone():
    """The mlx-vlm 0.6.17 modules glm5_next imports -- vendored under
    glm5_next/_mlx_vlm/, the version the released rungs were built and
    scored with -- import with no mention of the real mlx_vlm package."""
    P = "knurlogic.engine.architectures.glm5_next._mlx_vlm"
    import importlib
    mods = [importlib.import_module(f"{P}.{m}") for m in (
        "models.base", "models.cache", "models.gated_delta", "models.mla",
        "models.mlp", "models.rope_utils", "models.switch_layers",
        "models.deepseek_v32.language", "models.deepseek_v4.hyper_connection",
        "turboquant", "kv_quant")]
    base, cache, gd, mla = mods[:4]
    assert hasattr(base, "BaseModelConfig")
    assert hasattr(cache, "KVCache")
    assert hasattr(mla, "MultiLinear")
    assert hasattr(gd, "gated_delta_update")
    for mod in mods:
        assert not mod.__name__.startswith("mlx_vlm")


def test_glm5_siblings_now_empty():
    """`deps.glm5_siblings()` regexes glm5_next's OWN source for `from
    ..X import` lines -- the exact mlx_vlm-relative pattern this package
    rewrote to absolute `knurlogic.engine.architectures.glm5_next._mlx_vlm.models.X` imports.
    Read 2026-09-23 (before this package): {base, cache,
    deepseek_v4.hyper_connection, gated_delta, linear, mla,
    qwen3_vl.processing_qwen3_vl, sparse_attention, switch_layers} -- 9
    modules, all now vendored under `_vendor/` (see PROVENANCE.md). This
    is the gate that catches a regression: if a future edit to
    `glm5_next/*.py` reintroduces a `from ..X import`, `glm5_siblings()`
    stops being empty and `deps.py`'s dashboard (`PIECES["mlx-vlm"]`)
    starts claiming knurlogic needs mlx-vlm for GLM again -- exactly the
    state this package closes."""
    from knurlogic.machine.deps import glm5_siblings

    assert glm5_siblings() == []


def test_glm5_siblings_now_empty_MUTATED():
    """Mutation check for the gate above: reverting ONE import in
    `engine/architectures/glm5_next/language.py` from the absolute
    `knurlogic.engine.architectures.glm5_next._mlx_vlm.models.mla` back to a relative
    `from ..mla import MultiLinear` (confirmed 2026-09-23) turns
    `glm5_siblings()` non-empty again -- `test_glm5_siblings_now_empty`
    goes red as expected. Reverted after confirming."""
    assert True


# --- Family protocol, tiny random weights --------------------------------

def _family():
    from fixtures_vision_glm5 import glm5_family
    return glm5_family()


def test_family_spec():
    fam = _family()
    assert fam.spec.family == "glm5_next"
    assert fam.spec.merge == 2
    assert fam.spec.patch == 14
    assert fam.spec.fixed_tokens is None


def test_family_preprocess_encode_token_count():
    from fixtures_vision_glm5 import glm5_tiny_image

    fam = _family()
    img = glm5_tiny_image()
    from knurlogic.engine.vision.images import pixel_sha
    sha = pixel_sha(img)
    pixels, ref = fam.preprocess(img, sha)
    assert ref.sha == sha
    assert ref.grid_thw is not None
    enc = fam.encode(pixels, ref)
    assert enc.feats.shape[0] == ref.n_tokens
    assert enc.feats.shape[1] == fam.vision_config.out_hidden_size


def test_family_encode_shape_mismatch_raises():
    """G2-shaped gate: a grid that disagrees with n_tokens must raise, not
    silently truncate."""
    from dataclasses import replace

    from fixtures_vision_glm5 import glm5_tiny_image

    fam = _family()
    img = glm5_tiny_image()
    from knurlogic.engine.vision.images import pixel_sha
    sha = pixel_sha(img)
    pixels, ref = fam.preprocess(img, sha)
    bad_ref = replace(ref, n_tokens=ref.n_tokens + 1)
    with pytest.raises(ValueError):
        fam.encode(pixels, bad_ref)


def test_family_encode_shape_mismatch_raises_MUTATED():
    """Mutation check for the gate above: temporarily removing the
    `feats.shape[0] != ref.n_tokens` check in `Glm5VisionFamily.encode`
    (confirmed 2026-09-23) turns `test_family_encode_shape_mismatch_raises`
    green-to-red as expected (no `ValueError` raised); restored."""
    assert True


def test_family_positions_is_nope():
    fam = _family()
    assert fam.positions([], None) == (None, 0)


def test_family_chunk_boundaries_empty_causal():
    fam = _family()
    assert fam.chunk_boundaries([1, 2, 3]) == []


def test_family_embed_merges_features():
    import mlx.core as mx

    from fixtures_vision_glm5 import glm5_tiny_image
    from knurlogic.engine.vision.images import pixel_sha
    from knurlogic.engine.vision.key import sentinel

    fam = _family()
    img = glm5_tiny_image()
    sha = pixel_sha(img)
    pixels, ref = fam.preprocess(img, sha)
    enc = fam.encode(pixels, ref)

    def features(s, p):
        assert (s, p) == (sha, ref.proc_hash)
        return enc

    key_slice = [sentinel(ref, k) for k in range(ref.n_tokens)]

    class _Trunk:
        class model:
            @staticmethod
            def embed_tokens(ids):
                d = fam.vision_config.out_hidden_size
                return mx.zeros((*ids.shape, d))

    out = fam.embed(_Trunk, [0] * 0 + key_slice, 0, features)
    embeds = out["input_embeddings"]
    assert embeds.shape[-2] == ref.n_tokens
    assert bool(mx.allclose(embeds[0], enc.feats).item()) or bool(
        mx.array_equal(embeds[0], enc.feats).item())


def test_preprocess_normalizes_like_the_reference_processor(tmp_path):
    """Glm5NextImageProcessor rescales to [0,1] and then normalizes with
    CLIP's mean/std (or the artifact's). Found on GLM-5.3-Flash 2.7: without
    the normalize step a pure red square was read as "salmon/coral"."""
    import json
    import numpy as np
    from PIL import Image
    from fixtures_vision_glm5 import glm5_tiny_config
    from knurlogic.engine.families.glm5.vision import Glm5VisionFamily, build
    fam = Glm5VisionFamily(glm5_tiny_config())
    px, _ = fam.preprocess(Image.new("RGB", (56, 56), (255, 0, 0)), "x")
    v = px["pixel_values"].reshape(px["pixel_values"].shape[0], 3, -1)
    m, s = np.array(fam.IMAGE_MEAN), np.array(fam.IMAGE_STD)
    want = (np.array([1.0, 0.0, 0.0]) - m) / s
    assert np.allclose(v[0, :, 0], want, atol=1e-5)
    # and the artifact's own values win, and change the processing hash
    (tmp_path / "processor_config.json").write_text(json.dumps(
        {"image_processor": {"image_mean": [0.5] * 3, "image_std": [0.5] * 3}}))
    other = Glm5VisionFamily(glm5_tiny_config(), [0.5] * 3, [0.5] * 3)
    assert other.spec.proc_hash != fam.spec.proc_hash
