"""The vendored gemma4_text (engine/families/gemma4/architecture) held to
the maker's reference: HF transformers' modeling_gemma4.py run on the same
tiny random weights under torch in float32
(tests/support/goldens/build_gemma4_text.py).

Two configs: the e-style dense model (per-layer embeddings, KV sharing,
double-wide MLP, sliding/full mix, proportional partial RoPE) and the
26B-A4B-style MoE (router, per-expert scale, K=V full layers). Each runs
an 11-token prefill past the 6-token window, then 5 decode steps through
the model's own make_cache -- the rotating sliding cache wraps."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

mx = pytest.importorskip("mlx.core")

GOLD = np.load(ROOT / "tests/support/goldens/gemma4_text.npz")
#: float32 both sides; the reference and MLX differ only in summation order
ATOL = 2e-4


def _model(name):
    from mlx.utils import tree_flatten

    from knurlogic.engine import register
    # override: the vendored file, even if something imported mlx-lm's own
    register.register("gemma4_text", override=True)
    from mlx_lm.models import gemma4_text as arch
    assert "families/gemma4/architecture" in arch.__file__

    cfg = json.loads(str(GOLD[f"{name}/config"]))
    model = arch.Model(arch.ModelArgs.from_dict(cfg))
    pre = f"{name}/w/"
    weights = {k[len(pre):]: mx.array(GOLD[k]) for k in GOLD.files
               if k.startswith(pre)}
    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()), strict=True)
    assert len(weights) == len(tree_flatten(model.parameters()))
    mx.eval(model.parameters())
    return model


@pytest.mark.parametrize("name", ["dense", "moe"])
def test_gemma4_text_matches_the_reference(name):
    model = _model(name)
    cache = model.make_cache()
    prompt = mx.array(GOLD["prompt"])[None]
    prefill = model(prompt, cache=cache)[0]
    np.testing.assert_allclose(np.array(prefill), GOLD[f"{name}/prefill"],
                               atol=ATOL, rtol=0)
    for i, t in enumerate(GOLD["decode_ids"].tolist()):
        out = model(mx.array([[t]]), cache=cache)[0, -1]
        np.testing.assert_allclose(np.array(out), GOLD[f"{name}/decode"][i],
                                   atol=ATOL, rtol=0, err_msg=f"step {i}")


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
