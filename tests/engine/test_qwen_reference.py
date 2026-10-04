"""knurlogic's vendored Qwen trunks compute what Qwen's reference computes.

tests/support/goldens/qwen_reference.npz holds the logits of HF
transformers' own Qwen3.5 / Qwen3.5-MoE / Qwen4-Exp text models (float32,
CPU, eager) on a tiny random config, for a 21-token prefill and 4 decode
steps through the reference cache (build_qwen_reference.py says what the
config makes run). Here the same weights, re-made from numpy by name, go
through knurlogic's sanitize into its MLX trunk, float32, through its own
cache, and the logits must agree.
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


def _model(family, meta):
    from knurlogic.engine import register
    register.register(family, override=True)
    arch = importlib.import_module(f"mlx_lm.models.{family}")
    cfg = dict(meta[f"{family}/config"])
    top = cfg["model_type"].removesuffix("_text")
    model = arch.Model(arch.ModelArgs.from_dict(
        {"model_type": top, "text_config": cfg}))
    model.set_dtype(mx.float32)
    w = G.init_weights({k: tuple(v) for k, v in
                        meta[f"{family}/shapes"].items()})
    if family == "qwen4_exp":
        w = _shard_ngram(w, cfg)
    w = model.sanitize({k: mx.array(v) for k, v in w.items()})
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
    n = cfg["split_ngram_parts"]
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
    return np.array(mx.concatenate(out, axis=0))


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
