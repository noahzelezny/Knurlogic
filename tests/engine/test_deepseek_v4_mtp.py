"""deepseek_v4's MTP head: a random head packed the way vqlab packs one
(`mtp.0.*`, experts mxfp4) beside the tiny deepseek_v4 is found, bound,
and drafts -- and greedy output with drafting is the plain steps' output,
token for token, through rejections that roll the untrimmable
DeepseekV4Cache back. A header-only check binds the real VQ Lab head."""
import json
import os
import shutil
import struct
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests" / "support" / "goldens"))

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

import build_deepseek_v4 as G  # noqa: E402

#: VQ Lab's head beside its Flash build (optional; headers only)
REAL = Path(os.environ.get("KNURLOGIC_TEST_DEEPSEEK_V4_HEAD") or
            "/Volumes/Thunderbay SSD/Exo Models/"
            "TheDrainFlorist--DeepSeek-V4-Flash-VQ-3.2bpw")


def _load():
    from mlx_lm.utils import load_model

    from knurlogic.interfaces import loading
    from knurlogic.machine.artifact import Artifact
    assert loading.register(Artifact.load(str(G.TINY))) == []
    model, _ = load_model(G.TINY)
    return model


def _arch(model):
    import importlib
    return importlib.import_module(type(model.model).__module__)


def _pack(model, path):
    """A random head, experts mxfp4 like the trunk's, saved as a sidecar."""
    from knurlogic.engine.families.deepseek.heads.deepseek_v4 import MTPHead
    mx.random.seed(7)
    h = MTPHead(model, _arch(model))
    from mlx.utils import tree_flatten, tree_unflatten
    new = []
    for k, v in tree_flatten(h.m.parameters()):
        if k.endswith("norm.weight"):
            new.append((k, mx.ones(v.shape)))
        else:
            new.append((k, 0.15 * mx.random.normal(v.shape)))
    h.m.update(tree_unflatten(new))
    nn.quantize(h.m, class_predicate=lambda p, m: (
        "switch_mlp" in p and hasattr(m, "to_quantized")
        and {"group_size": 32, "bits": 4, "mode": "mxfp4"}))
    return h.save(path)


def _run(gen, prompts, max_tokens):
    uids = gen.insert_segments(
        segments=[[p] for p in prompts], max_tokens=[max_tokens] * len(prompts),
        caches=[None] * len(prompts), all_tokens=[[] for _ in prompts])
    out, done = {u: [] for u in uids}, set()
    for _ in range(10_000):
        _, grs = gen.next()
        for r in grs:
            out[r.uid].append(r.token)
            if r.finish_reason is not None:
                done.add(r.uid)
        if len(done) == len(uids):
            break
    gen.close()
    return [out[u] for u in uids]


def test_the_sidecar_is_found_as_a_deepseek_v4_head(tmp_path):
    from knurlogic.engine import mtp
    model = _load()
    flat = _pack(model, tmp_path / "mtp-head-mxfp4.safetensors")
    assert all(k.startswith("mtp.0.") for k in flat)
    assert "mtp.0.hc_head_fn" in flat and "mtp.0.e_proj.weight" in flat
    assert flat["mtp.0.ffn.switch_mlp.gate_proj.weight"].dtype == mx.uint32
    head = mtp.find_head(tmp_path)
    assert head is not None and head.family == "deepseek_v4"


@pytest.mark.parametrize("prompts", [
    [G.PROMPT],
    [G.PROMPT, G.PROMPT[:3], G.PROMPT[2:9] + G.DECODE],
], ids=["one-row", "three-rows"])
@pytest.mark.parametrize("guess", [False, True], ids=["head", "repeat"])
def test_greedy_drafting_is_the_plain_steps_token_for_token(
        prompts, guess, tmp_path, monkeypatch):
    """`head`: the random head's own drafts, nearly all rejected (every
    step rolls the trunk back). `repeat`: the head still runs and fills its
    cache, but drafts "the last token again", which the tiny model's
    greedy loops accept -- so accepted steps keep the verify's cache."""
    from knurlogic.engine.mtp import registry
    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    monkeypatch.setenv("KNURLOGIC_MTP_BATCH_MAX_ROWS", "8")   # draft always
    model = _load()
    shutil.copytree(G.TINY, tmp_path / "m")
    _pack(model, tmp_path / "m" / "mtp-head-mxfp4.safetensors")
    head, spec = registry.load_head(model, model_path=tmp_path / "m")
    assert spec.name == "deepseek_v4"
    if guess:
        real = head.draft_logits

        def draft_logits(h, ids, cache=None):
            out = real(h, ids, cache)
            return out + 100.0 * (mx.arange(out.shape[-1])
                                  == ids[:, :, None]).astype(out.dtype)
        head.draft_logits = draft_logits
    plain = _run(MTPBatchGenerator(model, None, prefill_step_size=4),
                 prompts, 24)
    stats = {}
    draft = _run(MTPBatchGenerator(model, head, stats=stats,
                                   prefill_step_size=4), prompts, 24)
    assert stats["steps"] > 0
    if guess:
        assert stats["accepted"] > 0
    assert draft == plain
    assert all(len(t) == 24 for t in draft)


def test_no_head_leaves_the_trunk_as_it_was():
    """The trunk still drops `mtp.*` and declares nothing new."""
    model = _load()
    from mlx.utils import tree_flatten
    assert not [k for k, _ in tree_flatten(model.parameters())
                if {"mtp", "e_proj", "h_proj"} & set(k.split("."))]


def _header(f):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))


@pytest.mark.skipif(not (REAL / "mtp-head-mxfp4.safetensors").is_file(),
                    reason="VQ Lab's DeepSeek-V4-Flash head is not mounted")
def test_the_real_vqlab_head_binds_by_its_header_alone():
    """Every name and shape of the packed Flash head, against a head built
    from the build's own config -- lazily, nothing loaded."""
    from knurlogic.engine import register
    from knurlogic.engine.families.deepseek.heads.deepseek_v4 import MTPHead
    register.register("deepseek_v4")
    import mlx_lm.models.deepseek_v4 as M
    hdr = _header(REAL / "mtp-head-mxfp4.safetensors")
    hdr.pop("__metadata__", None)
    cfg = json.loads((REAL / "config.json").read_text())
    args = M.ModelArgs.from_dict(cfg)
    layer = M.DeepseekV4Block.__new__(M.DeepseekV4Block)
    core = types.SimpleNamespace(args=args, layers=[layer])
    head = MTPHead(types.SimpleNamespace(model=core), M)
    shapes = {k: types.SimpleNamespace(shape=tuple(v["shape"]))
              for k, v in hdr.items()}
    mw = head.bind(shapes)
    assert len(mw) == len(hdr) == 34
    assert type(head.m.block.ffn.switch_mlp.gate_proj).__name__ \
        == "QuantizedSwitchLinear"
    assert head.m.block.ffn.switch_mlp.gate_proj.mode == "mxfp4"
    assert isinstance(head.m.e_proj, nn.Linear)        # bf16, not packed
