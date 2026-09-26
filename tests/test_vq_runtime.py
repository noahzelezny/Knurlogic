"""knurlogic's own VQ runtime (design D1), and the numerics-default fix.

Three layers, cheapest first:

1. The record: the vendored files are the pinned vqlab 42df84f bytes, and
   rungs.json says what each released rung's PUBLISHED model.py ships --
   the three generations the design names, read off the Hub 2026-09-23.
2. The resolver: a rung's numerics come from the rung. The v2 rungs keep
   bf16 I/O on unless someone asks for a profile. (The bug: a v1.5 default
   forced them off on every VQ artifact.)
3. The runtime, on a tiny random-weight qwen3_moe with real VQ expert
   modules: knurlogic's runtime with a rung's knobs is token- and
   logit-identical to a bundle built the way vqlab builds one -- and NOT
   identical when the knobs are dropped, which is the bug, and is what
   proves the identity check can fail. The real-rung version of this is
   tools/vq_gate.py (G-VQ), run by the orchestrator; it is not run here.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from knurlogic.engine.vq import rungs as RG  # noqa: E402
from knurlogic.machine.artifact import Artifact  # noqa: E402
from knurlogic.tuning import settings as S  # noqa: E402
from knurlogic.tuning.resolve import numerics_for, resolve  # noqa: E402

VQ = ROOT / "src/knurlogic/engine/vq"
GIB = 1 << 30
ORG = "TheDrainFlorist"

V2 = {"Qwen3.8-Flash-Next-VQ-2.1bpw", "Qwen3.6-35B-A3B-VQ-3.8bpw",
      "Qwen3.6-35B-A3B-VQ-4.6bpw", "Qwen3.6-35B-A3B-VQ-5.4bpw"}
ARC6 = {"GLM-5.3-Flash-VQ-2.7bpw", "GLM-5.3-Flash-VQ-3.1bpw",
        "GLM-5.3-Flash-VQ-3.6bpw", "Qwen3.5-397B-A17B-VQ-2.4bpw",
        "Qwen3.5-397B-A17B-VQ-2.6bpw", "Qwen3.5-397B-A17B-VQ-3.1bpw"}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Every VQ flag unset, so each runtime reads its own defaults."""
    import os
    for k in list(os.environ):
        if k.startswith(("VQ_", "VQLAB_")):
            monkeypatch.delenv(k)
    RG.reload()
    yield


def _table():
    return json.loads(RG.RUNGS_JSON.read_text())


# --- 1. the record -----------------------------------------------------------

def test_vendored_runtime_is_the_pinned_bytes():
    """Verbatim vqlab 42df84f: the pin in runtime.py, rungs.json and
    PROVENANCE.md all name the same digests as the files on disk."""
    from knurlogic.engine.vq import runtime
    prov = (VQ / "PROVENANCE.md").read_text()
    for f, want in runtime.RUNTIME_FILES.items():
        got = hashlib.sha256((VQ / f).read_bytes()).hexdigest()
        assert got == want, f"{f} drifted from the vendored pin"
        assert _table()["runtime"]["files"][f] == want
        assert want in prov
    assert runtime.VQLAB_COMMIT == _table()["runtime"]["vqlab_commit"]
    assert runtime.VQLAB_COMMIT in prov


def test_every_released_rung_is_listed_and_starts_unverified():
    rows = _table()["rungs"]
    assert len(rows) == 20 and all(r.startswith(ORG + "/") for r in rows)
    ungated = [r for r, v in rows.items() if v["gate"] is None]
    assert all(not rows[r]["verified"] for r in ungated), \
        "a rung is verified only by a recorded G-VQ pass"
    for r, v in rows.items():
        if v["verified"]:
            assert v["gate"] and v["gate"]["pass"], r


def test_generations_are_what_the_hub_ships():
    """Design D1's three published generations, as recorded."""
    rows = {r.split("/", 1)[1]: v for r, v in _table()["rungs"].items()}
    assert {n for n, v in rows.items() if v["generation"] == "v2"} == V2
    assert {n for n, v in rows.items()
            if v["generation"] == "arc6-no-flags"} == ARC6
    assert all(v["generation"] == "v1.5" for n, v in rows.items()
               if n not in V2 | ARC6)
    assert all(v["runtime_change"] == (n in ARC6) for n, v in rows.items())


def test_knobs_reproduce_each_published_default_on_head():
    """HEAD's defaults + a rung's knobs == that rung's published defaults,
    for every flag both read. This is the claim the knobs exist to make."""
    from knurlogic.engine.vq import runtime
    head = runtime.head_defaults()
    for repo, v in _table()["rungs"].items():
        eff = runtime.effective_flags(v["knobs"])
        for flag, pub in v["published_defaults"].items():
            if flag in head:
                assert eff[flag] == pub, (repo, flag)
        for flag in v["inferred_knobs"]:
            assert eff[flag] == "0", (repo, flag)


def test_numerics_flag_names_have_one_home():
    assert RG._NUMERICS == S.NUMERICS_FLAGS


def test_generation_classifier():
    assert RG.generation({}) == "arc6-no-flags"
    assert RG.generation({f: "1" for f in S.NUMERICS_FLAGS}) == "v2"
    assert RG.generation({f: "0" for f in S.NUMERICS_FLAGS}) == "v1.5"
    assert RG.generation({S.NUMERICS_FLAGS[0]: "1",
                          S.NUMERICS_FLAGS[1]: "0"}) == "mixed"


def test_repo_of_reads_exo_and_hf_cache_names():
    assert RG.repo_of("/x/.exo/models/TheDrainFlorist--gemma-4-e4b-it-VQ-PLE") \
        == "TheDrainFlorist/gemma-4-e4b-it-VQ-PLE"
    assert RG.repo_of("/c/hub/models--TheDrainFlorist--Qwen3.8-27B-VQ-3.9bpw"
                      "/snapshots/abc") == "TheDrainFlorist/Qwen3.8-27B-VQ-3.9bpw"
    assert RG.repo_of("/some/local/build") is None


def test_flag_defaults_reads_multiline_gets():
    src = ('A = os.environ.get("VQ_A", "1") == "1"\n'
           'B = int(os.environ.get("VQ_B",\n        4096))\n'
           'C = os.environ.get("VQ_C")\n')
    assert RG.flag_defaults(src) == {"VQ_A": "1", "VQ_B": "4096"}


# --- 2. the resolver: a rung's numerics are the rung's -----------------------

def _art(name, tmp_path=None, **kw):
    base = dict(path=Path(f"/nonexistent/{ORG}--{name}"),
                model_type="qwen3_5_moe", model_file="model.py",
                bytes_on_disk=20 * GIB, hidden_size=2048,
                moe_intermediate_size=512,
                vq_modules={"m": {"d": 4, "K": 256}})
    base.update(kw)
    return Artifact(**base)


@pytest.mark.parametrize("name", sorted(V2))
def test_v2_rungs_keep_bf16_io_on_by_default(name):
    """THE BUG: resolve() with no profile used to force these to 0."""
    r = resolve(_art(name), 96 * GIB)
    assert all(r.env[f] == "1" for f in S.NUMERICS_FLAGS), r.env
    assert any("as shipped" in n for n in r.notes)


@pytest.mark.parametrize("name", ["Qwen3.8-Flash-Next-VQ-4.4bpw",
                                  "Qwen3.8-27B-VQ-3.9bpw",
                                  "Qwen3.5-397B-A17B-VQ-2.6bpw"])
def test_v15_and_arc6_rungs_resolve_off(name):
    r = resolve(_art(name), 96 * GIB)
    assert all(r.env[f] == "0" for f in S.NUMERICS_FLAGS)


def test_a_profile_applies_only_when_asked_and_says_what_it_overrode():
    r = resolve(_art("Qwen3.8-Flash-Next-VQ-2.1bpw"), 96 * GIB, "v1.5")
    assert all(r.env[f] == "0" for f in S.NUMERICS_FLAGS)
    note = next(n for n in r.notes if "runtime profile" in n)
    assert "asked for" in note and "1->0" in note


def test_declared_knobs_outrank_the_published_record():
    a = _art("Qwen3.8-Flash-Next-VQ-2.1bpw",
             knobs={"VQ_GEMMSEG_BF16IO": {"default": "0"}})
    env, note = numerics_for(a)
    assert env["VQ_GEMMSEG_BF16IO"] == "0"      # declared
    assert env["VQ_DECODE_BF16IO"] == "1"       # published
    assert "declared" in note and "published" in note


def test_unlisted_rung_falls_back_to_its_bundled_runtime(tmp_path):
    d = tmp_path / "local-build"
    d.mkdir()
    (d / "model.py").write_text(
        'import os\n_G = os.environ.get("VQ_GEMMSEG_BF16IO", "1") == "1"\n'
        '_D = os.environ.get("VQ_DECODE_BF16IO", "1") == "1"\n')
    env, note = numerics_for(_art("x", path=d))
    assert env == {f: "1" for f in S.NUMERICS_FLAGS} and "bundled" in note


def test_nothing_declared_emits_nothing(tmp_path):
    env, note = numerics_for(_art("x", path=tmp_path, model_file=None))
    assert env == {} and "runtime's own defaults" in note


def test_verified_switch_defaults_off_and_follows_the_record(monkeypatch):
    from knurlogic.engine.vq import runtime
    p = f"/m/{ORG}--Qwen3.8-27B-VQ-3.9bpw"
    t = json.loads(json.dumps(RG.table()))
    monkeypatch.setattr(RG, "_table", lambda: t)
    t["rungs"][f"{ORG}/Qwen3.8-27B-VQ-3.9bpw"]["verified"] = False
    assert runtime.serves(p) is False, "an unverified rung stays on its bundle"
    t["rungs"][f"{ORG}/Qwen3.8-27B-VQ-3.9bpw"]["verified"] = True
    assert runtime.serves(p) is True
    assert runtime.serves("/m/some-other-model") is False


# --- 3. the runtime on a tiny VQ model ---------------------------------------

mx = pytest.importorskip("mlx.core")

E, D, K, G = 4, 4, 256, 64
TINY = dict(model_type="qwen3_moe", vocab_size=512, hidden_size=256,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=64, intermediate_size=256, moe_intermediate_size=128,
            num_experts=E, num_experts_per_tok=2, decoder_sparse_step=1,
            mlp_only_layers=[], norm_topk_prob=True, rms_norm_eps=1e-6,
            rope_theta=10000.0, max_position_embeddings=512,
            tie_word_embeddings=False, torch_dtype="bfloat16")

#: The MoE bundle shim, reduced to the mlx-lm branch: what vqlab's
#: add_model_file.py appends to vq_switch.py for a text load. Test-only --
#: it stands in for a published bundle so the identity check needs no rung.
SHIM = '''

import importlib as _importlib, json as _json, pathlib as _pathlib
_cfg = _json.load(open(_pathlib.Path(__file__).parent / "config.json"))
_arch = _importlib.import_module("mlx_lm.models." + _cfg["model_type"])
ModelArgs = _arch.ModelArgs


class Model(_arch.Model):
    def __init__(self, args):
        super().__init__(args)
        for _path, _m in _cfg.get("vq_modules", {}).items():
            _obj = self
            _parts = _path.split(".")
            for _c in _parts[:-1]:
                _obj = _obj[int(_c)] if _c.isdigit() else getattr(_obj, _c)
            _ncol = _m["in"] // _m["dim"]
            setattr(_obj, _parts[-1], VQSwitchLinear(
                mx.zeros((_m["experts"], _m["out"], _ncol), dtype=mx.uint8),
                mx.zeros((_m["k"], _m["dim"]), dtype=mx.float16),
                mx.zeros((_m["experts"], _m["out"], _m["in"] // _m["group"]),
                         dtype=mx.float16),
                group_size=_m["group"]))
'''


def _bake(src: str, defaults: dict) -> str:
    """Rewrite flag defaults in the runtime text, as vqlab's
    runtime_profile does when it bundles a rung."""
    for flag, v in defaults.items():
        src, n = re.subn(
            r'(os\.environ\.get\(\s*"%s"\s*,\s*)"[^"]*"' % flag,
            r'\g<1>"%s"' % v, src)
        assert n, flag
    return src


def _tiny_artifact(root: Path, baked: dict) -> Path:
    """config.json + random weights (bf16: the numerics flags act on bf16
    I/O only) + a bundle model.py baked with `baked` defaults."""
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.models import qwen3_moe
    from knurlogic.engine.vq import runtime

    mx.random.seed(0)
    root.mkdir()
    model = qwen3_moe.Model(qwen3_moe.ModelArgs.from_dict(TINY))
    weights = dict(tree_flatten(model.parameters()))
    vq_modules = {}
    for layer in range(TINY["num_hidden_layers"]):
        for proj, (out, inn) in {"gate_proj": (128, 256), "up_proj": (128, 256),
                                 "down_proj": (256, 128)}.items():
            p = f"model.layers.{layer}.mlp.switch_mlp.{proj}"
            weights.pop(p + ".weight")
            weights[p + ".codes"] = mx.random.randint(
                0, K, (E, out, inn // D)).astype(mx.uint8)
            weights[p + ".codebook"] = (
                mx.random.normal((K, D)) * 0.2).astype(mx.float16)
            weights[p + ".vq_scales"] = mx.random.uniform(
                0.5, 1.5, (E, out, inn // G)).astype(mx.float16)
            vq_modules[p] = {"experts": E, "out": out, "in": inn, "k": K,
                             "dim": D, "group": G}
    weights = {k: (v.astype(mx.bfloat16) if v.dtype == mx.float32 else v)
               for k, v in weights.items()}
    mx.save_safetensors(str(root / "model.safetensors"), weights)
    cfg = dict(TINY, model_file="model.py", vq_modules=vq_modules)
    (root / "config.json").write_text(json.dumps(cfg))
    (root / "model.py").write_text(_bake(runtime.source(), baked) + SHIM)
    del nn
    return root


def _run(model, n=40):
    """Prompt logits (float32) and n greedy tokens."""
    import numpy as np
    from mlx_lm.models.cache import make_prompt_cache
    ids = mx.array([[1, 7, 42, 99, 3, 250, 17, 88, 5, 311, 2, 64]])
    cache = make_prompt_cache(model)
    logits = model(ids, cache=cache)
    prompt = np.array(logits.astype(mx.float32))
    nxt = mx.argmax(logits[:, -1, :], axis=-1)
    toks = []
    for _ in range(n):
        toks.append(int(nxt.item()))
        logits = model(nxt[:, None], cache=cache)
        nxt = mx.argmax(logits[:, -1, :], axis=-1)
    return prompt, toks


def _bundle(path):
    from mlx_lm.utils import load_model
    return load_model(path)[0]       # executes the artifact's model.py


V2_KNOBS = {"VQ_GEMMSEG_BF16IO": "1", "VQ_DECODE_BF16IO": "1",
            "VQ_DENSE_SS": "0"}


@pytest.fixture(scope="module")
def v2_rung(tmp_path_factory):
    return _tiny_artifact(tmp_path_factory.mktemp("vq") / "v2", V2_KNOBS)


def test_runtime_attaches_every_vq_module(v2_rung):
    from knurlogic.engine.vq import runtime
    m, _ = runtime.load_model(v2_rung, knobs=V2_KNOBS)
    assert m._vq_attached == 6
    assert type(m.model.layers[0].mlp.switch_mlp.gate_proj).__name__ \
        == "VQSwitchLinear"


def test_identity_with_the_rungs_knobs_and_not_without(v2_rung, tmp_path):
    """G-VQ in miniature, and its mutation in the same breath.

    knurlogic's runtime given the v2 rung's knobs == the v2 bundle (logits
    atol 1e-5, 40 greedy tokens). Given NO knobs -- HEAD's v1.5 defaults,
    i.e. what the old resolver forced on this rung -- the logits move past
    the tolerance, so the identity check can fail."""
    import numpy as np
    from knurlogic.engine.vq import runtime
    from vq_gate import compare

    ref = _run(_bundle(v2_rung))
    ours = _run(runtime.load_model(v2_rung, knobs=V2_KNOBS)[0])
    wrong = _run(runtime.load_model(v2_rung, knobs={})[0])

    def npz(name, r):
        f = tmp_path / f"{name}.npz"
        np.savez(f, logits=r[0], tokens=np.array(r[1]))
        return f

    good = compare(npz("ref", ref), npz("ours", ours))
    assert good["pass"], good
    assert good["max_abs_logit_diff"] == 0.0
    bad = compare(npz("ref2", ref), npz("wrong", wrong))
    assert not bad["pass"] and bad["max_abs_logit_diff"] > 1e-5, bad


def test_env_set_by_a_person_wins_over_the_rung(monkeypatch):
    from knurlogic.engine.vq import runtime
    assert runtime.runtime_module(V2_KNOBS)._GEMMSEG_BF16IO is True
    monkeypatch.setenv("VQ_GEMMSEG_BF16IO", "0")
    assert runtime.runtime_module(V2_KNOBS)._GEMMSEG_BF16IO is False


def test_each_knob_set_gets_its_own_module():
    """Flags are frozen at import; one shared module would give every rung
    the first rung's numerics."""
    from knurlogic.engine.vq import runtime
    a = runtime.runtime_module(V2_KNOBS)
    b = runtime.runtime_module({})
    assert a is not b and a is runtime.runtime_module(dict(V2_KNOBS))
    assert (a._DECODE_BF16IO, b._DECODE_BF16IO) == (True, False)
    import os
    assert "VQ_DECODE_BF16IO" not in os.environ, "overlay must be undone"


def test_serve_loads_a_verified_rung_on_knurlogics_runtime(tmp_path,
                                                           monkeypatch):
    """The routing (engine/serve/load.load_unlocked, which the model host
    calls): a rung rungs.json lists as verified loads through knurlogic's
    runtime; anything else through the artifact's own loader, bundled
    model.py and all."""
    # by module path: the package re-exports a FUNCTION named `load`
    from knurlogic.engine.serve.load import load_unlocked
    from knurlogic.engine.serve import state
    from knurlogic.engine.vq import runtime
    import mlx_lm.utils as mu

    calls = []
    verified = tmp_path / "verified"
    other = tmp_path / "other"
    verified.mkdir(), other.mkdir()
    monkeypatch.setattr(runtime, "serves", lambda p: p.name == "verified")
    monkeypatch.setattr(runtime, "load_model", lambda p, lazy=False: (
        calls.append(("knurlogic", str(p))) or ("M", {"eos_token_id": 1})))
    monkeypatch.setattr(mu, "load_tokenizer", lambda *a, **k: "T")
    monkeypatch.setattr(mu, "load", lambda p, **k: (
        calls.append(("bundled", str(p))) or ("m", "t")))

    assert load_unlocked(str(verified)) == ("M", "T")
    assert state.SERVED["runtime"] == "knurlogic"
    assert load_unlocked(str(other)) == ("m", "t")
    assert [c[0] for c in calls] == ["knurlogic", "bundled"]
    assert state.SERVED["runtime"] == "bundled"

