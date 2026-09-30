"""The model's own launch settings: MTP drafting on/off, dynamic MTP, and
KV-cache precision -- resolved per family, counted in memory, refused
with a reason where a family cannot take them, and passed ring-wide."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from knurlogic.machine.artifact import Artifact
from knurlogic.tuning import settings as S
from knurlogic.tuning.resolve import (context_room, kv_bytes_per_token,
                                      kv_refusal, resolve)

GIB = 1 << 30
TC = {"num_hidden_layers": 8, "full_attention_interval": 4,
      "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 256,
      "max_position_embeddings": 262144}


def _art(path, model_type="qwen3_5_moe_text", **kw):
    base = dict(path=Path(path), model_type=model_type, model_file=None,
                bytes_on_disk=20 * GIB, raw_config=dict(TC),
                hidden_size=4096, moe_intermediate_size=1024)
    base.update(kw)
    return Artifact(**base)


def _with_head(tmp_path):
    from test_mtp import _safetensors
    d = tmp_path / "m"
    d.mkdir()
    _safetensors(d / "mtp-head-q6.safetensors",
                 ["block.x", "fc.weight", "norm_e.weight", "norm_h.weight",
                  "norm_out.weight"], {"vqlab_mtp": "{}"})
    return d


# --- resolved per artifact -------------------------------------------------------

def test_mtp_knobs_appear_only_where_a_head_ships(tmp_path):
    r = resolve(_art(_with_head(tmp_path)), 96 * GIB)
    assert r.env["KNURLOGIC_MTP"] == "on"
    assert r.env["KNURLOGIC_MTP_DYNAMIC"] == "on"
    bare = resolve(_art(tmp_path / "none"), 96 * GIB)
    assert "KNURLOGIC_MTP" not in bare.env
    assert "KNURLOGIC_MTP_DYNAMIC" not in bare.env


@pytest.mark.parametrize("mt,bits", [("qwen3_5_text", ["8", "6", "4"]),
                                     ("qwen3_5_moe", ["8", "6", "4"]),
                                     ("gemma4", ["8", "6", "4"]),
                                     ("gemma4_text", ["8", "6", "4"]),
                                     ("qwen4_exp_text", ["8", "6", "4"]),
                                     ("glm5_next", ["8"])])
def test_kv_bits_offered_per_family(tmp_path, mt, bits):
    r = resolve(_art(tmp_path, mt), 96 * GIB)
    assert r.env["KNURLOGIC_KV_BITS"] == "bf16"
    assert r.ranges["KNURLOGIC_KV_BITS"] == ["bf16"] + bits
    assert kv_refusal(_art(tmp_path, mt), 8) is None
    for b in ("6", "4"):
        assert (kv_refusal(_art(tmp_path, mt), int(b)) is None) == (b in bits)
    assert kv_refusal(_art(tmp_path, mt), None) is None


def test_the_launch_knobs_need_a_reload_and_are_reach_tier():
    from knurlogic.engine.serve.load import LIVE_KNOBS
    for k in S.MODEL_KNOBS:
        assert k in S.ENGINE_KNOB_NAMES and k not in LIVE_KNOBS
        assert S.knob_tier(k) == "reach" and k in S.KNOB_DOC


def test_engine_settings_reads_them():
    got = S.engine_settings({"KNURLOGIC_MTP": "off",
                             "KNURLOGIC_MTP_DYNAMIC": "on",
                             "KNURLOGIC_KV_BITS": "6"})
    assert got == {"mtp": False, "mtp_dynamic": True, "kv_bits": 6}
    assert S.engine_settings({"KNURLOGIC_KV_BITS": "bf16"}) == {
        "kv_bits": None}
    with pytest.raises(ValueError):
        S.engine_settings({"KNURLOGIC_KV_BITS": "5"})
    with pytest.raises(ValueError):
        S.engine_settings({"KNURLOGIC_MTP": "maybe"})


def test_the_launch_allowlist_takes_them():
    from knurlogic.interfaces.page.server import clean_sets
    ok, bad = clean_sets({"KNURLOGIC_MTP": "off",
                          "KNURLOGIC_MTP_DYNAMIC": "off",
                          "KNURLOGIC_KV_BITS": "8"})
    assert not bad and len(ok) == 3


# --- memory ------------------------------------------------------------------------

@pytest.mark.parametrize("bits,el", [(8, 1.0625), (6, 0.8125), (4, 0.5625)])
def test_kv_bytes_per_token_count_the_bits(bits, el):
    bf16, _ = kv_bytes_per_token(TC)
    q, why = kv_bytes_per_token(TC, bits)
    assert bf16 == 2 * 2 * 2 * 256 * 2
    assert q == int(2 * 2 * 2 * 256 * el) and f"{bits}-bit" in why


def test_the_room_left_holds_more_tokens_at_fewer_bits():
    a = context_room(64 * GIB, 40 * GIB, {"text_config": TC})
    b = context_room(64 * GIB, 40 * GIB, {"text_config": TC}, kv_bits=4)
    assert b["tokens"] > 3 * a["tokens"]


def test_an_mla_latent_is_counted_at_the_bits_and_its_keys_are_not():
    tc = {"num_hidden_layers": 2, "kv_lora_rank": 512,
          "qk_rope_head_dim": 64, "index_head_dim": 128,
          "layer_types": ["linear_attention", "deepseek_sparse_attention"]}
    assert kv_bytes_per_token(tc)[0] == (512 + 192) * 2
    assert kv_bytes_per_token(tc, 8)[0] == int(512 * 1.0625 + 192 * 2)


def test_the_scheduler_costs_the_first_prompt_at_the_bits(tmp_path):
    import json
    from knurlogic.engine.runtime.scheduler import _kv_from_config
    (tmp_path / "config.json").write_text(json.dumps(
        {"model_type": "qwen3_5_moe", "text_config": TC}))
    full = _kv_from_config(tmp_path)
    q8 = _kv_from_config(tmp_path, 8)
    assert full and q8 and q8[1] == pytest.approx(full[1] * 1.0625 / 2)


# --- the engine's switches --------------------------------------------------------

def test_dynamic_off_drafts_every_step(monkeypatch):
    pytest.importorskip("mlx.core")
    from knurlogic.engine.mtp.batch_loop import (ALWAYS_DRAFT,
                                                 default_draft_max_rows)
    monkeypatch.delenv("KNURLOGIC_MTP_BATCH_MAX_ROWS", raising=False)
    monkeypatch.delenv("KNURLOGIC_MTP_DYNAMIC", raising=False)
    assert default_draft_max_rows() is None
    monkeypatch.setenv("KNURLOGIC_MTP_DYNAMIC", "off")
    assert default_draft_max_rows() == ALWAYS_DRAFT
    monkeypatch.setenv("KNURLOGIC_MTP_DYNAMIC", "on")
    assert default_draft_max_rows() is None
    # an explicit ceiling still wins
    monkeypatch.setenv("KNURLOGIC_MTP_DYNAMIC", "off")
    monkeypatch.setenv("KNURLOGIC_MTP_BATCH_MAX_ROWS", "2")
    assert default_draft_max_rows() == 2


def test_the_host_refuses_bits_on_a_model_it_cannot_quantize(monkeypatch):
    pytest.importorskip("mlx.core")
    from knurlogic.engine.runtime.host import ModelHost

    class M:
        def make_cache(self):
            return [object()]
    h = ModelHost(kv_bits=8)
    h.model = M()
    with pytest.raises(RuntimeError, match="bf16"):
        h._quantize_kv()


# --- serve --------------------------------------------------------------------------

def _serve_until_resolve(monkeypatch, art, argv):
    """Run serve.main up to the resolver; return what it was given."""
    from knurlogic.interfaces import loading, serve
    seen = {}
    monkeypatch.setattr(serve.Artifact, "load", staticmethod(lambda p: art))
    monkeypatch.setattr(loading, "register", lambda a: [])

    def stop(*a, **k):
        seen.update(k)
        raise SystemExit(0)
    monkeypatch.setattr(serve, "resolve", stop)
    monkeypatch.setattr(serve.wired, "load_budget",
                        lambda: {"bytes": 0, "limited_by": "",
                                 "working_set_bytes": 0,
                                 "available_bytes": 0})
    try:
        rc = serve.main(argv)
    except SystemExit:
        rc = None
    return rc, seen


def test_serve_refuses_kv_bits_for_a_family_that_cannot(tmp_path,
                                                         monkeypatch,
                                                         capsys):
    art = _art(tmp_path, "glm5_next")
    rc, seen = _serve_until_resolve(monkeypatch, art,
                                    [str(tmp_path), "--kv-bits", "4"])
    assert rc == 2 and not seen
    assert "MLA latent" in capsys.readouterr().err


def test_serve_counts_the_bits_it_launches_with(tmp_path, monkeypatch):
    art = _art(tmp_path)
    rc, seen = _serve_until_resolve(
        monkeypatch, art, [str(tmp_path), "--set", "KNURLOGIC_KV_BITS=4"])
    assert seen["kv_bits"] == 4


def test_a_cluster_job_passes_the_same_sets_to_every_rank():
    """Ring-wide: one `sets` dict in the job's base spec, the same --set
    flags on every rank's argv."""
    from knurlogic.cluster.launch import rank_argv
    sets = {"KNURLOGIC_KV_BITS": "8", "KNURLOGIC_MTP_DYNAMIC": "off"}
    argvs = [rank_argv("/m", {"rank": r, "world": 2, "split": "pipeline",
                              "link": "ring", "job": "ab", "hosts": ["a", "b"],
                              "prefill_chunk": 512, "sets": sets}, {})
             for r in (0, 1)]
    for a in argvs:
        assert "KNURLOGIC_KV_BITS=8" in a and "KNURLOGIC_MTP_DYNAMIC=off" in a


def test_settings_offer_a_family_only_the_bits_it_takes(tmp_path):
    """Settings -> MODELS: one row per launch knob, needs reload, and the
    KV control narrowed to what the family allows."""
    from knurlogic.interfaces.page import documents
    for mt, want in (("glm5_next", ["bf16", "8"]),
                     ("qwen3_5_text", ["bf16", "8", "6", "4"])):
        (tmp_path / mt).mkdir()
        a = _art(_with_head(tmp_path / mt), mt)
        doc = documents.settings_document(
            a, live_env={}, live_tune="balanced", live_working_set=96 * GIB,
            resolve_fn=lambda ws, t: resolve(a, ws, tune=t))({})
        ks = {k["name"]: k for k in doc["knobs"]}
        assert ks["KNURLOGIC_KV_BITS"]["values"] == want
        # the 8-bit decode kernel's switch is offered only at 8 bits
        assert "KNURLOGIC_KV_KERNEL" not in ks
        for k in S.MODEL_KNOBS:
            if k == "KNURLOGIC_LONG_CONTEXT" and \
                    S.long_context_family(mt) is None:
                # offered only where a model card documents YaRN
                assert k not in ks
            elif k != "KNURLOGIC_KV_KERNEL":
                assert ks[k]["reach"] == "restart"


def test_the_kv_kernel_switch_is_emitted_only_at_8_bits(tmp_path):
    a = _art(tmp_path, "qwen3_5_text")
    assert "KNURLOGIC_KV_KERNEL" not in resolve(a, 96 * GIB).env
    r = resolve(a, 96 * GIB, kv_bits=8)
    assert r.env["KNURLOGIC_KV_KERNEL"] == "on"
    assert "KNURLOGIC_KV_KERNEL" not in resolve(a, 96 * GIB, kv_bits=4).env


def test_bf16_rows_do_not_follow_the_vision_allowance_dtype(monkeypatch):
    """DeepSeek-V4's pools and MLA rope keys are bf16 in their own right;
    changing the vision allowance's dtype must not move them."""
    from knurlogic.tuning import settings as S
    v4 = {"model_type": "deepseek_v4", "num_hidden_layers": 3,
          "head_dim": 512, "index_head_dim": 128, "compress_ratios": [0, 4, 128],
          "sliding_window": 128}
    mla = {"num_hidden_layers": 2, "kv_lora_rank": 512,
           "qk_rope_head_dim": 64, "index_head_dim": 128,
           "layer_types": ["linear_attention", "deepseek_sparse_attention"]}
    before = kv_bytes_per_token(v4)[0], kv_bytes_per_token(mla)[0]
    assert before[0] == round((512 / 4 + 128 / 4 + 512 / 128) * 2)
    monkeypatch.setattr(S, "VISION_KV_DTYPE_BYTES", 4)
    assert (kv_bytes_per_token(v4)[0], kv_bytes_per_token(mla)[0]) == before


def test_serve_says_when_an_environment_knob_is_ignored():
    from knurlogic.interfaces.serve import ignored_env
    env = {"KNURLOGIC_KV_BITS": "bf16", "KNURLOGIC_MTP": "on"}
    lines = ignored_env(env, {}, {"KNURLOGIC_KV_BITS": "8",
                                  "KNURLOGIC_MTP": "on"})
    assert len(lines) == 1
    assert "KNURLOGIC_KV_BITS=8" in lines[0] and "--kv-bits" in lines[0]
    # a --set / flag value is applied, not ignored: nothing to say
    assert ignored_env(env, {"KNURLOGIC_KV_BITS": "8"},
                       {"KNURLOGIC_KV_BITS": "8"}) == []
    assert "--set KNURLOGIC_X=" in ignored_env(
        {"KNURLOGIC_X": "1"}, {}, {"KNURLOGIC_X": "2"})[0]


def test_the_picker_offers_only_the_splits_a_launch_takes():
    from knurlogic.interfaces.page.documents import splits_of
    tc = {"model_type": "qwen3_5_moe_text", "num_hidden_layers": 40,
          "num_attention_heads": 32, "num_key_value_heads": 4,
          "num_experts": 8, "moe_intermediate_size": 512,
          "shared_expert_intermediate_size": 512}
    assert splits_of({"model_type": "qwen3_5_moe", "text_config": tc}) == [
        "tensor", "pipeline"]
    # a family only the pipeline knows: no Tensor offered
    assert splits_of({"model_type": "glm5_next",
                      "num_hidden_layers": 40}) == ["pipeline"]
    # neither knows it: nothing to offer across machines
    assert splits_of({"model_type": "llama", "num_hidden_layers": 40}) == []
