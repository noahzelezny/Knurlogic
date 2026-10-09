"""Thinking effort: OpenAI's ladder -> each chat template's own controls.

The detection and the rendering tests read the RELEASED templates from
~/.exo/models, or KNURLOGIC_MODELS (skipped where a rung is absent): the
claim is about those files, so a synthetic template would test nothing.
"""
import json
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from knurlogic.engine.model import thinking as T  # noqa: E402

MODELS = Path(os.environ.get("KNURLOGIC_MODELS",
                           Path.home() / ".exo" / "models"))
RUNGS = {
    "TheDrainFlorist--Qwen3.8-27B-VQ-3.9bpw": "qwen_effort",
    "TheDrainFlorist--Qwen3.8-Flash-Next-VQ-4.4bpw": "qwen_effort",
    "TheDrainFlorist--Qwen3.6-35B-A3B-VQ-3.4bpw": "qwen_toggle",
    "TheDrainFlorist--Qwen3.5-397B-A17B-VQ-2.2bpw": "qwen_toggle",
    "TheDrainFlorist--gemma-4-e4b-it-VQ-PLE": "gemma_toggle",
    "TheDrainFlorist--gemma-4-26b-a4b-it-VQ-6.2bpw": "gemma_toggle",
    "TheDrainFlorist--GLM-5.3-Flash-VQ-2.7bpw": "glm_effort",
}


def _template(rung):
    d = MODELS / rung
    j = d / "chat_template.jinja"
    if j.is_file():
        return j.read_text()
    c = d / "tokenizer_config.json"
    if c.is_file():
        t = json.loads(c.read_text()).get("chat_template")
        return t if isinstance(t, str) else None
    return None


@pytest.mark.parametrize("rung,dialect", sorted(RUNGS.items()))
def test_each_released_template_is_its_dialect(rung, dialect):
    t = _template(rung)
    if t is None:
        pytest.skip(f"{rung} not on this box")
    assert T.detect(t)[0] == dialect


def test_a_template_with_no_controls_is_no_dialect():
    assert T.detect("{{ messages[0].content }}") == (None, None)
    kw, rep = T.resolve("low", None, None)
    assert kw == {} and rep["applied"] == "not controllable"


# ladder level -> native name applied, per dialect
EXPECT = {
    "qwen_effort": {"none": "off", "minimal": "low", "low": "low",
                    "medium": "medium", "high": "xhigh", "xhigh": "xhigh"},
    "qwen_toggle": {"none": "off", "minimal": "on", "low": "on",
                    "medium": "on", "high": "on", "xhigh": "on"},
    "gemma_toggle": {"none": "off", "minimal": "on", "low": "on",
                     "medium": "on", "high": "on", "xhigh": "on"},
    "glm_effort": {"none": "off", "minimal": "low", "low": "low",
                   "medium": "high", "high": "max", "xhigh": "max"},
}


@pytest.mark.parametrize("dialect", sorted(EXPECT))
def test_every_level_maps_to_a_native_one_and_says_so(dialect):
    from knurlogic.engine import families
    spec = families.build_maps()["thinking"][dialect]
    for level, want in EXPECT[dialect].items():
        kw, rep = T.resolve(level, dialect, spec)
        assert rep["applied"] == want, (dialect, level, rep)
        exact = any(n[0] == level for n in spec["native"])
        assert rep["native"] == exact and bool(rep["note"]) == (not exact)
        assert kw, (dialect, level)
    kw, rep = T.resolve(None, dialect, spec)
    assert kw == {} and rep["applied"] == spec["default"]


def test_an_unknown_level_is_refused_not_guessed():
    with pytest.raises(ValueError):
        T.requested({"reasoning_effort": "maximum"})
    assert T.requested({"reasoning_effort": "HIGH"}) == "high"
    assert T.requested({"reasoning": {"effort": "low"}}) == "low"
    assert T.requested({}) is None


# --- the kwargs are ones the real templates read -------------------------------

def _render(rung, **kw):
    tr = pytest.importorskip("transformers")
    d = MODELS / rung
    if not d.is_dir():
        pytest.skip(f"{rung} not on this box")
    tok = tr.AutoTokenizer.from_pretrained(str(d))
    return tok.apply_chat_template([{"role": "user", "content": "hi"}],
                                   add_generation_prompt=True, tokenize=False,
                                   **kw)


def _kw(dialect, level):
    from knurlogic.engine import families
    spec = families.build_maps()["thinking"][dialect]
    return T.resolve(level, dialect, spec)[0]


def test_qwen38_effort_reaches_the_prompt():
    r = "TheDrainFlorist--Qwen3.8-27B-VQ-3.9bpw"
    low = _render(r, **_kw("qwen_effort", "low"))
    top = _render(r, **_kw("qwen_effort", "xhigh"))
    off = _render(r, **_kw("qwen_effort", "none"))
    assert "Reasoning effort is set to low" in low
    assert "Reasoning effort is set to xhigh" in top
    assert off.rstrip().endswith("</think>")          # closed: no thinking
    assert low != top != off


def test_qwen36_toggle_reaches_the_prompt():
    r = "TheDrainFlorist--Qwen3.6-35B-A3B-VQ-3.4bpw"
    assert _render(r, **_kw("qwen_toggle", "none")).rstrip().endswith(
        "</think>")
    assert not _render(r, **_kw("qwen_toggle", "high")).rstrip().endswith(
        "</think>")


def test_gemma_on_reaches_the_prompt():
    r = "TheDrainFlorist--gemma-4-e4b-it-VQ-PLE"
    assert "<|think|>" in _render(r, **_kw("gemma_toggle", "low"))
    assert "<|think|>" not in _render(r, **_kw("gemma_toggle", "none"))


def test_glm_effort_reaches_the_prompt():
    r = "TheDrainFlorist--GLM-5.3-Flash-VQ-2.7bpw"
    assert "Reasoning Effort: Low" in _render(r, **_kw("glm_effort", "low"))
    assert "Reasoning Effort: High" in _render(r, **_kw("glm_effort",
                                                        "medium"))
    assert "Reasoning Effort: Max" in _render(r, **_kw("glm_effort",
                                                       "high"))
    assert "Reasoning Effort: Max" in _render(r, **_kw("glm_effort",
                                                       "xhigh"))


# --- the server hook ------------------------------------------------------------

QWEN38 = "enable_thinking ... reasoning_effort ... 'xhigh' <think>"


class _Qwen38Tok:
    """Renders like Qwen3.8's template behind mlx-lm's TokenizerWrapper,
    which injects enable_thinking=<has_thinking> when a request is silent."""
    chat_template = QWEN38

    def apply_chat_template(self, msgs, add_generation_prompt=True,
                            tokenize=False, **kw):
        kw.setdefault("enable_thinking", True)
        if kw["enable_thinking"] is False:
            return "<think></think>"
        return "effort=" + kw.get("reasoning_effort", "xhigh")


def _serve(tok):
    """The served tokenizer the translation reads (as the model host
    registers it)."""
    import types

    from knurlogic.engine.model import state
    T._probe_cache.clear()
    state.SERVED["provider"] = types.SimpleNamespace(tokenizer=tok)


def test_translate_maps_the_level_and_reports_it():
    _serve(_Qwen38Tok())
    kwargs, rep = T.translate({"reasoning_effort": "medium"}, None)
    assert kwargs == {"reasoning_effort": "medium"}
    assert rep["applied"] == "medium"


def test_a_silent_request_reports_what_the_server_renders():
    _serve(_Qwen38Tok())
    kwargs, rep = T.translate({}, None)
    assert kwargs is None
    assert rep["applied"] == "xhigh"                    # the bare render


def test_the_thinking_default_serves_a_silent_request(monkeypatch):
    """KNURLOGIC_THINKING_DEFAULT: a request that names no level is served
    at the per-model default (GLM-5.3's own is max, and a client whose
    effort control broke left every long conversation thinking for hours);
    one that names a level still wins; "model" is the template's own."""
    _serve(_Qwen38Tok())
    monkeypatch.setenv(T.DEFAULT_ENV, "low")
    kwargs, rep = T.translate({}, None)
    assert kwargs == {"reasoning_effort": "low"}
    assert rep["applied"] == "low" and "Thinking default" in rep["note"]
    kwargs, rep = T.translate({"reasoning_effort": "medium"}, None)
    assert kwargs == {"reasoning_effort": "medium"}
    assert "Thinking default" not in rep["note"]
    monkeypatch.setenv(T.DEFAULT_ENV, "model")
    kwargs, rep = T.translate({}, None)
    assert kwargs is None and rep["applied"] == "xhigh"


def test_the_thinking_default_is_live_and_checked(monkeypatch):
    import os

    from knurlogic.engine.model.load import apply_live
    from knurlogic.tuning import checks, knobs
    monkeypatch.delenv(T.DEFAULT_ENV, raising=False)
    done = apply_live({T.DEFAULT_ENV: "high"})
    assert done[T.DEFAULT_ENV].startswith("applied")
    assert os.environ[T.DEFAULT_ENV] == "high"
    done = apply_live({T.DEFAULT_ENV: "hi gh"})
    assert done[T.DEFAULT_ENV].startswith("failed")
    assert os.environ[T.DEFAULT_ENV] == "high"
    assert checks.check_knob(T.DEFAULT_ENV, "max") is None
    assert checks.check_knob(T.DEFAULT_ENV, "4") is not None
    assert "model" not in knobs.KNOB_RANGE[T.DEFAULT_ENV][0]   # that is unset


def test_the_thinking_default_reads_the_templates_own_names(monkeypatch):
    """The page offers a model's own level names (GLM: off, low, high,
    max), whose words mean other ladder levels there: GLM's "high" is the
    ladder's medium and its "max" the ladder's high."""
    from knurlogic.engine import families
    spec = families.build_maps()["thinking"]["glm_effort"]
    for name, level in (("off", "none"), ("low", "low"), ("high", "medium"),
                        ("max", "high")):
        monkeypatch.setenv(T.DEFAULT_ENV, name)
        assert T.default_level(spec) == level, name
    monkeypatch.setenv(T.DEFAULT_ENV, "model")
    assert T.default_level(spec) is None
    monkeypatch.setenv(T.DEFAULT_ENV, "turbo")      # not this template's
    assert T.default_level(spec) is None
    monkeypatch.setenv(T.DEFAULT_ENV, "low")
    assert T.default_level(None) is None            # no controls


def test_a_client_override_is_reported_by_what_it_renders():
    """The case: reasoning_effort low, but the client's own kwargs turn
    thinking off. The prompt is closed-think; the report must say off."""
    _serve(_Qwen38Tok())
    kwargs, rep = T.translate({"reasoning_effort": "low"},
                              {"enable_thinking": False})
    assert kwargs == {"reasoning_effort": "low", "enable_thinking": False}
    assert rep["applied"] == "off" and "won" in rep["note"]
    assert "translation alone gave low" in rep["note"]


def test_a_template_that_only_mentions_the_controls_is_not_trusted():
    class Deaf(_Qwen38Tok):
        def apply_chat_template(self, msgs, **kw):
            return "same every time"
    _serve(Deaf())
    kwargs, rep = T.translate({"reasoning_effort": "low"}, None)
    assert kwargs is None
    assert rep["applied"] == "not controllable"


def test_the_wire_refuses_a_bad_level_with_400_and_honours_exclude():
    from knurlogic.interfaces.http import openai as O
    _serve(_Qwen38Tok())
    msgs = [{"role": "user", "content": "hi"}]
    with pytest.raises(O.ApiError) as e:
        O.build_job({"messages": msgs, "reasoning_effort": "lots"},
                    chat=True, translate=T.translate)
    assert e.value.status == 400 and e.value.param == "reasoning_effort"
    _, ctx = O.build_job({"messages": msgs, "reasoning": {"exclude": True}},
                         chat=True, translate=T.translate)
    assert ctx["exclude"] is True


# --- the probe on the REAL templates, through mlx-lm's own wrapper -----------------

def _wrapper(rung):
    d = MODELS / rung
    if not d.is_dir():
        pytest.skip(f"{rung} not on this box")
    from mlx_lm.utils import load_tokenizer
    return load_tokenizer(d)


@pytest.mark.parametrize("rung,dialect,served_default", [
    ("TheDrainFlorist--gemma-4-e4b-it-VQ-PLE", "gemma_toggle", "on"),
    ("TheDrainFlorist--Qwen3.6-35B-A3B-VQ-3.4bpw", "qwen_toggle", "on"),
    ("TheDrainFlorist--Qwen3.8-27B-VQ-3.9bpw", "qwen_effort", "xhigh"),
    ("TheDrainFlorist--GLM-5.3-Flash-VQ-2.7bpw", "glm_effort", "max"),
])
def test_the_probe_verifies_each_released_template_and_finds_its_default(
        rung, dialect, served_default):
    """Gemma's TEMPLATE defaults off, but mlx-lm injects enable_thinking
    when the request is silent -- served, it thinks."""
    tok = _wrapper(rung)
    name, spec = T.detect(tok.chat_template)
    assert name == dialect
    T._probe_cache.clear()
    p = T.probe(tok, tok.chat_template, spec)
    assert p["verified"] and p["default"] == served_default
    assert spec["default"] == served_default     # the manifest agrees


def test_messages_thinking_disabled_asks_for_none():
    from knurlogic.interfaces.http.messages import to_openai
    b = to_openai({"messages": [{"role": "user", "content": "hi"}],
                   "thinking": {"type": "disabled"}})
    assert b["reasoning_effort"] == "none"
    b = to_openai({"messages": [{"role": "user", "content": "hi"}],
                   "thinking": {"type": "enabled", "budget_tokens": 9000}})
    assert "reasoning_effort" not in b          # budgets are not honoured


def test_the_mcp_models_tool_lists_each_models_thinking_levels():
    d = MODELS / "TheDrainFlorist--GLM-5.3-Flash-VQ-2.7bpw"
    if not d.is_dir():
        pytest.skip("GLM not on this box")
    lv = T.levels(T.template_of(d))
    assert lv["dialect"] == "glm_effort" and lv["default"] == "max"
    assert [n["name"] for n in lv["native"]] == ["off", "low", "high", "max"]
    assert T.levels(T.template_of("/nonexistent")) == {
        "dialect": None, "native": [], "default": None}


def test_messages_returns_thinking_blocks_only_when_enabled():
    from knurlogic.interfaces.http.messages import from_openai, to_openai
    on = to_openai({"messages": [{"role": "user", "content": "hi"}],
                    "thinking": {"type": "enabled", "budget_tokens": 2048}})
    off = to_openai({"messages": [{"role": "user", "content": "hi"}]})
    assert on["reasoning"] == {"exclude": False}
    assert off["reasoning"] == {"exclude": True}
    r = from_openai({"choices": [{"message": {"content": "4",
                                              "reasoning_content": "2+2"},
                                  "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 3, "completion_tokens": 2,
                               "knurlogic": {"thinking": {"applied": "low"}}}},
                    "m")
    assert r["content"][0] == {"type": "thinking", "thinking": "2+2",
                               "signature": ""}
    assert r["content"][1]["text"] == "4"
    assert r["knurlogic"]["thinking"]["applied"] == "low"


def test_messages_streams_thinking_then_text():
    from knurlogic.interfaces.http.messages import stream
    lines = [f"data: {json.dumps(c)}" for c in (
        {"choices": [{"delta": {"reasoning_content": "let me "}}]},
        {"choices": [{"delta": {"reasoning_content": "see"}}]},
        {"choices": [{"delta": {"content": "4"}, "finish_reason": "stop"}]},
    )] + ["data: [DONE]"]
    evs = [json.loads(e.decode().split("data: ", 1)[1])
           for e in stream(lines, "m")]
    kinds = [(e["type"], (e.get("content_block") or e.get("delta") or {})
              .get("type")) for e in evs]
    assert kinds == [
        ("message_start", None),
        ("content_block_start", "thinking"),
        ("content_block_delta", "thinking_delta"),
        ("content_block_delta", "thinking_delta"),
        ("content_block_delta", "signature_delta"),
        ("content_block_stop", None),
        ("content_block_start", "text"),
        ("content_block_delta", "text_delta"),
        ("content_block_stop", None),
        ("message_delta", None), ("message_stop", None)]


def test_probe_holds_when_requests_race_it():
    """The first requests after a load probe together. A tokenizer that
    fails when entered twice at once (as a real one does) must
    not turn a verified template into 'not controllable'."""
    import threading
    import time

    from knurlogic.engine.model import thinking as T

    class Tok:
        busy = False

        def apply_chat_template(self, msgs, **kw):
            if Tok.busy:
                raise RuntimeError("Already borrowed")
            Tok.busy = True
            try:
                time.sleep(0.002)
                return "on" if kw.get("enable_thinking", True) else "off"
            finally:
                Tok.busy = False

    spec = {"default": "on",
            "native": [["none", "off", {"enable_thinking": False}],
                       ["xhigh", "on", {"enable_thinking": True}]]}
    T._probe_cache.clear()
    got = []
    ts = [threading.Thread(target=lambda: got.append(
        T.probe(Tok(), "race-template", spec))) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert all(p["verified"] and p["default"] == "on" for p in got)


def test_a_request_during_load_still_gets_its_level(monkeypatch):
    """A request that arrives while the model loads has no served
    tokenizer yet; it is translated from the artifact's own tokenizer on
    disk, not served the model's default."""
    import mlx_lm.utils

    from knurlogic.engine.model import state
    state.SERVED["provider"] = types.SimpleNamespace(tokenizer=None)
    state.SERVED["path"] = "/artifact/still-loading"
    T._probe_cache.clear()
    T._disk_tok.clear()
    loads = []
    monkeypatch.setattr(mlx_lm.utils, "load_tokenizer",
                        lambda p: loads.append(p) or _Qwen38Tok())
    kwargs, rep = T.translate({"reasoning_effort": "none"}, None)
    assert kwargs == {"enable_thinking": False}
    assert rep["applied"] == "off"
    # loaded once, and let go once the served tokenizer exists
    T.translate({"reasoning_effort": "low"}, None)
    assert len(loads) == 1
    state.SERVED["provider"] = types.SimpleNamespace(tokenizer=_Qwen38Tok())
    T._served_tokenizer()
    assert T._disk_tok == {}
    state.SERVED["path"] = None


def test_glm_off_closes_the_think_block_so_the_answer_starts_normal():
    """GLM's `none` is the template's own no-thinking format: the prompt
    ends `<think></think>`, and mlx-lm's rfind sees a closed block."""
    tok = _wrapper("TheDrainFlorist--GLM-5.3-Flash-VQ-2.7bpw")
    kw = _kw("glm_effort", "none")
    assert T.CLOSE in kw
    text = T._render(tok, kw)
    assert text.endswith("<think></think>")
    assert "Reasoning Effort: Low" in text
    ids = T._Closing(tok).apply_chat_template(
        [{"role": "user", "content": "hi"}], add_generation_prompt=True,
        tokenize=True, **kw)
    assert tok.rfind_think_end(ids) > tok.rfind_think_start(ids)
    # the system-prompt split mlx-lm renders WITHOUT a generation prompt
    # is untouched, so its segments still line up with the prompt
    plain = T._Closing(tok).apply_chat_template(
        [{"role": "user", "content": "hi"}], add_generation_prompt=False,
        tokenize=True, **kw)
    assert plain == tok.apply_chat_template(
        [{"role": "user", "content": "hi"}], add_generation_prompt=False,
        tokenize=True, reasoning_effort="low")
    T._probe_cache.clear()
    name, spec = T.detect(tok.chat_template)
    p = T.probe(tok, tok.chat_template, spec)
    assert p["verified"] and p["renders"]["off"] != p["renders"]["low"]


def test_every_way_of_saying_no_thinking_is_none():
    from knurlogic.engine.model.thinking import requested
    assert requested({"reasoning_effort": "none"}) == "none"
    assert requested({"reasoning": {"effort": "none"}}) == "none"
    assert requested({"reasoning": {"enabled": False}}) == "none"
    # enabled without a level leaves the model's own default
    assert requested({"reasoning": {"enabled": True}}) is None


QWEN35_DEFAULTS = {"temp": 0.6, "top_p": 0.95, "top_k": 20,
                   "non_thinking": {"temp": 0.7, "top_p": 0.8, "top_k": 20,
                                    "min_p": 0.0, "presence_penalty": 1.5}}


def test_thinking_off_takes_the_makers_non_thinking_sampling():
    # Qwen3.5's card: thinking 0.6/0.95/20, off 0.7/0.8/20 + presence 1.5;
    # generation_config carries only the thinking set (exo's cards do both)
    from knurlogic.interfaces.http import openai as O
    _serve(_Qwen38Tok())
    msgs = [{"role": "user", "content": "hi"}]
    off = {"messages": msgs, "chat_template_kwargs": {"enable_thinking": False}}
    job, ctx = O.build_job(off, chat=True, translate=T.translate,
                           sampling_defaults=QWEN35_DEFAULTS)
    assert ctx["thinking"]["applied"] == "off"
    assert job.sampling["temp"] == 0.7 and job.sampling["top_p"] == 0.8
    assert job.penalties["presence_penalty"] == 1.5
    # the request's own values still win
    job, _ = O.build_job(dict(off, temperature=0.2, presence_penalty=0.5),
                         chat=True, translate=T.translate,
                         sampling_defaults=QWEN35_DEFAULTS)
    assert job.sampling["temp"] == 0.2 and job.penalties["presence_penalty"] == 0.5
    # thinking on keeps generation_config's set and no penalty
    job, _ = O.build_job({"messages": msgs}, chat=True, translate=T.translate,
                         sampling_defaults=QWEN35_DEFAULTS)
    assert job.sampling["temp"] == 0.6 and "presence_penalty" not in job.penalties


def test_sampling_defaults_carry_qwen35s_non_thinking_set(tmp_path):
    import json

    from knurlogic.machine.artifact import sampling_defaults
    (tmp_path / "generation_config.json").write_text(json.dumps(
        {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20}))
    (tmp_path / "config.json").write_text('{"model_type": "qwen3_5_moe"}')
    d = sampling_defaults(tmp_path)
    assert d["temp"] == 0.6 and d["non_thinking"]["presence_penalty"] == 1.5
    (tmp_path / "config.json").write_text('{"model_type": "qwen4_exp"}')
    assert sampling_defaults(tmp_path)["non_thinking"] == {
        "temp": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
        "presence_penalty": 1.5}   # Qwen3.8-Flash-Next's README
    (tmp_path / "config.json").write_text('{"model_type": "qwen3_next"}')
    assert "non_thinking" not in sampling_defaults(tmp_path)
