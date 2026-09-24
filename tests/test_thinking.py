"""Thinking effort: OpenAI's ladder -> each chat template's own controls.

The detection and the rendering tests read the RELEASED templates from
~/.exo/models (skipped where a rung is not on this box): the claim is about
those files, so a synthetic template would test nothing.
"""
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from knurlogic.engine.serve import thinking as T  # noqa: E402

MODELS = Path.home() / ".exo" / "models"
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
    "glm_effort": {"none": "low", "minimal": "low", "low": "low",
                   "medium": "high", "high": "high", "xhigh": "max"},
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
                                                       "xhigh"))


# --- the server hook ------------------------------------------------------------

def _fake_srv(template):
    seen = {}

    class H:
        def handle_completion(self, request, stop_words):
            seen["kwargs"] = self.chat_template_kwargs
            return "served"

        def generate_response(self, *a, **k):
            return {"choices": [{"message": {"content": "x",
                                             "reasoning": "r"}}],
                    "usage": {"prompt_tokens": 1}}

        def completion_usage_response(self, *a, **k):
            return {"choices": [], "usage": {"prompt_tokens": 1}}

    from knurlogic.engine.serve import state
    state.SERVED["provider"] = types.SimpleNamespace(
        tokenizer=types.SimpleNamespace(chat_template=template))
    srv = types.SimpleNamespace(APIHandler=H)
    T.install(srv)
    return srv, seen


def _handler(srv, body, client_kwargs=None):
    h = srv.APIHandler()
    h.body = body
    h.chat_template_kwargs = client_kwargs
    return h


QWEN38 = "enable_thinking ... reasoning_effort ... 'xhigh' <think>"


def test_the_hook_translates_reports_and_lets_the_client_win():
    srv, seen = _fake_srv(QWEN38)
    req = types.SimpleNamespace(request_type="chat")
    h = _handler(srv, {"reasoning_effort": "medium"})
    assert h.handle_completion(req, []) == "served"
    assert seen["kwargs"] == {"reasoning_effort": "medium"}
    resp = h.generate_response()
    assert resp["usage"]["knurlogic"]["thinking"]["applied"] == "medium"
    assert resp["choices"][0]["message"]["reasoning_content"] == "r"

    h = _handler(srv, {"reasoning_effort": "low"},
                 client_kwargs={"reasoning_effort": "xhigh", "x": 1})
    h.handle_completion(req, [])
    assert seen["kwargs"] == {"reasoning_effort": "xhigh", "x": 1}
    assert "won" in h._knurlogic_thinking["note"]


def test_the_hook_refuses_a_bad_level_with_400():
    srv, seen = _fake_srv(QWEN38)
    sent = {}
    h = _handler(srv, {"reasoning_effort": "lots"})
    h.send_response = lambda c: sent.setdefault("code", c)
    h.send_header = lambda *a: None
    h.end_headers = lambda: None
    import io
    h.wfile = io.BytesIO()
    h.handle_completion(types.SimpleNamespace(request_type="chat"), [])
    assert sent["code"] == 400 and "kwargs" not in seen


def test_messages_thinking_disabled_asks_for_none():
    from knurlogic.interfaces.messages import to_openai
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
    assert [n["name"] for n in lv["native"]] == ["low", "high", "max"]
    assert T.levels(T.template_of("/nonexistent")) == {
        "dialect": None, "native": [], "default": None}
