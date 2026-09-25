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


def _fake_srv(tok):
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

    class RG:
        def generate(self, request, args, **k):
            return "ctx", iter([types.SimpleNamespace(state=s) for s in
                                ("reasoning", "reasoning", "normal")])

    from knurlogic.engine.serve import state
    T._probe_cache.clear()
    state.SERVED["provider"] = types.SimpleNamespace(tokenizer=tok)
    srv = types.SimpleNamespace(APIHandler=H, ResponseGenerator=RG)
    T.install(srv)
    return srv, seen


def _handler(srv, body, client_kwargs=None):
    h = srv.APIHandler()
    h.body = body
    h.chat_template_kwargs = client_kwargs
    return h


REQ = types.SimpleNamespace(request_type="chat")


def test_the_hook_translates_and_reports():
    srv, seen = _fake_srv(_Qwen38Tok())
    h = _handler(srv, {"reasoning_effort": "medium"})
    assert h.handle_completion(REQ, []) == "served"
    assert seen["kwargs"] == {"reasoning_effort": "medium"}
    resp = h.generate_response()
    assert resp["usage"]["knurlogic"]["thinking"]["applied"] == "medium"
    assert resp["choices"][0]["message"]["reasoning_content"] == "r"


def test_a_silent_request_reports_what_the_server_renders():
    srv, seen = _fake_srv(_Qwen38Tok())
    h = _handler(srv, {})
    h.handle_completion(REQ, [])
    assert seen["kwargs"] is None
    assert h._knurlogic_thinking["applied"] == "xhigh"   # the bare render


def test_a_client_override_is_reported_by_what_it_renders():
    """Fable's case: reasoning_effort low, but the client's own kwargs turn
    thinking off. The prompt is closed-think; the report must say off."""
    srv, seen = _fake_srv(_Qwen38Tok())
    h = _handler(srv, {"reasoning_effort": "low"},
                 client_kwargs={"enable_thinking": False})
    h.handle_completion(REQ, [])
    rep = h._knurlogic_thinking
    assert seen["kwargs"] == {"reasoning_effort": "low",
                              "enable_thinking": False}
    assert rep["applied"] == "off" and "won" in rep["note"]
    assert "translation alone gave low" in rep["note"]


def test_a_template_that_only_mentions_the_controls_is_not_trusted():
    class Deaf(_Qwen38Tok):
        def apply_chat_template(self, msgs, **kw):
            return "same every time"
    srv, seen = _fake_srv(Deaf())
    h = _handler(srv, {"reasoning_effort": "low"})
    h.handle_completion(REQ, [])
    assert seen["kwargs"] is None
    assert h._knurlogic_thinking["applied"] == "not controllable"


def test_exclude_strips_reasoning_and_tokens_are_counted():
    srv, seen = _fake_srv(_Qwen38Tok())
    h = _handler(srv, {"reasoning": {"exclude": True}})
    h.handle_completion(REQ, [])
    msg = h.generate_response()["choices"][0]["message"]
    assert "reasoning" not in msg and "reasoning_content" not in msg
    _, it = srv.ResponseGenerator().generate(REQ, None)
    list(it)
    usage = h.completion_usage_response()["usage"]
    assert usage["completion_tokens_details"]["reasoning_tokens"] == 2


def test_the_hook_refuses_a_bad_level_with_400():
    srv, seen = _fake_srv(_Qwen38Tok())
    sent = {}
    h = _handler(srv, {"reasoning_effort": "lots"})
    h.send_response = lambda c: sent.setdefault("code", c)
    h.send_header = lambda *a: None
    h.end_headers = lambda: None
    import io
    h.wfile = io.BytesIO()
    h.handle_completion(REQ, [])
    assert sent["code"] == 400 and "kwargs" not in seen


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
    when the request is silent -- served, it thinks (Fable 5.1's finding)."""
    tok = _wrapper(rung)
    name, spec = T.detect(tok.chat_template)
    assert name == dialect
    T._probe_cache.clear()
    p = T.probe(tok, tok.chat_template, spec)
    assert p["verified"] and p["default"] == served_default
    assert spec["default"] == served_default     # the manifest agrees


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


def test_messages_returns_thinking_blocks_only_when_enabled():
    from knurlogic.interfaces.messages import from_openai, to_openai
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
    from knurlogic.interfaces.messages import stream
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
    fails when entered twice at once (as the real one did on the M4) must
    not turn a verified template into 'not controllable'."""
    import threading
    import time
    from knurlogic.engine.serve import thinking as T

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
    """mlx-lm answers HTTP before the model is loaded. A request then has
    no served tokenizer; it must be translated from the artifact's own
    tokenizer on disk, not served the model's default."""
    import mlx_lm.utils
    from knurlogic.engine.serve import state
    srv, seen = _fake_srv(_Qwen38Tok())
    state.SERVED["provider"] = types.SimpleNamespace(tokenizer=None)
    state.SERVED["path"] = "/artifact/still-loading"
    T._disk_tok.clear()
    loads = []
    monkeypatch.setattr(mlx_lm.utils, "load_tokenizer",
                        lambda p: loads.append(p) or _Qwen38Tok())
    h = _handler(srv, {"reasoning_effort": "none"})
    h.handle_completion(REQ, [])
    assert seen["kwargs"] == {"enable_thinking": False}
    assert h._knurlogic_thinking["applied"] == "off"
    # loaded once, and let go once the served tokenizer exists
    h2 = _handler(srv, {"reasoning_effort": "low"})
    h2.handle_completion(REQ, [])
    assert len(loads) == 1
    state.SERVED["provider"] = types.SimpleNamespace(tokenizer=_Qwen38Tok())
    T._served_tokenizer()
    assert T._disk_tok == {}
    state.SERVED["path"] = None
