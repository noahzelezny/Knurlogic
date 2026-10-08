"""tool_choice on every API: "none" means no tool call comes back, streamed
or not (the tools stay in the prompt so its prefix, and the prompt cache,
match the turns that offer them; a call the model makes anyway is
dropped). "required"/"any" and a named tool reach a chat template that
reads `tool_choice`; nothing enforces them."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

from knurlogic.engine.runtime import prompt as P  # noqa: E402
from knurlogic.interfaces.http import messages as M  # noqa: E402
from knurlogic.interfaces.http import openai as O  # noqa: E402
from knurlogic.interfaces.http import responses as R  # noqa: E402

TOOLS = [{"type": "function", "function": {
    "name": "f", "parameters": {"type": "object", "properties": {}}}}]
NAMED = {"type": "function", "function": {"name": "f"}}


def _choice(body):
    job, _ = O.build_job(dict(body, messages=[{"role": "user",
                                               "content": "hi"}]),
                         chat=True, sampling_defaults={})
    return job.request.tool_choice


@pytest.mark.parametrize("tc", [None, "auto", "none", "required", NAMED])
def test_openai_chat_takes_each_form(tc):
    assert _choice({"tools": TOOLS, "tool_choice": tc}) == tc


@pytest.mark.parametrize("tc", ["sometimes", {"type": "function"}, 3])
def test_a_bad_tool_choice_is_a_400(tc):
    with pytest.raises(O.ApiError) as e:
        _choice({"tools": TOOLS, "tool_choice": tc})
    assert e.value.status == 400


@pytest.mark.parametrize("anth,oai", [
    ({"type": "auto"}, "auto"), ({"type": "any"}, "required"),
    ({"type": "none"}, "none"), ({"type": "tool", "name": "f"}, NAMED)])
def test_messages_tool_choice_is_translated(anth, oai):
    o = M.to_openai({"messages": [], "tool_choice": anth, "tools": [
        {"name": "f", "input_schema": {"type": "object"}}]})
    assert o["tool_choice"] == oai
    assert _choice(o) == oai


@pytest.mark.parametrize("resp,oai", [
    ("none", "none"), ("required", "required"),
    ({"type": "function", "name": "f"}, NAMED)])
def test_responses_tool_choice_is_translated(resp, oai):
    o = R.to_chat({"model": "m", "input": "hi", "tool_choice": resp,
                   "tools": [{"type": "function", "name": "f",
                              "parameters": {"type": "object"}}]})
    assert o["tool_choice"] == oai
    assert _choice(o) == oai


def test_none_drops_a_tool_call_the_model_makes():
    from test_request import run
    parse = lambda text, tools: json.loads(text)  # noqa: E731
    got, _ = run([10, 3, 17, 4, 90], tool_parser=parse, no_tools=True)
    assert got["tool_calls"] == [] and got["finish"] == "stop"
    assert all(not d.tool_calls for d in got["deltas"])
    assert got["content"] == "The"


class _Tok:
    def __init__(self, template):
        self.chat_template = template
        self.seen = None

    def apply_chat_template(self, messages, **kw):
        self.seen = kw
        return "x"


def test_tool_choice_reaches_a_template_that_reads_it(monkeypatch):
    seen = {}
    monkeypatch.setattr(P, "_render",
                        lambda tok, m, render, close: seen.update(render))
    monkeypatch.setattr(P, "_segment", lambda *a: None)
    for template, want in (("{{ tool_choice }}", "required"),
                           ("{{ tools }}", None)):
        seen.clear()
        req = P.ChatRequest("chat", "", [{"role": "user", "content": "hi"}],
                            TOOLS, tool_choice="required")
        P.tokenize(None, _Tok(template), req, P.PromptArgs())
        assert seen.get("tool_choice") == want
        assert seen.get("tools")
