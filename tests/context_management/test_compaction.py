"""Server-side compaction (context_management/context_edits and
context_management/compaction): the request's shapes on both APIs, the history surgery and its invariants,
distillation and its fallback, the response shapes -- and one round trip
on the tiny model over real sockets."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from knurlogic.context_management import compaction as C  # noqa: E402
from knurlogic.context_management import context_edits as E  # noqa: E402
from knurlogic.interfaces.http import messages as M  # noqa: E402

ENV = {"KNURLOGIC_COMPACT_KEEP_TURNS": "2"}


def call(i, name="Grep", args=None):
    return {"id": f"c{i}", "type": "function",
            "function": {"name": name,
                         "arguments": json.dumps(args or {"pattern": f"p{i}"})}}


def agent_history(n=4):
    """system, goal, then n (assistant call, tool result) pairs, and a
    last user turn."""
    msgs = [{"role": "system", "content": "you are an agent"},
            {"role": "user", "content": "find where foo is defined"}]
    for i in range(n):
        msgs.append({"role": "assistant", "content": f"step {i}",
                     "tool_calls": [call(i)]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}",
                     "content": f"result {i} " + "x" * 400})
    msgs.append({"role": "user", "content": "go on"})
    return msgs


def count(msgs, tools=None):
    """four characters a token, as a fake tokenizer"""
    return sum(len(E._text(m.get("content")) or "") + 4 for m in msgs) // 4


# ------------------------------------------------------------ parsing

def test_the_documented_edits_parse():
    edits = E.parse({"edits": [
        {"type": "clear_thinking_20251015",
         "keep": {"type": "thinking_turns", "value": 2}},
        {"type": "clear_tool_uses_20250919",
         "trigger": {"type": "input_tokens", "value": 30000},
         "keep": {"type": "tool_uses", "value": 3},
         "clear_at_least": {"type": "input_tokens", "value": 5000},
         "exclude_tools": ["web_search"], "clear_tool_inputs": True},
        {"type": "compact_20260112",
         "trigger": {"type": "input_tokens", "value": 150000},
         "pause_after_compaction": True, "instructions": "be brief"}]})
    th, cl, co = edits
    assert th.keep == 2
    assert (cl.trigger, cl.keep, cl.clear_at_least, cl.exclude,
            cl.clear_inputs) == (30000, 3, 5000, {"web_search"}, True)
    assert (co.trigger, co.pause, co.instructions) == (150000, True,
                                                       "be brief")


@pytest.mark.parametrize("cm, says", [
    ({"edits": [{"type": "clear_everything"}]}, "clear_everything"),
    ({"edits": [{"type": "compact_20260112",
                 "trigger": {"type": "input_tokens", "value": -1}}]},
     "trigger"),
    ({"edits": [{"type": "compact_20260112",
                 "trigger": {"type": "turns", "value": 3}}]}, "trigger.type"),
    ({"edits": "compact"}, "list"),
])
def test_an_edit_it_cannot_honour_is_refused_by_name(cm, says):
    with pytest.raises(E.EditError, match=says):
        E.parse(cm)


# ------------------------------------------------------------ surgery

def test_the_plan_keeps_system_goal_and_tail():
    msgs = agent_history(4)
    p = E.plan(msgs, 2)
    out = E.compacted(msgs, p, "S")
    assert out[0] == msgs[0] and out[1] == msgs[1]
    assert out[2]["content"].endswith("S") and out[2]["role"] == "user"
    assert out[3:] == msgs[p.end:]


def test_the_tail_never_starts_on_an_orphaned_tool_result():
    msgs = agent_history(4)
    for keep in range(0, 8):
        p = E.plan(msgs, keep)
        if p is None:
            continue
        tail = msgs[p.end:]
        assert not tail or tail[0]["role"] != "tool"
        # every result in the tail has its call in the tail
        ids = {c["id"] for m in tail for c in m.get("tool_calls") or []}
        assert all(m["tool_call_id"] in ids for m in tail
                   if m["role"] == "tool")


def test_nothing_is_done_unless_two_messages_would_go():
    msgs = [{"role": "user", "content": "goal"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "b"}]
    assert E.plan(msgs, 1) is None


def test_a_resent_compaction_folds_to_exactly_what_the_model_saw():
    """The client appends the response (summary on the assistant message)
    and resends everything: the server's view is the compacted prompt, the
    same bytes each turn, so it is a prefix-cache hit."""
    msgs = agent_history(4)
    p = E.plan(msgs, 2)
    seen = E.compacted(msgs, p, "SUMMARY")
    resent = msgs + [{"role": "assistant", "content": "answer",
                      "compaction": "SUMMARY"},
                     {"role": "user", "content": "next"}]
    view, cuts = E.view(resent, 2)
    assert cuts == 1
    assert view == seen + [{"role": "assistant", "content": "answer"},
                           {"role": "user", "content": "next"}]


def test_a_second_compaction_folds_over_the_first():
    msgs = agent_history(4)
    v1 = E.compacted(msgs, E.plan(msgs, 2), "ONE")
    later = v1 + [{"role": "assistant", "content": "a1"},
                  {"role": "user", "content": "u1"},
                  {"role": "assistant", "content": "a2"},
                  {"role": "user", "content": "u2"}]
    v2 = E.compacted(later, E.plan(later, 2), "TWO")
    raw = msgs + [{"role": "assistant", "content": None,
                   "compaction": "ONE"}] + later[len(v1):] + [
        {"role": "assistant", "content": None, "compaction": "TWO"}]
    assert E.view(raw, 2)[0] == v2
    assert v2[1]["content"] == "find where foo is defined"      # the goal


def test_the_plain_text_form_is_a_cut_point_too():
    m = {"role": "system", "content": E.wrap("S")}
    assert E.cut_of(m) == "S"
    assert E.cut_of({"role": "system", "content": "not one"}) is None


def test_clearing_tool_uses_keeps_the_pairs_and_the_recent_ones():
    msgs = agent_history(5)
    e = E.ClearTools(trigger=1, keep=2, exclude={"Read"})
    out, applied = E.clear_tool_uses(msgs, e, 10_000, len)
    results = [m for m in out if m["role"] == "tool"]
    assert [r["content"] == E.CLEARED for r in results] == \
        [True, True, True, False, False]
    assert applied["cleared_tool_uses"] == 3
    assert sum(1 for m in out if m.get("tool_calls")) == 5
    # under the trigger, nothing
    assert E.clear_tool_uses(msgs, E.ClearTools(trigger=10**9), 5, len)[1] \
        is None
    # too little to be worth it, nothing
    assert E.clear_tool_uses(msgs, E.ClearTools(trigger=1, keep=2,
                                                clear_at_least=10**6),
                             10_000, len)[1] is None


# ------------------------------------------------------------ distilling

def test_findings_are_parsed_and_a_missed_one_falls_back_to_a_clear():
    uses = E.tool_uses(agent_history(3))
    text = ("## Goal\n- find foo\n\n## Tool findings\n"
            "T1: foo is defined at src/foo.py:120\n"
            "- T3: nothing relevant\nT9: not a listed call")
    summary, found = E.parse_output(text, len(uses))
    assert summary == "## Goal\n- find foo"
    assert found == {1: "foo is defined at src/foo.py:120",
                     3: "nothing relevant"}
    final, distilled, cleared = E.render(summary, uses, found)
    assert (distilled, cleared) == (2, 1)
    assert 'Grep {"pattern":"p0"} -> foo is defined at src/foo.py:120' \
        in final
    assert f'Grep {{"pattern":"p1"}} -> {E.CLEARED}' in final
    # distill off: every call only named
    assert E.render(summary, uses, found, distill=False)[1:] == (0, 3)


def test_the_summary_prompt_is_extended_and_lists_the_calls():
    uses = E.tool_uses(agent_history(2))
    p = E.prompt(uses)
    for h in ("Decisions made", "Information gathered", "Files and "
              "identifiers touched", "Current step", "Next step",
              "Open errors"):
        assert h in p
    assert "never longer than the conversation you replace" in p
    assert "tokens" not in p and 'T2: Grep {"pattern":"p1"}' in p
    # instructions replace the summary prompt; the findings are still asked
    q = E.prompt(uses, "custom")
    assert q.startswith("custom") and "Decisions made" not in q
    assert "T1:" in q


def test_the_prompts_ship_as_markdown_beside_the_module():
    try:
        import tomllib
    except ModuleNotFoundError:            # 3.10: pytest depends on tomli
        import tomli as tomllib
    from pathlib import Path
    here = Path(E.__file__).with_name("prompts")
    for name in ("compact.md", "findings.md"):
        assert (here / name).is_file()
    assert "{calls}" in E.FINDINGS_PROMPT and "{" not in E.SUMMARY_PROMPT
    root = Path(__file__).resolve().parents[2]
    data = tomllib.loads((root / "pyproject.toml").read_text())
    assert "context_management/prompts/*.md" in \
        data["tool"]["setuptools"]["package-data"]["knurlogic"]


# ------------------------------------------------------------ orchestration

def _body(msgs, **cm):
    return {"model": "m", "messages": msgs,
            "context_management": {"edits": [dict(
                {"type": "compact_20260112",
                 "trigger": {"type": "input_tokens", "value": 100}}, **cm)]}}


def test_below_the_trigger_nothing_happens():
    run, out, pending = C.prepare(_body(agent_history(1), trigger={
        "type": "input_tokens", "value": 10**6}), count=count, env=ENV)
    assert pending is None and out.applied == []
    assert "context_management" not in run


def test_a_summary_pass_is_a_continuation_and_distills_in_one_call():
    body = _body(agent_history(4))
    run, out, pending = C.prepare(body, count=count, env=ENV)
    assert pending is not None and len(pending.uses) == 3
    # no budget: never longer than the span it replaces, plus the findings
    assert C.summary_body(body, pending)["max_tokens"] == \
        pending.dropped_tokens + C.TOKENS_PER_FINDING * 3
    calls = []

    def gen(b):
        calls.append(b)
        return {"choices": [{"message": {"content":
                "## Goal\n- foo\n## Tool findings\nT1: foo at a.py:1\n"
                "T2: nothing relevant"}}],
                "usage": {"prompt_tokens": 900, "completion_tokens": 40,
                          "prompt_tokens_details": {"cached_tokens": 880}}}
    run = C.summarize(run, pending, out, gen, env=ENV)
    assert len(calls) == 1                         # batched: one model call
    sent = calls[0]["messages"]
    assert sent[:-1] == pending.view               # the history, unchanged
    assert sent[-1]["role"] == "user" and "T3:" in sent[-1]["content"]
    assert calls[0]["temperature"] == C.SUMMARY_TEMPERATURE
    a = out.applied[0]
    assert a["type"] == "compact_20260112" and not a.get("fallback")
    assert (a["distilled_tool_uses"], a["cleared_tool_uses"]) == (2, 1)
    assert out.iteration == {"type": "compaction", "input_tokens": 900,
                             "output_tokens": 40,
                             "cache_read_input_tokens": 880}
    assert run["messages"][2]["content"].endswith(out.compaction)
    assert len(run["messages"]) < len(body["messages"])


@pytest.mark.parametrize("gen", [
    lambda b: (_ for _ in ()).throw(RuntimeError("ring failed")),
    lambda b: {"choices": [{"message": {"content": ""}}], "usage": {}},
    lambda b: {"choices": [{"message": {"content": "y" * 10**6}}],
               "usage": {}},
])
def test_a_failed_summary_falls_back_to_the_marker_never_the_turn(gen):
    run, out, pending = C.prepare(_body(agent_history(4)), count=count,
                                  env=ENV)
    run = C.summarize(run, pending, out, gen, env=ENV)
    a = out.applied[0]
    assert a["fallback"] is True and a["reason"]
    assert "no summary could be made" in out.compaction
    assert a["cleared_tool_uses"] == 3 and a["distilled_tool_uses"] == 0


def test_automatic_compaction_is_off_by_default_and_on_asks_nothing():
    body = {"messages": agent_history(4)}
    assert C.prepare(body, count=count, window=200, env={})[2] is None
    run, out, pending = C.prepare(body, count=count, window=200, env={
        "KNURLOGIC_COMPACT_AUTO": "on", "KNURLOGIC_COMPACT_TRIGGER": "0.5",
        "KNURLOGIC_COMPACT_KEEP_TURNS": "2"})
    assert pending is not None and pending.edit.auto


def test_the_settings_fall_back_to_their_defaults():
    from knurlogic.tuning import settings as S
    cfg = S.compact_settings({"KNURLOGIC_COMPACT_TRIGGER": "lots"})
    assert cfg == {"auto": False, "trigger": 0.8, "keep": 6,
                   "distill": True}
    assert S.check_knob("KNURLOGIC_COMPACT_TOOL_RESULTS", "burn")
    assert S.check_knob("KNURLOGIC_COMPACT_TRIGGER", "0.7") is None


# ------------------------------------------------------------ Anthropic shape

def test_a_resent_compaction_block_becomes_the_cut_point():
    req = {"messages": [
        {"role": "user", "content": "goal"},
        {"role": "assistant", "content": [
            {"type": "compaction", "content": "S"},
            {"type": "text", "text": "and then"}]},
        {"role": "user", "content": "next"}],
        "context_management": {"edits": [{"type": "compact_20260112"}]}}
    oai = M.to_openai(req)
    assert oai["context_management"] == req["context_management"]
    assert oai["messages"][1] == {"role": "assistant", "content": None,
                                  "compaction": "S"}
    assert oai["messages"][2] == {"role": "assistant", "content": "and then"}


def _oai(compaction="S", finish="stop", content="hi"):
    return {"choices": [{"finish_reason": finish, "message": {
        "role": "assistant", "content": content, "compaction": compaction}}],
        "context_management": {"applied_edits": [
            {"type": "compact_20260112"}]},
        "usage": {"prompt_tokens": 100, "completion_tokens": 5,
                  "prompt_tokens_details": {"cached_tokens": 60},
                  "knurlogic": {"context": {"tokens": 100, "window": 1000},
                                "compaction": {"input_tokens": 900,
                                               "output_tokens": 30,
                                               "cache_read_input_tokens":
                                                   880}}}}


def test_the_messages_response_leads_with_the_compaction_block():
    r = M.from_openai(_oai(), "m")
    assert r["content"][0] == {"type": "compaction", "content": "S"}
    assert r["content"][1] == {"type": "text", "text": "hi"}
    assert r["context_management"]["applied_edits"][0]["type"] == \
        "compact_20260112"
    u = r["usage"]
    assert (u["input_tokens"], u["cache_read_input_tokens"]) == (40, 60)
    assert [i["type"] for i in u["iterations"]] == ["compaction", "message"]
    assert u["iterations"][0]["input_tokens"] == 20
    assert u["knurlogic"]["context"] == {"tokens": 100, "window": 1000}


def test_paused_after_compaction_is_the_block_alone():
    r = M.from_openai(_oai(finish="compaction", content=None), "m")
    assert r["stop_reason"] == "compaction"
    assert r["content"] == [{"type": "compaction", "content": "S"}]


def test_the_stream_sends_the_block_first_whole_then_the_text():
    base = {"id": "x", "object": "chat.completion.chunk"}
    lines = [": keepalive compaction",
             "data: " + json.dumps(dict(base, choices=[{"index": 0, "delta": {
                 "role": "assistant", "content": ""}}])),
             "data: " + json.dumps(dict(base, choices=[{"index": 0, "delta": {
                 "compaction": "S"}}])),
             "data: " + json.dumps(dict(base, choices=[],
                                        context_management={
                                            "applied_edits": [{"type": "c"}]})),
             "data: " + json.dumps(dict(base, choices=[{"index": 0, "delta": {
                 "content": "hi"}, "finish_reason": "stop"}])),
             "data: [DONE]"]
    evs = []
    for raw in M.stream(lines, "m"):
        ev, data = raw.decode().strip().split("\n")
        evs.append((ev[7:], json.loads(data[6:])))
    kinds = [e for e, _ in evs]
    assert kinds[:5] == ["message_start", "ping", "content_block_start",
                         "content_block_delta", "content_block_stop"]
    assert evs[2][1]["content_block"]["type"] == "compaction"
    assert evs[3][1]["delta"] == {"type": "compaction_delta",
                                  "content": "S"}
    assert evs[5][1]["index"] == 1 and \
        evs[5][1]["content_block"]["type"] == "text"
    final = [d for e, d in evs if e == "message_delta"][0]
    assert final["context_management"]["applied_edits"] == [{"type": "c"}]


# ------------------------------------------------------------ end to end

mx = None


@pytest.fixture(scope="module")
def server():
    """The tiny model behind the real server, with a tokenizer that takes
    words (so the summary prompt can be rendered)."""
    global mx
    mx = pytest.importorskip("mlx.core")
    import threading
    import mlx.nn as nn
    from test_batch_drafting import _tiny
    from test_scheduler import Host, Tok
    from knurlogic.engine.runtime.scheduler import Scheduler
    from knurlogic.engine.serve import state
    from knurlogic.interfaces.http.server import App, make_server

    class WordTok(Tok):
        def encode(self, s):
            return [int(x) if x.isdigit() and int(x) < 512
                    else 10 + sum(map(ord, x)) % 400 for x in s.split()]

        def decode(self, ids):
            return "".join(f"<{i}>" for i in ids)

        def apply_chat_template(self, messages, **kw):
            out = []
            for m in messages:
                out += self.encode(m.get("content") or "")
            return out

    model, head, prompts = _tiny(512)
    mx.eval([v if isinstance(v, mx.array) else v.parameters()
             for v in vars(head).values()
             if isinstance(v, (mx.array, nn.Module))])
    state.DRAFT.update(head=None, on=False)
    host = Host(model, WordTok(prompts))
    host.path, host.loaded_at = "", 0.0
    host.status = lambda: {"state": "ready", "model": "", "error": ""}
    sched = Scheduler(host, prefill_step_size=16).start()
    app = App(sched, served=lambda: {"id": "tiny", "capabilities": ["text"],
                                     "size_bytes": 1})
    app.translate = None
    srv = make_server(app, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"


def _post(url, path, body):
    import urllib.request
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def test_a_round_trip_on_the_tiny_model(server, monkeypatch):
    monkeypatch.setenv("KNURLOGIC_COMPACT_KEEP_TURNS", "2")
    words = " ".join(f"w{i}" for i in range(60))
    msgs = [{"role": "user", "content": "the goal " + words}]
    for i in range(6):
        msgs += [{"role": "assistant", "content": f"answer {i} " + words},
                 {"role": "user", "content": f"question {i} " + words}]
    cm = {"edits": [{"type": "compact_20260112",
                     "trigger": {"type": "input_tokens", "value": 200}}]}
    r = _post(server, "/v1/messages", {"model": "tiny", "max_tokens": 4,
                                       "messages": msgs,
                                       "context_management": cm})
    assert r["content"][0]["type"] == "compaction"
    assert r["content"][0]["content"]
    edit = r["context_management"]["applied_edits"][0]
    assert edit["type"] == "compact_20260112"
    assert edit["summarized_messages"] >= 2
    before = edit["input_tokens_before"]
    after = r["usage"]["knurlogic"]["context"]["tokens"]
    assert after < before
    assert r["usage"]["iterations"][0]["type"] == "compaction"

    # the client appends the response and resends: the server sees the
    # compacted prompt again, below the trigger this time
    msgs2 = msgs + [{"role": "assistant", "content": r["content"]},
                    {"role": "user", "content": "next"}]
    cm2 = {"edits": [{"type": "compact_20260112",
                      "trigger": {"type": "input_tokens", "value": before}}]}
    r2 = _post(server, "/v1/messages", {"model": "tiny", "max_tokens": 4,
                                        "messages": msgs2,
                                        "context_management": cm2})
    assert r2["content"][0]["type"] != "compaction"
    assert r2["usage"]["knurlogic"]["context"]["tokens"] < before
    # ...and its prefix is the compacted prompt the continuation prefilled
    assert r2["usage"]["cache_read_input_tokens"] > 0

    # the OpenAI side: same object, the summary as message.compaction
    o = _post(server, "/v1/chat/completions", {
        "model": "tiny", "max_tokens": 4, "messages": msgs,
        "context_management": {"edits": [dict(
            cm["edits"][0], pause_after_compaction=True)]}})
    ch = o["choices"][0]
    assert ch["finish_reason"] == "compaction"
    assert ch["message"]["compaction"] and ch["message"]["content"] is None
    assert o["context_management"]["applied_edits"][0]["type"] == \
        "compact_20260112"
