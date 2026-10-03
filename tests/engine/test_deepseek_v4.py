"""engine/templates: DeepSeek-V4's chat encoding as knurlogic serves it.

The mlx-community DeepSeek-V4-Flash conversion ships a stub template (no
tools, no tool calls, no tool results). knurlogic replaces it with
deepseek_v4.jinja, a port of DeepSeek's own encoder -- vendored unchanged
in tests/support/fixtures_deepseek_v4/ (MIT) -- and these tests hold the port to
that encoder: its four golden outputs, and agent conversations rendered
both ways. Then the prompt stage's prefix rule on it, and the DSML
tool-call parser. No model."""
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
FIX = Path(__file__).resolve().parents[1] / "support" / "fixtures_deepseek_v4"
sys.path.insert(0, str(FIX))

cu = pytest.importorskip("transformers.utils.chat_template_utils")

import encoding_dsv4 as official  # noqa: E402

from knurlogic.engine import templates  # noqa: E402
from knurlogic.engine.runtime import prompt as P  # noqa: E402

TEMPLATE = templates.text("deepseek_v4")
BOS, EOS = "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"
D = "｜DSML｜"
# A local DeepSeek-V4-Flash artifact directory; the tests that need it skip
# without it.
STUB = Path(os.environ.get("KNURLOGIC_TEST_DEEPSEEK_V4") or "/nonexistent")


def render(messages, tools=None, gen=True, **kw):
    """The port, rendered the way transformers renders it."""
    extra = {"tools": tools} if tools else {}
    out, _ = cu.render_jinja_template(
        conversations=[copy.deepcopy(messages)], chat_template=TEMPLATE,
        add_generation_prompt=gen, bos_token=BOS, **extra, **kw)
    return out[0]


def encode(messages, tools=None, mode="chat"):
    """DeepSeek's encoder on the same conversation: tools on a system
    message, tool-call arguments as JSON strings (OpenAI's wire form;
    knurlogic's flatten decodes them to objects before rendering)."""
    ms = copy.deepcopy(messages)
    if tools:
        if not ms or ms[0]["role"] != "system":
            ms.insert(0, {"role": "system", "content": ""})
        ms[0]["tools"] = tools
    for m in ms:
        for tc in m.get("tool_calls") or []:
            a = tc["function"]["arguments"]
            if not isinstance(a, str):
                tc["function"]["arguments"] = json.dumps(a,
                                                         ensure_ascii=False)
    return official.encode_messages(ms, thinking_mode=mode)


TOOLS = [{"type": "function", "function": {
    "name": "read_file", "description": "Read a file.",
    "parameters": {"type": "object",
                   "properties": {"path": {"type": "string"},
                                  "lines": {"type": "integer"}},
                   "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "grep", "description": "Search.",
        "parameters": {"type": "object",
                       "properties": {"pattern": {"type": "string"},
                                      "flags": {"type": "array"}}}}}]


def _call(i, name="read_file", **args):
    return {"id": f"t{i}", "type": "function",
            "function": {"name": name, "arguments": args or {"path": f"f{i}"}}}


def agent(rounds=3, reasoning=True):
    """goal, then tool rounds (one parallel), then the answer."""
    h = [{"role": "system", "content": "You are a coding agent."},
         {"role": "user", "content": "Goal: read the files."}]
    for i in range(rounds):
        a = {"role": "assistant", "content": "" if i else "Reading.",
             "tool_calls": [_call(i, lines=5 * i, path=f"f{i}")]}
        if i == 1:
            a["tool_calls"].append(_call(10 + i, "grep", pattern="x|y",
                                         flags=["-n", True, None]))
        if reasoning:
            a["reasoning_content"] = f"step {i}"
        h.append(a)
        if i == 1:     # results out of call order: sorted back
            h.append({"role": "tool", "tool_call_id": "t11",
                      "content": "grep hit"})
        h.append({"role": "tool", "tool_call_id": f"t{i}",
                  "content": f"contents of f{i}"})
    h.append({"role": "assistant", "content": "Done: three files.",
              **({"reasoning_content": "all read"} if reasoning else {})})
    return h


# ------------------------------------------------------------- equivalence

@pytest.mark.parametrize("case,mode", [(1, "thinking"), (2, "thinking"),
                                       (3, "thinking"), (4, "chat")])
def test_the_port_renders_deepseeks_golden_outputs(case, mode):
    """DeepSeek's own encoder test cases (encoding/tests), byte for byte:
    tools + thinking, drop_thinking, developer + latest_reminder with
    search tools, a quick-instruction task."""
    data = json.loads((FIX / f"test_input_{case}.json").read_text())
    if case == 1:
        messages = data["messages"]
        messages[0]["tools"] = data["tools"]
    else:
        messages = data
    gold = (FIX / f"test_output_{case}.txt").read_text()
    for m in messages:
        for tc in m.get("tool_calls") or []:
            tc["function"]["arguments"] = json.loads(
                tc["function"]["arguments"])
    assert render(messages, thinking_mode=mode) == gold


def _conversations():
    h = agent()
    yield "agent", h, TOOLS
    yield "agent, open", h[:-1], TOOLS
    yield "agent, next question", h + [
        {"role": "user", "content": "And f3?"}], TOOLS
    yield "no system", h[1:], TOOLS
    yield "no tools", [{"role": "system", "content": "Be brief."},
                       {"role": "user", "content": "Hi"},
                       {"role": "assistant", "content": "Hello.",
                        "reasoning_content": "greet"},
                       {"role": "user", "content": "Capital of France?"}], None
    yield "two users in a row", [{"role": "user", "content": "a"},
                                 {"role": "user", "content": "b"}], None


@pytest.mark.parametrize("mode", ["chat", "thinking"])
def test_the_port_equals_deepseeks_encoder_on_agent_conversations(mode):
    for name, messages, tools in _conversations():
        want = encode(messages, tools, mode)
        got = render(messages, tools, thinking_mode=mode)
        assert got == want, name


def test_enable_thinking_picks_the_mode_when_thinking_mode_is_absent():
    """mlx-lm passes enable_thinking=<has_thinking> on every render; a
    request's thinking_mode (the stub's own switch) still wins."""
    h = agent()[:-1]
    assert render(h, TOOLS, enable_thinking=True) == \
        encode(h, TOOLS, "thinking")
    assert render(h, TOOLS, enable_thinking=False) == encode(h, TOOLS)
    assert render(h, TOOLS, enable_thinking=True, thinking_mode="chat") == \
        encode(h, TOOLS)
    assert render(h, TOOLS) == encode(h, TOOLS)          # default: chat


# ------------------------------------------------------ what it renders

def test_tools_are_rendered_in_the_system_prompt():
    s = render(agent()[:2], TOOLS)
    head = s.split("<｜User｜>")[0]
    assert head.startswith(BOS + "You are a coding agent.\n\n## Tools\n")
    assert json.dumps(TOOLS[0]["function"], ensure_ascii=False) in head
    assert f"<{D}tool_calls>" in head


def test_assistant_tool_calls_and_tool_results_are_rendered():
    s = render(agent(), TOOLS, thinking_mode="thinking")
    assert (f'Reading.\n\n<{D}tool_calls>\n<{D}invoke name="read_file">\n'
            f'<{D}parameter name="lines" string="false">0</{D}parameter>\n'
            f'<{D}parameter name="path" string="true">f0</{D}parameter>\n'
            f'</{D}invoke>\n</{D}tool_calls>{EOS}') in s
    assert (f'<{D}parameter name="flags" string="false">'
            f'["-n", true, null]</{D}parameter>') in s
    # the parallel round's results, in call order, one user turn
    assert ("<｜User｜><tool_result>contents of f1</tool_result>\n\n"
            "<tool_result>grep hit</tool_result><｜Assistant｜><think>") in s


def test_reasoning_follows_deepseeks_rules():
    h = agent()
    # thinking + tools: every turn keeps its reasoning
    s = render(h, TOOLS, thinking_mode="thinking")
    assert all(f"step {i}</think>" in s for i in range(3))
    # chat: no reasoning, every turn opens closed
    s = render(h, TOOLS, thinking_mode="chat")
    assert "step 0" not in s and "<think>" not in s.split("## Tools")[0]
    # thinking, no tools: reasoning before the last user turn is dropped
    s = render(h[:1] + [m for m in h if m["role"] in ("user",)]
               + [h[-1], {"role": "user", "content": "next"}],
               thinking_mode="thinking")
    assert "all read" not in s and s.endswith("next<｜Assistant｜><think>")


def test_content_before_tool_calls_loses_the_parsers_separator():
    """The engine streams the "\\n\\n" before the tool block as content;
    rendered back, the block adds its own."""
    a = agent()[:3]
    b = copy.deepcopy(a)
    b[2]["content"] = "Reading.\n\n"
    assert render(a, TOOLS) == render(b, TOOLS)


# ----------------------------------------------------- the prompt stage

class DSTok:
    """A tokenizer carrying a template, rendered as transformers renders
    it; a token is a character."""
    has_chat_template = True
    has_thinking = False
    added_tokens_decoder = {i: t for i, t in enumerate(
        [BOS, EOS, "<｜User｜>", "<｜Assistant｜>", "<think>", "</think>", D])}

    def __init__(self, template=TEMPLATE, name="x/DeepSeek-V4-Flash"):
        self.chat_template = template
        self.name_or_path = name

    def apply_chat_template(self, messages, add_generation_prompt=False,
                            tokenize=True, chat_template=None, tools=None,
                            **kw):
        extra = {"tools": tools} if tools else {}
        out, _ = cu.render_jinja_template(
            conversations=[messages],
            chat_template=chat_template or self.chat_template,
            add_generation_prompt=add_generation_prompt, bos_token=BOS,
            **extra, **kw)
        return [ord(c) for c in out[0]]


def _tok(tok, messages, **kw):
    return P.tokenize(None, tok, P.ChatRequest(messages=messages,
                                               tools=TOOLS),
                      P.PromptArgs(kw or None))


def _stored(segs):
    """The checkpoints a request stores: the end of each segment but the
    generation prompt's."""
    ends, n = [], 0
    for s in segs[:-1]:
        n += len(s)
        ends.append(n)
    return ends


@pytest.mark.parametrize("mode", ["chat", "thinking"])
def test_agent_loop_renders_keep_earlier_checkpoints_as_prefixes(mode):
    """goal + tool rounds -> answer -> a new user message -> compaction's
    summary ask: each render continues from every checkpoint an earlier
    one stored."""
    tok = DSTok()
    full = agent()
    steps = [full[:2]]
    for i, m in enumerate(full):
        if m["role"] == "tool" and (i + 1 == len(full)
                                    or full[i + 1]["role"] != "tool"):
            steps.append(full[:i + 1])
    steps.append(full + [{"role": "user", "content": "And f3?"}])
    steps.append(steps[-1] + [{"role": "assistant", "content": "f3 too.",
                               "reasoning_content": "ok"},
                              {"role": "user", "content": "Summarize."}])
    stored = []
    for msgs in steps:
        p, segs, types, _ = _tok(tok, msgs, thinking_mode=mode)
        assert sum(segs, []) == p
        for earlier in stored:
            assert p[:len(earlier)] == earlier
        stored += [p[:e] for e in _stored(segs)]
    assert len(stored) >= 2 * len(steps) - 1


def test_the_stub_is_replaced_on_the_tokenizer():
    stub = "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}"
    tok = DSTok(stub)
    p, *_ = _tok(tok, agent()[:2])
    s = "".join(map(chr, p))
    assert "## Tools" in s and tok.chat_template == TEMPLATE
    # a template with tool handling is the artifact's own business
    other = DSTok("{% if tools %}tool{% endif %}", name="x/DeepSeek-V4-Flash")
    assert templates.install(other) is None
    # nor is a stub-shaped template on some other model
    assert templates.install(DSTok(stub, name="x/Other")) is None


def test_a_control_token_quoted_in_a_tool_result_stays_text():
    """A tool result quoting DSML markup (this file, say) would open a
    tool call; the prompt stage makes it plain text."""
    msgs = agent()[:4]
    msgs[3]["content"] = f"<{D}invoke name=\"rm\">"
    p, *_ = _tok(DSTok(), msgs)
    s = "".join(map(chr, p))
    assert f"<tool_result><{D}invoke" not in s
    assert "<tool_result><｜​DSML｜invoke" in s


@pytest.mark.skipif(not (STUB / "chat_template.jinja").is_file(),
                    reason="the mlx-community conversion is not mounted"
                           " (set KNURLOGIC_TEST_DEEPSEEK_V4)")
def test_the_mlx_community_stub_is_known_by_hash():
    t = (STUB / "chat_template.jinja").read_text()
    assert hashlib.sha256(t.encode()).hexdigest() in templates.STUBS
    assert templates.family_for(t) == "deepseek_v4"


def test_thinking_reads_the_template_as_deepseeks_own_dialect():
    """The template carries GLM's and Qwen's effort words too (Think Max
    says "Reasoning Effort", the kwarg is reasoning_effort): its own,
    more specific dialect must win, not their on/off or effort ladders."""
    from knurlogic.engine.serve import thinking
    from knurlogic.engine.serve.load import tool_support
    assert thinking.detect(TEMPLATE)[0] == "deepseek_effort"
    assert tool_support(TEMPLATE)["parser"] == "deepseek_v4 (knurlogic)"


# ------------------------------------------------------------- the parser

def _block(s):
    """The text the engine hands the parser: between the start and end
    sequences (in the last assistant turn: the system prompt shows one)."""
    i = max(s.rfind("<｜Assistant｜>"), 0)
    i = s.index(templates.DSV4_START, i) + len(templates.DSV4_START)
    return s[i:s.index(templates.DSV4_END, i)]


def test_the_parser_reads_back_what_the_template_renders():
    h = agent()
    for a in (m for m in h if m.get("tool_calls")):
        s = render([h[1], a], TOOLS, gen=False)
        got = templates.parse_deepseek_v4(_block(s))
        assert got == [{"name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"]}
                       for tc in a["tool_calls"]]


def test_the_parser_agrees_with_deepseeks():
    s = render(agent()[:3], TOOLS, gen=False, thinking_mode="thinking")
    text = s[s.index("<｜Assistant｜><think>") + 20:]
    ref = official.parse_message_from_completion_text(text, "thinking")
    ours = templates.parse_deepseek_v4(_block(text))
    assert [(c["function"]["name"], json.loads(c["function"]["arguments"]))
            for c in ref["tool_calls"]] == \
        [(c["name"], c["arguments"]) for c in ours]
    assert ref["content"] == "Reading."


def test_the_parser_refuses_a_block_with_no_call():
    with pytest.raises(ValueError):
        templates.parse_deepseek_v4(">\nnothing here\n")


def test_install_gives_a_wrapper_the_dsml_parser():
    class Enc:
        def encode(self, s, add_special_tokens=False):
            return [ord(c) for c in s]

    class Wrapper(DSTok):
        _tool_parser = None
        _tokenizer = Enc()

    stub = "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}"
    w = Wrapper(stub)
    assert templates.install(w) == "deepseek_v4"
    assert w._tool_parser is templates.parse_deepseek_v4
    assert w._tool_call_start == templates.DSV4_START
    assert w._tool_call_start_tokens == tuple(map(ord, templates.DSV4_START))
    assert templates.install(w) == "deepseek_v4"      # idempotent



def test_a_shipped_copy_of_the_template_gets_ours_and_the_parser():
    """A release ships (an older) copy of knurlogic's template: it speaks
    DSML and handles tools, so it is no stub -- and was kept with no
    parser, the calls coming back as text. It is replaced by ours, current,
    with the parser."""
    class Enc:
        def encode(self, s, add_special_tokens=False):
            return [ord(c) for c in s]

    class Wrapper(DSTok):
        _tool_parser = None
        _tokenizer = Enc()

    older = TEMPLATE.replace("Think Max", "an older copy")
    assert older != TEMPLATE
    w = Wrapper(older)
    assert templates.install(w) == "deepseek_v4"
    assert w.chat_template == TEMPLATE
    assert w._tool_parser is templates.parse_deepseek_v4

_REAL = STUB / "tokenizer.json"


@pytest.mark.skipif(not _REAL.is_file(),
                    reason="the DeepSeek-V4 tokenizer is not mounted"
                           " (set KNURLOGIC_TEST_DEEPSEEK_V4)")
def test_a_dsml_tool_call_parses_through_the_engine(tmp_path):
    """DeepSeek-V4's real tokenizer: a model turn with reasoning, text and
    a DSML call, fed token by token through the engine's control machine
    and Request, comes back as reasoning, content and one tool call."""
    pytest.importorskip("mlx_lm")
    import shutil

    from mlx_lm.utils import load_tokenizer

    from knurlogic.engine.runtime.executor import Token
    from knurlogic.engine.runtime.request import Request, control_machine
    d = tmp_path / "DeepSeek-V4-Flash"
    d.mkdir()
    for f in ("tokenizer.json", "tokenizer_config.json",
              "chat_template.jinja"):
        shutil.copy(_REAL.parent / f, d / f)
    tok = load_tokenizer(d)
    assert templates.install(tok) == "deepseek_v4" and tok.has_tool_calling
    out = (f"Needs a file.</think>Let me look.\n\n<{D}tool_calls>\n"
           f'<{D}invoke name="read_file">\n'
           f'<{D}parameter name="path" string="true">a/b.py</{D}parameter>\n'
           f'<{D}parameter name="lines" string="false">5</{D}parameter>\n'
           f"</{D}invoke>\n</{D}tool_calls>{EOS}")
    ids = tok.encode(out, add_special_tokens=False)
    sm, seqs = control_machine(tok, "reasoning")
    req = Request(tok.detokenizer, sequences=seqs,
                  tool_parser=tok.tool_parser, tools=TOOLS)
    st = sm.make_state()
    got = {"r": "", "c": "", "t": [], "f": None}
    for i, t in enumerate(ids):
        st, match, cur = sm.match(st, t)
        finish = "stop" if (match is not None and cur is None) else (
            "length" if i == len(ids) - 1 else None)
        d_ = req.feed(Token(0, t, -0.5, finish, cur, match))
        got["r"] += d_.reasoning
        got["c"] += d_.content
        got["t"] += d_.tool_calls
        got["f"] = d_.finish or got["f"]
        if req.finished:
            break
    assert got["r"] == "Needs a file."
    assert got["c"] == "Let me look.\n\n"
    assert got["f"] == "tool_calls" and len(got["t"]) == 1
    fn = got["t"][0]["function"]
    assert fn["name"] == "read_file"
    assert json.loads(fn["arguments"]) == {"path": "a/b.py", "lines": 5}


@pytest.mark.parametrize("mode", ["chat", "thinking"])
def test_the_latest_reminder_kwarg_is_deepseeks_reminder_before_the_last_user_turn(mode):
    """The chat page sends the date and language as a kwarg (without a
    language the model reasons in Chinese): it renders as the official
    latest_reminder message before the last user turn, and a request that
    carries its own reminder keeps it."""
    r = "2026-10-02,Friday,en-US"
    for h in ([{"role": "user", "content": "hello there!"}],
              [{"role": "system", "content": "Be brief."},
               {"role": "user", "content": "hi"},
               {"role": "assistant", "content": "Hello."},
               {"role": "user", "content": "and now?"}]):
        k = max(i for i, m in enumerate(h) if m["role"] == "user")
        want = encode(h[:k] + [{"role": "latest_reminder", "content": r}]
                      + h[k:], mode=mode)
        assert render(h, thinking_mode=mode, latest_reminder=r) == want
        own = h[:k] + [{"role": "latest_reminder", "content": "x,de"}] + h[k:]
        assert render(own, thinking_mode=mode, latest_reminder=r) == \
            encode(own, mode=mode)
    assert render(h, thinking_mode=mode) == encode(h, mode=mode)


def test_think_max_is_deepseeks_prefix_and_the_three_modes_are_the_dialect():
    """reasoning_effort max renders the official Think Max prefix (thinking
    mode only); the dialect offers off / high / max."""
    h = [{"role": "system", "content": "Be brief."},
         {"role": "user", "content": "hi"}]
    want = official.encode_messages(copy.deepcopy(h), thinking_mode="thinking",
                                    reasoning_effort="max")
    assert render(h, thinking_mode="thinking", reasoning_effort="max") == want
    assert render(h, thinking_mode="chat", reasoning_effort="max") == \
        encode(h)                           # chat mode: no prefix
    from knurlogic.engine.serve import thinking
    name, spec = thinking.detect(TEMPLATE)
    assert name == "deepseek_effort"
    assert [n[1] for n in spec["native"]] == ["off", "high", "max"]
