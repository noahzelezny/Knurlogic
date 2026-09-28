"""engine/runtime/request.py: tokens -> reasoning, answer, tool calls, stop
strings and usage, with no model. Token events are driven through the same
state machine the engine uses, built by `control_machine`."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("mlx_lm")

from knurlogic.engine.runtime.executor import Token  # noqa: E402
from knurlogic.engine.runtime.request import (Request,  # noqa: E402
                                              control_machine)

# A toy vocabulary: one string per id. 90 = end of turn.
VOCAB = {1: "<think>", 2: "</think>", 3: "<tool_call>", 4: "</tool_call>",
         10: "The", 11: " answer", 12: " is", 13: " D", 14: ".", 15: " Hmm",
         16: " so", 17: "{\"name\": \"f\", \"arguments\": {\"x\": 1}}",
         18: " STO", 19: "P", 20: " more", 21: "<", 22: "end>",
         90: "<eot>"}


class Detok:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tokens, self.text, self.offset = [], "", 0

    def add_token(self, t):
        self.tokens.append(t)
        self.text += VOCAB[t]

    def finalize(self):
        pass

    @property
    def last_segment(self):
        seg, self.offset = self.text[self.offset:], len(self.text)
        return seg


class Tok:
    eos_token_ids = [90]
    has_thinking = True
    think_start, think_end = "<think>", "</think>"
    think_start_tokens, think_end_tokens = [1], [2]
    has_tool_calling = True
    tool_call_start, tool_call_end = "<tool_call>", "</tool_call>"
    tool_call_start_tokens, tool_call_end_tokens = [3], [4]

    def convert_ids_to_tokens(self, t):
        return VOCAB[t]


def run(ids, *, initial="normal", tok=None, **kw):
    """Feed ids through the machine and a Request; (deltas joined, req)."""
    tok = tok or Tok()
    sm, seqs = control_machine(tok, initial)
    req = Request(Detok(), sequences=seqs, **kw)
    st = sm.make_state()
    got = {"reasoning": "", "content": "", "tool_calls": [], "finish": None,
           "deltas": []}
    for i, t in enumerate(ids):
        st, match, cur = sm.match(st, t)
        finish = "stop" if (match is not None and cur is None) else (
            "length" if i == len(ids) - 1 else None)
        d = req.feed(Token(0, t, -0.5, finish, cur, match))
        _add(got, d)
        if req.finished:
            break
    return got, req


def _add(got, d):
    got["reasoning"] += d.reasoning
    got["content"] += d.content
    got["tool_calls"] += d.tool_calls
    got["finish"] = d.finish or got["finish"]
    got["deltas"].append(d)


def test_reasoning_and_answer_split_and_markers_never_show():
    got, req = run([1, 15, 16, 2, 10, 11, 12, 13, 14, 90])
    assert got["reasoning"] == " Hmm so"
    assert got["content"] == "The answer is D."
    assert got["finish"] == "stop"
    assert req.reasoning_tokens == 3 and req.completion_tokens == 10


def test_a_prompt_that_opens_the_think_block_starts_in_reasoning():
    got, _ = run([15, 2, 10, 90], initial="reasoning")
    assert got["reasoning"] == " Hmm" and got["content"] == "The"


def test_a_stop_string_across_a_token_boundary_ends_the_answer_before_it():
    # "STOP" arrives as " STO" + "P": token-id matching never sees it
    got, req = run([10, 11, 18, 19, 20, 90], stops=["STOP"])
    assert got["content"] == "The answer "
    assert got["finish"] == "stop" and req.finished == "stop"


def test_a_stop_string_never_reaches_a_streamed_delta():
    got, _ = run([10, 11, 18, 19, 20, 90], stops=["STOP"])
    for d in got["deltas"]:
        assert "STO" not in d.content


def test_a_stop_string_inside_reasoning_does_not_stop():
    got, _ = run([1, 13, 2, 10, 14, 90], stops=[" D"])
    assert got["reasoning"] == " D" and got["content"] == "The."
    assert got["finish"] == "stop"


def test_held_text_comes_out_when_there_is_no_stop():
    got, _ = run([10, 11, 12, 90], stops=["NEVER-SEEN"])
    assert got["content"] == "The answer is"


def test_a_stop_found_the_moment_the_text_completes_it():
    got, _ = run([10, 21, 22, 20, 90], stops=["<end>"])
    assert got["content"] == "The" and got["finish"] == "stop"


def test_a_tool_call_is_parsed_and_the_finish_says_tool_calls():
    parse = lambda text, tools: json.loads(text)  # noqa: E731
    got, req = run([10, 3, 17, 4, 90], tool_parser=parse)
    assert got["content"] == "The"
    [tc] = got["tool_calls"]
    assert tc["function"] == {"name": "f", "arguments": "{\"x\": 1}"}
    assert tc["type"] == "function" and tc["id"] and tc["index"] == 0
    assert got["finish"] == "tool_calls"


def test_an_unclosed_tool_call_is_still_reported_at_the_end():
    parse = lambda text, tools: json.loads(text)  # noqa: E731
    got, _ = run([3, 17], tool_parser=parse)
    assert len(got["tool_calls"]) == 1 and got["finish"] == "length"


def test_logprobs_ride_with_the_text_they_belong_to():
    got, _ = run([10, 11, 90], logprobs=True)
    lps = [lp for d in got["deltas"] for lp in d.logprobs]
    assert [t for t, _, _ in lps] == [10, 11, 90]


def test_usage_counts_and_carries_the_cache_report():
    _, req = run([1, 15, 2, 10, 90], prompt_tokens=7)
    u = req.usage({"used": 5, "offered": 5})
    assert u["prompt_tokens"] == 7 and u["completion_tokens"] == 5
    assert u["total_tokens"] == 12
    assert u["completion_tokens_details"]["reasoning_tokens"] == 2
    assert u["prompt_tokens_details"]["cached_tokens"] == 5
    assert u["knurlogic"]["cache"]["used"] == 5


def test_nothing_after_the_finish():
    sm, seqs = control_machine(Tok())
    req = Request(Detok(), sequences=seqs, stops=["is"])
    req.feed(Token(0, 10, 0.0, state="normal"))
    d = req.feed(Token(0, 12, 0.0, state="normal"))
    assert d.finish == "stop"
    assert not req.feed(Token(0, 11, 0.0, state="normal"))
    assert not req.finish()


class HoldingDetok(Detok):
    """Like mlx-lm's BPE detokenizer: a lone space is held and flushed
    together with the next token's text."""

    def add_token(self, t):
        self.tokens.append(t)
        piece = VOCAB[t]
        if piece == " ":
            self._held = " "
            return
        self.text += getattr(self, "_held", "") + piece
        self._held = ""


def test_a_held_space_flushed_with_a_marker_stays_in_the_answer():
    VOCAB[30] = " "
    sm, seqs = control_machine(Tok())
    req = Request(HoldingDetok(), sequences=seqs)
    st, text = sm.make_state(), ""
    for t in (10, 30, 3, 17, 4, 90):          # "The" " " <tool_call> ...
        st, match, cur = sm.match(st, t)
        d = req.feed(Token(0, t, 0.0, "stop" if cur is None and match
                           else None, cur, match))
        text += d.content
    assert text == "The "


def test_an_engine_error_mid_stream_is_an_anthropic_error_event():
    import json as _j
    from knurlogic.interfaces.http import messages
    lines = [b'data: {"choices": [{"delta": {"content": "par"}}]}\n\n',
             b'data: {"error": {"message": "non-finite logits"}}\n\n',
             b"data: [DONE]\n\n"]
    evs = list(messages.stream(lines, "m"))
    kinds = [e.split(b"\n", 1)[0] for e in evs]
    assert b"event: error" in kinds
    assert not any(b"end_turn" in e for e in evs)
    err = [e for e in evs if e.startswith(b"event: error")][0]
    assert b"non-finite" in err
