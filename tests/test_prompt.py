"""engine/runtime/prompt.py: the prompt stage's rules, with a stand-in
template (no model)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knurlogic.engine.runtime import prompt as P  # noqa: E402
from knurlogic.engine.serve import thinking  # noqa: E402

SYS, USR, END, GEN, TS, TE = 100, 200, 300, 400, 1, 2


class Tok:
    """<sys>content<end> <usr>content<end> <gen><think>: role and message
    tokens are fixed ids; a message's content is its length in 7s."""
    has_chat_template = True
    has_thinking = True
    think_start, think_end = "<think>", "</think>"
    _think_end_tokens = [TE]

    def apply_chat_template(self, messages, add_generation_prompt=False,
                            tokenize=True, **kw):
        assert thinking.CLOSE not in kw, "the flag reached the template"
        out = []
        for m in messages:
            out += [SYS if m["role"] == "system" else USR]
            out += [7] * len(m["content"]) + [END]
        if add_generation_prompt:
            out += [GEN, TS]
        return out

    def rfind_think_start(self, p, start=0):
        return max((i for i, t in enumerate(p) if t == TS and i >= start),
                   default=-1)

    def rfind_think_end(self, p, start=0):
        return max((i for i, t in enumerate(p) if t == TE and i >= start),
                   default=-1)


def _req(*msgs):
    return P.ChatRequest(messages=[{"role": r, "content": c}
                                   for r, c in msgs])


def test_an_open_think_block_starts_in_reasoning():
    p, segs, types, state = P.tokenize(None, Tok(), _req(("user", "hi")),
                                       P.PromptArgs())
    assert p[-1] == TS and state == "reasoning"
    assert sum(segs, []) == p


def test_close_think_ends_the_prompt_closed_and_starts_normal():
    args = P.PromptArgs({thinking.CLOSE: True})
    p, segs, types, state = P.tokenize(None, Tok(), _req(("user", "hi")),
                                       args)
    assert p[-2:] == [TS, TE] and state == "normal"
    assert sum(segs, []) == p


def test_the_system_prompt_gets_its_own_segment():
    p, segs, types, _ = P.tokenize(None, Tok(),
                                   _req(("system", "abcd"), ("user", "hi")),
                                   P.PromptArgs())
    # up to where the system render and the prompt first differ: the
    # empty user turn's header is shared too (mlx-lm's rule)
    assert types[0] == "system" and segs[0] == [SYS, 7, 7, 7, 7, END, USR]
    assert sum(segs, []) == p


def test_list_content_is_joined_and_non_text_refused():
    import pytest
    r = P.ChatRequest(messages=[{"role": "user", "content": [
        {"type": "text", "text": "ab"}, {"type": "text", "text": "c"}]}])
    p, *_ = P.tokenize(None, Tok(), r, P.PromptArgs())
    assert p.count(7) == 3
    r.messages[0]["content"].append({"type": "image_url"})
    with pytest.raises(P.PromptError):
        P.tokenize(None, Tok(), r, P.PromptArgs())
