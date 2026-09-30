"""engine/serve/segments: a system prompt whose render is a pure prefix of
the prompt still gets its own segment, so its checkpoint can be reused."""
from knurlogic.engine.serve.segments import split_system


class GLMish:
    """Renders like GLM-5.3: the empty user turn is just the <|user|> tag."""
    def apply_chat_template(self, msgs, add_generation_prompt=False,
                            tokenize=True, **kw):
        out = [1, 2]                                   # [gMASK]<sop>
        for m in msgs:
            out += [10 if m["role"] == "system" else 20]
            out += [ord(c) for c in m["content"]]
        if add_generation_prompt:
            out += [30, 40]                            # <|assistant|><think>
        return out


MSGS = [{"role": "system", "content": "schema"}, {"role": "user",
                                                   "content": "item"}]


def test_a_prefix_render_splits_off_the_system_segment():
    tok = GLMish()
    prompt = tok.apply_chat_template(MSGS, add_generation_prompt=True)
    segs, types = split_system(tok, MSGS, prompt, [prompt[:-1], prompt[-1:]],
                               ["user", "assistant"], {})
    assert types == ["system", "user", "assistant"]
    sys_ = tok.apply_chat_template(MSGS[:1] + [{"role": "user",
                                                "content": ""}])
    assert segs[0] == sys_ and sum(map(len, segs)) == len(prompt)


def test_left_alone_when_mlx_lm_already_split_or_no_system():
    tok = GLMish()
    p = tok.apply_chat_template(MSGS, add_generation_prompt=True)
    assert split_system(tok, MSGS, p, [p[:3], p[3:]], ["system", "user"],
                        {}) == ([p[:3], p[3:]], ["system", "user"])
    u = MSGS[1:]
    q = tok.apply_chat_template(u, add_generation_prompt=True)
    assert split_system(tok, u, q, [q], ["user"], {}) == ([q], ["user"])
