"""DeepSeek's encoders put the reasoning-effort text at index 0; knurlogic's
template opens the last user turn with it instead (the prompt cache keeps
the conversation when the level changes; families/deepseek/templates/
PROVENANCE.md, "effort"). `moved` turns an encoder's prompt into that one,
so a parity test still compares every other byte with the maker's."""

#: the user turn's tag (its bars are FULLWIDTH VERTICAL LINE, U+FF5C)
USER = "<\uff5cUser\uff5c>"


def moved(prompt: str, effort_text: str, bos: str) -> str:
    """`prompt` (an encoder's, `effort_text` right after `bos`) with the
    effort text moved to the start of its last user turn."""
    if not effort_text:
        return prompt
    head = bos + effort_text
    assert prompt.startswith(head), "the encoder's effort text is not first"
    rest = bos + prompt[len(head):]
    i = rest.rindex(USER) + len(USER)
    return rest[:i] + effort_text + rest[i:]
