"""DeepSeek-V4's tool-call dialect (DSML): the block the engine's state
machine cuts out and the parser that reads it. The template that writes
it is templates/deepseek_v4.jinja (templates/PROVENANCE.md); the manifest's
`chat_templates` names both.

Stdlib only.
"""
from __future__ import annotations

import json
import re

_D = "｜DSML｜"
#: The tool block's start and end as the state machine matches them.
#: The start stops before its ">": the model writes ">\n" there, one token
#: (">Ċ"), so the string with ">" never matches the generated tokens. The
#: end keeps it: end-of-sentence follows, and ">" stands alone.
DSV4_START = f"<{_D}tool_calls"
DSV4_END = f"</{_D}tool_calls>"
_INVOKE = re.compile(rf'<{_D}invoke\s+name="(.*?)">\n?(.*?)</{_D}invoke>',
                     re.S)
_PARAM = re.compile(rf'<{_D}parameter\s+name="(.*?)"\s+string="(true|false)">'
                    rf'(.*?)</{_D}parameter>', re.S)


def parse_deepseek_v4(text: str, tools=None):
    """The calls in a DSML tool block (the text between DSV4_START and
    DSV4_END) -> [{"name", "arguments": dict}], after DeepSeek's
    parse_tool_calls: string="true" values are raw strings, the rest JSON.
    Raises ValueError when there is no call in it."""
    calls = []
    for name, body in _INVOKE.findall(text):
        args = {}
        for key, is_str, val in _PARAM.findall(body):
            if key in args:
                raise ValueError(f"duplicate parameter {key!r}")
            if is_str == "true":
                args[key] = val
            else:
                try:
                    args[key] = json.loads(val)
                except ValueError:
                    args[key] = val
        calls.append({"name": name, "arguments": args})
    if not calls:
        raise ValueError("no DSML invoke in the tool block")
    return calls


#: (tool block start, end, parser(text, tools)), as engine/templates
#: installs it on a tokenizer
PARSER = (DSV4_START, DSV4_END, parse_deepseek_v4)
