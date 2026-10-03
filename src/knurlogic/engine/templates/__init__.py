"""Chat templates knurlogic supplies in place of an artifact's own.

Some conversions ship a template that cannot carry an agent: the
mlx-community DeepSeek-V4-Flash conversion's chat_template.jinja renders no
tool definitions, ignores assistant tool_calls and role=tool messages, and
drops reasoning outside thinking_mode='thinking'. `deepseek_v4.jinja` is a
port of DeepSeek's Python encoder (encoding/encoding_dsv4.py in
deepseek-ai/DeepSeek-V4-Flash, MIT), checked against its golden outputs
(tests/test_deepseek_v4.py, PROVENANCE.md).

`install(tokenizer)` swaps the template in when the artifact's is a known
stub, on the tokenizer itself, so every render and the tool-call parser
agree. It is idempotent and cheap; the loaders and the prompt stage both
call it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

import jinja2

logger = logging.getLogger(__name__)

#: What rendering a third-party chat template can raise: Jinja's own errors
#: (including the template's raise_exception) and the plain ones a template
#: meets in an odd message shape.
TEMPLATE_ERRORS = (jinja2.TemplateError, ValueError, TypeError, KeyError,
                   IndexError, AttributeError)

_HERE = Path(__file__).resolve().parent

#: sha256 of an artifact template -> the family whose template replaces it
STUBS = {
    # mlx-community/DeepSeek-V4-Flash(-8bit) chat_template.jinja, 2026-09
    "718a756ad62609c2539a4a4c0fa4279306773e47df105674c01f10fcb0f18c6e":
        "deepseek_v4",
}


#: DeepSeek-V4-Flash-Vision-Exp's encoder is Flash's with other reasoning
#: effort prefixes: its "high" is Flash's "max", its "max" a new one. Its
#: template is Flash's with this line first.
DSV4_VISION = "{%- set dsv4_vision = true -%}\n"


def text(family: str) -> str:
    if family == "deepseek_v4_vision":
        return DSV4_VISION + text("deepseek_v4")
    return (_HERE / f"{family}.jinja").read_text(encoding="utf-8")


def family_for(template, name: str = "") -> str | None:
    """The family whose template should replace `template`, or None.
    A known stub by hash; any template that speaks DeepSeek-V4's DSML (a
    copy of knurlogic's own, shipped in a release, or an older one: ours,
    current, replaces it and its parser is installed -- a copy kept as it
    was had no parser, and its tool calls came back as text); or, for an
    artifact named DeepSeek-V4, no template or one with no tool handling.
    An artifact named DeepSeek-V4 and Vision gets the Vision-Exp variant."""
    low = (name or "").lower()
    named = "deepseek-v4" in low
    ours = "deepseek_v4_vision" if named and "vision" in low else "deepseek_v4"
    if not isinstance(template, str) or not template:
        # no template at all (deepseek-ai's own MLX conversion ships none):
        # chat was refused outright
        return ours if named else None
    fam = STUBS.get(hashlib.sha256(template.encode()).hexdigest())
    if fam:
        return fam
    if template.startswith(DSV4_VISION):
        return "deepseek_v4_vision"
    if "｜DSML｜" in template:
        return ours
    if named and "tool" not in template.lower():
        return ours
    return None


def served_family(template, name: str = "") -> str | None:
    """The family knurlogic serves for `template`: a stub it replaces, or
    one of its own templates already in place; else None."""
    fam = family_for(template, name)
    if fam:
        return fam
    for f in PARSERS:
        if template == text(f):
            return f
    return None


def override(template, name: str = "") -> str | None:
    """The replacement template text, or None to keep the artifact's."""
    fam = family_for(template, name)
    return text(fam) if fam else None


def install(tokenizer) -> str | None:
    """Replace a stub template on `tokenizer` (an mlx-lm TokenizerWrapper or
    an HF tokenizer) and give it the family's tool-call parser when it has
    none. Returns the family installed, or None."""
    t = getattr(tokenizer, "chat_template", None)
    done = getattr(tokenizer, "_knurlogic_template", None)
    if done is not None and t is done[1]:
        return done[0]
    name = str(getattr(tokenizer, "name_or_path", "") or "")
    fam = family_for(t, name)
    if fam is None:
        return None
    new = text(fam)
    try:
        tokenizer.chat_template = new
    except (AttributeError, TypeError):
        logger.warning("could not replace the chat template on %r", name)
        return None
    # mlx-lm's wrapper fixes this at construction: an artifact with no
    # template read "no chat template" and chat was refused after this
    if getattr(tokenizer, "has_chat_template", True) is False:
        tokenizer.has_chat_template = True
    parser = PARSERS.get(fam)
    if parser is not None and hasattr(tokenizer, "_tool_parser") and \
            getattr(tokenizer, "_tool_parser", None) is None:
        start, end, fn = parser
        tokenizer._tool_parser = fn
        tokenizer._tool_call_start = start
        tokenizer._tool_call_end = end
        enc = getattr(tokenizer, "_tokenizer", tokenizer)
        tokenizer._tool_call_start_tokens = tuple(
            enc.encode(start, add_special_tokens=False))
        tokenizer._tool_call_end_tokens = tuple(
            enc.encode(end, add_special_tokens=False))
    try:
        tokenizer._knurlogic_template = (fam, tokenizer.chat_template)
    except (AttributeError, TypeError):
        pass    # a tokenizer that refuses attributes: re-detect next time
    logger.info("%s: the artifact's chat template is a known stub, a copy or "
                "missing; using knurlogic's %s template", name or "tokenizer",
                fam)
    return fam


# ------------------------------------------------------ DeepSeek-V4 (DSML)

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


#: family -> (tool block start, end, parser(text, tools))
PARSERS = {"deepseek_v4": (DSV4_START, DSV4_END, parse_deepseek_v4),
           "deepseek_v4_vision": (DSV4_START, DSV4_END, parse_deepseek_v4)}
