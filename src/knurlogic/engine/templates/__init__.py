"""Chat templates knurlogic supplies in place of an artifact's own.

Some conversions ship a template that cannot carry an agent: the
mlx-community DeepSeek-V4-Flash conversion's chat_template.jinja renders no
tool definitions, ignores assistant tool_calls and role=tool messages, and
drops reasoning outside thinking_mode='thinking' -- an agent on it never
sees its own tool calls or their results. DeepSeek publishes V4's encoding
as Python (encoding/encoding_dsv4.py in deepseek-ai/DeepSeek-V4-Flash, MIT),
not as a template; `deepseek_v4.jinja` here is a port of it, checked
against DeepSeek's own encoder and golden outputs (tests/test_deepseek_v4.py,
PROVENANCE.md).

`install(tokenizer)` swaps the template in when the artifact's is a known
stub, on the tokenizer itself, so every render (the prompt, the segment
probes, the thinking probe) and the tool-call parser agree. It is
idempotent and cheap; the loaders and the prompt stage both call it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent

#: sha256 of an artifact template -> the family whose template replaces it
STUBS = {
    # mlx-community/DeepSeek-V4-Flash(-8bit) chat_template.jinja, 2026-09
    "718a756ad62609c2539a4a4c0fa4279306773e47df105674c01f10fcb0f18c6e":
        "deepseek_v4",
}


def text(family: str) -> str:
    return (_HERE / f"{family}.jinja").read_text(encoding="utf-8")


def family_for(template, name: str = "") -> Optional[str]:
    """The family whose template should replace `template`, or None.
    A known stub by hash; or, for an artifact named DeepSeek-V4, any
    template with no tool handling at all."""
    if not isinstance(template, str) or not template:
        return None
    fam = STUBS.get(hashlib.sha256(template.encode()).hexdigest())
    if fam:
        return fam
    if "deepseek-v4" in (name or "").lower() and \
            "tool" not in template.lower():
        return "deepseek_v4"
    return None


def served_family(template, name: str = "") -> Optional[str]:
    """The family knurlogic serves for `template`: a stub it replaces, or
    one of its own templates already in place; else None."""
    fam = family_for(template, name)
    if fam:
        return fam
    for f in PARSERS:
        if template == text(f):
            return f
    return None


def override(template, name: str = "") -> Optional[str]:
    """The replacement template text, or None to keep the artifact's."""
    fam = family_for(template, name)
    return text(fam) if fam else None


def install(tokenizer) -> Optional[str]:
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
    except Exception:
        logger.warning("could not replace the chat template on %r", name)
        return None
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
    logger.info("%s: the artifact's chat template is a known stub; using "
                "knurlogic's %s template", name or "tokenizer", fam)
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
PARSERS = {"deepseek_v4": (DSV4_START, DSV4_END, parse_deepseek_v4)}
