"""Chat templates knurlogic supplies in place of an artifact's own.

Some conversions ship a template that cannot carry an agent: the
mlx-community DeepSeek-V4-Flash conversion's chat_template.jinja renders no
tool definitions, ignores assistant tool_calls and role=tool messages, and
drops reasoning outside thinking_mode='thinking'.

The templates, their variants, what selects them and their tool-call
parsers are the families' (`chat_templates` in a family's MANIFEST,
engine/families/; DeepSeek-V4's port and its provenance are
families/deepseek/templates/). This module is the generic half: which
one an artifact gets, its text, and putting it on a tokenizer.

`install(tokenizer)` swaps the template in when the artifact's is a known
stub, on the tokenizer itself, so every render and the tool-call parser
agree. It is idempotent and cheap; the loaders and the prompt stage both
call it.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
from pathlib import Path

import jinja2

from knurlogic.engine import families as _families

logger = logging.getLogger(__name__)

#: What rendering a third-party chat template can raise: Jinja's own errors
#: (including the template's raise_exception) and the plain ones a template
#: meets in an odd message shape.
TEMPLATE_ERRORS = (jinja2.TemplateError, ValueError, TypeError, KeyError,
                   IndexError, AttributeError)

#: template name -> its manifest entry (with "family"), every family's
SPECS: dict = _families.build_maps()["chat_templates"]


def _base(name: str) -> str:
    return SPECS[name].get("base", name)


#: sha256 of an artifact template -> the template that replaces it
STUBS = {h: n for n, s in SPECS.items() for h in s.get("stubs", ())}


def _resolve(target: str):
    mod, _, attr = target.partition(":")
    return getattr(importlib.import_module(mod), attr)


#: template -> (tool block start, end, parser(text, tools)); a variant
#: takes its base's
PARSERS = {n: _resolve(SPECS[_base(n)]["parser"]) for n in SPECS
           if SPECS[_base(n)].get("parser")}


def text(name: str) -> str:
    s = SPECS[name]
    if "base" in s:
        return s.get("prefix", "") + text(s["base"])
    return (_families.HERE / s["family"] / s["file"]).read_text(
        encoding="utf-8")


def part_separator(name: str | None, role: str | None = None) -> str:
    """What a message's list of text parts is joined with under template
    `name`, for a message of `role`: its maker's encoder's joiner; ""
    (mlx-lm's) where the manifest says nothing."""
    seps = (SPECS.get(name) or {}).get("part_separator") or {}
    return seps.get(role, seps.get("default", "")) if role \
        else seps.get("default", "")


def _config(name: str) -> dict | None:
    """The artifact's config.json, `name` being its folder; else None."""
    if not name:
        return None
    try:
        c = json.loads((Path(name) / "config.json").read_text())
    except (OSError, ValueError):
        return None
    return c if isinstance(c, dict) else None


def _positive(v) -> bool:
    try:
        return v is not None and not isinstance(v, bool) and float(v) > 0
    except (TypeError, ValueError):
        return False


def _for_config(base: str, config: dict | None) -> str:
    """`base` or the variant of it the artifact's config.json selects."""
    if config:
        for n, s in SPECS.items():
            if s.get("base") == base and any(
                    _positive(config.get(k)) for k in s.get("when_config", ())):
                return n
    return base


def family_for(template, name: str = "") -> str | None:
    """The template that should replace `template`, or None. `name` is the
    artifact's folder: its config.json (model_type, and the fields a
    variant is chosen by) says which family's template and which variant,
    never the folder's name. Replaced: a known stub by hash; any template
    carrying a family's marker (a copy of knurlogic's own, shipped in a
    release, or an older one: ours, current, replaces it and its parser is
    installed -- a copy kept as it was had no parser, and its tool calls
    came back as text); or, for an artifact of a model_type a family
    serves a template for, no template or one with no tool handling."""
    config = _config(name)
    mt = (config or {}).get("model_type")
    bases = [n for n, s in SPECS.items() if "base" not in s]
    ours = next((_for_config(n, config) for n in bases
                 if mt and mt in SPECS[n].get("model_types", ())), None)
    if not isinstance(template, str) or not template:
        # no template at all (deepseek-ai's own MLX conversion ships none):
        # chat was refused outright
        return ours
    stub = STUBS.get(hashlib.sha256(template.encode()).hexdigest())
    if stub:
        return _for_config(stub, config)
    for n, s in SPECS.items():
        if s.get("prefix") and template.startswith(s["prefix"]):
            return n
    for n in bases:
        marker = SPECS[n].get("marker")
        if marker and marker in template:
            return _for_config(n, config)
    if ours and "tool" not in template.lower():
        return ours
    return None


def served_family(template, name: str = "") -> str | None:
    """The template knurlogic serves for `template`: a stub it replaces, or
    one of its own templates already in place; else None."""
    fam = family_for(template, name)
    if fam:
        return fam
    for f in SPECS:
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
    none. Returns the template installed, or None."""
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
