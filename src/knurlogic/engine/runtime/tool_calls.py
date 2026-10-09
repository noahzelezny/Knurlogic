"""Tool-call arguments the tokenizer's own parser dropped.

mlx-lm's qwen3_coder parser reads `<parameter=NAME>VALUE</parameter>` and
nothing else: a call whose model left out a `</parameter>` (ending the value
at the next `<parameter=` or at `</function>`), or wrote its arguments as a
JSON object inside `<function=...>`, comes back with `arguments: {}` and no
error. Qwen3.6 does both late in long agent conversations, and an agent then
calls a tool with nothing in it. `recover` reads those two forms back from the
call's text; a call that still has no arguments while its tool requires some
is logged with the text, so the next form shows up instead of an empty call.
No mlx here (request.py's rule).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_PARAM = re.compile(r"<parameter=\s*([^\s<>]+?)\s*>(.*?)"
                    r"(?=</parameter>|<parameter=|</function>|$)", re.DOTALL)
_STRING = {"string", "str", "text", "varchar", "char", "enum"}


def _schema(name: str, tools) -> dict:
    for t in tools or []:
        f = t.get("function") if isinstance(t, dict) else None
        if isinstance(f, dict) and f.get("name") == name:
            return f.get("parameters") or {}
    return {}


def _value(raw: str, spec: dict) -> Any:
    v = raw.strip("\n")
    if str((spec or {}).get("type", "string")).lower() in _STRING:
        return v
    try:
        return json.loads(v)
    except ValueError:
        return v


def recover(text: str, call: dict, tools) -> dict:
    """`call` ({"name", "arguments"}) with the arguments its parser dropped
    read back from `text`; unchanged when it has some or none are found."""
    args = call.get("arguments")
    if args not in ({}, None, "", "{}"):
        return call
    name = call.get("name") or ""
    schema = _schema(name, tools)
    props = schema.get("properties") or {}
    got = {k: _value(v, props.get(k, {})) for k, v in _PARAM.findall(text)}
    if not got:
        body = text.split(f"<function={name}>", 1)[-1]
        a, b = body.find("{"), body.rfind("}")
        if 0 <= a < b:
            try:
                obj = json.loads(body[a:b + 1])
                got = obj if isinstance(obj, dict) else {}
            except ValueError:
                got = {}
    if got:
        return {**call, "arguments": got}
    if schema.get("required"):
        logger.warning("tool call %r has no arguments but its tool requires "
                       "%s; the model wrote: %r", name, schema["required"],
                       text[-400:])
    return call
