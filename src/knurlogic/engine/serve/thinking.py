"""Thinking effort: one control for every model, translated to each chat
template's own.

A client asks with OpenAI's `reasoning_effort` (or OpenRouter's
`reasoning: {"effort": ...}`), on the standard ladder

    none < minimal < low < medium < high < xhigh

and this module turns it into the `chat_template_kwargs` the served
model's template actually reads. Which controls a template has is its
DIALECT, detected from the template text -- not from the architecture: one
module (qwen3_5) ships templates with on/off only (Qwen3.6) and with graded
effort (Qwen3.8). The dialects and their native levels live in the family
manifests (engine/families/<family>/__init__.py, "thinking").

Native controls only, never a token budget. A level the template cannot
express goes to the nearest native level AT OR ABOVE it (never less thought
than asked), or the highest there is; "none" on a template with no off
switch goes to its lowest level. Every response says what was applied, in
usage.knurlogic.thinking. Deciding HOW MUCH to think for a given question
is the harness's call; this is only the translation.

A client that sends `chat_template_kwargs` itself wins over the translation,
key by key -- it asked for something specific.
"""

from __future__ import annotations

import functools
import hashlib
import json

from . import state

LADDER = ("none", "minimal", "low", "medium", "high", "xhigh")

_dialect_cache: dict = {}


def _rank(level: str) -> int:
    return LADDER.index(level)


def detect(template: str | None):
    """(dialect name, spec) the template speaks, or (None, None)."""
    if not template:
        return None, None
    h = hashlib.sha256(template.encode()).hexdigest()
    if h in _dialect_cache:
        return _dialect_cache[h]
    from knurlogic.engine import families
    dialects = families.build_maps()["thinking"]
    # Most specific first: the more a dialect requires, the earlier it is
    # tried, so "enable_thinking + reasoning_effort" beats "enable_thinking".
    order = sorted(dialects.items(),
                   key=lambda kv: -len(kv[1]["detect"].get("all", [])))
    found = (None, None)
    for name, spec in order:
        d = spec["detect"]
        if all(s in template for s in d.get("all", [])) and \
                not any(s in template for s in d.get("none", [])):
            found = (name, spec)
            break
    _dialect_cache[h] = found
    return found


def requested(body: dict):
    """The level a request asks for, or None. Raises ValueError on a level
    that is not on the ladder -- a typo is refused, not guessed at."""
    level = body.get("reasoning_effort")
    r = body.get("reasoning")
    if level is None and isinstance(r, dict):
        level = r.get("effort")
    if level is None:
        return None
    level = str(level).strip().lower()
    if level not in LADDER:
        raise ValueError(f"reasoning_effort {level!r} is not one of "
                         f"{', '.join(LADDER)}")
    return level


def resolve(level, dialect: str | None, spec: dict | None):
    """(template kwargs, report) for a requested level on a dialect."""
    if spec is None:
        return {}, {"requested": level, "dialect": None,
                    "applied": None if level is None else "not controllable",
                    "native": level is None,
                    "note": "" if level is None else
                    "this model's chat template has no thinking control "
                    "knurlogic knows; the request was served unchanged"}
    if level is None:
        return {}, {"requested": None, "dialect": dialect,
                    "applied": spec["default"], "native": True,
                    "note": "the model's own default"}
    natives = sorted(spec["native"], key=lambda n: _rank(n[0]))
    exact = [n for n in natives if n[0] == level]
    if exact:
        pick, note = exact[0], ""
    elif level == "none":
        pick = natives[0]
        note = (f"this template has no off switch; its lowest level "
                f"({pick[1]}) was used")
    else:
        above = [n for n in natives
                 if n[0] != "none" and _rank(n[0]) >= _rank(level)]
        pick = above[0] if above else natives[-1]
        note = (f"{level} is not a level this template has; the nearest "
                f"at or above it ({pick[1]}) was used")
    return dict(pick[2]), {"requested": level, "dialect": dialect,
                           "applied": pick[1], "native": not note,
                           "note": note}


def levels(template: str | None) -> dict:
    """What a served template offers, for the MCP and /status.json."""
    name, spec = detect(template)
    if spec is None:
        return {"dialect": None, "native": [], "default": None}
    return {"dialect": name, "default": spec["default"],
            "native": [{"level": n[0], "name": n[1]} for n in spec["native"]]}


def template_of(path) -> str | None:
    """An artifact's chat template, read off disk (chat_template.jinja, else
    tokenizer_config.json) -- what `levels` needs without loading anything."""
    from pathlib import Path
    d = Path(str(path))
    j = d / "chat_template.jinja"
    if j.is_file():
        return j.read_text()
    c = d / "tokenizer_config.json"
    if c.is_file():
        try:
            t = json.loads(c.read_text()).get("chat_template")
        except ValueError:
            return None
        return t if isinstance(t, str) else None
    return None


def _served_template() -> str | None:
    prov = state.SERVED.get("provider")
    tok = getattr(prov, "tokenizer", None) if prov is not None else None
    return getattr(tok, "chat_template", None) if tok is not None else None


def _refuse(handler, msg: str) -> None:
    body = json.dumps({"error": msg}).encode()
    handler.send_response(400)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def install(srv) -> None:
    """Translate the request's effort before mlx-lm builds its generation
    arguments (APIHandler.handle_completion reads self.chat_template_kwargs),
    report it in usage, and mirror `reasoning` as `reasoning_content`, the
    name DeepSeek and vLLM clients read. Each hook guards itself."""
    H = srv.APIHandler

    real_hc = H.handle_completion
    if not getattr(real_hc, "_knurlogic_thinking", False):
        @functools.wraps(real_hc)
        def handle_completion(self, request, *a, **k):
            body = getattr(self, "body", None) or {}
            self._knurlogic_thinking = None
            if getattr(request, "request_type", "chat") == "chat":
                try:
                    level = requested(body)
                except ValueError as e:
                    return _refuse(self, str(e))
                name, spec = detect(_served_template())
                kwargs, report = resolve(level, name, spec)
                client = dict(self.chat_template_kwargs or {})
                overridden = sorted(set(kwargs) & set(client))
                if overridden:
                    report["note"] = (report["note"] + "; " if report["note"]
                                      else "") + (
                        f"the request's own chat_template_kwargs set "
                        f"{', '.join(overridden)} and won")
                self.chat_template_kwargs = {**kwargs, **client} or None
                self._knurlogic_thinking = report
            return real_hc(self, request, *a, **k)
        handle_completion._knurlogic_thinking = True
        H.handle_completion = handle_completion

    def _with_thinking(real):
        @functools.wraps(real)
        def wrapped(self, *a, **k):
            resp = real(self, *a, **k)
            if not isinstance(resp, dict):
                return resp
            for choice in resp.get("choices") or []:
                for key in ("message", "delta"):
                    part = choice.get(key)
                    if isinstance(part, dict) and "reasoning" in part:
                        part.setdefault("reasoning_content",
                                        part["reasoning"])
            report = getattr(self, "_knurlogic_thinking", None)
            usage = resp.get("usage")
            if report is not None and isinstance(usage, dict):
                usage.setdefault("knurlogic", {})["thinking"] = report
            return resp
        wrapped._knurlogic_thinking = True
        return wrapped

    for name in ("generate_response", "completion_usage_response"):
        if not getattr(getattr(H, name), "_knurlogic_thinking", False):
            setattr(H, name, _with_thinking(getattr(H, name)))
