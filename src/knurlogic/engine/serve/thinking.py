"""Thinking effort: one control for every model, translated to each chat
template's own.

A client asks with OpenAI's `reasoning_effort` (or OpenRouter's
`reasoning: {"effort": ...}`, and its `{"enabled": false}` for none), on
the standard ladder

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
key by key -- it asked for something specific -- and the report then says
what the MERGED kwargs render to, not what was asked.

THE TEMPLATE TEXT PROPOSES, RENDERING DECIDES. Detection is a substring
pre-filter on the template; with a loaded tokenizer, `probe` renders a tiny
conversation through mlx-lm's own TokenizerWrapper once per native level
and once bare. The dialect is only trusted when every level renders
differently, and the model's default is whichever level the BARE render
equals -- because mlx-lm injects enable_thinking=<has_thinking> into any
request that is silent about it (tokenizer_utils.apply_chat_template), so
gemma, whose template defaults off, thinks by default when served.

Reasoning is streamed by default; `reasoning: {"exclude": true}`
(OpenRouter's spelling) strips it from messages and deltas. Its token
count goes in usage.completion_tokens_details.reasoning_tokens.
"""

from __future__ import annotations

import hashlib
import json
import threading

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


def excluded(body: dict) -> bool:
    r = body.get("reasoning")
    return isinstance(r, dict) and bool(r.get("exclude"))


def requested(body: dict):
    """The level a request asks for, or None. Raises ValueError on a level
    that is not on the ladder -- a typo is refused, not guessed at."""
    level = body.get("reasoning_effort")
    r = body.get("reasoning")
    if level is None and isinstance(r, dict):
        level = r.get("effort")
        # OpenRouter's switch: some clients say "no thinking" only this way
        if level is None and r.get("enabled") is False:
            level = "none"
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


_PROBE_MSGS = [{"role": "user", "content": "hi"}]
_probe_cache: dict = {}
# One render at a time. Requests arrive on the server's handler threads,
# and the first few after a load probe together: concurrent renders through
# the same tokenizer failed, the probe read those failures as "the template
# does not act on its controls", and whichever thread finished last decided
# what was cached. Measured on an M4 Max (128 GB) (2026-09-25): the first four
# concurrent requests of every bench arm were served "not controllable" --
# a `none` request reasoned for 242 tokens.
_render_lock = threading.RLock()


def _render(tokenizer, kwargs: dict):
    with _render_lock:
        return _render_unlocked(tokenizer, kwargs)


def _render_unlocked(tokenizer, kwargs: dict):
    try:
        return _Closing(tokenizer).apply_chat_template(
            _PROBE_MSGS, add_generation_prompt=True, tokenize=False,
            **kwargs)
    except Exception:
        return None


#: A native level may CLOSE the think block instead of passing a template
#: kwarg: the template's own format for a turn that did not think (GLM-5.3
#: writes `<think></think>` into every past assistant turn; its template has
#: no off switch). Measured before it was offered: GLM 2.7, 12/12 right, 0
#: reasoning tokens. Never reaches the
#: template -- `_Closing` strips it and appends the tokenizer's think_end
#: to the generation prompt, so mlx-lm's own rfind sees a closed block and
#: starts the response in its normal state.
CLOSE = "_knurlogic_close_think"


class _Closing:
    """A tokenizer whose generation prompt ends with the think block
    closed when CLOSE is asked for; every other call passes through."""

    def __init__(self, tok):
        self._tok = tok

    def __getattr__(self, name):
        return getattr(self._tok, name)

    def apply_chat_template(self, *a, **kw):
        close = kw.pop(CLOSE, False)
        out = self._tok.apply_chat_template(*a, **kw)
        if not (close and kw.get("add_generation_prompt")):
            return out
        end = getattr(self._tok, "think_end", None)
        if not end:
            raise ValueError("closing the think block needs a tokenizer "
                             "that knows its think_end")
        if kw.get("tokenize", True) and not isinstance(out, str):
            return list(out) + list(self._tok._think_end_tokens)
        return out + end


def thinking_keys(spec: dict) -> set:
    return {k for n in spec["native"] for k in n[2]}


def probe(tokenizer, template: str, spec: dict) -> dict:
    """Render once per native level and once bare, through the tokenizer the
    server uses. {verified, default, renders}. Cached per template."""
    h = hashlib.sha256((template or "").encode()).hexdigest()
    with _render_lock:
        if h not in _probe_cache:
            _probe_cache[h] = _probe(tokenizer, spec)
        return _probe_cache[h]


def _probe(tokenizer, spec: dict) -> dict:
    renders = {n[1]: _render(tokenizer, dict(n[2])) for n in spec["native"]}
    bare = _render(tokenizer, {})
    ok = (None not in renders.values() and bare is not None
          and len(set(renders.values())) == len(renders))
    default = next((name for name, r in renders.items() if r == bare), None)
    return {"verified": ok, "default": default, "renders": renders}


def applied_by_render(tokenizer, spec: dict, merged: dict, p: dict):
    """Which native level the merged kwargs actually render to, judged on
    the thinking keys only (other client kwargs are not ours to weigh)."""
    keys = thinking_keys(spec)
    r = _render(tokenizer, {k: v for k, v in merged.items() if k in keys})
    return next((name for name, x in p["renders"].items() if x == r), None)


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
    from knurlogic.engine import templates
    d = Path(str(path))
    j = d / "chat_template.jinja"
    t = None
    if j.is_file():
        t = j.read_text()
    else:
        c = d / "tokenizer_config.json"
        if c.is_file():
            try:
                t = json.loads(c.read_text()).get("chat_template")
            except ValueError:
                return None
    if not isinstance(t, str):
        return None
    # the template knurlogic serves in place of a known stub
    return templates.override(t, d.name) or t


_disk_tok: dict = {}


def _served_tokenizer():
    """The server's tokenizer -- or, while the model is still loading, the
    same artifact's tokenizer read off disk.

    mlx-lm answers HTTP before its generation thread has loaded the model,
    so the first requests to a slow-loading artifact arrived with no
    tokenizer, found no template, and were served the model's own default
    level whatever they asked for (Flash-Next 2.1 on an M4 Max (128 GB), 2026-09-25:
    four `none` requests reasoned 89-232 tokens). The template on disk is
    the one the server will use; the tokenizer is loaded once, cheaply,
    and dropped when the served one appears."""
    prov = state.SERVED.get("provider")
    tok = getattr(prov, "tokenizer", None) if prov is not None else None
    if tok is not None:
        with _render_lock:        # a reader inside it is mid-lookup
            _disk_tok.clear()
        return tok
    path = state.served_path()
    if not path:
        return None
    with _render_lock:
        if _disk_tok.get("path") != path:
            try:
                from pathlib import Path
                from mlx_lm.utils import load_tokenizer
                from knurlogic.engine import templates
                tok = load_tokenizer(Path(path))
                templates.install(tok)
                _disk_tok.update(path=path, tok=tok)
            except Exception:
                _disk_tok.update(path=path, tok=None)
        return _disk_tok.get("tok")


def _served_template() -> str | None:
    tok = _served_tokenizer()
    return getattr(tok, "chat_template", None) if tok is not None else None


def status() -> dict:
    """For /status.json: what reasoning_effort means on the served model,
    the same answer the MCP's `models` gives, plus the rendered default."""
    tmpl = _served_template()
    out = levels(tmpl)
    name, spec = detect(tmpl)
    tok = _served_tokenizer()
    if spec is not None and tok is not None:
        p = probe(tok, tmpl, spec)
        out.update(verified=p["verified"], served_default=p["default"])
    return out


def translate(body: dict, client_kwargs: dict | None):
    """A chat request's reasoning level -> (chat_template_kwargs or None,
    report). The request's own chat_template_kwargs win over the
    translation, and the report says so. ValueError: a level off the
    ladder, for the caller to refuse with 400."""
    level = requested(body)
    tmpl = _served_template()
    name, spec = detect(tmpl)
    tok = _served_tokenizer()
    p = probe(tok, tmpl, spec) if (spec is not None and
                                   tok is not None) else None
    if p is not None and not p["verified"]:
        # The text named controls the template does not act on.
        name, spec = None, None
    kwargs, report = resolve(level, name, spec)
    if level is None and p is not None and spec is not None:
        report["applied"] = p["default"] or "the model's own"
        report["note"] = ("what the server renders when the "
                          "request is silent")
    client = dict(client_kwargs or {})
    merged = {**kwargs, **client}
    touched = spec is not None and bool(set(client) & thinking_keys(spec))
    if touched:
        was = report["applied"]
        got = (applied_by_render(tok, spec, merged, p)
               if p is not None else None)
        report["applied"] = got or "set by the request"
        report["native"] = got is not None
        report["note"] = (
            f"the request's own chat_template_kwargs set "
            f"{', '.join(sorted(set(client) & thinking_keys(spec)))}"
            f" and won" + (f" (translation alone gave {was})"
                           if was != report["applied"] else ""))
    return merged or None, report
