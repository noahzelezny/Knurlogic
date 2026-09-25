"""The system prompt gets its own segment, whatever the template.

mlx-lm finds where the system prompt ends by rendering the system
messages plus an EMPTY user turn and taking the first token where that
differs from the real prompt. When the empty turn renders as a pure prefix
of the real one -- GLM-5.3: `<|user|>` and nothing after it -- nothing
differs, the system segment is never made, and no checkpoint is stored at
its end: a long shared system prompt (Scout's ingest schema) was
re-prefilled on every request (GLM 2.7, 482 tokens, 0 reused;
2026-09-25). Qwen's templates escape it by accident (the empty turn ends in
`<|im_end|>`, which differs).

The fix is the rule mlx-lm meant: if the system render is a prefix of the
prompt, the system segment ends where it ends.
"""

from __future__ import annotations


def split_system(tokenizer, messages, prompt, segments, types, kwargs):
    """(segments, types) with a leading system segment added when mlx-lm
    found none although the prompt starts with the system render."""
    if not segments or "system" in types or not messages:
        return segments, types
    n_sys = 0
    for m in messages:
        if m.get("role") != "system":
            break
        n_sys += 1
    if n_sys == 0 or messages[-1].get("role") != "user":
        return segments, types
    try:
        sys_tokens = list(tokenizer.apply_chat_template(
            list(messages[:n_sys]) + [{"role": "user", "content": ""}],
            add_generation_prompt=False, tokenize=True, **kwargs))
    except Exception:
        return segments, types
    k = len(sys_tokens)
    first = list(segments[0])
    if not (0 < k < len(first)) or list(prompt[:k]) != sys_tokens:
        return segments, types
    return [first[:k], first[k:], *segments[1:]], ["system", *types]


def install(srv) -> None:
    RG = srv.ResponseGenerator
    real = getattr(RG, "_tokenize", None)
    if real is None or getattr(real, "_knurlogic_segments", False):
        return

    def _tokenize(self, tokenizer, request, args):
        prompt, segments, types, state = real(self, tokenizer, request, args)
        if getattr(request, "request_type", "chat") != "chat":
            return prompt, segments, types, state
        msgs = getattr(request, "messages", None) or []
        # the kwargs mlx-lm rendered the prompt with: its CLI defaults, then
        # the request's own
        prov = getattr(self, "model_provider", None)
        kw = dict(getattr(getattr(prov, "cli_args", None),
                          "chat_template_args", None) or {})
        kw.update(getattr(args, "chat_template_kwargs", None) or {})
        kw.pop("_knurlogic_close_think", None)   # generation prompt only
        if getattr(request, "tools", None):
            kw["tools"] = request.tools
        segments, types = split_system(tokenizer, msgs, prompt,
                                       segments, types, kw)
        return prompt, segments, types, state
    _tokenize._knurlogic_segments = True
    RG._tokenize = _tokenize
