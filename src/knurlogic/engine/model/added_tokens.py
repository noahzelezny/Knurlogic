"""An artifact's added (special) tokens, read off its tokenizer files.
Stdlib only, so the page and the prompt cache's fingerprint read them
without a tokenizer."""
from __future__ import annotations

import json
from pathlib import Path


def listed(tokenizer_json: dict) -> list:
    """[(id, content, special)] of a parsed tokenizer.json's `added_tokens`,
    in file order."""
    return [(a.get("id"), a.get("content"), bool(a.get("special")))
            for a in tokenizer_json.get("added_tokens") or []]


def added_tokens(model_path) -> dict:
    """{id: (content, special)} of the artifact's added tokens, from
    tokenizer_config.json `added_tokens_decoder` and tokenizer.json
    `added_tokens` (tokenizer.json wins where both name an id)."""
    p = Path(str(model_path))
    out: dict = {}
    tc = p / "tokenizer_config.json"
    if tc.is_file():
        for i, d in (json.loads(tc.read_text()).get("added_tokens_decoder")
                     or {}).items():
            out[int(i)] = (d.get("content"), bool(d.get("special", False)))
    tj = p / "tokenizer.json"
    if tj.is_file():
        for i, content, special in listed(json.loads(tj.read_text())):
            out[int(i)] = (content, special)
    return out


def token_id(model_path, content: str) -> int | None:
    """The id of the added token whose text is `content`, or None."""
    return next((i for i, (c, _) in added_tokens(model_path).items()
                 if c == content), None)
