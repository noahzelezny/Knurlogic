"""Which released rung is which, and what numerics it SHIPPED with.

A rung is one published quantization level of a model (one Hub repo, e.g.
a 3.4-bit and a 4.6-bit upload of the same model are two rungs).

Stdlib only: the resolver (tuning/) asks this before anything loads, and
asking must not import an engine.

THE RECORD IS THE PUBLISHED BUNDLE. Each released rung ships a `model.py`
whose flag defaults ARE its numerics. `rungs.json` holds, per Hub repo, the
defaults read out of that rung's PUBLISHED model.py (`hf download <repo>
model.py`) -- never out of a local copy, which may have drifted from the
Hub. `tools/vq_gate.py knobs` regenerates it from
downloaded bundles; nothing here is typed by hand.

Two facts per rung that other code acts on:

  * `knobs` -- the environment knurlogic's vendored runtime (vqlab d271035)
    must be given to reproduce the published defaults: only the flags whose
    published default differs from HEAD's. Measured 2026-09-23: Flash-Next
    2.1's whole runtime body differs from HEAD by exactly its three flag
    defaults; the 27B's by nothing at all.
  * `verified` -- knurlogic serves a rung on its OWN runtime only after the
    identity gate (tools/vq_gate.py) passed on that rung against its
    published bundle. Everything starts False; an unverified rung keeps
    loading the model.py it ships, exactly as before.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

RUNGS_JSON = Path(__file__).with_name("rungs.json")

#: The two numerics-active flags (F103/F105, up to +0.97% ppl). The one
#: home of the names is tuning/settings.NUMERICS_FLAGS; repeated here only
#: as the generation classifier's input, and a test holds them equal.
_NUMERICS = ("VQ_GEMMSEG_BF16IO", "VQ_DECODE_BF16IO")

#: How the runtime spells an env flag and its default in source.
FLAG_DEFAULT = re.compile(
    r'os\.environ\.get\(\s*"([A-Z][A-Z0-9_]+)"\s*,\s*\n?\s*("?[^")\s]*"?)\s*\)')


def flag_defaults(src: str) -> dict:
    """{flag: default string} for every `os.environ.get("X", d)` in a
    runtime's source. First occurrence wins (a flag re-read later with the
    same default changes nothing). Flags read with no default are skipped:
    they have no value to reproduce."""
    out: dict = {}
    for m in FLAG_DEFAULT.finditer(src):
        out.setdefault(m.group(1), m.group(2).strip('"'))
    return out


def generation(published_defaults: dict) -> str:
    """v2 (both bf16-I/O on), v1.5 (both off), or arc6 (the flags do not
    exist in the bundle at all). Anything else is mixed and named as such
    rather than rounded to one of the three."""
    vals = {published_defaults.get(f) for f in _NUMERICS}
    if vals == {None}:
        return "arc6-no-flags"
    if vals == {"1"}:
        return "v2"
    if vals == {"0"}:
        return "v1.5"
    return "mixed"


@lru_cache(maxsize=1)
def _table() -> dict:
    try:
        return json.loads(RUNGS_JSON.read_text())
    except (OSError, ValueError):
        return {"rungs": {}}


def reload() -> None:
    """Forget the cached table (tests, and after vq_gate records a pass)."""
    _table.cache_clear()


def table() -> dict:
    return _table()


def repo_of(path) -> Optional[str]:
    """The Hub repo id an artifact directory was downloaded from, if its
    name says: exo spells it `Org--Name`, the HF cache
    `models--Org--Name/snapshots/<rev>`. None when the name does not say --
    a rung is never guessed from anything but its repo."""
    p = Path(str(path))
    for part in reversed(p.parts):
        if part.startswith("models--") and part.count("--") >= 2:
            _, org, name = part.split("--", 2)
            return f"{org}/{name}"
    name = p.name
    if "--" in name:
        org, rest = name.split("--", 1)
        return f"{org}/{rest}"
    return None


def rung(path_or_repo) -> Optional[dict]:
    """This rung's row, by repo id or by artifact path; None if unlisted."""
    rows = _table().get("rungs", {})
    key = str(path_or_repo)
    if key in rows:
        return rows[key]
    repo = repo_of(path_or_repo)
    return rows.get(repo) if repo else None


def knobs(path_or_repo) -> dict:
    """The env knurlogic's runtime needs to reproduce this rung's published
    numerics; {} for an unlisted rung."""
    r = rung(path_or_repo)
    return dict(r.get("knobs", {})) if r else {}


def published_default(path_or_repo, flag: str) -> Optional[str]:
    """The default this rung's PUBLISHED bundle gives `flag`, or None when
    the rung is unlisted or its bundle predates the flag."""
    r = rung(path_or_repo)
    if not r:
        return None
    return r.get("published_defaults", {}).get(flag)


def verified(path_or_repo) -> bool:
    """Has the identity gate passed for this rung? Only then may knurlogic's runtime
    serve it instead of its bundled model.py."""
    r = rung(path_or_repo)
    return bool(r and r.get("verified") is True)
