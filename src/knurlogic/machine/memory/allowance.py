"""The knurlogic allowance: the most memory knurlogic may use on this box.

`serve --working-set-gib` said this once per launch; the allowance says it
for the machine and is remembered, because the reason for it is the
machine's -- a Mac that also runs exo, or a desk machine whose owner wants
room for everything else -- not any one model's. It only ever LOWERS what
knurlogic would otherwise take: the load budget (`wired.load_budget`, and so
the fit check in `interfaces/load_checks.prepare`) and the model server's
working set, which is what the scheduler's memory guard counts against.

Kept in ~/.config/knurlogic/allowance.json (XDG_CONFIG_HOME honoured): it
is a setting a person chose, so it lives where settings live -- not in the
cache directory beside servers.json, which clearing caches would wipe.
Stdlib only: the page server and the CLI read it without the engine.
"""
from __future__ import annotations

import json
from pathlib import Path

GIB = 1 << 30


def path() -> Path:
    import os
    root = Path(os.environ.get("XDG_CONFIG_HOME",
                               Path.home() / ".config")) / "knurlogic"
    root.mkdir(parents=True, exist_ok=True)
    return root / "allowance.json"


def get() -> int:
    """Bytes, or 0 for no allowance (take what the machine gives). A file
    that cannot be read is no allowance, not an error: a setting that
    stopped a model loading because its file was damaged would be worse
    than the setting not being there."""
    try:
        v = json.loads(path().read_text()).get("allowance_bytes")
    except (OSError, ValueError, AttributeError):
        return 0
    return int(v) if isinstance(v, (int, float)) and v > 0 else 0


def set(nbytes) -> int:
    """Remember `nbytes` (0 or None clears it). -> what is now stored."""
    n = int(nbytes or 0)
    if n < 0:
        raise ValueError("an allowance cannot be negative")
    p = path()
    if n == 0:
        p.unlink(missing_ok=True)
        return 0
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"allowance_bytes": n}) + "\n")
    tmp.replace(p)      # a reader never sees half a file
    return n


def cap(nbytes: int) -> int:
    """`nbytes` lowered to the allowance; 0 (unknown) becomes the allowance,
    since that is then the only limit known."""
    a = get()
    if not a:
        return int(nbytes or 0)
    return min(int(nbytes), a) if nbytes else a
