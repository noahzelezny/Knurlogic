"""The knurlogic-wide settings: chosen once, for every model.

Beside the strategy (the default launch preset): a custom set is that preset
plus explicit values for its rows (tuning/settings.PRESET_ROWS), and compaction
(tuning/settings.COMPACT_KNOBS, read per request by every server) and
identical results across chips (KNURLOGIC_CROSS_CHIP, read at launch;
unset means the preset decides). Kept in ~/.config/knurlogic/settings.json
(XDG_CONFIG_HOME honoured). A saved value beats the same name in a
server's environment; an explicit --set of the cross-chip knob still beats
it at that launch. Stdlib only.

Design: docs/design/settings.md (preferences).
"""
from __future__ import annotations

import json
from pathlib import Path

from knurlogic.tuning.settings import (COMPACT_KNOBS, DECODE_SCALE, MTP_MODE,
                                       PRESET_ROW_NAMES, check_knob)

CROSS_CHIP = "KNURLOGIC_CROSS_CHIP"
#: every name kept here
NAMES = (CROSS_CHIP,) + PRESET_ROW_NAMES + tuple(COMPACT_KNOBS)


def path() -> Path:
    import os
    root = Path(os.environ.get("XDG_CONFIG_HOME",
                               Path.home() / ".config")) / "knurlogic"
    root.mkdir(parents=True, exist_ok=True)
    return root / "settings.json"


def get() -> dict:
    """{name: value} as saved; a damaged file or an unknown name or value is
    left out, never an error: a launch or a request must not fail on it."""
    try:
        raw = json.loads(path().read_text())
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {k: str(v) for k, v in raw.items()
            if k in NAMES and str(v).strip() and check_knob(k, v) is None}


def set(values: dict) -> dict:
    """Merge {name: value} in ('' or None clears that name); refuses the
    whole change (ValueError) on an unknown name or a value it may not take.
    -> what is saved after."""
    if not isinstance(values, dict):
        raise ValueError("send {name: value}")
    bad = [k for k in values if k not in NAMES]
    if bad:
        raise ValueError(f"not a knurlogic-wide setting: {', '.join(bad)}")
    cur = get()
    for k, v in values.items():
        s = str(v if v is not None else "").strip()
        if not s:
            cur.pop(k, None)
            continue
        why = check_knob(k, s)
        if why:
            raise ValueError(why)
        cur[k] = s
    p = path()
    if not cur:
        p.unlink(missing_ok=True)
        return {}
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(cur, sort_keys=True) + "\n")
    tmp.replace(p)
    return cur


def launch_sets(sets: dict) -> dict:
    """A launch's explicit settings with the saved knurlogic-wide ones a
    launch reads added where the launch names none: identical results
    across chips, and the custom preset values (a launch that names its own
    preset takes none of those). Saved beats the preset's value, an
    explicit set beats saved."""
    out = dict(sets or {})
    saved = get()
    v = saved.get(CROSS_CHIP)
    if v and CROSS_CHIP not in out:
        out[CROSS_CHIP] = v
    if "KNURLOGIC_PRESET" in out:
        return out
    for k, v in saved.items():
        if k in PRESET_ROW_NAMES and k not in (DECODE_SCALE, MTP_MODE):
            out.setdefault(k, v)
    mode = saved.get(MTP_MODE)
    if mode:
        out.setdefault("KNURLOGIC_MTP", "off" if mode == "off" else "on")
        if mode != "off":
            out.setdefault("KNURLOGIC_MTP_DYNAMIC",
                           "on" if mode == "dynamic" else "off")
    return out


def decode_scale(sets: dict | None = None):
    """The saved expert chunk scale (0.5), or None: the preset's own. A
    launch that names its own preset takes none."""
    if "KNURLOGIC_PRESET" in (sets or {}):
        return None
    v = get().get(DECODE_SCALE)
    return float(v) if v else None


def compaction_env(env=None) -> dict:
    """The environment compaction reads: `env` (os.environ by default) with
    the saved knurlogic-wide values over it."""
    import os
    base = dict(os.environ if env is None else env)
    base.update({k: v for k, v in get().items() if k in COMPACT_KNOBS})
    return base
