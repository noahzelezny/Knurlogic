"""The knurlogic strategy: the launch preset this machine uses by default.

One choice for the machine, like a broker asking for risk tolerance: most
people want one answer to "speed or headroom?", not a preset per model
family. It is the tune a launch takes when none is named; a per-model
KNURLOGIC_PRESET (Settings -> Models -> Launch preset) still beats it.

Kept in ~/.config/knurlogic/strategy.json (XDG_CONFIG_HOME honoured), beside
the allowance, for the same reason: a setting a person chose. Stdlib only.
"""
from __future__ import annotations

from pathlib import Path

from knurlogic.machine import files, paths
from knurlogic.tuning.presets import PRESET_DEFAULT, PRESETS, preset_of


def path() -> Path:
    root = paths.config_dir()
    root.mkdir(parents=True, exist_ok=True)
    return root / "strategy.json"


def get() -> str:
    """The chosen preset, or the default. A damaged file or an unknown name
    is the default, never an error: a launch must not fail on it."""
    v = files.read_json(path(), {}).get("preset")
    try:
        return preset_of(v)
    except ValueError:
        return PRESET_DEFAULT


def set(name) -> str:
    """Remember `name` (the default, or empty, clears the file)."""
    try:
        s = preset_of(name)
    except ValueError:
        raise ValueError(f"strategy {name!r}: one of {list(PRESETS)}") from None
    p = path()
    if s == PRESET_DEFAULT:
        p.unlink(missing_ok=True)
        return s
    files.write_json(p, {"preset": s}, end="\n")
    return s
