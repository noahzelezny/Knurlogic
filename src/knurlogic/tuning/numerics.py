"""A VQ model's numerics flags: the profiles a person may name, and where
a model's own numerics come from when none is named (its config's
`knobs`, then the defaults in the model.py it ships).
"""

from __future__ import annotations

import re

from knurlogic.machine.artifact import Artifact

# Numerics-active flags: family-local, up to +0.97% ppl.
# v1.5 = both off (bit-exact vs the published arc6 runtime); v2 = both on.
#
# A VQ MODEL'S NUMERICS ARE ITS OWN. It runs the model.py it ships, whose
# defaults are what its weights were fitted under, and that is not uniform
# across releases. So the resolver takes the numerics from the artifact
# (NUMERICS_SOURCES, in order) and applies a profile ONLY when a person
# names one.
NUMERICS_FLAGS = ("VQ_GEMMSEG_BF16IO", "VQ_DECODE_BF16IO")

RUNTIME_PROFILES = {
    "v1.5": {f: "0" for f in NUMERICS_FLAGS},
    "v2": {f: "1" for f in NUMERICS_FLAGS},
}

# Where a model's numerics come from when no profile is asked for, first
# match wins, per flag:
#   declared   config.json `knobs` -- the artifact's own record
#   bundled    the default in the artifact's own model.py
# Nothing found means nothing is emitted: the runtime's own default stands.
NUMERICS_SOURCES = ("declared", "bundled")


#: How a runtime spells an env flag and its default in source.
_FLAG_DEFAULT = re.compile(
    r'os\.environ\.get\(\s*"([A-Z][A-Z0-9_]+)"\s*,\s*\n?\s*("?[^")\s]*"?)\s*\)')


def _flag_defaults(src: str) -> dict:
    """{flag: default string} for every `os.environ.get("X", d)` in a
    runtime's source; the first occurrence wins."""
    out: dict = {}
    for m in _FLAG_DEFAULT.finditer(src):
        out.setdefault(m.group(1), m.group(2).strip('"'))
    return out


def _numerics_source(artifact: Artifact, flag: str, source: str):
    if source == "declared":
        d = artifact.declared_knobs().get(flag)
        if isinstance(d, dict) and d.get("default") is not None:
            return str(d["default"])
        return None
    if source == "bundled":
        return _flag_defaults(artifact.runtime_source()).get(flag)
    raise ValueError(source)


def numerics_for(artifact: Artifact, profile: str | None = None):
    """(env, note): the numerics-active flags for this artifact.

    A profile someone ASKED for wins, and the note names what it overrode.
    Otherwise every flag comes from the artifact itself, first source in
    S.NUMERICS_SOURCES that answers. The bug this replaces: a v1.5 default
    applied to every VQ artifact, forcing Flash-Next 2.1 and Qwen3.6-35B-A3B
    3.8/4.6/5.4 -- published with both flags ON -- to run off (F103/F105:
    numerics-active, up to +0.97% ppl)."""
    own, where = {}, {}
    for flag in NUMERICS_FLAGS:
        for src in NUMERICS_SOURCES:
            v = _numerics_source(artifact, flag, src)
            if v is not None:
                own[flag], where[flag] = v, src
                break
    if profile is not None:
        forced = dict(RUNTIME_PROFILES[profile])
        changed = {f: (own[f], v) for f, v in forced.items()
                   if f in own and own[f] != v}
        note = f"runtime profile {profile} (asked for)"
        if changed:
            note += " -- overrides what this rung shipped: " + ", ".join(
                f"{f} {a}->{b}" for f, (a, b) in changed.items())
        return forced, note
    if not own:
        return {}, ("numerics: nothing declares them -- the runtime's own "
                    "defaults stand")
    srcs = sorted(set(where.values()))
    return own, ("numerics as shipped (" + ", ".join(
        f"{f}={v}" for f, v in own.items()) + f"; from {'/'.join(srcs)})")
