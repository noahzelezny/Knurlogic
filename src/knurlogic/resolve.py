"""Turn an artifact + a memory budget into runtime settings.

The resolver owns the FINAL value of every knob. It does not write env files
and it does not hope one wins: vqlab F33 recorded an experiment that set
RTILE in `exo-env.sh`, which is sourced BEFORE a `ring-env.sh` that assigns
RTILE=32 unconditionally -- so the run benchmarked 32 twice and was reported
as "no difference." Anything that resolves settings must hand back one dict
and be the last word on it.

Headroom is an INPUT, not something this package detects. Machine inventory
belongs to whatever manages the machines.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import settings as S
from .artifact import Artifact

GIB = 1 << 30


@dataclass
class Resolution:
    env: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def as_exports(self) -> str:
        return "\n".join(f"export {k}={v}" for k, v in sorted(self.env.items()))


def decode_chunk_for(headroom_bytes: int, known: bool = True) -> int:
    """Chunk width that keeps the largest dense-expert transient bounded.

    transient = chunk * out * in * 2 bytes, and on a box where the model
    nearly fills RAM this is what caps context length -- it grew 3.35 MB/token
    on the 397B where KV-cache theory predicted 0.059. Smaller is also faster
    (128 -> 32 is 1.37x), so there is no speed/memory tradeoff to negotiate
    below the default; it is capped at the default rather than raised.
    """
    if not known:
        return S.DECODE_CHUNK_DEFAULT      # no budget given: leave the default
    if headroom_bytes <= 0:
        return S.DECODE_CHUNK_MIN          # does not fit: tightest, not default
    per = S.DECODE_CHUNK_BYTES_PER_UNIT * S.DECODE_CHUNK_HEADROOM_DIVISOR
    return max(S.DECODE_CHUNK_MIN,
               min(S.DECODE_CHUNK_DEFAULT, int(headroom_bytes / per)))


def resolve(artifact: Artifact, working_set_bytes: int,
            profile: str = "v1.5") -> Resolution:
    """Resolve every knob for this artifact on a box with this working set.

    `profile` is v1.5 (bit-exact vs the published runtime) or v2 (the two
    numerics-active flags on). An artifact shipping UNCHANGED weights gets
    v1.5: there is no quality gain to offset a numerics regression, however
    small. Only a repo shipping improved weights may take v2.
    """
    r = Resolution()

    if profile not in S.RUNTIME_PROFILES:
        raise ValueError(f"profile must be one of {sorted(S.RUNTIME_PROFILES)}")

    if not artifact.is_vq:
        r.notes.append("not a VQ artifact -- kernel knobs do not apply")
    elif not artifact.model_file:
        r.warnings.append(
            "VQ modules declared but config.json names no `model_file`: the "
            "bundled kernels will not be found and the load will fail")

    headroom = working_set_bytes - artifact.bytes_on_disk

    # --- the two knobs that decide runnable-vs-not --------------------------
    chunk = decode_chunk_for(headroom, known=working_set_bytes > 0)
    r.env["VQ_DECODE_CHUNK"] = str(chunk)
    fits = working_set_bytes <= 0 or headroom > 0
    if chunk < S.DECODE_CHUNK_DEFAULT and fits:
        r.notes.append(
            f"VQ_DECODE_CHUNK lowered to {chunk} ({headroom/GIB:.1f} GiB "
            f"headroom): bounds the dense-expert transient, which is what "
            f"caps context length on a full box")

    tight = working_set_bytes > 0 and headroom < S.TIGHT_HEADROOM_GIB * GIB
    r.env["VQLAB_PREFILL_CHUNK"] = str(
        S.PREFILL_CHUNK_TIGHT if tight else S.PREFILL_CHUNK_DEFAULT)
    if tight:
        r.notes.append(
            "prompt chunk narrowed: token-identical at every width, so this "
            "costs nothing but peak memory")

    r.env["VQLAB_CACHE_LIMIT_GB"] = str(S.CACHE_LIMIT_GB_DEFAULT)

    if working_set_bytes > 0 and headroom <= 0:
        r.warnings.append(
            f"artifact is {artifact.gib:.1f} GiB against a "
            f"{working_set_bytes/GIB:.1f} GiB working set -- it does not fit "
            f"this box. No setting fixes that; it needs a bigger box or more "
            f"than one.")

    # --- performance knobs, each with a finding behind it -------------------
    if artifact.is_vq:
        for k, (v, why) in S.PERFORMANCE_DEFAULTS.items():
            r.env[k] = v
        r.env.update(S.RUNTIME_PROFILES[profile])
        r.notes.append(f"runtime profile {profile}")

    return r
