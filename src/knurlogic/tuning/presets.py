"""The launch presets (the tune axis): what each preset sets, and the
rows a custom set is saved as (tuning/preferences).

How a preset is applied to a launch (preset_env, apply_preset_overrides)
is the resolver's: tuning/resolve.
"""

from __future__ import annotations

from knurlogic.tuning import measured

# --- the tuning axis, and what it is NOT allowed to do ----------------------
# "Less prefill spike at the cost of speed" and "I have headroom, go fast" are
# the two things a person actually wants to say. The axis exists so they can
# say it without learning what any of these names mean.
#
# THE CAPS ARE THE POINT. A tuning axis that just scales every knob would
# happily walk into settings that are measured to be WORSE at both ends:
#
#   * VQ_DECODE_CHUNK: smaller is faster AND smaller in memory (128 -> 32 is
#     1.37x on every rung). There is no tradeoff on this knob, so no preset
#     may raise it. It is capped at the default in both directions.
#
# So a preset moves only the knobs where headroom actually buys something, or
# tightens the ones that bound peak memory. Anything a profile asks for
# beyond a cap is refused and the refusal is printed, never silently clamped.
TUNE_PROFILES: dict = {
    # prefill chunk, cache limit GiB, and whether to bound the transient
    # harder than headroom requires
    "default": {
        "decode_chunk_scale": 1.0,
        "why": "the measured defaults",
    },
    "lean": {
        "KNURLOGIC_PREFILL_CHUNK": measured.PREFILL_CHUNK_LOW_HEADROOM,
        "KNURLOGIC_CACHE_LIMIT_GB": 1.0,
        "launch": {"kv_bits": "8", "mtp": "off"},
        "why": "most context and most agents: 8-bit KV where the family "
               "takes it (bf16 where it does not, and said), 512-token "
               "prompt chunks, MTP off so the head's memory is free",
    },
}

#: The launch presets ARE the tune axis: one named bundle per value, the
#: default "default" (the measured defaults, unchanged). A per-model
#: KNURLOGIC_PRESET (Settings -> Models) picks one for that base model;
#: any explicit knob set beside it beats the preset's value for that knob.
PRESETS = ("default", "lean")
PRESET_DEFAULT = "default"


def preset_of(v, default: str = PRESET_DEFAULT) -> str:
    s = str(v or "").strip().lower()
    if not s:
        return default
    # the names the presets once had: safe is lean now, the rest the default
    s = {"safe": "lean", "fast": "default", "stable": "default",
         "balanced": "default"}.get(s, s)
    if s not in TUNE_PROFILES:
        raise ValueError(f"Preset: {v!r} isn't default or lean")
    return s


def preset_or(v, default: str) -> str:
    """The preset `v` names, or `default` when it names none."""
    try:
        return preset_of(v, default)
    except ValueError:
        return default


def preset_arg(v) -> str:
    """argparse `type` for a --tune value: default or lean, or a name a
    preset once had."""
    import argparse
    try:
        return preset_of(v)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


def preset_launch(tune: str, model_type: str = "") -> tuple:
    """({logical: value}, [notes]) -- the model launch settings a preset
    asks for, narrowed to what this family takes (a KV precision it
    refuses falls back to bf16, and the note says so)."""
    want = dict(TUNE_PROFILES[tune].get("launch") or {})
    notes = []
    if want.get("kv_bits") not in (None, "bf16"):
        bits, why = measured.kv_quant_for(model_type)
        if int(want["kv_bits"]) not in bits:
            notes.append(f"preset {tune}: KV cache stays bf16 -- "
                         f"{want['kv_bits']}-bit is not taken here ({why})")
            want["kv_bits"] = "bf16"
    return want, notes

#: What a preset sets, starting from the default's values: prompt chunk (None:
#: read from the room free at launch), cache GiB, memory transient scale,
#: and the model launch settings.
PRESET_BASE = {"prefill": None, "cache": measured.CACHE_LIMIT_GB_DEFAULT, "scale": 1.0,
               "mtp": "on", "mtp_dynamic": "on", "kv_bits": "bf16"}


def preset_values(name: str) -> dict:
    """PRESET_BASE with the preset's own values over it."""
    t = TUNE_PROFILES[name]
    return {**PRESET_BASE,
            **({"prefill": t["KNURLOGIC_PREFILL_CHUNK"]}
               if "KNURLOGIC_PREFILL_CHUNK" in t else {}),
            **({"cache": t["KNURLOGIC_CACHE_LIMIT_GB"]}
               if "KNURLOGIC_CACHE_LIMIT_GB" in t else {}),
            "scale": t.get("decode_chunk_scale", 1.0),
            **(t.get("launch") or {})}


#: The knurlogic-wide settings a preset sets, as rows: what is saved beside
#: the strategy to make a custom set (tuning/preferences). Each has its
#: saved name, its title, one plain sentence, and the values it takes as
#: (saved value, label); "" is the resolver's own choice.
MTP_MODE = "KNURLOGIC_MTP_MODE"
MTP_MODES = ("dynamic", "every", "off")
PRESET_ROWS = (
    {"name": "KNURLOGIC_PREFILL_CHUNK", "title": "Prompt chunk",
     "help": "Tokens processed per step while reading a prompt; bigger is "
             "faster but needs more memory. Auto uses smaller chunks when "
             "headroom is low.",
     "options": [("", "auto")]
                + [(str(v), str(v)) for v in (512, 1024, 2048, 4096)]},
    {"name": "KNURLOGIC_CACHE_LIMIT_GB", "title": "Cache reserve",
     "help": "Freed memory held back for reuse instead of returned to the "
             "system. No measured speed difference; less leaves more memory "
             "free.",
     "options": [(str(v), f"{v} GiB") for v in (1, 2, 4, 8)]},
    {"name": MTP_MODE, "title": "MTP",
     "help": "For models with an MTP head: drafts tokens ahead to speed "
             "up decoding, and loading the head takes more memory. Dynamic "
             "turns drafting off when it doesn't help.",
     "options": [("dynamic", "dynamic"), ("every", "every step"),
                 ("off", "off")]},
    {"name": "KNURLOGIC_KV_BITS", "title": "KV cache",
     "help": "8-bit holds about twice the context in the same memory; ~5% "
             "slower decode and slightly less precise.",
     "options": [("bf16", "bf16"), ("8", "8-bit")]},
)
PRESET_ROW_NAMES = tuple(r["name"] for r in PRESET_ROWS)

def preset_row_values(name: str) -> dict:
    """What a preset sets on each PRESET_ROWS row, as saved strings."""
    v = preset_values(name)
    return {
        "KNURLOGIC_PREFILL_CHUNK": str(v["prefill"] or ""),
        "KNURLOGIC_CACHE_LIMIT_GB": f"{v['cache']:g}",
        MTP_MODE: ("off" if v["mtp"] == "off" else
                   "dynamic" if v["mtp_dynamic"] == "on" else "every"),
        "KNURLOGIC_KV_BITS": str(v["kv_bits"]),
    }
