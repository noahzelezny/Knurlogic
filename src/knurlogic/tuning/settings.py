"""Measured runtime constants, with the provenance that paid for them.

Every number here came off a run. The point of Knurlogic is that a downloader
should never have to know them: the resolver turns them into defaults. Each
constant carries the measurement that established it, so a future change
has to argue with a measurement rather than a preference. The VQ kernel
numbers were measured in the VQ runtime's upstream project.
"""

from __future__ import annotations

from typing import Any

# --- the two knobs that decide runnable-vs-not ------------------------------

# Experts decoded to dense fp16 per prefill chunk. THIS IS THE MEMORY KNOB,
# not the KV cache: measured on a 128 GB M4 Max running the
# 110.8 GiB 397B, prefill grew 3.35 MB/token where KV-cache theory predicts
# 0.059 -- a 57x gap owned entirely by these buffers.
#   transient = chunk * out * in * 2 bytes
# It is also FASTER small: 128 -> 32 is 1.37x on every rung measured, and the
# knee does not move with codebook size (K128/K256/K2048 identical).
DECODE_CHUNK_DEFAULT = 32
DECODE_CHUNK_MIN = 4
# Bytes of transient per unit of chunk, per expert tensor:
#
#     transient = chunk * out * in * 2      (gate_up, the larger of the two)
#
# out and in are the MODEL'S shape, not the box's: gate_up is [2 * M, H] for
# hidden size H and moe_intermediate_size M. This constant is that formula
# frozen for ONE model (H=4096, M=1024), which is why it was a constant at
# all -- the auto-sizer it came from only ever ran on that rung.
#
# Keeping it frozen makes the resolver blind to the thing that actually
# moves the spike. Measured across boxes and families: the same prefill
# spike appeared on M4 and on M3, and DeepSeek V4 was far more dramatic than
# Qwen3.5 -- so the FAMILY is the bigger factor and the BOX is close to
# irrelevant once headroom is equal. That is exactly what this formula
# predicts, since H and M differ per family and do not depend on the machine
# at all. `expert_transient_bytes_per_unit` reads them off the artifact; this
# constant is now only the fallback for a config that declares neither.
DECODE_CHUNK_BYTES_PER_UNIT = 2048 * 4096 * 2
#: The shape the fallback constant encodes, so a note can say whose it is.
DECODE_CHUNK_ASSUMED_SHAPE = (2048, 4096)

# May the artifact's own shape make the chunk LARGER than the frozen constant
# would have? Not yet, and the asymmetry is deliberate.
#
# Sizing from the model tightens the knob for a family with bigger experts
# (DeepSeek V4) and loosens it for one with smaller experts (Qwen3.5). The
# tightening direction is protective: it is the case that was under-served by
# a constant, and it is the one the measurements describe as "more dramatic".
# The loosening direction is the one where being wrong means an OOM -- the
# exact failure this package exists to prevent -- and it has not been
# measured yet. So the formula may only reduce the chunk until a run says
# otherwise. Flipping this to True is a one-line change and should be made
# by a measurement, not by a preference.
DECODE_CHUNK_SHAPE_MAY_LOOSEN = False
# keep the largest transient under this fraction of remaining headroom
DECODE_CHUNK_HEADROOM_DIVISOR = 8

# Prompt chunk width. Token-identical at every value (gated upstream by
# the VQ runtime's prefill tests) -- purely a memory knob. mlx-lm's
# server does not expose it, which is why it must be resolved here.
#
# Chosen from the room ACTUALLY free at launch (the load budget: the
# smaller of the working set and what macOS would hand over now), never
# from installed RAM. "No reason to leave headroom unused, but read the
# room": take the widest width on the ladder, capped at the family's
# measured best, whose predicted step transient fits in the memory the
# launch RESERVES for transients (the step margin, or 1.25x the first
# request's transient when larger -- the reserve the fit already holds
# free); else step down, floor 512. (It used to be 10% of the room left
# after weights, KV and cache: 0.00 GiB on a fitting box with low headroom, which
# pinned 512 and cost 2.4x prefill -- GLM-5.3 Flash 2.7bpw on 128 GB:
# 93-98 tok/s at 512 against 220-236 at 2048, step transient 1.39 GiB
# measured at 512 over 3018 tokens, 480*hidden*512 predicts ~1.4 GiB.)
# A family with no measurement stays 512.
#
# M4 Max 128 GB sweep, prefill tok/s at 4k/16k-token prompts (median of 3,
# one server per arm), then the step transient:
#   Qwen3.8 Flash 4.4bpw:  512 565/484 0.79 GiB | 1024 528/490 0.99
#                          2048 552/524 2.11    | 4096 551/499 4.05
#   Qwen3.5-397B VQ 2.2:   512 194/155 0.33 GiB | 1024 224/186 1.15-1.54
#                          2048 244/205 2.95    | 4096 249/206 5.6-7.5
# M4 Max 128 GB, Qwen3.6-35B-A3B VQ 3.4 (13.8 GiB), 28,727-token
# prompt, interleaved, n=3, prefill tok/s:
#   512 642.7/642.6/649.5 | 2048 1182.9/1164.5/1141.9
#   4096 1173.2/1166.5/1133.8
# So width is worth ~1.8x on the 35B-A3B up to 2048 and nothing past it;
# ~30% on the 397B VQ for a 9-23x larger transient -- 4096 aborted Metal
# with one agent at 25k tokens on a box with ~14 GiB left. A chunk is
# now allowed only when its predicted transient fits the memory reserved
# for transients (tuning/resolve.prefill_chunk_by_room).
PREFILL_CHUNK_DEFAULT = 512
PREFILL_CHUNK_LOW_HEADROOM = 512
#: The widths the room rule may choose from (and the knob's native range).
PREFILL_CHUNK_LADDER = (512, 1024, 2048, 4096)
#: Predicted step transient per prompt token per unit of hidden size:
#: transient(width) = width * hidden_size * this. Calibrated to the WORST
#: measured case, the 397B VQ at 4096 (7.5 GiB, hidden 4096): 7.5 GiB /
#: (4096 * 4096) = 480 bytes. It over-predicts the milder rungs (0.94 vs
#: 0.33 GiB at 512 on the 397B), which is the safe direction.
PREFILL_TRANSIENT_BYTES_PER_TOKEN_HIDDEN = 480
#: KV held back before the rule sizes a chunk: one long conversation.
PREFILL_KV_ALLOWANCE_TOKENS = 32768

# What a rank must keep free AFTER its weights for a model to be "placed"
# there (resolve.fit_reserve): the first request's transient at the
# smallest prompt chunk, plus the KV of a minimum context. A share that
# fills the working set to its step margin fits on paper and then dies on
# the first prompt.
#
# MEASURED on an M3 Ultra (96 GB), mlx 0.31.2, by loading each model,
# evaluating all its parameters, then serving one prompt (peak memory
# above the weights, mx.get_peak_memory):
#   Qwen3.6-35B-A3B VQ 3.4bpw (12.96 GiB, hidden 2048):
#     evaluating every weight at once      +0.00 GiB (peak == active: no
#                                          mmap copy, no second copy)
#     binding the vision tower             +0.83 GiB (rank 0 only, counted
#                                          as leader_bytes)
#     64-token first request               +0.22 GiB
#     4096-token prompt, chunk  512        +0.93 GiB
#     4096-token prompt, chunk 4096        +3.45 GiB
# and, from the sweep above, Qwen3.8 Flash 4.4bpw: 512 0.79 GiB. So the
# transient at the smallest chunk is about 1 GiB whatever the model, and
# PREFILL_TRANSIENT_BYTES_PER_TOKEN_HIDDEN under-reads it there (0.47 GiB
# predicted for the 35B against 0.93 measured): FIT_TRANSIENT_FLOOR is the
# measured flat part.
FIT_TRANSIENT_FLOOR = 1 << 30
#: the chunk the fit is asked at: the smallest the room rule can fall to
FIT_PREFILL_CHUNK = PREFILL_CHUNK_DEFAULT
#: the context every rank must be able to hold beyond its weights
FIT_MIN_CONTEXT_TOKENS = 8192

# Per-ARCHITECTURE prompt chunk: a measurement with its run, kept in each
# family's manifest (engine/families/<family>/__init__.py, `prefill_chunk`)
# beside the rest of what that family is.
# Not measured means PREFILL_CHUNK_DEFAULT. A measured width is a CAP, not
# a default: the room rule (above) decides how much of it this launch takes.
def _measured_widths() -> dict:
    from knurlogic.engine import families
    return families.build_maps()["prefill_chunk"]


#: architecture -> (width, evidence), from each family's manifest.
PREFILL_CHUNK_MEASURED = _measured_widths()


def prefill_chunk_for(model_type: str) -> tuple:
    """(width, source) for a family: the manifest's measured value with its
    evidence, or the default and the fact that nothing was measured."""
    # Through the architecture map, because configs spell one family several
    # ways -- a qwen3_5 27B reports `qwen3_5_text` -- and a miss here quietly
    # hands a measured family the unmeasured default.
    from knurlogic.engine.arch import ARCH_FOR_MODEL_TYPE
    family = ARCH_FOR_MODEL_TYPE.get(model_type, model_type)
    if family in PREFILL_CHUNK_MEASURED:
        width, why = PREFILL_CHUNK_MEASURED[family]
        return width, f"measured for {family}: {why}"
    return PREFILL_CHUNK_DEFAULT, "default: no measured width for this family"


# How many prompts are prefilled in ONE forward. mlx-lm's server defaulted
# to 8 (--prompt-concurrency), each with its own transient. knurlogic's own
# engine admits ONE row per step (engine/mtp/batch_generator._next), so
# prompts are always prefilled one at a time and there is nothing to set:
# KNURLOGIC_PROMPT_CONCURRENCY went with mlx-lm's server (a64bba7) and is
# no longer emitted or shown. It is still ACCEPTED (KNOB_ALIASES) so a
# launch setting saved before cannot fail a launch; it does nothing.

# Freed MLX buffers pile up invisibly -- they do not appear in "active
# memory." The biggest single memory win measured, zero measured speed
# cost at 26k-token prefill.
CACHE_LIMIT_GB_DEFAULT = 4.0

# Below this much free headroom after the weights, treat the box as low on headroom and
# resolve the memory knobs down rather than leaving performance defaults:
# the larger of a floor and a fraction of the working set. A fixed 12 GiB
# lets the 397B on a 128 GB M4 Max (~14 GiB above its weights) take the
# 4096-token prefill chunk; its first step alone measured 8.1 GiB of
# transient, and one agent at a 25k-token context aborted Metal. A share of the
# working set scales with the machine.
LOW_HEADROOM_GIB = 12.0
LOW_HEADROOM_SHARE = 0.20


def low_headroom_bytes(working_set_bytes: int) -> int:
    return int(max(LOW_HEADROOM_GIB * (1 << 30),
                   LOW_HEADROOM_SHARE * working_set_bytes))

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
        "KNURLOGIC_PREFILL_CHUNK": PREFILL_CHUNK_LOW_HEADROOM,
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
        bits, why = kv_quant_for(model_type)
        if int(want["kv_bits"]) not in bits:
            notes.append(f"preset {tune}: KV cache stays bf16 -- "
                         f"{want['kv_bits']}-bit is not taken here ({why})")
            want["kv_bits"] = "bf16"
    return want, notes

#: What a preset sets, starting from the default's values: prompt chunk (None:
#: read from the room free at launch), cache GiB, memory transient scale,
#: and the model launch settings.
PRESET_BASE = {"prefill": None, "cache": CACHE_LIMIT_GB_DEFAULT, "scale": 1.0,
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
#: the strategy to make a custom set (machine/preferences). Each has its
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

#: the page's name for each launch setting (Settings -> Models / Knurlogic;
#: views/settings/knobs.js KNOB_TITLE): what every refusal calls it
KNOB_TITLES: dict = {
    **{r["name"]: r["title"] for r in PRESET_ROWS},
    "KNURLOGIC_PRESET": "Preset",
    "KNURLOGIC_CONTEXT_LENGTH": "Context length",
    "KNURLOGIC_CROSS_CHIP": "Per-chip rounding",
    "KNURLOGIC_MTP": "MTP",
    "KNURLOGIC_MTP_DYNAMIC": "MTP dynamic",
    "KNURLOGIC_VISION": "Vision",
    "KNURLOGIC_KV_KERNEL": "KV kernel",
    "KNURLOGIC_LONG_CONTEXT": "Long context",
    "KNURLOGIC_SPARSE_PREFILL_FROM": "Sparse prefill from",
}
KNOB_TITLES["VQLAB_PREFILL_CHUNK"] = KNOB_TITLES["KNURLOGIC_PREFILL_CHUNK"]
KNOB_TITLES["VQ_CACHE_LIMIT_GB"] = KNOB_TITLES["KNURLOGIC_CACHE_LIMIT_GB"]
KNOB_TITLES["VQLAB_CACHE_LIMIT_GB"] = KNOB_TITLES["KNURLOGIC_CACHE_LIMIT_GB"]


def knob_title(name: str) -> str:
    return KNOB_TITLES.get(name, name)


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


#: Hard ceiling on the reclaimable cache, whatever the profile asks. Freed
#: buffers are reclaimable but they are still resident, and a cache larger
#: than this has never been measured to buy anything.
CACHE_LIMIT_GB_MAX = 16.0


# --- what each knob IS, in one line, for anything that shows it to a person -
# A settings panel that lists names and values is a config file with a
# stylesheet. The reason to show a knob at all is the sentence next to it:
# what it does, and what run says so. Anything added to the resolver should
# be added here too, or it will appear in the UI as a bare string.
KNOB_DOC = {
    # (what it is, its TRADE-OFF: what a change buys and what it costs).
    # Numbers are measurements with their run; where none exists the cost
    # is said in words, never invented.
    "VQ_DECODE_CHUNK": (
        "experts decoded to dense fp16 per prefill chunk",
        "THE memory knob: prefill grew 3.35 MB/token where KV-cache theory "
        "predicted 0.059. No trade at the top: smaller is also faster (128 "
        "-> 32 is 1.37x), so it is capped at 32. Lower still shrinks the "
        "prefill memory spike further; the speed below 32 is not measured."),
    "KNURLOGIC_PREFILL_CHUNK": (
        "how many prompt tokens are processed at once",
        "wider can prefill faster -- ~1.8x from 512 to 2048 on the "
        "35B-A3B VQ (645 -> 1163 tok/s at 28.7k tokens, M4, 4096 no "
        "better), up to ~30% on the 397B VQ, nothing on Qwen3.8 Flash -- "
        "but each step's memory spike grows 9-23x from 512 to 4096, which "
        "is what runs a long prompt or parallel agents out of memory. "
        "Unset, it is read from the room free at launch: the widest "
        "width up to the family's measured best whose predicted spike "
        "fits in 10% of the room left after weights, KV and cache, else "
        "512 (lean: always 512). Output is identical at "
        "every width."),
    "KNURLOGIC_CONTEXT_LENGTH": (
        "the longest conversation (prompt + answer, in tokens) a request may "
        "use",
        "longer lets a request run longer, and every token of it holds KV "
        "memory while it runs, so fewer long conversations or agents fit at "
        "once. Shorter caps that memory; a longer prompt is refused (400) "
        "and max_tokens trimmed to fit. Nothing is reserved up front. The "
        "default is the model's own window."),
    "KNURLOGIC_MTP": (
        "multi-token prediction: draft with the head packed beside the "
        "weights",
        "on: steps that draft can emit several tokens, usually faster "
        "replies, with the same output distribution -- at the cost of "
        "keeping the head in memory. Off frees that memory and every step "
        "is a plain, slower one."),
    "KNURLOGIC_VISION": (
        "load the vision tower, its image store and the image KV allowance",
        "on (the default): a model with a vision tower takes images. Off: "
        "the tower is not loaded and no image store or image KV is held, "
        "so that memory goes to headroom (a wider prompt chunk, or a fit "
        "that would not otherwise); a request with an image gets a 400 "
        "saying vision is off for this launch."),
    "KNURLOGIC_MTP_DYNAMIC": (
        "switch between drafting and plain steps by their measured cost",
        "on: each regime is timed per batch width and the cheaper one "
        "taken, so it is usually faster -- but timing varies as it "
        "switches. Off: every step drafts while a head is bound, steady "
        "timing, paying for drafts even where a plain step would be "
        "cheaper. Only with MTP on."),
    "KNURLOGIC_KV_BITS": (
        "precision of the attention KV cache: bf16, or 8, 6 or 4 bits",
        "fewer bits hold more context and more agents: 8-bit takes about "
        "53% of bf16's memory (6: 41%, 4: 28%). The cost is speed and a "
        "little precision: at 8 bits decode reads the cache through a "
        "fused kernel (KNURLOGIC_KV_KERNEL), measured ~5% slower than bf16 "
        "at 6k tokens of context and ~6% at 16k (M4, kernel on; ~7% / ~18% "
        "with it off), no measured prefill cost; 6/4 dequantize K/V every "
        "step and are not measured. 8-bit moves the tiny test models' "
        "logits by ~0.3% of their range; a planted-needle answer stayed "
        "exact on Qwen3.6-35B at 26k tokens. Attention K/V only (GLM: its "
        "MLA latent; GLM takes 8 only)."),
    "KNURLOGIC_KV_KERNEL": (
        "8-bit KV decode through the fused kernel (on) or dequantize + "
        "attention (off)",
        "only with an 8-bit KV cache: decode steps read the 8-bit K/V "
        "directly instead of dequantizing the whole cache each step "
        "(engine/kvattn.py). No trade measured: M4 Qwen3.6-35B decode is "
        "+4% faster at 6k, +15% at 16k context than off, and the needle "
        "answer is exact either way. Off is for A/B -- check /status.json "
        "kv_kernel hits vs misses to see "
        "which path is actually live (GLM's MLA latent and gemma4's "
        "KV-shared layers always take dequantize + attention). A row with "
        "every key masked returns 0 here where mlx sdpa returns NaN. "
        "Prefill is dequantize + attention either way."),
    "KNURLOGIC_SPARSE_PREFILL_FROM": (
        "tokens of context from which GLM's prefill attends in the latent, "
        "over each query's own sparse selection (0: always; off: never)",
        "GLM-5.3 (glm5_next) only. Past the indexer's top-k, a prefill "
        "chunk otherwise expands every cached token's latent into per-head "
        "K/V and masks it down to the selection: memory and work grow with "
        "the whole context, and at 334k tokens the prefill chunk shrank to "
        "256 and an M3 Ultra (96 GB) was still killed at 98%. In the "
        "latent the chunk reads only what the indexer chose: the same "
        "keys and softmax (equal logits to rounding, "
        "tests/engine/test_glm5_sparse_prefill.py), memory independent "
        "of the context. Below the crossover the expanded path keeps the "
        "fused attention kernel; where it pays is measured by "
        "tools/bench_glm5_sparse_prefill.py, not assumed."),
    "KNURLOGIC_LONG_CONTEXT": (
        "reach past the model's trained window: off, or yarn (Qwen's "
        "documented YaRN rope scaling, factor 4 over 262,144 -> ~1M tokens)",
        "yarn raises this model's context cap to 1,048,576 tokens (Qwen "
        "documents 1,010,000 for Qwen3.5/3.6, 1,000,000 for Qwen3.8), set "
        "on the loaded config at load time, the artifact untouched. The "
        "cost, per Qwen: static YaRN applies the same scaling to every "
        "prompt, so short prompts may get slightly worse -- leave it off "
        "unless you need past 262k. And every token of context holds KV "
        "memory: at 1,048,576 tokens bf16 KV is 20 GiB on Qwen3.6-35B-A3B, "
        "24 GiB on Qwen3.8 Flash, 30 GiB on the 397B, 64 GiB on "
        "Qwen3.8-27B (all conversations together); 8-bit KV "
        "(KNURLOGIC_KV_BITS=8) takes 53% of that. The load is "
        "refused where the box cannot hold the chosen context's KV. Only "
        "for families whose model card documents it (qwen3_5, "
        "qwen3_5_moe, qwen4_exp)."),
    "KNURLOGIC_CROSS_CHIP": (
        "identical results across chips: a split over an M3 and an M4 "
        "gives the same tokens as a split over two of one",
        "on costs speed: mlx picks a different quantized-matmul kernel "
        "for 9-31 rows on each GPU generation, so on pads those calls to 32 "
        "rows, +2-6% time on them. Off is faster, and a mixed-chip split "
        "may then produce different (equally valid) tokens than a same-chip "
        "one -- it cannot desync, rank 0 samples every token. auto: on only "
        "when a cluster job's machines have different GPU architectures."),
    "KNURLOGIC_PRESET": (
        "launch preset for this model: default or lean -- the knurlogic "
        "strategy unless set here",
        "one named bundle of the settings below. default is the measured "
        "settings; lean buys context and agents with some speed and "
        "precision (8-bit KV, 512-token prompt chunks, MTP off). Any "
        "setting changed beside it beats the preset's value."),
    "VQ_CACHE_LIMIT_GB": (
        "how much freed-buffer cache the runtime may hold",
        "larger keeps more freed buffers for reuse, but they stay resident: "
        "memory a long context or another agent cannot use. Smaller frees "
        "it, with no measured speed cost at 26k-token prefill -- the "
        "biggest single win in the memory playbook."),
    "KNURLOGIC_CACHE_LIMIT_GB": (
        "how much freed-buffer cache the runtime may hold",
        "larger keeps more freed buffers for reuse, but they stay resident: "
        "memory a long context or another agent cannot use. Smaller frees "
        "it, with no measured speed cost at 26k-token prefill -- the "
        "biggest single win in the memory playbook."),
    "VQLAB_CACHE_LIMIT_GB": (
        "how much freed-buffer cache the runtime may hold",
        "larger keeps more freed buffers for reuse, but they stay resident: "
        "memory a long context or another agent cannot use. Smaller frees "
        "it, with no measured speed cost at 26k-token prefill -- the "
        "biggest single win in the memory playbook."),
    "VQ_GEMMSEG_BF16IO": ("bf16 IO in the segmented GEMM",
                          "numerics-active: changing it changes "
                          "the output, up to +0.97% ppl, on weights fitted "
                          "the other way. Each model keeps what it shipped."),
    "VQ_DECODE_BF16IO": ("bf16 IO on the decode path",
                         "numerics-active: changing it changes "
                         "the output (up to +0.97% ppl). Each model keeps "
                         "what it shipped."),
}


#: The (i) beside a knob in Settings -> Models: what it does for the person
#: choosing, short and plain. KNOB_DOC keeps the measurements behind it.
KNOB_HELP = {
    "VQ_DECODE_CHUNK": "How much of the model is unpacked at once; lower "
                       "uses less memory and is also faster.",
    "KNURLOGIC_PREFILL_CHUNK": "How many prompt tokens are read at once; "
                               "wider is faster on long prompts but needs "
                               "more memory.",
    "KNURLOGIC_CONTEXT_LENGTH": "How many tokens the model can hold in "
                                "memory; more uses more memory.",
    "KNURLOGIC_MTP": "Guesses several tokens per step: usually faster, for "
                     "a little more memory.",
    "KNURLOGIC_VISION": "Reads images. Off frees the vision tower's "
                        "memory for a model you only send text.",
    "KNURLOGIC_MTP_DYNAMIC": "Guesses ahead only where that is faster; off "
                             "gives steadier timing.",
    "KNURLOGIC_KV_BITS": "8-bit holds about twice the conversation in the "
                         "same memory, slightly slower.",
    "KNURLOGIC_KV_KERNEL": "A faster way to read an 8-bit cache; leave on.",
    "KNURLOGIC_SPARSE_PREFILL_FROM": "GLM: long prompts read only the "
                                     "tokens attention picks, in far less "
                                     "memory. 0 = always.",
    "KNURLOGIC_CACHE_LIMIT_GB": "Freed memory held back for reuse instead of "
                                "returned to the system. No measured speed "
                                "difference; less leaves more memory free.",
    "VQ_GEMMSEG_BF16IO": "Changes the output slightly; keep what the model "
                         "shipped with.",
    "VQ_DECODE_BF16IO": "Changes the output slightly; keep what the model "
                        "shipped with.",
    "KNURLOGIC_PRESET": "Default follows the Knurlogic presets.",
    "KNURLOGIC_CROSS_CHIP": "On: faster, but a split across different "
                            "chips can give different (still valid) tokens. "
                            "Turn off if you need identical output; "
                            "slightly slower.",
    "KNURLOGIC_COMPACT_AUTO": "Compacts the conversation automatically "
                              "when it reaches the Auto compact point, even if "
                              "the client didn't ask.",
    "KNURLOGIC_COMPACT_TRIGGER": "How full the context window gets before "
                                 "automatic compaction starts.",
    "KNURLOGIC_COMPACT_KEEP_TURNS": "How many of the most recent messages "
                                    "stay word-for-word when the rest is "
                                    "summarized.",
    "KNURLOGIC_COMPACT_TOOL_RESULTS": "Distill keeps a summary of tool "
                                      "results, making compaction slower. "
                                      "Clear is faster, but the model is "
                                      "more likely to repeat calls.",
}


# --- the names are the ARTIFACT'S, not ours --------------------------------
# A knob has a LOGICAL name here and a list of env names, preferred first.
# The resolver emits whichever one the target artifact's bundled runtime
# actually reads. The old names are still accepted from a saved setting or
# `--set`. The cache limit is one value, knurlogic's (mlx's free-buffer
# cache ceiling, which the engine sets): a VQ bundle's runtime sets the same
# ceiling from VQ_CACHE_LIMIT_GB (published ones: VQLAB_CACHE_LIMIT_GB), so
# the resolver passes knurlogic's value through under that name as well
# (resolve.emit_cache_limit).
KNOB_ALIASES = {
    "cache_limit_gb": ("KNURLOGIC_CACHE_LIMIT_GB", "VQ_CACHE_LIMIT_GB",
                       "VQLAB_CACHE_LIMIT_GB"),
    "prefill_chunk": ("KNURLOGIC_PREFILL_CHUNK", "VQLAB_PREFILL_CHUNK"),
    "decode_chunk": ("VQ_DECODE_CHUNK",),
    "prompt_concurrency": ("KNURLOGIC_PROMPT_CONCURRENCY",),
    "context_length": ("KNURLOGIC_CONTEXT_LENGTH",),
    "mtp": ("KNURLOGIC_MTP",),
    "mtp_dynamic": ("KNURLOGIC_MTP_DYNAMIC",),
    "vision": ("KNURLOGIC_VISION",),
    "kv_bits": ("KNURLOGIC_KV_BITS",),
    # the 8-bit KV decode kernel (engine/kvattn): on unless "off"; A/B knob
    "kv_kernel": ("KNURLOGIC_KV_KERNEL",),
    # GLM's prefill in the latent from this context (glm5_next edit 10)
    "sparse_prefill": ("KNURLOGIC_SPARSE_PREFILL_FROM",),
    "cross_chip": ("KNURLOGIC_CROSS_CHIP",),
    "long_context": ("KNURLOGIC_LONG_CONTEXT",),
    "preset": ("KNURLOGIC_PRESET",),
}

#: launch settings of the model itself, read when it loads: the same on
#: every rank of a split (cluster/launch passes them ring-wide)
MODEL_KNOBS = ("KNURLOGIC_MTP", "KNURLOGIC_MTP_DYNAMIC", "KNURLOGIC_VISION",
               "KNURLOGIC_KV_BITS",
               "KNURLOGIC_KV_KERNEL", "KNURLOGIC_CROSS_CHIP",
               "KNURLOGIC_LONG_CONTEXT", "KNURLOGIC_SPARSE_PREFILL_FROM",
               "KNURLOGIC_PRESET")


# --- which knobs the ENGINE consumes ----------------------------------------
# Most knobs are read by an artifact's bundled runtime. These are not: the
# scheduler takes the prompt chunk directly (interfaces/http.scheduler_
# options), and the buffer cache is a process-global mlx setting. Emitting
# them as environment variables and stopping there is how they were, for a
# while, settings that did nothing -- the resolver explained a prompt chunk
# the server never saw.
ENGINE_KNOB_NAMES = tuple(n for k in ("prefill_chunk", "cache_limit_gb",
                                      "context_length", "mtp",
                                      "mtp_dynamic", "vision", "kv_bits",
                                      "kv_kernel", "sparse_prefill",
                                      "cross_chip", "long_context",
                                      "preset")
                          for n in KNOB_ALIASES[k])


def vision_of(sets: dict | None) -> bool:
    """KNURLOGIC_VISION from launch settings: True (the default) unless
    set off; a bad value is the default here (serve refuses it)."""
    try:
        return on_off((sets or {}).get("KNURLOGIC_VISION"), True)
    except ValueError:
        return True


def mtp_of(sets: dict | None) -> bool:
    """KNURLOGIC_MTP from launch settings: True (the default) unless set
    off; a bad value is the default here (serve refuses it)."""
    try:
        return on_off((sets or {}).get("KNURLOGIC_MTP"), True)
    except ValueError:
        return True


def on_off(v, default: bool = True) -> bool:
    """'on'/'off' (and 1/0, true/false); None or '' is `default`."""
    s = str(v if v is not None else "").strip().lower()
    if not s:
        return default
    if s in ("on", "1", "true", "yes"):
        return True
    if s in ("off", "0", "false", "no"):
        return False
    raise ValueError(f"{v!r}: on or off")


#: what KNURLOGIC_KV_BITS may be; "bf16" is the unquantized cache
KV_BITS_VALUES = ["bf16", "8", "6", "4"]
#: what Settings offers: 6 and 4 have no fused decode kernel and were never
#: measured on a real model, so they are taken from env / --set only
KV_BITS_OFFERED = ("bf16", "8")


#: GLM's sparse-prefill crossover, in tokens of context: "0" always, "off"
#: never (glm5_next edit 10). The default is "0": in the latent a chunk
#: reads K rows where the expanded path reads the whole context; the
#: crossover is for where a fused kernel over a short context measures
#: faster (tools/bench_glm5_sparse_prefill.py).
SPARSE_PREFILL_DEFAULT = "0"
SPARSE_PREFILL_VALUES = ["0", "8192", "16384", "32768", "65536", "131072",
                         "off"]


def sparse_prefill_of(v) -> str:
    """'off' or a whole number of tokens >= 0, as a string."""
    s = str(v if v is not None else "").strip().lower()
    if s in ("", "default"):
        return SPARSE_PREFILL_DEFAULT
    if s in ("off", "never"):
        return "off"
    try:
        n = int(s)
    except ValueError:
        raise ValueError(f"{s!r} isn't a number of tokens or off") from None
    if n < 0:
        raise ValueError(f"{n} is below 0")
    return str(n)


def kv_bits_of(v):
    """'bf16'/''/None -> None; '8'/'6'/'4' -> int; anything else raises.
    The same reading as engine/kvquant.parse_bits, without mlx."""
    s = str(v if v is not None else "").strip().lower()
    if s in ("", "bf16", "16", "none", "off"):
        return None
    if s not in KV_BITS_VALUES:
        n = f"{s}-bit" if s.isdigit() else repr(s)
        raise ValueError(f"KV cache: {n} isn't supported; use bf16 or 8-bit")
    return int(s)


def kv_bytes_per_element(bits) -> float:
    """One stored K or V element: 2 bytes at bf16; bits/8 plus a bf16
    scale and bias per group of 64 when quantized (engine/kvquant.py;
    a head dim that is not a multiple of 64 groups by 32 and costs a
    little more)."""
    if bits is None:
        return 2.0
    return int(bits) / 8 + 4 / 64


def kv_quant_for(model_type: str) -> tuple:
    """([bits allowed], why) for a family: its manifest's `kv_quant`
    (engine/families), [] with the reason where it is refused."""
    from knurlogic.engine import families
    from knurlogic.engine.arch import ARCH_FOR_MODEL_TYPE
    arch = ARCH_FOR_MODEL_TYPE.get(model_type, model_type)
    spec = families.build_maps()["kv_quant"].get(arch)
    if not spec:
        return [], (f"{model_type or 'this model'}: no family declares "
                    f"whether its KV cache can be quantized")
    if spec.get("refused"):
        return [], spec["refused"]
    return [int(b) for b in spec["bits"]], spec.get("why", "")


def cross_chip_of(v) -> str:
    """'auto' | 'on' | 'off' (engine/crosschip.parse, without mlx)."""
    from knurlogic.engine.crosschip import parse
    return parse(v)


def engine_settings(env: dict) -> dict:
    """{prefill_step_size, cache_limit_gb, mtp, ...} from a
    resolved environment, whichever alias it was emitted under. Absent means
    the engine's own default stands."""
    out = {}
    for logical, key, cast in (("prefill_chunk", "prefill_step_size", int),
                               ("cache_limit_gb", "cache_limit_gb", float),
                               ("mtp", "mtp", on_off),
                               ("mtp_dynamic", "mtp_dynamic", on_off),
                               ("vision", "vision", on_off),
                               ("kv_bits", "kv_bits", kv_bits_of),
                               ("cross_chip", "cross_chip", cross_chip_of),
                               ("long_context", "long_context",
                                long_context_of)):
        for name in KNOB_ALIASES[logical]:
            if name in env:
                out[key] = (cast(float(env[name])) if cast in (int, float)
                            else cast(env[name]))
                break
    return out

def canonical_sets(sets: dict | None) -> dict:
    """Explicit settings with an accepted old name moved to the name the
    resolver emits first (VQLAB_PREFILL_CHUNK -> KNURLOGIC_PREFILL_CHUNK,
    VQ_CACHE_LIMIT_GB -> KNURLOGIC_CACHE_LIMIT_GB), so an explicit value cannot
    lose to the resolver's under another alias. Where both are given, the
    current name wins."""
    out = dict(sets or {})
    for names in KNOB_ALIASES.values():
        for old in names[1:]:
            if old in out:
                v = out.pop(old)
                out.setdefault(names[0], v)
    return out


def legacy_mirror(env: dict, forced: dict) -> dict:
    """An explicit value set under a knob's current name, copied to the old
    name the resolver emitted for this artifact's bundled runtime (which
    reads only that one). Returns the extra {name: value} to set."""
    out = {}
    for names in KNOB_ALIASES.values():
        if names[0] in forced:
            for old in names[1:]:
                if old in env and old not in forced:
                    out[old] = forced[names[0]]
    return out


#: The env name for a logical knob when no bundled runtime can be asked.
def default_alias(logical: str) -> str:
    return KNOB_ALIASES[logical][0]


# --- who is a knob FOR ------------------------------------------------------
# Those bundled runtimes read ~38 environment variables. Most are kernel
# internals -- tile widths, register buffers, SIMD group sizes -- and nobody
# outside the person writing the kernel has a reason to touch them. Showing
# all 38 would be the busy-panel mistake: every knob visible, none of them
# weighted, the eye with nowhere to go.
#
# So three tiers, by who would reach for it:
#
#   reach     will it run, will it OOM, how fast. The memory knobs and the
#             tune axis. These are the page.
#   deeper    measured performance and numerics flags. Real effects, real
#             findings behind them, but you go looking on purpose.
#   kernel    read by the runtime, no measured answer here. NOT defaulted and
#             NOT hidden: listed if someone digs, labelled as the runtime's
#             own business. Inventing defaults for unmeasured knobs is how
#             the frozen 2048*4096*2 constant happened.
KNOB_TIER_REACH = ("VQ_DECODE_CHUNK", "VQ_CACHE_LIMIT_GB",
                   "KNURLOGIC_CACHE_LIMIT_GB",
                   "VQLAB_CACHE_LIMIT_GB", "KNURLOGIC_PREFILL_CHUNK",
                   "VQLAB_PREFILL_CHUNK",
                   "KNURLOGIC_CONTEXT_LENGTH") + MODEL_KNOBS


def knob_tier(name: str) -> str:
    if name in KNOB_TIER_REACH:
        return "reach"
    if name in NUMERICS_FLAGS:
        return "deeper"
    return "kernel"


# --- what a knob can be turned TO -------------------------------------------
# Only the knobs a person reaches for get a range. The rest are named, not
# turned. A range is (values, unit): discrete, because these are discrete --
# a continuous slider over a chunk width would invent positions that no run
# ever measured.
#
# The maximum is a MEASUREMENT, not a taste. VQ_DECODE_CHUNK stops at 32
# because 128 -> 32 is 1.37x on every rung measured and nothing above it was
# ever better; the control should stop where the evidence stops.
KNOB_RANGE: dict = {
    # 4096 is the widest ever measured (qwen3_5); the table is where a wider
    # one would have to be earned first.
    "VQLAB_PREFILL_CHUNK": ([512, 1024, 2048, 4096], "tokens"),
    "KNURLOGIC_PREFILL_CHUNK": ([512, 1024, 2048, 4096], "tokens"),
    # powers of two up to the longest window a released model has; the
    # control stops at the model's own (the resolver's value)
    "KNURLOGIC_CONTEXT_LENGTH": ([8192, 16384, 32768, 65536, 131072,
                                  262144, 524288, 1048576], "tokens"),
    "VQ_DECODE_CHUNK": ([4, 8, 16, 32], ""),
    "KNURLOGIC_MTP": (["on", "off"], ""),
    "KNURLOGIC_MTP_DYNAMIC": (["on", "off"], ""),
    "KNURLOGIC_VISION": (["on", "off"], ""),
    # narrowed per family by the resolver (Resolution.ranges): a family
    # that refuses quantized KV offers bf16 alone
    "KNURLOGIC_KV_BITS": (KV_BITS_VALUES, "bits"),
    "KNURLOGIC_KV_KERNEL": (["on", "off"], ""),
    "KNURLOGIC_SPARSE_PREFILL_FROM": (SPARSE_PREFILL_VALUES, "tokens"),
    "KNURLOGIC_CROSS_CHIP": (["off", "on", "auto"], ""),
    # narrowed per family by the resolver: ["off"] where no model card
    # documents YaRN
    "KNURLOGIC_LONG_CONTEXT": (["off", "yarn"], ""),
    "KNURLOGIC_PRESET": (list(PRESETS), ""),
    "VQ_CACHE_LIMIT_GB": ([0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0],
                          "GiB"),
    "VQLAB_CACHE_LIMIT_GB": ([0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0],
                             "GiB"),
    "KNURLOGIC_CACHE_LIMIT_GB": ([0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0],
                                 "GiB"),
}



# --- the longest context a model was built for -------------------------------
# max_position_embeddings is the trained window. A config that declares YaRN
# rope scaling was built to run past it: the window is then
# original_max_position_embeddings * factor. Anything else -- rope_type
# "default", no scaling at all -- stops at max_position_embeddings. The
# engine does not refuse a position past it (rope is computed for any
# position), it just runs a model past the length it was trained on, which
# is quietly worse output rather than an error. So the cap is refused above
# this, not honoured.
def model_window(cfg: dict) -> tuple:
    """(tokens, why) for an artifact's config.json; (0, why) when it does
    not say."""
    cfg = cfg or {}
    text: Any = cfg.get("text_config") if isinstance(
        cfg.get("text_config"), dict) else {}
    mpe = int(text.get("max_position_embeddings")
              or cfg.get("max_position_embeddings") or 0)
    rope: Any = None
    for c in (text, cfg):
        for key in ("rope_scaling", "rope_parameters"):
            if isinstance(c.get(key), dict):
                rope = c[key]
                break
        if rope:
            break
    kind = str((rope or {}).get("rope_type") or (rope or {}).get("type")
               or "").lower()
    if kind == "yarn" and rope.get("factor"):
        orig = int(rope.get("original_max_position_embeddings") or mpe or 0)
        yarn = int(orig * float(rope["factor"]))
        if yarn > mpe:
            return yarn, (f"{orig:,} trained x YaRN factor "
                          f"{float(rope['factor']):g} (rope_scaling)")
    if mpe:
        return mpe, ("max_position_embeddings; no YaRN rope scaling in its "
                     "config, so nothing longer was trained")
    return 0, "its config does not say (no max_position_embeddings)"


# --- running past the trained window: YaRN, where the model card says so ----
# Qwen's model cards for the hybrid families knurlogic serves document one
# recipe for ~1M tokens: rope_parameters gains rope_type "yarn", factor 4.0,
# original_max_position_embeddings 262144 (mrope/partial rotary unchanged),
# and vLLM/sglang raise their max length (VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# --max-model-len 1010000; SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
# --context-length 1010000). Static YaRN: "potentially impacting performance
# on shorter texts". Sources:
#   https://huggingface.co/Qwen/Qwen3.5-397B-A17B   (1,010,000)
#   https://huggingface.co/Qwen/Qwen3.6-35B-A3B     (1,010,000)
#   https://huggingface.co/Qwen/Qwen3.8-27B         (1,000,000)
#   https://huggingface.co/Qwen/Qwen3.8-Flash-Next  (1,000,000)
# The cap here is the YaRN window itself (262144 x 4 = 1,048,576, what
# model_window reads off a yarn config); Qwen's tested lengths are above.
LONG_CONTEXT_VALUES = ("off", "yarn")
#: model_type -> (factor, original_max_position_embeddings, documented tokens)
LONG_CONTEXT_YARN = {
    "qwen3_5": (4.0, 262144, 1_010_000),
    "qwen3_5_moe": (4.0, 262144, 1_010_000),
    "qwen4_exp": (4.0, 262144, 1_000_000),
}


def long_context_family(model_type: str) -> str | None:
    """The LONG_CONTEXT_YARN key for a model_type (a text config's
    `qwen3_5_text` is its wrapper's), or None where no card documents it."""
    mt = str(model_type or "")
    mt = mt[:-5] if mt.endswith("_text") else mt
    return mt if mt in LONG_CONTEXT_YARN else None


def long_context_of(v) -> str:
    """'off' | 'yarn'; ''/None is off."""
    s = str(v if v is not None else "").strip().lower()
    if s in ("", "off", "0", "false", "no", "none"):
        return "off"
    if s not in LONG_CONTEXT_VALUES:
        raise ValueError(f"Long context: {v!r} isn't "
                         f"{' or '.join(LONG_CONTEXT_VALUES)}")
    return s


def long_context_refusal(model_type: str, mode) -> str | None:
    """Why `mode` cannot be taken for this family, or None."""
    if long_context_of(mode) == "off":
        return None
    if long_context_family(model_type) is None:
        return (f"KNURLOGIC_LONG_CONTEXT=yarn is refused for "
                f"{model_type or 'this model'}: only the Qwen families "
                f"whose model cards document YaRN take it "
                f"({', '.join(sorted(LONG_CONTEXT_YARN))})")
    return None


def context_ceiling(model_type: str, cfg: dict) -> int:
    """The most context a launch of this model can ask for: its YaRN
    window where the family documents YaRN, else its native window (0 when
    the config does not say)."""
    native, _ = model_window(cfg or {})
    if long_context_family(model_type):
        top, _ = model_window(with_long_context(
            {"model_type": model_type, **(cfg or {})}, "yarn"))
        return max(top, native)
    return native


def settle_context(model_type: str, cfg: dict, sets: dict) -> tuple:
    """(sets, notes): a launch's KNURLOGIC_CONTEXT_LENGTH made one the model
    can take, never a refusal -- a saved per-model value must not brick a
    launch.

    Asking for more than the model's native window IS asking for long
    context: where the family's card documents YaRN, KNURLOGIC_LONG_CONTEXT
    is turned on for the launch (its KV-room check still applies), and a
    value past even the YaRN window is lowered to it. A family without YaRN
    has the value lowered to its native window. Each change is one note."""
    out = dict(sets or {})
    raw = out.get("KNURLOGIC_CONTEXT_LENGTH")
    try:
        want = int(str(raw).strip())
    except (TypeError, ValueError):
        return out, []            # check_knob says what is wrong with it
    native, _ = model_window(cfg or {})
    if not native or want <= native:
        return out, []
    notes = []
    try:
        mode = long_context_of(out.get("KNURLOGIC_LONG_CONTEXT"))
    except ValueError:
        return out, []            # refused with its own reason
    if long_context_family(model_type):
        if mode == "off":
            out["KNURLOGIC_LONG_CONTEXT"] = "yarn"
            notes.append(f"context {want:,} is past the native "
                         f"{native:,}: long context (YaRN) is on for this "
                         f"launch")
        top, _ = model_window(with_long_context(
            {"model_type": model_type, **(cfg or {})}, "yarn"))
        if top and want > top:
            out["KNURLOGIC_CONTEXT_LENGTH"] = str(top)
            notes.append(f"context {want:,} lowered to {top:,}, the most "
                         f"long context (YaRN) reaches")
        return out, notes
    out["KNURLOGIC_CONTEXT_LENGTH"] = str(native)
    notes.append(f"context {want:,} lowered to {native:,}, this model's "
                 f"maximum ({model_type or 'this family'} has no documented "
                 f"long context)")
    return out, notes


def long_context_config(cfg: dict, mode) -> dict:
    """The top-level config keys to overlay at load (mlx-lm's
    `model_config`, a shallow update) for `mode`: {} when off, else the
    text config with rope_parameters carrying Qwen's YaRN. The artifact's
    config.json is never written. Raises ValueError where refused."""
    mode = long_context_of(mode)
    if mode == "off":
        return {}
    cfg = cfg or {}
    mt = str(cfg.get("model_type") or "")
    why = long_context_refusal(mt, mode)
    if why:
        raise ValueError(why)
    factor, orig, _doc = LONG_CONTEXT_YARN[long_context_family(mt) or ""]
    nested = isinstance(cfg.get("text_config"), dict)
    tc = dict(cfg["text_config"]) if nested else dict(cfg)
    key = "rope_parameters" if isinstance(tc.get("rope_parameters"), dict) \
        or not isinstance(tc.get("rope_scaling"), dict) else "rope_scaling"
    rp = dict(tc.get(key) or {})
    rp.pop("type", None)
    rp.update(rope_type="yarn", factor=factor,
              original_max_position_embeddings=orig)
    tc[key] = rp
    if nested:
        return {"text_config": tc}
    return {key: rp}


def with_long_context(cfg: dict, mode) -> dict:
    """`cfg` as the model will load under `mode` (a copy)."""
    out = dict(cfg or {})
    out.update(long_context_config(cfg, mode))
    return out


#: numeric knobs: (type, lowest, highest or None, what it counts)
KNOB_BOUNDS = {
    "KNURLOGIC_PREFILL_CHUNK": (int, 16, 4096, "tokens"),
    "VQLAB_PREFILL_CHUNK": (int, 16, 4096, "tokens"),
    "KNURLOGIC_CONTEXT_LENGTH": (int, 256, None, "tokens"),
    "VQ_DECODE_CHUNK": (int, DECODE_CHUNK_MIN, DECODE_CHUNK_DEFAULT, ""),
    "VQ_CACHE_LIMIT_GB": (float, 0.0, CACHE_LIMIT_GB_MAX, "GiB"),
    "VQLAB_CACHE_LIMIT_GB": (float, 0.0, CACHE_LIMIT_GB_MAX, "GiB"),
    "KNURLOGIC_CACHE_LIMIT_GB": (float, 0.0, CACHE_LIMIT_GB_MAX, "GiB"),
}


def check_knob(name: str, value, window: int = 0):
    """None when `value` is one `name` may take, else the sentence saying
    why not. `window`: the model's (model_window), which caps the context
    length. Enumerated knobs are checked against their values; numeric ones
    for type and range; anything else is the runtime's own business."""
    s = str(value if value is not None else "").strip()
    if name in KNOB_BOUNDS:
        cast, lo, hi, unit = KNOB_BOUNDS[name]
        try:
            v = cast(s)
        except ValueError:
            return (f"{knob_title(name)}: {s!r} isn't a "
                    f"{'whole number' if cast is int else 'number'}")
        if name == "KNURLOGIC_CONTEXT_LENGTH" and window:
            hi = window
        if v < lo or (hi is not None and v > hi):
            where = (f"this model's maximum is {hi:,} tokens" if
                     name == "KNURLOGIC_CONTEXT_LENGTH" and window else
                     f"between {lo:g} and {hi:g}{' ' + unit if unit else ''}"
                     if hi is not None else f"at least {lo:g}")
            return f"{knob_title(name)}: {s} isn't allowed; {where}"
        return None
    if name in COMPACT_KNOBS:
        return check_compact_knob(name, value)
    try:
        if name in ("KNURLOGIC_MTP", "KNURLOGIC_MTP_DYNAMIC",
                    "KNURLOGIC_VISION"):
            if s.lower() not in ("", "on", "off", "1", "0", "true", "false",
                                 "yes", "no"):
                raise ValueError(f"{s!r} isn't on or off")
        elif name == "KNURLOGIC_KV_BITS":
            kv_bits_of(s)
        elif name == "KNURLOGIC_PRESET":
            preset_of(s)
        elif name == MTP_MODE and s not in MTP_MODES:
            raise ValueError(f"{s!r} isn't {', '.join(MTP_MODES)}")
        elif name == "KNURLOGIC_CROSS_CHIP":
            cross_chip_of(s)
        elif name == "KNURLOGIC_LONG_CONTEXT":
            long_context_of(s)
        elif name == "KNURLOGIC_SPARSE_PREFILL_FROM":
            sparse_prefill_of(s)
    except ValueError as e:
        m = str(e)
        return m if m.startswith(knob_title(name) + ":") \
            else f"{knob_title(name)}: {m}"
    return None

# --- what a VISION rung holds besides its weights ---------------------------
# A model with a vision tower needs three things a text model does not, and
# the resolver must count them BEFORE a load, not discover them as an OOM on
# the first screenshot:
#
# 1. The TOWER'S WEIGHTS. Read from the safetensors headers (tensor names
#    under these prefixes), never guessed. Every family keeps them in the
#    artifact's own directory -- in the shards, or Qwen's
#    `model-vision-graft.safetensors` sidecar -- so `bytes_on_disk` already
#    includes them; the term is shown so nobody has to take that on faith,
#    and is added only if the scan finds tower tensors outside what
#    bytes_on_disk counted. The prefixes are the union of the families'
#    (each manifest's vision `signature`, engine/families/; plain data,
#    importing them imports no mlx).
def _tower_prefixes() -> tuple:
    from knurlogic.engine import families
    return families.build_maps()["vision_tower_prefixes"]


VISION_TOWER_PREFIXES = _tower_prefixes()
#
# 2. The IMAGE STORE'S bound. The number is engine/vision/store.py's
#    DEFAULT_MAX_BYTES (one home), or a live store's budget_bytes() when a
#    store exists -- which can exceed the bound while prompt-cache entries
#    pin images in use (engine/vision/cachehook.py).
#
# 3. KV for IMAGE SPANS. An image is hundreds to thousands of tokens of
#    context that a text chat would not have had. The allowance reserves KV
#    for this many images of this many tokens each. 4096 tokens: GLM's
#    largest images run to ~8000, gemma's are 280, a default-sized Qwen
#    screenshot ~1000-2500; 4096 is the middle of the families' upper
#    ranges. 4 images: a conversation's worth, the same figure the store's
#    256 MiB default was sized for. An ALLOWANCE, not a measurement -- the
#    note on the resolution says so.
VISION_KV_IMAGES = 4
VISION_KV_TOKENS_PER_IMAGE = 4096
#: Bytes per bf16 element: the size of any cache row kept in bf16 (a
#: text model's unquantized KV, MLA rope keys, DeepSeek-V4's pools). Its
#: own name so a change to the vision allowance cannot move these.
BF16_BYTES = 2
#: bf16 KV, the dtype every served rung's cache runs in.
VISION_KV_DTYPE_BYTES = 2


# --- server-side context compaction (context_management/compaction.py) ----
# The operator's defaults for what a request's `context_management` leaves
# out, and whether the server compacts a request that asks for nothing.
# Read per request from the environment, so each applies live on a running
# server (POST /settings.json) and, set before a launch, from its start.
# Not measured: these are policy, ported from an earlier agent-loop
# compactor (keep_recent=6). There is no summary
# budget: a summary is only never longer than what it replaces.
COMPACT_KNOBS = {
    # name: (default, values, unit, what, why)
    "KNURLOGIC_COMPACT_AUTO": (
        "off", ["off", "on"], "",
        "compact a request that asks for nothing, once its prompt passes "
        "the trigger",
        "on keeps a client that never asks inside the window, at a cost: "
        "the request waits for a summary pass (one extra model call), and "
        "older turns survive only as that summary -- detail is lost. Off: "
        "only a harness that asks (context_management) is compacted; one "
        "that does not is refused past the window. The summary goes back "
        "in the response for the client to resend; nothing is kept here."),
    "KNURLOGIC_COMPACT_TRIGGER": (
        "0.8", ["0.5", "0.6", "0.7", "0.8", "0.9"], "of the window",
        "where compaction starts, as a share of the model's context window",
        "lower compacts earlier: shorter prompts, faster prefill and less "
        "KV memory, but more summary passes and detail lost sooner. Higher "
        "keeps more word for word, and runs closer to the window. Used by "
        "automatic compaction, and by a compact edit that names no trigger "
        "when the window is below the API's 150k default."),
    "KNURLOGIC_COMPACT_KEEP_TURNS": (
        "6", ["2", "4", "6", "8", "12", "16"], "messages",
        "the most recent messages kept word for word",
        "more keeps more recent work exact, at the cost of a longer "
        "compacted prompt (more prefill and KV memory). Fewer shrinks it "
        "further and leans harder on the summary, so more detail is lost. "
        "The kept tail never starts on a tool result (it is "
        "widened back to the call that asked for it); the first message "
        "and the goal turn are always kept."),
    "KNURLOGIC_COMPACT_TOOL_RESULTS": (
        "distill", ["distill", "clear"], "",
        "what becomes of a dropped tool result: a one-line finding, or "
        "nothing",
        "distill keeps what each dropped tool call established (a Grep for "
        "X -> 'X is defined at src/foo.py:120'), at the cost of more output "
        "from the same summary pass -- a slower compaction. clear is the "
        "cheapest, and the model loses what those calls found: it may run "
        "them again."),
}


def compact_settings(env: dict) -> dict:
    """The operator's compaction defaults from an environment, each
    falling back to its default when absent or unreadable:
    {auto, trigger, keep, distill}."""
    def get(name):
        v = str((env or {}).get(name, "") or "").strip()
        return v or COMPACT_KNOBS[name][0]

    def num(name, cast, lo, hi):
        try:
            v = cast(get(name))
        except ValueError:
            v = cast(COMPACT_KNOBS[name][0])
        return min(max(v, lo), hi)
    try:
        auto = on_off(get("KNURLOGIC_COMPACT_AUTO"), False)
    except ValueError:
        auto = False
    return {"auto": auto,
            "trigger": num("KNURLOGIC_COMPACT_TRIGGER", float, 0.05, 0.99),
            "keep": num("KNURLOGIC_COMPACT_KEEP_TURNS", int, 0, 1000),
            "distill": get("KNURLOGIC_COMPACT_TOOL_RESULTS") != "clear"}


def check_compact_knob(name: str, value):
    """None when `value` is one compaction knob `name` may take, else why
    not."""
    s = str(value if value is not None else "").strip()
    if name == "KNURLOGIC_COMPACT_AUTO":
        try:
            on_off(s)
            return None
        except ValueError as e:
            return f"{name}: {e}"
    if name == "KNURLOGIC_COMPACT_TOOL_RESULTS":
        return None if s in ("distill", "clear") else \
            f"{name}={s!r}: distill or clear"
    cast, lo, hi = {"KNURLOGIC_COMPACT_TRIGGER": (float, 0.05, 0.99),
                    "KNURLOGIC_COMPACT_KEEP_TURNS": (int, 0, 1000)}[name]
    try:
        v = cast(s)
    except ValueError:
        return f"{name}={s!r}: not a number"
    if not lo <= v <= hi:
        return f"{name}={s}: between {lo:g} and {hi:g}"
    return None


# --- what a launch request may carry ----------------------------------------
# The page, a forwarded load and a cluster job all check a request against
# these; they are settings facts, so they live here.

#: request keys that would name a place on disk; refused outright, never
#: ignored, so a coordinator that sends one learns it is wrong
PATH_KEYS = ("path", "target", "artifact", "where", "dir", "directory")


def launch_knobs() -> frozenset:
    """The knob names a forwarded load may set: the ones knurlogic documents
    (KNOB_DOC) and their aliases. Nothing else is passed on:
    `--set` puts it in the child's environment."""
    return frozenset(KNOB_DOC) | frozenset(
        n for v in KNOB_ALIASES.values() for n in v) | frozenset(
        NUMERICS_FLAGS)


def clean_sets(sets) -> tuple:
    """(allowed {name: value}, [refused names]). Values are short plain
    tokens: digits, letters, '.', '-', '_'."""
    import re
    ok, bad = {}, []
    allowed = launch_knobs()
    for k, v in (sets.items() if isinstance(sets, dict) else ()):
        v = str(v)
        if k in allowed and len(v) <= 64 and re.fullmatch(r"[\w.\-]*", v):
            ok[k] = v
        else:
            bad.append(str(k)[:64])
    return ok, bad
