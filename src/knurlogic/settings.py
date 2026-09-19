"""Measured runtime constants, with the provenance that paid for them.

Every number here came off a run. The point of Knurlogic is that a downloader
should never have to know them: the resolver turns them into defaults. Each
constant carries the finding that established it, so a future change has to
argue with a measurement rather than a preference.

Source: vqlab docs/RUNTIME-SETTINGS.md, docs/FINDINGS-LOG.md.
"""

from __future__ import annotations

# --- the two knobs that decide runnable-vs-not ------------------------------

# Experts decoded to dense fp16 per prefill chunk. THIS IS THE MEMORY KNOB,
# not the KV cache: measured 2026-08-15 on a 128 GB M4 Max running the
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

# Prompt chunk width. Token-identical at every value (vqlab
# tests/test_mtp_prefill.py gates this) -- purely a memory knob. mlx-lm's
# server does not expose it, which is why it must be resolved here.
PREFILL_CHUNK_DEFAULT = 2048
PREFILL_CHUNK_TIGHT = 512

# Freed MLX buffers pile up invisibly -- they do not appear in "active
# memory." Biggest single win in vqlab's memory playbook, zero measured speed
# cost at 26k-token prefill.
CACHE_LIMIT_GB_DEFAULT = 4.0

# Below this much free headroom after the weights, treat the box as tight and
# resolve the memory knobs down rather than leaving performance defaults.
TIGHT_HEADROOM_GIB = 12.0

# --- performance knobs with a measured basis --------------------------------
# value -> (default, why). Anything not listed should not be set by a
# resolver; it exists in the runtime so a finding stays reproducible.
PERFORMANCE_DEFAULTS = {
    # F124: device codebook beats threadgroup by 20.9% on prefill at
    # d4-K2048. 'auto' lets the runtime's own selector decide per module;
    # ~447 fleet modules ride on it.
    "VQ_MOE_GEMMSEG_CBDEV": ("auto", "F124: device arm +20.9% prefill at d4-K2048"),
    # F25/F33: RTILE=64 is SLOWER everywhere (0.75-0.97x), confirmed on both
    # exo and local. The one 'win' was an env-ordering bug that benchmarked
    # 32 twice. DO NOT SET 64.
    "VQ_MOE_GEMMSEG_RTILE": ("32", "F25/F33: 64 is 0.75-0.97x, never faster"),
    # F54 arm 1: +5.1-6.6% prefill, bit-exact.
    "VQ_GEMMSEG_OTILE64": ("1", "F54: +5.1-6.6% prefill, bit-exact"),
    # F56: the v2 stack reaches +11.9% over shipped.
    "VQ_GEMMSEG_PH2V": ("1", "F56: part of the +11.9% stack"),
    "VQ_D4_WALK": ("1", "F56: part of the +11.9% stack"),
    # Arm 1.5 measured NEGATIVE (-1.8-2%).
    "VQ_GEMMSEG_PIPE": ("0", "F56 arm 1.5: measured -1.8-2%"),
}

# Numerics-active flags (F103/F105): family-local, up to +0.97% ppl.
# v1.5 = both off (bit-exact vs the published arc6 runtime); v2 = both on.
# An artifact shipping UNCHANGED weights gets v1.5 -- there is no quality gain
# to offset a numerics regression, however small. See vqlab
# docs/RUNTIME-SHIP-PLAN.md.
NUMERICS_FLAGS = ("VQ_GEMMSEG_BF16IO", "VQ_DECODE_BF16IO")

RUNTIME_PROFILES = {
    "v1.5": {f: "0" for f in NUMERICS_FLAGS},
    "v2": {f: "1" for f in NUMERICS_FLAGS},
}


# --- the tuning axis, and what it is NOT allowed to do ----------------------
# "Less prefill spike at the cost of speed" and "I have headroom, go fast" are
# the two things a person actually wants to say. The axis exists so they can
# say it without learning what any of these names mean.
#
# THE CAPS ARE THE POINT. A tuning axis that just scales every knob would
# happily walk into settings that are measured to be WORSE at both ends:
#
#   * VQ_DECODE_CHUNK: smaller is faster AND smaller in memory (128 -> 32 is
#     1.37x on every rung). There is no tradeoff on this knob, so "fast" must
#     NOT raise it. It is capped at the default in both directions.
#   * VQ_MOE_GEMMSEG_RTILE=64 is 0.75-0.97x and never faster (F25/F33), so no
#     setting of this axis may reach it.
#
# So `fast` moves only the knobs where headroom actually buys something, and
# `safe` tightens the ones that bound peak memory. Anything a profile asks for
# beyond a cap is refused and the refusal is printed, never silently clamped.
TUNE_PROFILES = {
    # prefill chunk, cache limit GiB, and whether to bound the transient
    # harder than headroom requires
    "safe": {
        "VQLAB_PREFILL_CHUNK": PREFILL_CHUNK_TIGHT,
        "VQLAB_CACHE_LIMIT_GB": 1.0,
        "decode_chunk_scale": 0.5,   # bound the transient below what fits
        "why": "lowest peak memory: narrow prompt chunks, a small reclaimable "
               "cache, and a transient bounded tighter than headroom requires",
    },
    "balanced": {
        "decode_chunk_scale": 1.0,
        "why": "the measured defaults",
    },
    "fast": {
        "VQLAB_PREFILL_CHUNK": PREFILL_CHUNK_DEFAULT,
        "VQLAB_CACHE_LIMIT_GB": 8.0,
        "decode_chunk_scale": 1.0,   # capped: smaller is already faster
        "why": "spends headroom where it actually buys speed -- wider prompt "
               "chunks and a larger reclaimable cache. It does NOT raise the "
               "decode chunk, because smaller is faster there as well as "
               "smaller in memory",
    },
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
    "VQ_DECODE_CHUNK": (
        "experts decoded to dense fp16 per prefill chunk",
        "THE memory knob: prefill grew 3.35 MB/token where KV-cache theory "
        "predicted 0.059. Smaller is also faster (128 -> 32 is 1.37x), so it "
        "is capped at 32 and never raised."),
    "VQLAB_PREFILL_CHUNK": (
        "how many prompt tokens are processed at once",
        "token-identical at every width, so it is purely a memory knob -- "
        "narrowing costs nothing but peak."),
    "VQLAB_CACHE_LIMIT_GB": (
        "how much freed-buffer cache the runtime may hold",
        "biggest single win in the memory playbook, no measured speed cost at "
        "26k-token prefill. Reclaimable, but still resident."),
    "VQ_MOE_GEMMSEG_CBDEV": (
        "where the codebook lives during the MoE GEMM",
        "F124: the device arm is +20.9% on prefill at d4-K2048; 'auto' lets "
        "the runtime choose per module."),
    "VQ_MOE_GEMMSEG_RTILE": (
        "row tile width in the segmented GEMM",
        "F25/F33: 64 is 0.75-0.97x and NEVER faster. The one 'win' was an "
        "env-ordering bug that benchmarked 32 twice."),
    "VQ_GEMMSEG_OTILE64": (
        "64-wide output tiling in the segmented GEMM",
        "F54: +5.1-6.6% prefill, bit-exact."),
    "VQ_GEMMSEG_PH2V": ("phase-2 vectorization",
                        "F56: part of the +11.9% stack."),
    "VQ_D4_WALK": ("d4 codebook walk", "F56: part of the +11.9% stack."),
    "VQ_GEMMSEG_PIPE": ("software pipelining in the segmented GEMM",
                        "F56 arm 1.5: measured NEGATIVE, -1.8-2%. Off."),
    "VQ_GEMMSEG_BF16IO": ("bf16 IO in the segmented GEMM",
                          "F103/F105 numerics-active: family-local, up to "
                          "+0.97% ppl. Off at v1.5 (bit-exact vs shipped)."),
    "VQ_DECODE_BF16IO": ("bf16 IO on the decode path",
                         "F103/F105 numerics-active. Off at v1.5."),
}


# --- the names are the ARTIFACT'S, not ours --------------------------------
# `VQLAB_CACHE_LIMIT_GB` is a knurlogic-shaped name for something read by 24
# of the 37 bundled runtimes on this machine. Those files are published. A
# tidy-up rename in the resolver would not tidy anything -- it would emit a
# name nobody reads and silently stop bounding the cache on every artifact
# already shipped, which is precisely the failure mode this package exists to
# end, dressed as housekeeping.
#
# So a knob has a LOGICAL name here and a list of env names, preferred first.
# The resolver emits whichever one the target artifact actually reads. A new
# rung can bundle a runtime reading the new name and every published rung
# keeps the one it shipped with -- the same per-artifact boundary `model_file`
# already establishes, used for the interface rather than the engine.
KNOB_ALIASES = {
    "cache_limit_gb": ("KNURLOGIC_CACHE_LIMIT_GB", "VQLAB_CACHE_LIMIT_GB"),
    "prefill_chunk": ("KNURLOGIC_PREFILL_CHUNK", "VQLAB_PREFILL_CHUNK"),
    "decode_chunk": ("VQ_DECODE_CHUNK",),
}

#: When no bundled runtime can be asked, emit this one. The LAST alias, not
#: the first: the legacy name is the one with 24 artifacts behind it, and a
#: guess should fail towards what exists rather than towards what is planned.
def default_alias(logical: str) -> str:
    return KNOB_ALIASES[logical][-1]
