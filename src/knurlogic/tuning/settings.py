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

# Per-FAMILY prompt chunk, keyed by config.json `model_type`. Carried from the
# exo fork (worker/engines/mlx/constants.py, PREFILL_STEP_SIZE_BY_FAMILY),
# where it was keyed by a model-id substring; model_type is the same fact
# without guessing from a name. An entry is a MEASUREMENT with its run:
#
#   glm5_next  2048. 34 deltanet layers hold per-token recurrent
#              intermediates (16.8 MB/layer) across a chunk, so the transient
#              scales with chunk x state. 4096 OOMed BOTH boxes of a 224 GB
#              pair on the 135 GB 3.6bpw (2026-09-01) -- but before the
#              per-chunk eval fix. 512 after it was over-caution costing 4x
#              the chunks; 2048 is the post-fix value.
#   qwen3_5    4096. MEASURED, not the default leaking through: 2026-06-19
#   qwen3_5_moe      A/B, +115% prefill tok/s at 11k tokens vs a 512 cap, no
#              peak-memory cost, bit-identical output. Hybrid attention, 45/60
#              layers recurrent, so there is no chunk x seq^2 transient to
#              cap. It carries ArraysCache entries, which is why any "has SSM
#              caches -> small chunk" heuristic catches it wrongly: a blanket
#              SSM->512 on 2026-09-02 made its prefill 8x the chunks.
#
# Not here means PREFILL_CHUNK_DEFAULT. A tight box still wins over the table:
# a measured width is a width that fit on the box it was measured on.
PREFILL_CHUNK_BY_FAMILY = {
    "glm5_next": 2048,
    "qwen3_5": 4096,
    "qwen3_5_moe": 4096,
}


def prefill_chunk_for(model_type: str) -> tuple:
    """(width, source) for a family: the table's measured value, or the
    default and the fact that nothing was measured."""
    # Through the architecture map, because configs spell one family several
    # ways -- a qwen3_5 27B reports `qwen3_5_text` -- and a miss here quietly
    # hands a measured family the unmeasured default.
    from knurlogic.engine.arch import ARCH_FOR_MODEL_TYPE
    family = ARCH_FOR_MODEL_TYPE.get(model_type, model_type)
    if family in PREFILL_CHUNK_BY_FAMILY:
        return PREFILL_CHUNK_BY_FAMILY[family], f"measured for {family}"
    return PREFILL_CHUNK_DEFAULT, "default: no measured width for this family"


# How many prompts the engine prefills in ONE forward. mlx-lm defaults to 8,
# and its prefill transient is per row -- eight long prompts arriving together
# is eight chunk transients at once, the spike a single-prompt test never
# shows. Arithmetic, not a measurement: on a tight box it is 1; elsewhere the
# engine's own default stands, because nothing here has measured better.
PROMPT_CONCURRENCY_TIGHT = 1

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
#
# A RUNG'S NUMERICS ARE THE RUNG'S (design D1). What a released rung computes
# with is what its PUBLISHED model.py defaults to, and that is not uniform:
# Flash-Next 2.1 and Qwen3.6-35B-A3B 3.8/4.6/5.4 shipped v2, the rest v1.5 or
# the arc6-era runtime with no flags at all (docs/design/vq-rung-knobs.md,
# read off the Hub 2026-09-23). This table used to be applied to EVERY VQ
# artifact with v1.5 as the default, which forced the v2 rungs' two flags
# to 0 -- a numerics change nobody asked for, on exactly the rungs whose
# weights were fitted under v2. So the resolver now takes a rung's numerics
# from the rung (NUMERICS_SOURCES, in order) and applies a profile ONLY when
# a person names one. See vqlab docs/RUNTIME-SHIP-PLAN.md for the profiles.
NUMERICS_FLAGS = ("VQ_GEMMSEG_BF16IO", "VQ_DECODE_BF16IO")

RUNTIME_PROFILES = {
    "v1.5": {f: "0" for f in NUMERICS_FLAGS},
    "v2": {f: "1" for f in NUMERICS_FLAGS},
}

# Where a rung's numerics come from when no profile is asked for, first
# match wins, per flag:
#   declared   config.json `knobs` -- the artifact's own record, which
#              Artifact.declared_knobs() already ranks above everything
#   published  engine/vq/rungs.json -- read from the rung's PUBLISHED
#              model.py (never an ~/.exo copy: those drifted)
#   bundled    the default in the artifact's own model.py, for a rung not
#              in rungs.json (a local build, a new upload)
# Nothing found means nothing is emitted: the runtime's own default stands,
# and the note says so rather than inventing one.
NUMERICS_SOURCES = ("declared", "published", "bundled")


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
    "KNURLOGIC_PROMPT_CONCURRENCY": (
        "how many prompts are prefilled together in one forward",
        "the prefill transient is per prompt, so 8 arriving together is 8x "
        "the spike a one-prompt test shows. 1 on a tight box."),
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
                          "+0.97% ppl. Each rung keeps what it shipped: on "
                          "for the v2 rungs, off for v1.5."),
    "VQ_DECODE_BF16IO": ("bf16 IO on the decode path",
                         "F103/F105 numerics-active. Each rung keeps what "
                         "it shipped."),
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
    "prompt_concurrency": ("KNURLOGIC_PROMPT_CONCURRENCY",),
}


# --- which knobs the ENGINE consumes ----------------------------------------
# Most knobs are read by an artifact's bundled runtime. These three are not:
# mlx-lm's server takes the prompt chunk and concurrency as argv, and the
# buffer cache is a process-global mlx setting. Emitting them as environment
# variables and stopping there is how they were, for a while, settings that
# did nothing -- the resolver explained a prompt chunk the server never saw.
# `engine.serve` is the only thing that turns these into argv and calls.
ENGINE_KNOB_NAMES = tuple(n for k in ("prefill_chunk", "cache_limit_gb",
                                      "prompt_concurrency")
                          for n in KNOB_ALIASES[k])


# --- the same knobs, spelled the way the exo fork reads them ---------------
# When exo is the engine, the prompt chunk and buffer cache reach its runners
# through ITS variables (worker/engines/mlx/constants.py and utils_mlx.py).
# Emitting knurlogic's names into exo's environment reaches nothing -- the
# same fault as an env var mlx-lm's server never reads.
#
# RING-WIDE means every rank must hold the same value or the ranks desync:
# the fork found this live when one launcher dropped EXO_PREFILL_STEP_SIZE
# and its mirror did not (GLM-5.3 at 2048 on one rank, 4096 on the other).
# `resolve_cluster` gives those one value on every node.
EXO_NAMES = {
    "prefill_step_size": ("EXO_PREFILL_STEP_SIZE", "ring"),
    "cache_limit_gb": ("EXO_MLX_CACHE_LIMIT_GB", "node"),
}


def exo_env(env: dict) -> dict:
    """The EXO_* variables for a resolved node environment."""
    out = {}
    for key, v in engine_settings(env).items():
        if key in EXO_NAMES:
            out[EXO_NAMES[key][0]] = (str(int(v)) if isinstance(v, int)
                                      else f"{v:g}")
    return out


def engine_settings(env: dict) -> dict:
    """{prefill_step_size, prompt_concurrency, cache_limit_gb} from a
    resolved environment, whichever alias it was emitted under. Absent means
    the engine's own default stands."""
    out = {}
    for logical, key, cast in (("prefill_chunk", "prefill_step_size", int),
                               ("prompt_concurrency", "prompt_concurrency", int),
                               ("cache_limit_gb", "cache_limit_gb", float)):
        for name in KNOB_ALIASES[logical]:
            if name in env:
                out[key] = cast(float(env[name]))
                break
    return out

#: When no bundled runtime can be asked, emit this one. The LAST alias, not
#: the first: the legacy name is the one with 24 artifacts behind it, and a
#: guess should fail towards what exists rather than towards what is planned.
def default_alias(logical: str) -> str:
    return KNOB_ALIASES[logical][-1]


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
KNOB_TIER_REACH = ("VQ_DECODE_CHUNK", "KNURLOGIC_CACHE_LIMIT_GB",
                   "VQLAB_CACHE_LIMIT_GB", "KNURLOGIC_PREFILL_CHUNK",
                   "VQLAB_PREFILL_CHUNK", "KNURLOGIC_PROMPT_CONCURRENCY")


def knob_tier(name: str) -> str:
    if name in KNOB_TIER_REACH:
        return "reach"
    if name in PERFORMANCE_DEFAULTS or name in NUMERICS_FLAGS:
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
KNOB_RANGE = {
    # 4096 is the widest ever measured (qwen3_5); the table is where a wider
    # one would have to be earned first.
    "VQLAB_PREFILL_CHUNK": ([512, 1024, 2048, 4096], "tokens"),
    "KNURLOGIC_PREFILL_CHUNK": ([512, 1024, 2048, 4096], "tokens"),
    "KNURLOGIC_PROMPT_CONCURRENCY": ([1, 2, 4, 8], "prompts"),
    "VQ_DECODE_CHUNK": ([4, 8, 16, 32], ""),
    "VQLAB_CACHE_LIMIT_GB": ([0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0],
                             "GiB"),
    "KNURLOGIC_CACHE_LIMIT_GB": ([0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0],
                                 "GiB"),
}


# --- what a VISION rung holds besides its weights ---------------------------
# A model with a vision tower needs three things a text model does not, and
# the resolver must count them BEFORE a load (critique issue 10, Flash-Next
# review point 4), not discover them as an OOM on the first screenshot:
#
# 1. The TOWER'S WEIGHTS. Read from the safetensors headers (tensor names
#    under these prefixes), never guessed. Every family keeps them in the
#    artifact's own directory -- in the shards, or Qwen's
#    `model-vision-graft.safetensors` sidecar -- so `bytes_on_disk` already
#    includes them; the term is shown so nobody has to take that on faith,
#    and is added only if the scan finds tower tensors outside what
#    bytes_on_disk counted. The prefixes are the union of the families'
#    loaders (gemma4 vision_tower./embed_vision., glm5 vision_model./
#    vision_tower., qwen's four namings).
VISION_TOWER_PREFIXES = ("vision_tower.", "embed_vision.", "vision_model.",
                         "visual.", "model.visual.",
                         "model.language_model.visual.",
                         "model.vision_tower.", "model.embed_vision.",
                         "multi_modal_projector.")
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
#: bf16 KV, the dtype every served rung's cache runs in.
VISION_KV_DTYPE_BYTES = 2
