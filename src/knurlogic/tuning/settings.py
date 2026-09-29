"""Measured runtime constants, with the provenance that paid for them.

Every number here came off a run. The point of Knurlogic is that a downloader
should never have to know them: the resolver turns them into defaults. Each
constant carries the measurement that established it, so a future change
has to argue with a measurement rather than a preference. The VQ kernel
numbers were measured in vqlab, where the runtime is developed.
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
#
# Chosen from the room ACTUALLY free at launch (the load budget: the
# smaller of the working set and what macOS would hand over now), never
# from installed RAM. "No reason to leave headroom unused, but read the
# room": take the widest width on the ladder, capped at the family's
# measured best, whose predicted step transient fits in
# PREFILL_TRANSIENT_ROOM_SHARE of the room left after the weights, a KV
# allowance and the reclaimable (prompt) cache; else step down, floor 512.
# A family with no measurement stays 512.
#
# M4 sweep 2026-09-26, prefill tok/s at 4k/16k-token prompts (median of 3,
# one server per arm), then the step transient:
#   Qwen3.8 Flash 4.4bpw:  512 565/484 0.79 GiB | 1024 528/490 0.99
#                          2048 552/524 2.11    | 4096 551/499 4.05
#   Qwen3.5-397B VQ 2.2:   512 194/155 0.33 GiB | 1024 224/186 1.15-1.54
#                          2048 244/205 2.95    | 4096 249/206 5.6-7.5
# M4 Max 128 GB 2026-09-29, Qwen3.6-35B-A3B VQ 3.4 (13.8 GiB), 28,727-token
# prompt, interleaved, n=3, prefill tok/s:
#   512 642.7/642.6/649.5 | 2048 1182.9/1164.5/1141.9 | 4096 1173.2/1166.5/1133.8
# So width is worth ~1.8x on the 35B-A3B up to 2048 and nothing past it;
# ~30% on the 397B VQ for a 9-23x larger transient -- 4096 aborted Metal
# with one agent at 25k tokens on a box with ~14 GiB left, which is what
# the room rule keeps at 512.
PREFILL_CHUNK_DEFAULT = 512
PREFILL_CHUNK_TIGHT = 512
#: The widths the room rule may choose from (and the knob's native range).
PREFILL_CHUNK_LADDER = (512, 1024, 2048, 4096)
#: Predicted step transient per prompt token per unit of hidden size:
#: transient(width) = width * hidden_size * this. Calibrated to the WORST
#: measured case, the 397B VQ at 4096 (7.5 GiB, hidden 4096): 7.5 GiB /
#: (4096 * 4096) = 480 bytes. It over-predicts the milder rungs (0.94 vs
#: 0.33 GiB at 512 on the 397B), which is the safe direction.
PREFILL_TRANSIENT_BYTES_PER_TOKEN_HIDDEN = 480
#: The transient may take at most this share of the room left.
PREFILL_TRANSIENT_ROOM_SHARE = 0.10
#: KV held back before the rule sizes a chunk: one long conversation.
PREFILL_KV_ALLOWANCE_TOKENS = 32768

# Per-ARCHITECTURE prompt chunk: a measurement with its run, kept in each
# family's manifest (engine/families/<family>/__init__.py, `prefill_chunk`)
# beside the rest of what that family is. Carried from the exo fork
# (PREFILL_STEP_SIZE_BY_FAMILY), where it was keyed by a model-id substring.
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
# memory." Biggest single win in vqlab's memory playbook, zero measured speed
# cost at 26k-token prefill.
CACHE_LIMIT_GB_DEFAULT = 4.0

# Below this much free headroom after the weights, treat the box as tight and
# resolve the memory knobs down rather than leaving performance defaults:
# the larger of a floor and a fraction of the working set. A fixed 12 GiB let
# 397B on the 128 GB M4 (~14 GiB above its weights) take the measured
# 4096-token prefill chunk; its first step alone measured 8.1 GiB of
# transient, and one agent at a 25k-token context aborted Metal
# (2026-09-26). A share of the working set scales with the machine.
TIGHT_HEADROOM_GIB = 12.0
TIGHT_HEADROOM_SHARE = 0.20


def tight_headroom_bytes(working_set_bytes: int) -> int:
    return int(max(TIGHT_HEADROOM_GIB * (1 << 30),
                   TIGHT_HEADROOM_SHARE * working_set_bytes))

# --- performance knobs with a measured basis --------------------------------
# value -> (default, why). Anything not listed should not be set by a
# resolver; it exists in the runtime so a finding stays reproducible.
PERFORMANCE_DEFAULTS = {
    # device codebook beats threadgroup by 20.9% on prefill at
    # d4-K2048. 'auto' lets the runtime's own selector decide per module;
    # ~447 fleet modules ride on it.
    "VQ_MOE_GEMMSEG_CBDEV": ("auto", "device arm +20.9% prefill at d4-K2048"),
    # RTILE=64 is SLOWER everywhere measured (0.75-0.97x). DO NOT SET 64.
    "VQ_MOE_GEMMSEG_RTILE": ("32", "64 is 0.75-0.97x, never faster"),
    # +5.1-6.6% prefill, bit-exact.
    "VQ_GEMMSEG_OTILE64": ("1", "+5.1-6.6% prefill, bit-exact"),
    # the v2 stack reaches +11.9% over shipped.
    "VQ_GEMMSEG_PH2V": ("1", "part of the +11.9% stack"),
    "VQ_D4_WALK": ("1", "part of the +11.9% stack"),
    # Arm 1.5 measured NEGATIVE (-1.8-2%).
    "VQ_GEMMSEG_PIPE": ("0", "measured -1.8-2%"),
}

# Numerics-active flags: family-local, up to +0.97% ppl.
# v1.5 = both off (bit-exact vs the published arc6 runtime); v2 = both on.
#
# A RUNG'S NUMERICS ARE THE RUNG'S. What a released rung computes
# with is what its PUBLISHED model.py defaults to, and that is not uniform:
# Flash-Next 2.1 and Qwen3.6-35B-A3B 3.8/4.6/5.4 shipped v2, the rest v1.5 or
# the arc6-era runtime with no flags at all (docs/design/vq-rung-knobs.md,
# read off the Hub 2026-09-23). This table used to be applied to EVERY VQ
# artifact with v1.5 as the default, which forced the v2 rungs' two flags
# to 0 -- a numerics change nobody asked for, on exactly the rungs whose
# weights were fitted under v2. So the resolver now takes a rung's numerics
# from the rung (NUMERICS_SOURCES, in order) and applies a profile ONLY when
# a person names one.
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
#              model.py (never a local copy: those may drift)
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
#   * VQ_MOE_GEMMSEG_RTILE=64 is 0.75-0.97x and never faster, so no
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
        "VQLAB_CACHE_LIMIT_GB": 8.0,
        "decode_chunk_scale": 1.0,   # capped: smaller is already faster
        "launch": {"mtp": "on", "mtp_dynamic": "on", "kv_bits": "bf16",
                   "cross_chip": "off"},
        "why": "spends headroom where it actually buys speed -- a "
               "larger reclaimable cache, MTP with its dynamic controller, "
               "bf16 KV. It does NOT raise the decode chunk, because smaller "
               "is faster there as well as smaller in memory",
    },
    "stable": {
        "VQLAB_PREFILL_CHUNK": PREFILL_CHUNK_TIGHT,
        "VQLAB_CACHE_LIMIT_GB": 2.0,
        "decode_chunk_scale": 0.5,   # more room left for a step's spike
        "launch": {"cross_chip": "on", "mtp": "on", "mtp_dynamic": "off",
                   "kv_bits": "bf16"},
        "why": "repeatable: 512-token prompt chunks, identical rounding "
               "across chips, MTP drafting every step (no controller "
               "switching regimes) for steady timing, bf16 KV, and "
               "conservative memory -- a smaller reclaimable cache and a "
               "transient bounded tighter than headroom requires",
    },
    "lean": {
        "VQLAB_PREFILL_CHUNK": PREFILL_CHUNK_TIGHT,
        "launch": {"kv_bits": "8", "mtp": "off"},
        "why": "most context and most agents: 8-bit KV where the family "
               "takes it (bf16 where it does not, and said), 512-token "
               "prompt chunks, MTP off so the head's memory is free",
    },
}

#: The launch presets ARE the tune axis: one named bundle per value, the
#: default "balanced" (the measured defaults, unchanged). A per-model
#: KNURLOGIC_PRESET (Settings -> Models) picks one for that base model;
#: any explicit knob set beside it beats the preset's value for that knob.
PRESETS = ("balanced", "fast", "stable", "lean", "safe")
PRESET_DEFAULT = "balanced"


#: Each preset in plain words, for the Knurlogic tab: what it trades, what
#: it changes, and who should pick it. Order is the order shown.
PRESET_GUIDE = {
    "balanced": {
        "title": "Balanced",
        "trades": "Neither extreme: the measured defaults. Leaves speed on "
                  "the table on a roomy machine, and more memory in use "
                  "than lean or safe on a tight one.",
        "changes": "Prompt chunk read from the room free at launch: the "
                   "widest up to the family's measured best whose step "
                   "spike fits 10% of the room left, else 512 (35B-A3B "
                   "on a roomy M4: 2048, ~1.8x prefill). 4 GiB cache, "
                   "MTP as the family ships it.",
        "who": "Most people. Start here and move only for a reason.",
    },
    "fast": {
        "title": "Fast",
        "trades": "Memory headroom for speed: faster replies, less room "
                  "left for long contexts and parallel agents; timing "
                  "varies as dynamic MTP switches.",
        "changes": "The same room-read prompt chunk as balanced (its "
                   "larger cache leaves slightly less room for it), an "
                   "8 GiB reclaimable cache, MTP with its dynamic "
                   "controller, bf16 KV cache, no cross-chip padding.",
        "who": "One person, one conversation at a time, on a machine with "
               "memory to spare.",
    },
    "stable": {
        "title": "Stable",
        "trades": "Some speed for repeatability: the same answer and "
                  "steady timing, run after run and across machines. Costs "
                  "cross-chip padding (+2-6% on small matmuls) and drafts "
                  "even where a plain step is cheaper.",
        "changes": "512-token prompt chunks whatever the room, identical "
                   "rounding across chips, MTP drafting every step (no "
                   "controller), bf16 KV, a 2 GiB reclaimable cache and a "
                   "memory transient bounded tighter than needed.",
        "who": "Clusters of mixed Macs, benchmarks, and anyone debugging "
               "or comparing outputs.",
    },
    "lean": {
        "title": "Lean",
        "trades": "Some speed and a little precision for capacity: the "
                  "most context and the most agents at once. 8-bit KV "
                  "decodes ~7% slower at 6k tokens of context, ~18% at "
                  "16k (M4); with MTP off every step is a plain one.",
        "changes": "8-bit KV cache where the family takes it (bf16 where "
                   "not), 512-token prompt chunks whatever the room, the "
                   "default 4 GiB cache, MTP off so its memory is free.",
        "who": "Long documents, many parallel agents, or a big model on a "
               "machine it only just fits.",
    },
    "safe": {
        "title": "Safe",
        "trades": "Speed for the lowest peak memory: the least likely to "
                  "run the machine out of memory, and the slowest -- a "
                  "small cache and a tight transient give back speed "
                  "headroom would have bought.",
        "changes": "512-token prompt chunks whatever the room, a 1 GiB "
                   "reclaimable cache, and a memory transient bounded "
                   "tighter than needed; MTP and KV as the family ships "
                   "them.",
        "who": "A machine that also does other work, or after a load has "
               "run out of memory.",
    },
}


def preset_of(v, default: str = PRESET_DEFAULT) -> str:
    s = str(v or "").strip().lower()
    if not s:
        return default
    if s not in TUNE_PROFILES:
        raise ValueError(f"preset {v!r}: one of {list(PRESETS)}")
    return s


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
        "512 (safe, stable, lean: always 512). Output is identical at "
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
        "answer is exact either way. Off is for A/B -- check /status.json kv_kernel hits vs misses to see "
        "which path is actually live (GLM's MLA latent and gemma4's "
        "KV-shared layers always take dequantize + attention). A row with "
        "every key masked returns 0 here where mlx sdpa returns NaN. "
        "Prefill is dequantize + attention either way."),
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
        "launch preset for this model: balanced, fast, stable, lean (or "
        "safe) -- the knurlogic strategy unless set here",
        "one named bundle of the settings below. fast buys speed with "
        "memory headroom; stable buys repeatability with some speed; lean "
        "buys context and agents with some speed and precision (8-bit KV). "
        "Any setting changed beside it beats the preset's value."),
    "VQLAB_CACHE_LIMIT_GB": (
        "how much freed-buffer cache the runtime may hold",
        "larger keeps more freed buffers for reuse, but they stay resident: "
        "memory a long context or another agent cannot use. Smaller frees "
        "it, with no measured speed cost at 26k-token prefill -- the "
        "biggest single win in the memory playbook."),
    "VQ_MOE_GEMMSEG_CBDEV": (
        "where the codebook lives during the MoE GEMM",
        "the device arm is +20.9% on prefill at d4-K2048, same "
        "output; 'auto' lets the runtime choose per module. Forcing an arm "
        "risks the slower one on modules it does not suit."),
    "VQ_MOE_GEMMSEG_RTILE": (
        "row tile width in the segmented GEMM",
        "no trade -- 64 is 0.75-0.97x and NEVER faster. The one "
        "'win' was an env-ordering bug that benchmarked 32 twice."),
    "VQ_GEMMSEG_OTILE64": (
        "64-wide output tiling in the segmented GEMM",
        "+5.1-6.6% prefill, bit-exact; no measured cost, so on."),
    "VQ_GEMMSEG_PH2V": ("phase-2 vectorization",
                        "part of the +11.9% stack; off gives that "
                        "speed back, no measured gain."),
    "VQ_D4_WALK": ("d4 codebook walk",
                   "part of the +11.9% stack; off gives that speed "
                   "back, no measured gain."),
    "VQ_GEMMSEG_PIPE": ("software pipelining in the segmented GEMM",
                        "on costs 1.8-2% and buys nothing. "
                        "Off."),
    "VQ_GEMMSEG_BF16IO": ("bf16 IO in the segmented GEMM",
                          "numerics-active: changing it changes "
                          "the output, up to +0.97% ppl, on weights fitted "
                          "the other way. Each rung keeps what it shipped: "
                          "on for the v2 rungs, off for v1.5."),
    "VQ_DECODE_BF16IO": ("bf16 IO on the decode path",
                         "numerics-active: changing it changes "
                         "the output (up to +0.97% ppl). Each rung keeps "
                         "what it shipped."),
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
    "context_length": ("KNURLOGIC_CONTEXT_LENGTH",),
    "mtp": ("KNURLOGIC_MTP",),
    "mtp_dynamic": ("KNURLOGIC_MTP_DYNAMIC",),
    "kv_bits": ("KNURLOGIC_KV_BITS",),
    # the 8-bit KV decode kernel (engine/kvattn): on unless "off"; A/B knob
    "kv_kernel": ("KNURLOGIC_KV_KERNEL",),
    "cross_chip": ("KNURLOGIC_CROSS_CHIP",),
    "long_context": ("KNURLOGIC_LONG_CONTEXT",),
    "preset": ("KNURLOGIC_PRESET",),
}

#: launch settings of the model itself, read when it loads: the same on
#: every rank of a split (cluster/launch passes them ring-wide)
MODEL_KNOBS = ("KNURLOGIC_MTP", "KNURLOGIC_MTP_DYNAMIC", "KNURLOGIC_KV_BITS",
               "KNURLOGIC_KV_KERNEL", "KNURLOGIC_CROSS_CHIP",
               "KNURLOGIC_LONG_CONTEXT", "KNURLOGIC_PRESET")


# --- which knobs the ENGINE consumes ----------------------------------------
# Most knobs are read by an artifact's bundled runtime. These are not: the
# scheduler takes the prompt chunk directly (interfaces/http.scheduler_
# options), and the buffer cache is a process-global mlx setting. Emitting
# them as environment variables and stopping there is how they were, for a
# while, settings that did nothing -- the resolver explained a prompt chunk
# the server never saw.
ENGINE_KNOB_NAMES = tuple(n for k in ("prefill_chunk", "cache_limit_gb",
                                      "context_length", "mtp",
                                      "mtp_dynamic", "kv_bits", "kv_kernel",
                                      "cross_chip", "long_context",
                                      "preset")
                          for n in KNOB_ALIASES[k])


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


def kv_bits_of(v):
    """'bf16'/''/None -> None; '8'/'6'/'4' -> int; anything else raises.
    The same reading as engine/kvquant.parse_bits, without mlx."""
    s = str(v if v is not None else "").strip().lower()
    if s in ("", "bf16", "16", "none", "off"):
        return None
    if s not in KV_BITS_VALUES:
        raise ValueError(f"KV bits {v!r}: one of {KV_BITS_VALUES}")
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

#: Logical knobs whose LEGACY name is read by published bundled runtimes
#: (VQLAB_CACHE_LIMIT_GB: 24 of the 37). The prompt chunk is not one of
#: them: no bundled runtime reads VQLAB_PREFILL_CHUNK -- only the engine
#: does (engine_settings, under either name) -- so it is emitted under
#: knurlogic's own name, and the legacy one is only still ACCEPTED (a saved
#: launch setting, a --set, a ring spec may carry it).
LEGACY_EMITTED = ("cache_limit_gb",)


def canonical_sets(sets: dict) -> dict:
    """Explicit settings with an accepted legacy name moved to the name
    the resolver emits (VQLAB_PREFILL_CHUNK -> KNURLOGIC_PREFILL_CHUNK),
    so an explicit value cannot lose to the resolver's under the other
    alias (engine_settings takes the first alias present). Where both are
    given, knurlogic's own name wins."""
    out = dict(sets or {})
    for logical, names in KNOB_ALIASES.items():
        if logical in LEGACY_EMITTED:
            continue
        for old in names[1:]:
            if old in out:
                v = out.pop(old)
                out.setdefault(names[0], v)
    return out


#: When no bundled runtime can be asked, emit this one. For a knob in
#: LEGACY_EMITTED, the LAST alias: the legacy name is the one with artifacts
#: behind it, and a guess should fail towards what exists rather than
#: towards what is planned. Otherwise knurlogic's own (first) name.
def default_alias(logical: str) -> str:
    names = KNOB_ALIASES[logical]
    return names[-1] if logical in LEGACY_EMITTED else names[0]


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
                   "VQLAB_PREFILL_CHUNK",
                   "KNURLOGIC_CONTEXT_LENGTH") + MODEL_KNOBS


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
    # powers of two up to the longest window a released model has; the
    # control stops at the model's own (the resolver's value)
    "KNURLOGIC_CONTEXT_LENGTH": ([8192, 16384, 32768, 65536, 131072,
                                  262144, 524288, 1048576], "tokens"),
    "VQ_DECODE_CHUNK": ([4, 8, 16, 32], ""),
    "KNURLOGIC_MTP": (["on", "off"], ""),
    "KNURLOGIC_MTP_DYNAMIC": (["on", "off"], ""),
    # narrowed per family by the resolver (Resolution.ranges): a family
    # that refuses quantized KV offers bf16 alone
    "KNURLOGIC_KV_BITS": (KV_BITS_VALUES, "bits"),
    "KNURLOGIC_KV_KERNEL": (["on", "off"], ""),
    "KNURLOGIC_CROSS_CHIP": (["off", "on", "auto"], ""),
    # narrowed per family by the resolver: ["off"] where no model card
    # documents YaRN
    "KNURLOGIC_LONG_CONTEXT": (["off", "yarn"], ""),
    "KNURLOGIC_PRESET": (list(PRESETS), ""),
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
    text = cfg.get("text_config") if isinstance(
        cfg.get("text_config"), dict) else {}
    mpe = int(text.get("max_position_embeddings")
              or cfg.get("max_position_embeddings") or 0)
    rope = None
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
# on shorter texts". Sources (read 2026-09-29):
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
        raise ValueError(f"long context {v!r}: one of "
                         f"{list(LONG_CONTEXT_VALUES)}")
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
    factor, orig, _doc = LONG_CONTEXT_YARN[long_context_family(mt)]
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
            return (f"{name}={s!r}: not a "
                    f"{'whole number' if cast is int else 'number'}")
        if name == "KNURLOGIC_CONTEXT_LENGTH" and window:
            hi = window
        if v < lo or (hi is not None and v > hi):
            where = (f"this model's maximum is {hi:,} tokens" if
                     name == "KNURLOGIC_CONTEXT_LENGTH" and window else
                     f"between {lo:g} and {hi:g}{' ' + unit if unit else ''}"
                     if hi is not None else f"at least {lo:g}")
            return f"{name}={s}: {where}"
        return None
    if name in COMPACT_KNOBS:
        return check_compact_knob(name, value)
    try:
        if name in ("KNURLOGIC_MTP", "KNURLOGIC_MTP_DYNAMIC"):
            on_off(s)
        elif name == "KNURLOGIC_KV_BITS":
            kv_bits_of(s)
        elif name == "KNURLOGIC_PRESET":
            preset_of(s)
        elif name == "KNURLOGIC_CROSS_CHIP":
            cross_chip_of(s)
        elif name == "KNURLOGIC_LONG_CONTEXT":
            long_context_of(s)
    except ValueError as e:
        return f"{name}: {e}"
    return None

# --- what a VISION rung holds besides its weights ---------------------------
# A model with a vision tower needs three things a text model does not, and
# the resolver must count them BEFORE a load, not discover them as an OOM on the first screenshot:
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
        "further and leans harder on the summary, so more detail is lost. The kept tail never starts on a tool result (it is "
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

TUNES = ("balanced", "fast", "stable", "lean", "safe")  # the names of PRESETS
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
