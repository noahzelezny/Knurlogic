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
