"""Measured runtime constants, with the provenance that paid for them.

Every number here came off a run. The point of Knurlogic is that a downloader
should never have to know them: the resolver turns them into defaults. Each
constant carries the measurement that established it, so a future change
has to argue with a measurement rather than a preference. The VQ kernel
numbers were measured in the VQ runtime's upstream project.

Here: the decode and prefill chunks, the fit reserve, the cache limits,
low headroom, what one KV element costs and which KV precisions a family
takes (read off its manifest in engine/families), and what a vision rung
holds besides its weights.
"""

from __future__ import annotations

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
# for transients (tuning/fit.prefill_chunk_by_room).
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
# there (fit.fit_reserve): the first request's transient at the
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

#: Hard ceiling on the reclaimable cache, whatever the profile asks. Freed
#: buffers are reclaimable but they are still resident, and a cache larger
#: than this has never been measured to buy anything.
CACHE_LIMIT_GB_MAX = 16.0


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
