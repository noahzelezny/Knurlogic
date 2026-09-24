"""Qwen: Qwen3.5 / Qwen3.8 (qwen3_5), the MoE wrapper that is the 397B and
the 35B-A3B (qwen3_5_moe, subclasses qwen3_5), and Qwen3.8-Flash-Next
(qwen4_exp). One vision tower for all three; two MTP heads.
"""

# Prompt chunk 4096 -- MEASURED, not the default leaking through: an A/B on
# 2026-06-19 gave +115% prefill tok/s at 11k tokens vs a 512 cap, no
# peak-memory cost, bit-identical output. Hybrid attention, 45/60 layers
# recurrent, so there is no chunk x seq^2 transient to cap. It carries
# ArraysCache entries, which is why any "has SSM caches -> small chunk"
# heuristic catches it wrongly: a blanket SSM->512 on 2026-09-02 made its
# prefill 8x the chunks.
_QWEN35_PREFILL = (4096, "measured 2026-06-19 (see the comment above)")

# The Qwen3.5/3.8 head drafts from the activation going INTO the trunk's
# final norm (one residual stream).
#
# cache_semantics="reassign" is load-bearing for SPEED. The trunk's cache
# list is mostly recurrent (48 ArraysCache to 16 KVCache on the dense 27B,
# 45 to 15 on the 397B); under "copy" every GatedDeltaNet state is
# deep-copied ONCE PER SPECULATIVE STEP -- hundreds of MB per token, which
# does not fail, it crawls (a 12-prompt run made no visible progress in 30
# minutes). GatedDeltaNet reassigns its slots and mlx arrays are immutable,
# so the cheap path is correct: caches.check_snapshot_semantics returned
# True against a loaded 27B (2026-08-31).
_QWEN35_HEAD = dict(
    head="knurlogic.engine.families.qwen.heads.qwen35:MTPHeadQwen35",
    capture="norm", draft_cache="KVCache",
    sidecar_name="mtp-head-q6.safetensors",
    cache_semantics="reassign")

MANIFEST = {
    "name": "qwen",
    "architectures": {
        "qwen3_5": {
            "model_types": ["qwen3_5_text", "qwen3_5"],
            "prefill_chunk": _QWEN35_PREFILL,
            "head": dict(_QWEN35_HEAD, names=["qwen3_5"]),
        },
        "qwen3_5_moe": {
            "depends_on": ["qwen3_5"],
            "model_types": ["qwen3_5_moe_text", "qwen3_5_moe"],
            "prefill_chunk": _QWEN35_PREFILL,
            "head": dict(_QWEN35_HEAD, names=["qwen3_5_moe"]),
        },
        "qwen4_exp": {
            "model_types": ["qwen4_exp_text", "qwen4_exp"],
            "head": dict(
                names=["qwen4_exp"],
                head="knurlogic.engine.families.qwen.heads.qwen4_exp:MTPHead",
                # the activation INTO the hyper-connection mixer, the last
                # thing before the final norm + lm_head. Every qwen4_exp
                # cache slot is REASSIGNED, never mutated (verified by
                # caches.check_snapshot_semantics).
                capture="hyper_connection_mixer", draft_cache="_AttnCache",
                sidecar_name="mtp-head-q6.safetensors",
                cache_semantics="reassign"),
        },
    },
    # Thinking controls, per CHAT-TEMPLATE DIALECT (one module, qwen3_5,
    # serves both): read off the released templates 2026-09-24.
    "thinking": {
        # Qwen3.8 27B, Flash-Next: enable_thinking, then reasoning_effort
        # in {xhigh (default), medium, low}; anything else raises.
        "qwen_effort": {
            "detect": {"all": ["enable_thinking", "reasoning_effort",
                               "xhigh"]},
            "default": "xhigh",
            "native": [["none", "off", {"enable_thinking": False}],
                       ["low", "low", {"reasoning_effort": "low"}],
                       ["medium", "medium", {"reasoning_effort": "medium"}],
                       ["xhigh", "xhigh", {"reasoning_effort": "xhigh"}]],
        },
        # Qwen3.5 397B, Qwen3.6 35B-A3B: on or off, nothing graded.
        "qwen_toggle": {
            "detect": {"all": ["enable_thinking", "<think>"],
                       "none": ["reasoning_effort"]},
            "default": "on",
            "native": [["none", "off", {"enable_thinking": False}],
                       ["xhigh", "on", {"enable_thinking": True}]],
        },
    },
    "vision": {"build": "knurlogic.engine.families.qwen.vision:build",
               "architectures": ["qwen3_5", "qwen3_5_moe", "qwen4_exp"]},
}
