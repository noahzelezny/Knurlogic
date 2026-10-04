"""Qwen: Qwen3.5 / Qwen3.8 (qwen3_5), the MoE wrapper that is the 397B and
the 35B-A3B (qwen3_5_moe, subclasses qwen3_5), and Qwen3.8-Flash-Next
(qwen4_exp). One vision tower for all three; two MTP heads.
"""

# Prompt chunk 4096 -- MEASURED, not the default leaking through: an A/B
# gave +115% prefill tok/s at 11k tokens vs a 512 cap, no
# peak-memory cost, bit-identical output. Hybrid attention, 45/60 layers
# recurrent, so there is no chunk x seq^2 transient to cap. It carries
# ArraysCache entries, which is why any "has SSM caches -> small chunk"
# heuristic catches it wrongly: a blanket SSM->512 makes its prefill 8x
# the chunks.
_QWEN35_PREFILL = (4096, "measured: +115% prefill tok/s at 11k tokens vs "
                         "512, no peak-memory cost")

# qwen3_5_moe (the 35B-A3B and the 397B): 2048 is the measured best. M4 Max
# 128 GB, Qwen3.6-35B-A3B VQ 3.4 (13.8 GiB), 28,727-token
# prompt, interleaved order, n=3, prefill tok/s:
#   512: 642.7 / 642.6 / 649.5
#   2048: 1182.9 / 1164.5 / 1141.9
#   4096: 1173.2 / 1166.5 / 1133.8
# ~1.8x over 512 at 2048 and nothing more at 4096 (the 397B's 4096 was +2%
# over 2048 for a ~2x larger step transient). A cap, not a
# default: the resolver takes it only where the room free at launch holds
# its step transient.
_QWEN35_MOE_PREFILL = (2048, "measured on an M4 Max: 35B-A3B VQ 3.4 "
                             "at 28.7k tokens, 512 645 / 2048 1163 / 4096 "
                             "1158 tok/s (n=3)")

# qwen4_exp (Qwen3.8-Flash-Next): 2048 is the measured best. M4 Max 128 GB,
# Flash-Next VQ 4.4 (96.6 GiB), 28,727-token prompt, interleaved
# order, n=3, prefill tok/s:
#   512: 383.4 / 360.6 / 387.1   (peak 100.9 GiB)
#   1024: 439.8 / 432.4 / 409.6  (peak 101.3 GiB)
#   2048: 477.8 / 454.8 / 439.2  (peak 103.5 GiB)
# ~+19% at 2048 for +2.6 GiB of peak; the experts dominate its prefill, so
# width buys less than on the 35B. 4096 not measured.
_QWEN4_EXP_PREFILL = (2048, "measured on an M4 Max: Flash-Next VQ 4.4 "
                            "at 28.7k tokens, 512 383 / 1024 432 / 2048 "
                            "455 tok/s (n=3)")

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
# True against a loaded 27B.
_QWEN35_HEAD = dict(
    head="knurlogic.engine.families.qwen.heads.qwen35:MTPHeadQwen35",
    capture="norm", draft_cache="KVCache",
    sidecar_name="mtp-head-q6.safetensors",
    cache_semantics="reassign",
    # a sidecar's top-level tensor prefixes: how a head is recognised when
    # its metadata does not say (measured from the sidecars on disk)
    layout=("block", "fc", "norm_e", "norm_h", "norm_out"))

# KV precision: the full-attention layers are mlx-lm's plain KVCache, which
# engine/kvquant.py stores quantized; the GatedDeltaNet layers' recurrent
# state (ArraysCache) is not KV and stays as it is. Not measured on a real
# model yet (tiny fixtures only: tests/test_kvquant.py).
_QWEN35_KVQ = {"bits": [8, 6, 4],
               "why": "full-attention layers only (a quarter of the "
                      "layers); the deltanet state is not KV and stays "
                      "bf16. 8 is the recommendation"}

# The sampling Qwen publishes for thinking OFF, which generation_config.json
# (one set, the thinking one) cannot carry; machine/artifact.py's
# sampling_defaults reports it as "non_thinking". Sources:
# Qwen3.5: https://huggingface.co/Qwen/Qwen3.5-397B-A17B#best-practices
# (exo's cards for these builds carry the same set); Qwen3.8-Flash-Next
# (qwen4_exp) publishes the same set in its README.
_NON_THINKING = {"temp": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
                 "presence_penalty": 1.5}

MANIFEST = {
    "name": "qwen",
    "architectures": {
        "qwen3_5": {
            "model_types": ["qwen3_5_text", "qwen3_5"],
            "prefill_chunk": _QWEN35_PREFILL,
            "kv_quant": _QWEN35_KVQ,
            "head": dict(_QWEN35_HEAD, names=["qwen3_5"]),
            # a pipeline split: the trunk core's class (qwen3_5_moe's trunk
            # is this one's), and what re-indexes it to a stage's layers
            "pipeline": {"core": "Qwen3_5TextModel",
                         "restage": "knurlogic.engine.families.qwen."
                                    "pipeline_stage:restage_qwen3_5"},
        },
        "qwen3_5_moe": {
            "depends_on": ["qwen3_5"],
            "model_types": ["qwen3_5_moe_text", "qwen3_5_moe"],
            "prefill_chunk": _QWEN35_MOE_PREFILL,
            "kv_quant": _QWEN35_KVQ,
            "head": dict(_QWEN35_HEAD, names=["qwen3_5_moe"]),
        },
        "qwen4_exp": {
            "model_types": ["qwen4_exp_text", "qwen4_exp"],
            "prefill_chunk": _QWEN4_EXP_PREFILL,
            "pipeline": {"core": "Qwen4ExpModel",
                         "restage": "knurlogic.engine.families.qwen."
                                    "pipeline_stage:restage_qwen4_exp"},
            # its attention cache is its own (_AttnCache/_BatchAttnCache:
            # K/V plus the sparse indexer's keys and positions, moved
            # together); kvcache.py's subclasses store the K/V quantized and
            # keep the indexer's keys exact
            "kv_quant": {"bits": [8, 6, 4],
                         "why": "full-attention K/V only; the QSA indexer's "
                                "keys, the deltanet state and the PLE / "
                                "n-gram slots stay bf16",
                         "caches": {"_AttnCache": "knurlogic.engine.families."
                                    "qwen.kvcache:QuantAttnCache"}},
            "head": dict(
                names=["qwen4_exp"],
                head="knurlogic.engine.families.qwen.heads.qwen4_exp:MTPHead",
                # the activation INTO the hyper-connection mixer, the last
                # thing before the final norm + lm_head. Every qwen4_exp
                # cache slot is REASSIGNED, never mutated (verified by
                # caches.check_snapshot_semantics).
                capture="hyper_connection_mixer", draft_cache="_AttnCache",
                sidecar_name="mtp-head-q6.safetensors",
                cache_semantics="reassign",
                # the qwen4_exp packs predate the `family` metadata field
                layout=("block", "fc", "mixer", "norm_e", "norm_h")),
        },
    },
    # Thinking controls, per CHAT-TEMPLATE DIALECT (one module, qwen3_5,
    # serves both): read off the released templates.
    "thinking": {
        # Qwen3.8 27B, Flash-Next: enable_thinking, then reasoning_effort
        # in {xhigh (default), medium, low}; anything else raises.
        "qwen_effort": {
            "detect": {"all": ["enable_thinking", "reasoning_effort"]},
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
               "architectures": ["qwen3_5", "qwen3_5_moe", "qwen4_exp"],
               # how an artifact shows it has a tower: a nested
               # vision_config; the tower's tensors under these prefixes
               # (the loader's namings, vision/)
               "signature": {"config": "vision_config",
                             "tower_prefixes": [
                                 "visual.", "model.visual.",
                                 "model.language_model.visual.",
                                 "vision_tower."]}},
    "sampling": {"non_thinking": {
        t: _NON_THINKING for t in ("qwen3_5", "qwen3_5_moe", "qwen4_exp")}},
}
