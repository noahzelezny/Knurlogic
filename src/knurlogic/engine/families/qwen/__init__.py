"""Qwen: Qwen3.5 / Qwen3.8 (qwen3_5), the MoE wrapper that is the 397B and
the 35B-A3B (qwen3_5_moe, subclasses qwen3_5), and Qwen3.8-Flash-Next
(qwen4_exp). One vision tower for all three; two MTP heads.
"""

_QWEN35_PREFILL = (4096, "measured 2026-06-19: +115% prefill tok/s at 11k "
                         "tokens vs a 512 cap, no peak-memory cost, "
                         "bit-identical output; 45/60 layers recurrent, so "
                         "no chunk x seq^2 transient to cap")

_QWEN35_HEAD = dict(
    head="knurlogic.engine.mtp.heads.qwen35:MTPHeadQwen35",
    capture="norm", draft_cache="KVCache",
    sidecar_name="mtp-head-q6.safetensors",
    # load-bearing for speed: 'copy' deep-copies every GatedDeltaNet state
    # per speculative step. check_snapshot_semantics True on a loaded 27B
    # (2026-08-31).
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
                head="knurlogic.engine.mtp.heads.qwen4_exp:MTPHead",
                # the activation INTO the hyper-connection mixer
                capture="hyper_connection_mixer", draft_cache="_AttnCache",
                sidecar_name="mtp-head-q6.safetensors",
                cache_semantics="reassign"),
        },
    },
    "vision": {"build": "knurlogic.engine.vision.qwen:build",
               "architectures": ["qwen3_5", "qwen3_5_moe", "qwen4_exp"]},
}
