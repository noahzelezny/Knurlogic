"""DeepSeek-V4 (deepseek_v4): MLA-style shared-KV attention over a
128-token sliding window plus learned compressed pools (Compressor, and a
top-k Indexer on the ratio-4 layers), hash-routed first layers, mHC
hyper-connections. One MTP head (heads/deepseek_v4.py, a sidecar beside
the trunk; the trunk's sanitize still drops `mtp.*`); Vision-Exp's DSpark
block drafter instead (heads/deepseek_v4_dspark.py, its own sidecar).
Images: the DeepSeek-V4-Flash-Vision-Exp artifact (vision_n_layers > 0),
its tower in vision/ (docs/design/deepseek-vision.md).

The chat template is engine/templates/deepseek_v4.jinja (the conversion
ships a stub); its thinking levels are DeepSeek's three modes: Non-think,
Think High, Think Max (the official prefix), the "deepseek_effort" dialect.
Vision-Exp's variant has four (off, low, high, max): "deepseek_vision_effort".
"""

MANIFEST = {
    "name": "deepseek",
    "architectures": {
        # KV precision: every layer's cache is the module's own
        # DeepseekV4Cache (a bf16 RotatingKVCache window + compressor and
        # indexer pools), which engine/kvquant.py does not know how to
        # quantize. Refused until someone measures it.
        "deepseek_v4": {
            "model_types": ["deepseek_v4"],
            "kv_quant": {"refused": (
                "deepseek_v4 caches through its own DeepseekV4Cache (a "
                "128-token bf16 window plus compressed pools); knurlogic's "
                "KV quantization does not apply to it")},
            "head": dict(
                names=["deepseek_v4"],
                head="knurlogic.engine.families.deepseek.heads."
                     "deepseek_v4:MTPHead",
                # the [B, S, hc, D] streams INTO the trunk's hc_head: the
                # official MTPBlock takes all hc streams, not the collapse
                capture="hc_head", draft_cache="DeepseekV4Cache",
                sidecar_name="mtp-head-mxfp4.safetensors",
                # DeepseekV4Cache cannot trim (its pools); caches.py rolls an
                # untrimmable cache back by its whole state
                cache_semantics="copy",
                # every sidecar key is under mtp.0. (the official names)
                layout=("mtp",),
                # Vision-Exp's DSpark (3 stages under mtp.*, block 5):
                # drafted by engine/mtp/block_loop, captured at the outputs
                # of dspark_target_layer_ids (docs/design/deepseek-vision.md)
                block=dict(
                    head="knurlogic.engine.families.deepseek.heads."
                         "deepseek_v4_dspark:DSparkHead",
                    sidecar_name="mtp-head-dspark-mxfp4.safetensors",
                    config="dspark_block_size")),
        },
    },
    "thinking": {
        "deepseek_effort": {
            "detect": {"all": ["thinking_mode", "Reasoning Effort",
                               "enable_thinking"]},
            "default": "high",
            "native": [["none", "off", {"thinking_mode": "chat"}],
                       ["high", "high", {"thinking_mode": "thinking"}],
                       ["xhigh", "max", {"thinking_mode": "thinking",
                                         "reasoning_effort": "max"}]],
        },
        # DeepSeek-V4-Flash-Vision-Exp: four levels, its encoder's
        # REASONING_EFFORT_PROMPTS ("low", its default, adds no prefix);
        # its variant's first line sets dsv4_vision, so it is detected first
        "deepseek_vision_effort": {
            "detect": {"all": ["set dsv4_vision = true", "thinking_mode",
                               "Reasoning Effort", "enable_thinking"]},
            "default": "low",
            "native": [["none", "off", {"thinking_mode": "chat"}],
                       ["low", "low", {"thinking_mode": "thinking",
                                       "reasoning_effort": "low"}],
                       ["high", "high", {"thinking_mode": "thinking",
                                         "reasoning_effort": "high"}],
                       ["xhigh", "max", {"thinking_mode": "thinking",
                                         "reasoning_effort": "max"}]],
        },
    },
    "vision": {"build": "knurlogic.engine.families.deepseek.vision:build",
               "architectures": ["deepseek_v4"]},
}
