"""DeepSeek-V4 (deepseek_v4): MLA-style shared-KV attention over a
128-token sliding window plus learned compressed pools (Compressor, and a
top-k Indexer on the ratio-4 layers), hash-routed first layers, mHC
hyper-connections. No MTP head (sanitize drops `mtp.*`), no vision.

The chat template is engine/templates/deepseek_v4.jinja (the conversion
ships a stub); its thinking control (thinking_mode / enable_thinking) is
not declared here as a dialect yet.
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
        },
    },
    "vision": None,
}
