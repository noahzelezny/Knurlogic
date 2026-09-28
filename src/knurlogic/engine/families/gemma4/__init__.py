"""Gemma 4: the text model (gemma4_text) and the multimodal wrapper the
released rungs load through (gemma4). No MTP head.
"""

# KV precision: the full-attention layers' KVCache is stored quantized
# (engine/kvquant.py); the sliding-window layers (RotatingKVCache) are
# bounded by their window and stay bf16; the KV-shared layers reuse the
# dequantized arrays their source layer returns. Unmeasured on a real model.
_KVQ = {"bits": [8, 6, 4],
        "why": "full-attention layers only; sliding-window layers are "
               "bounded by their window and stay bf16. 8 is the "
               "recommendation"}

MANIFEST = {
    "name": "gemma4",
    "architectures": {
        "gemma4_text": {"model_types": ["gemma4_text"],
                        "kv_quant": _KVQ},
        "gemma4": {"depends_on": ["gemma4_text"], "model_types": ["gemma4"],
                   "kv_quant": _KVQ},
    },
    # The TEMPLATE defaults off (enable_thinking must be true), but mlx-lm
    # passes enable_thinking=True to any request that is silent about it
    # when the tokenizer has thinking tokens -- gemma's does -- so SERVED,
    # the default is on. `default` states what knurlogic serves; the render
    # probe (engine/serve/thinking.probe) confirms it on the loaded model.
    "thinking": {
        "gemma_toggle": {
            "detect": {"all": ["enable_thinking", "<|think|>"]},
            "default": "on",
            "native": [["none", "off", {"enable_thinking": False}],
                       ["xhigh", "on", {"enable_thinking": True}]],
        },
    },
    # A text-config artifact reports gemma4_text; its images are the same
    # family's, so both modules name the builder.
    "vision": {"build": "knurlogic.engine.families.gemma4.vision:build",
               "architectures": ["gemma4", "gemma4_text"]},
}
