"""Gemma 4: the text model (gemma4_text) and the multimodal wrapper the
released rungs load through (gemma4). No MTP head.
"""

MANIFEST = {
    "name": "gemma4",
    "architectures": {
        "gemma4_text": {"model_types": ["gemma4_text"]},
        "gemma4": {"depends_on": ["gemma4_text"], "model_types": ["gemma4"]},
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
