"""Gemma 4: the text model (gemma4_text) and the multimodal wrapper the
released rungs load through (gemma4). No MTP head.
"""

MANIFEST = {
    "name": "gemma4",
    "architectures": {
        "gemma4_text": {"model_types": ["gemma4_text"]},
        "gemma4": {"depends_on": ["gemma4_text"], "model_types": ["gemma4"]},
    },
    # Thinking is OFF unless the template gets enable_thinking=true, the
    # opposite default to Qwen's (read off the e4b and 26b templates).
    "thinking": {
        "gemma_toggle": {
            "detect": {"all": ["enable_thinking", "<|think|>"]},
            "default": "off",
            "native": [["none", "off", {"enable_thinking": False}],
                       ["xhigh", "on", {"enable_thinking": True}]],
        },
    },
    # A text-config artifact reports gemma4_text; its images are the same
    # family's, so both modules name the builder.
    "vision": {"build": "knurlogic.engine.families.gemma4.vision:build",
               "architectures": ["gemma4", "gemma4_text"]},
}
