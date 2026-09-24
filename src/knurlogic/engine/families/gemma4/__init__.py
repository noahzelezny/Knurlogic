"""Gemma 4: the text model (gemma4_text) and the multimodal wrapper the
released rungs load through (gemma4). No MTP head.
"""

MANIFEST = {
    "name": "gemma4",
    "architectures": {
        "gemma4_text": {"model_types": ["gemma4_text"]},
        "gemma4": {"depends_on": ["gemma4_text"], "model_types": ["gemma4"]},
    },
    # A text-config artifact reports gemma4_text; its images are the same
    # family's, so both modules name the builder.
    "vision": {"build": "knurlogic.engine.families.gemma4.vision:build",
               "architectures": ["gemma4", "gemma4_text"]},
}
