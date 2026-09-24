"""Gemma 4: the text model (gemma4_text) and the multimodal wrapper the
released rungs load through (gemma4). No MTP head.
"""

MANIFEST = {
    "name": "gemma4",
    "architectures": {
        "gemma4_text": {"model_types": ["gemma4_text"]},
        "gemma4": {"depends_on": ["gemma4_text"], "model_types": ["gemma4"]},
    },
    "vision": {"build": "knurlogic.engine.vision.gemma4:build",
               "architectures": ["gemma4"]},
}
