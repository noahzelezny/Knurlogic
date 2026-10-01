from .config import ModelConfig, TextConfig, VisionConfig
from .glm5_next import Model
from .language import LanguageModel
from .vision import VisionModel

__all__ = [
    "ModelArgs",
    "Model",
    "ModelConfig",
    "TextConfig",
    "VisionConfig",
    "LanguageModel",
    "VisionModel",
]

# mlx-lm's loader (`mlx_lm.utils._get_classes`) asks the architecture module
# for `Model` and `ModelArgs`; this package is mlx-vlm-shaped and names its
# config `ModelConfig` (a from_dict config over the whole config.json, text
# and vision halves nested). The same object, so every rank's text host --
# rank 0 and a follower alike -- builds the same class from the same config.
ModelArgs = ModelConfig
