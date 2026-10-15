"""
Legacy wrapper for DeepLabV3 model initialization.
Maintained for backward compatibility. Direct imports from `src.model` are recommended.
"""

import torch.nn as nn
from src.model import prepare_model, build_deeplabv3, LandcoverDeepLabV3

__all__ = ["prepare_model", "build_deeplabv3", "LandcoverDeepLabV3"]

if __name__ == "__main__":
    print("Testing DeepLabV3 model construction...")
    model = prepare_model(num_classes=3)
    print("Model initialized successfully:", type(model).__name__)
