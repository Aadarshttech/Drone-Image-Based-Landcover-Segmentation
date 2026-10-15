"""
DeepLabV3 Model Architecture for Landcover Segmentation.
Supports configurable backbone, output classes, and auxiliary classifiers.
"""

from typing import Optional
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights


class LandcoverDeepLabV3(nn.Module):
    """
    Wrapper for DeepLabV3 with ResNet50 backbone modified for multi-class drone segmentation.
    """
    def __init__(self, num_classes: int = 3, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        self.num_classes = num_classes

        # Load backbone with pre-trained weights if requested
        if pretrained:
            try:
                weights = DeepLabV3_ResNet50_Weights.DEFAULT
                self.model = deeplabv3_resnet50(weights=weights)
            except Exception:
                # Fallback for older torchvision versions
                self.model = deeplabv3_resnet50(pretrained=True)
        else:
            self.model = deeplabv3_resnet50(weights=None)

        # Optionally freeze backbone feature extractor
        if freeze_backbone:
            for param in self.model.backbone.parameters():
                param.requires_grad = False

        # Replace final classification heads
        in_channels = 256
        self.model.classifier[-1] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        if self.model.aux_classifier is not None:
            self.model.aux_classifier[-1] = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        return self.model(x)


def build_deeplabv3(num_classes: int = 3, pretrained: bool = True, freeze_backbone: bool = False) -> LandcoverDeepLabV3:
    """
    Factory function to construct a configured LandcoverDeepLabV3 model.
    """
    return LandcoverDeepLabV3(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )


def prepare_model(num_classes: int = 3, pretrained: bool = True):
    """
    Backward-compatible function matching original codebase signature.
    Default updated to num_classes=3 to align with dataset classes (Background, Building, Woodland).
    """
    try:
        weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        model = deeplabv3_resnet50(weights=weights)
    except Exception:
        model = deeplabv3_resnet50(pretrained=pretrained)

    model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
    if model.aux_classifier is not None:
        model.aux_classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)

    return model
