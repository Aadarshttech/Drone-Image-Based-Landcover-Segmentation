"""
Drone Image-Based Landcover Segmentation Package
"""

from .model import build_deeplabv3, prepare_model
from .dataset import LandcoverDataset, NepalDataset, NepalDataGenerator, get_dataloader
from .metrics import compute_iou, compute_metrics
from .utils import create_color_mask, overlay_mask_on_image, COLORMAP_DEFAULT

__version__ = "1.0.0"
__all__ = [
    "build_deeplabv3",
    "prepare_model",
    "LandcoverDataset",
    "NepalDataset",
    "NepalDataGenerator",
    "get_dataloader",
    "compute_iou",
    "compute_metrics",
    "create_color_mask",
    "overlay_mask_on_image",
    "COLORMAP_DEFAULT",
]
