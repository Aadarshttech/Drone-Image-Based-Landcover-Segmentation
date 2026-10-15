"""
Visualization, colormap handling, and GIS geometric utilities.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Default Landcover Colormap
# 0: Background/Unlabeled (Black)
# 1: Buildings (Red)
# 2: Woodland/Vegetation (Green)
COLORMAP_DEFAULT: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),        # Background / Roads / Soil
    1: (255, 0, 0),      # Buildings / Built-up
    2: (0, 255, 0),      # Woodland / Vegetation
}

CLASS_NAMES: Dict[int, str] = {
    0: "Background / Unclassified",
    1: "Buildings / Structures",
    2: "Woodland / Trees",
}


def create_color_mask(
    mask: np.ndarray,
    colormap: Optional[Dict[int, Tuple[int, int, int]]] = None
) -> np.ndarray:
    """
    Converts a single-channel integer mask [H, W] or multi-channel mask [C, H, W]
    into an RGB image [H, W, 3].
    """
    cmap = colormap or COLORMAP_DEFAULT

    if mask.ndim == 3 and mask.shape[0] in (2, 3, 4):
        # One-hot or multi-channel mask [C, H, W] -> argmax to [H, W]
        mask = np.argmax(mask, axis=0)

    h, w = mask.shape[-2], mask.shape[-1]
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in cmap.items():
        color_mask[mask == class_id] = color

    return color_mask


def overlay_mask_on_image(
    image: Union[np.ndarray, Image.Image],
    mask: np.ndarray,
    alpha: float = 0.45,
    colormap: Optional[Dict[int, Tuple[int, int, int]]] = None,
    ignore_background: bool = True
) -> np.ndarray:
    """
    Overlays a segmentation mask on top of an aerial/drone RGB image with alpha blending.
    """
    if isinstance(image, Image.Image):
        image_np = np.array(image.convert("RGB"))
    else:
        image_np = image.copy()
        if image_np.ndim == 3 and image_np.shape[0] == 3:  # CHW to HWC
            image_np = np.transpose(image_np, (1, 2, 0))

    if image_np.dtype != np.uint8:
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)

    color_mask = create_color_mask(mask, colormap=colormap)

    if mask.ndim == 3 and mask.shape[0] in (2, 3, 4):
        mask_2d = np.argmax(mask, axis=0)
    else:
        mask_2d = mask

    blended = image_np.astype(np.float32).copy()

    if ignore_background:
        # Only blend where class != 0
        foreground = (mask_2d > 0)
        blended[foreground] = (
            (1.0 - alpha) * image_np[foreground] + alpha * color_mask[foreground]
        )
    else:
        blended = (1.0 - alpha) * image_np + alpha * color_mask

    return np.clip(blended, 0, 255).astype(np.uint8)


def visualize_prediction(
    image: np.ndarray,
    mask_gt: Optional[np.ndarray] = None,
    mask_pred: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "Landcover Segmentation Results"
):
    """
    Renders side-by-side plots of Drone Image, Ground Truth, and Prediction Overlay.
    """
    num_cols = 1 + (1 if mask_gt is not None else 0) + (1 if mask_pred is not None else 0)
    fig, axes = plt.subplots(1, num_cols, figsize=(6 * num_cols, 6))

    if num_cols == 1:
        axes = [axes]

    idx = 0
    axes[idx].imshow(image)
    axes[idx].set_title("Input Drone Imagery")
    axes[idx].axis("off")
    idx += 1

    if mask_gt is not None:
        gt_color = create_color_mask(mask_gt)
        axes[idx].imshow(gt_color)
        axes[idx].set_title("Ground Truth Mask")
        axes[idx].axis("off")
        idx += 1

    if mask_pred is not None:
        overlay = overlay_mask_on_image(image, mask_pred)
        axes[idx].imshow(overlay)
        axes[idx].set_title("Prediction Overlay")
        axes[idx].axis("off")

    plt.suptitle(title, fontsize=14, weight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved visualization to: {save_path}")
    plt.close(fig)


# ==============================================================================
# Backward Compatibility GIS helpers
# ==============================================================================

def poly_from_utm(polygon, transform):
    """Transforms UTM Shapely polygon coords using affine transform."""
    try:
        from shapely.ops import unary_union
        from shapely.geometry import Polygon
        poly = unary_union(polygon)
        poly_pts = [~transform * tuple(i) for i in np.array(poly.exterior.coords)]
        return Polygon(poly_pts)
    except Exception as e:
        return polygon


def visualize(image, mask):
    """Backward-compatible visualization matching original utils.py."""
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    visualize_prediction(image, mask_gt=mask)
