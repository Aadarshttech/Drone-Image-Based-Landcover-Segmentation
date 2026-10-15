"""
Geospatial and visualization utility functions.
Maintained for backward compatibility. Direct imports from `src.utils` and `src.preprocess` are recommended.
"""

import os
from typing import Generator, Tuple
import numpy as np
import matplotlib.pyplot as plt

from src.utils import (
    create_color_mask,
    overlay_mask_on_image,
    visualize_prediction,
    poly_from_utm,
    visualize,
    COLORMAP_DEFAULT,
)
from src.preprocess import generate_patch_coordinates
from src.dataset import NepalDataset, NepalDataGenerator


def rasterize_masks(image_meta, intersecting_masks, image_height, image_width):
    """Rasterizes Shapely vector polygons into a raster mask aligned with image metadata."""
    try:
        from rasterio.features import rasterize
        poly_shp = []
        im_size = (image_height, image_width)
        for _, row in intersecting_masks.iterrows():
            geom = row['geometry']
            if geom.geom_type == 'Polygon':
                poly = poly_from_utm(geom, image_meta['transform'])
                poly_shp.append(poly)
            else:
                for p in geom.geoms if hasattr(geom, 'geoms') else geom:
                    poly = poly_from_utm(p, image_meta['transform'])
                    poly_shp.append(poly)

        mask = rasterize(shapes=poly_shp, out_shape=im_size, transform=image_meta['transform'])
        return mask.astype(np.uint16)
    except Exception as e:
        print(f"[Warning] rasterize_masks failed: {e}")
        return np.zeros((image_height, image_width), dtype=np.uint16)


def save_image_patch(patch, patch_path, image_meta):
    """Saves an image patch as a GeoTIFF using rasterio if available."""
    os.makedirs(os.path.dirname(patch_path), exist_ok=True)
    try:
        import rasterio
        with rasterio.open(patch_path, 'w', **image_meta) as dst:
            dst.write(patch)
    except ImportError:
        from PIL import Image
        if patch.ndim == 3 and patch.shape[0] in (1, 3, 4):
            patch = np.transpose(patch, (1, 2, 0))
        Image.fromarray(patch.astype(np.uint8)).save(patch_path)


def save_mask_patch(patch, patch_path, mask_meta):
    """Saves a mask patch as a GeoTIFF using rasterio if available."""
    os.makedirs(os.path.dirname(patch_path), exist_ok=True)
    try:
        import rasterio
        with rasterio.open(patch_path, 'w', **mask_meta) as dst:
            dst.write(patch)
    except ImportError:
        from PIL import Image
        if patch.ndim == 3 and patch.shape[0] in (1, 3, 4):
            patch = np.transpose(patch, (1, 2, 0))
        Image.fromarray(patch.astype(np.uint8)).save(patch_path)


def visualize_image_mask(image_array, mask_array):
    """Displays multi-channel image alongside its mask."""
    if image_array.shape[-1] == 5:
        fig, axes = plt.subplots(1, 6, figsize=(15, 5))
        fig.suptitle('Multi-Spectral Image Bands & Mask')
        for i in range(5):
            axes[i].imshow(image_array[:, :, i], cmap='gray')
            axes[i].set_title(f'Band {i + 1}')
            axes[i].axis('off')
        axes[-1].imshow(mask_array, cmap='gray')
        axes[-1].set_title('Mask')
        axes[-1].axis('off')
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image_array)
        axes[0].set_title('Image')
        axes[0].axis('off')
        axes[1].imshow(mask_array, cmap='gray')
        axes[1].set_title('Mask')
        axes[1].axis('off')
    plt.tight_layout()
    plt.show()


__all__ = [
    "poly_from_utm",
    "rasterize_masks",
    "save_image_patch",
    "save_mask_patch",
    "generate_patch_coordinates",
    "visualize",
    "visualize_image_mask",
    "create_color_mask",
    "overlay_mask_on_image",
    "visualize_prediction",
    "COLORMAP_DEFAULT",
    "NepalDataset",
    "NepalDataGenerator",
]
