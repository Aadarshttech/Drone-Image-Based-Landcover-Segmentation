"""
Preprocessing pipeline for drone orthomosaics and GIS vector masks.
Handles sliding window patch extraction, nodata filtering, and train/val/test splits.
"""

import os
import shutil
import random
from typing import Generator, List, Optional, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import rasterio
    from rasterio.features import rasterize
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def generate_patch_coordinates(
    image_width: int,
    image_height: int,
    patch_size: int = 512,
    stride: int = 256
) -> Generator[Tuple[int, int], None, None]:
    """Generates top-left (x, y) coordinates for sliding window patch extraction."""
    for y in range(0, image_height - patch_size + 1, stride):
        for x in range(0, image_width - patch_size + 1, stride):
            yield x, y


def clean_dataset_patches(
    images_dir: str,
    masks_dir: str,
    empty_threshold: float = 0.50,
    dry_run: bool = False
) -> int:
    """
    Cleans dataset by identifying and removing patches that are dominated by background / nodata.

    Args:
        images_dir: Path to image patches directory.
        masks_dir: Path to mask patches directory.
        empty_threshold: Maximum allowed fraction of background pixels (default 50%).
        dry_run: If True, only reports without deleting files.

    Returns:
        Number of removed/flagged patches.
    """
    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        print(f"[Error] Directory not found: {images_dir} or {masks_dir}")
        return 0

    files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.tif', '.tiff'))])
    removed_count = 0

    print(f"Scanning {len(files)} patches with empty threshold: {empty_threshold * 100:.1f}%...")
    for filename in tqdm(files):
        img_path = os.path.join(images_dir, filename)
        base, ext = os.path.splitext(filename)

        # Look for matching mask
        mask_candidates = [
            os.path.join(masks_dir, f"{base}_mask{ext}"),
            os.path.join(masks_dir, f"{base}_mask.jpg"),
            os.path.join(masks_dir, f"{base}_mask.png"),
            os.path.join(masks_dir, f"{base}_mask.tif"),
            os.path.join(masks_dir, f"{base}{ext}"),
        ]

        mask_path = next((m for m in mask_candidates if os.path.exists(m)), None)
        if not mask_path:
            continue

        try:
            with Image.open(mask_path) as m_img:
                mask_arr = np.array(m_img)

            total_pixels = mask_arr.size
            if total_pixels == 0:
                continue

            zero_pixels = np.count_nonzero(mask_arr == 0)
            zero_ratio = zero_pixels / total_pixels

            # If background exceeds threshold, remove both image and mask
            if zero_ratio >= empty_threshold:
                removed_count += 1
                if not dry_run:
                    os.remove(img_path)
                    os.remove(mask_path)
        except Exception as e:
            print(f"[Warning] Failed to process {mask_path}: {e}")

    action = "Flagged" if dry_run else "Removed"
    print(f"{action} {removed_count} background-dominated patches.")
    return removed_count


def split_dataset(
    dataset_dir: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    """
    Splits image-mask patches into train, val, and test partitions.
    """
    random.seed(seed)
    images_dir = os.path.join(dataset_dir, "images")
    masks_dir = os.path.join(dataset_dir, "masks")

    images = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.tif', '.tiff'))])
    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    for split_name, split_files in splits.items():
        dst_img_dir = os.path.join(output_dir, "images", split_name)
        dst_mask_dir = os.path.join(output_dir, "masks", split_name)
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_mask_dir, exist_ok=True)

        for f in split_files:
            src_img = os.path.join(images_dir, f)
            shutil.copy2(src_img, os.path.join(dst_img_dir, f))

            base, ext = os.path.splitext(f)
            # Find mask
            for mask_name in [f"{base}_mask{ext}", f"{base}_mask.jpg", f"{base}_mask.png", f"{base}{ext}"]:
                src_mask = os.path.join(masks_dir, mask_name)
                if os.path.exists(src_mask):
                    shutil.copy2(src_mask, os.path.join(dst_mask_dir, mask_name))
                    break

        print(f"Created {split_name} partition: {len(split_files)} patches")
