#!/usr/bin/env python3
"""
Dataset Integrity & Statistics Verifier.
Validates paired image-mask alignment, dimensions, channels, and class distributions.
"""

import os
import sys
import argparse
from collections import Counter
from typing import List, Tuple
import numpy as np
from PIL import Image

CLASS_NAMES = {
    0: "Background / Unclassified",
    1: "Buildings",
    2: "Woodland / Vegetation"
}


def parse_args():
    parser = argparse.ArgumentParser(description="Verify dataset structure and class distribution")
    parser.add_argument("--data-dir", type=str, default="./data/sample",
                        help="Root dataset directory containing 'images' and 'masks'")
    return parser.parse_args()


def scan_partition(partition_name: str, img_dir: str, mask_dir: str):
    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        return

    images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.tif', '.tiff'))])
    masks = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.jpg', '.png', '.tif', '.tiff'))])

    print(f"\n--- Partition: {partition_name.upper()} ---")
    print(f"Total Images: {len(images)} | Total Masks: {len(masks)}")

    pairs = []
    orphans = []

    for img_name in images:
        base, ext = os.path.splitext(img_name)
        candidates = [
            f"{base}_mask{ext}",
            f"{base}_mask.jpg",
            f"{base}_mask.png",
            f"{base}_mask.tif",
            f"{base}{ext}"
        ]
        matched = next((c for c in candidates if os.path.exists(os.path.join(mask_dir, c))), None)
        if matched:
            pairs.append((os.path.join(img_dir, img_name), os.path.join(mask_dir, matched)))
        else:
            orphans.append(img_name)

    print(f"Successfully Paired: {len(pairs)} | Orphan Images (no mask): {len(orphans)}")

    if orphans:
        print(f"  [Warning] Missing masks for: {orphans[:5]}")

    if not pairs:
        print("  [Notice] No valid pairs found in this partition.")
        return

    # Check dimensions and class distribution
    class_counter = Counter()
    dimensions = set()
    channels = set()

    for img_path, mask_path in pairs:
        try:
            with Image.open(img_path) as im:
                dimensions.add(im.size)
                channels.add(len(im.getbands()))

            with Image.open(mask_path) as mk:
                mk_arr = np.array(mk)
                if mk_arr.ndim == 3:
                    mk_arr = mk_arr[..., 0]
                uniques, counts = np.unique(mk_arr, return_counts=True)
                for u, c in zip(uniques, counts):
                    class_counter[int(u)] += int(c)
        except Exception as e:
            print(f"  [Error] Failed to read {img_path}: {e}")

    print(f"Resolution / Dimensions found: {list(dimensions)}")
    print(f"Image Band counts: {list(channels)}")

    total_pixels = sum(class_counter.values())
    print("\nClass Pixel Distribution:")
    for class_id in sorted(class_counter.keys()):
        count = class_counter[class_id]
        pct = (count / total_pixels) * 100 if total_pixels > 0 else 0
        name = CLASS_NAMES.get(class_id, f"Class {class_id}")
        print(f"  Class {class_id} ({name:<25}): {count:12,d} pixels ({pct:5.2f}%)")


def main():
    args = parse_args()
    data_dir = args.data_dir

    print("=" * 60)
    print(f"  DRONE LANDCOVER DATASET VERIFIER: {data_dir}")
    print("=" * 60)

    if not os.path.exists(data_dir):
        print(f"[Error] Directory '{data_dir}' does not exist.")
        sys.exit(1)

    # Check for split subdirectories
    has_splits = False
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(data_dir, "images", split)
        mask_dir = os.path.join(data_dir, "masks", split)
        if os.path.exists(img_dir) and os.path.exists(mask_dir):
            has_splits = True
            scan_partition(split, img_dir, mask_dir)

    # If no splits, scan top-level images and masks
    if not has_splits:
        img_dir = os.path.join(data_dir, "images")
        mask_dir = os.path.join(data_dir, "masks")
        if os.path.exists(img_dir) and os.path.exists(mask_dir):
            scan_partition("all", img_dir, mask_dir)
        else:
            print(f"[Error] Neither split folders nor top-level 'images/' and 'masks/' found in {data_dir}.")

    print("\n" + "=" * 60)
    print("  Verification Complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
