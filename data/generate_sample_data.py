#!/usr/bin/env python3
"""
Synthetic Sample Data Generator for Drone Landcover Segmentation.
Generates small demonstration patches and ground-truth masks for instant pipeline testing.
"""

import os
import argparse
import numpy as np
from PIL import Image, ImageDraw


def generate_patch(patch_id: int, size: int = 512):
    """
    Synthesizes a realistic-looking 512x512 drone image patch and its corresponding
    landcover segmentation mask (0: Background, 1: Buildings, 2: Woodland).
    """
    np.random.seed(patch_id * 101)

    # 1. Base terrain background (soil / dry grass / roads)
    base_color = np.array([140, 135, 115], dtype=np.float32)
    noise = np.random.normal(0, 12, (size, size, 3))
    img_arr = np.clip(base_color + noise, 0, 255).astype(np.uint8)

    img = Image.fromarray(img_arr)
    draw_img = ImageDraw.Draw(img)

    mask = Image.new("L", (size, size), 0)  # Class 0: Background
    draw_mask = ImageDraw.Draw(mask)

    # Draw a roadway or path (still class 0)
    road_x = int(np.random.uniform(50, size - 50))
    road_w = int(np.random.uniform(30, 50))
    draw_img.line([(road_x, 0), (road_x + int(np.random.uniform(-40, 40)), size)], fill=(90, 88, 85), width=road_w)

    # 2. Add Woodland / Vegetation Clusters (Class 2)
    num_trees = np.random.randint(6, 14)
    for _ in range(num_trees):
        tx = int(np.random.uniform(30, size - 30))
        ty = int(np.random.uniform(30, size - 30))
        tr = int(np.random.uniform(25, 60))

        # Irregular foliage cluster
        bbox = [tx - tr, ty - tr, tx + tr, ty + tr]
        tree_color = (
            int(np.random.uniform(35, 65)),
            int(np.random.uniform(100, 160)),
            int(np.random.uniform(35, 65))
        )
        draw_img.ellipse(bbox, fill=tree_color)
        draw_mask.ellipse(bbox, fill=2)

    # 3. Add Buildings / Structures (Class 1)
    num_buildings = np.random.randint(3, 7)
    for _ in range(num_buildings):
        bx = int(np.random.uniform(40, size - 140))
        by = int(np.random.uniform(40, size - 140))
        bw = int(np.random.uniform(60, 110))
        bh = int(np.random.uniform(60, 110))

        # Roof tone (e.g. tin blue, terracotta red, concrete grey)
        roof_styles = [
            (180, 70, 55),    # Terracotta
            (80, 110, 145),   # Blue corrugated tin
            (195, 195, 190),  # Concrete roof
        ]
        roof_color = roof_styles[np.random.randint(0, len(roof_styles))]

        # Roof rectangle
        draw_img.rectangle([bx, by, bx + bw, by + bh], fill=roof_color, outline=(40, 40, 40), width=2)
        draw_mask.rectangle([bx, by, bx + bw, by + bh], fill=1)

    return img, mask


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic sample drone patches")
    parser.add_argument("--output-dir", type=str, default="./data/sample",
                        help="Target output directory")
    parser.add_argument("--patch-size", type=int, default=512, help="Patch width and height")
    args = parser.parse_args()

    splits = {
        "train": 4,
        "val": 2,
        "test": 2,
    }

    print(f"Generating synthetic demonstration dataset in '{args.output_dir}'...")

    idx = 1
    for split_name, count in splits.items():
        img_dir = os.path.join(args.output_dir, "images", split_name)
        mask_dir = os.path.join(args.output_dir, "masks", split_name)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        for i in range(count):
            img, mask = generate_patch(patch_id=idx, size=args.patch_size)

            img_filename = f"sample_drone_{split_name}_{i+1:02d}.png"
            mask_filename = f"sample_drone_{split_name}_{i+1:02d}_mask.png"

            img.save(os.path.join(img_dir, img_filename))
            mask.save(os.path.join(mask_dir, mask_filename))
            idx += 1

        print(f"  --> Created {count} sample image-mask pairs for '{split_name}'")

    print("\nSample dataset generation complete! You can now test train.py or inference.py immediately.")


if __name__ == "__main__":
    main()
