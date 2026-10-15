#!/usr/bin/env python3
"""
Legacy patch verification and cleaning utility (check2.py).
Maintained for backward compatibility. Direct use of `preprocess.py --mode clean` is recommended.
"""

import os
import sys
from src.preprocess import clean_dataset_patches


def main():
    images_dir = sys.argv[1] if len(sys.argv) > 1 else "./output/513x513/images"
    masks_dir = sys.argv[2] if len(sys.argv) > 2 else "./output/513x513/masks"

    if not os.path.exists(images_dir):
        print(f"Directory not found: {images_dir}. Pass custom path as argument: python check2.py <images_dir> <masks_dir>")
        return

    print(f"Running data cleaning on {images_dir} and {masks_dir}...")
    clean_dataset_patches(images_dir=images_dir, masks_dir=masks_dir, empty_threshold=0.50)


if __name__ == "__main__":
    main()
