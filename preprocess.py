#!/usr/bin/env python3
"""
CLI Preprocessing and Dataset Management Tool.
Supports patch quality cleaning, dataset partitioning, and coordinate extraction.
"""

import os
import argparse
from src.preprocess import clean_dataset_patches, split_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset preprocessing and maintenance")
    parser.add_argument("--mode", type=str, choices=["clean", "split"], default="clean",
                        help="Operation mode: 'clean' empty patches or 'split' into train/val/test")
    parser.add_argument("--images-dir", type=str, default="./output/513x513/images",
                        help="Path to image patches directory")
    parser.add_argument("--masks-dir", type=str, default="./output/513x513/masks",
                        help="Path to mask patches directory")
    parser.add_argument("--dataset-dir", type=str, default="./data/patches",
                        help="Root dataset directory containing 'images' and 'masks' for splitting")
    parser.add_argument("--output-dir", type=str, default="./data/splits",
                        help="Output directory for split partitions")
    parser.add_argument("--empty-threshold", type=float, default=0.50,
                        help="Threshold fraction above which a patch is considered empty/background (default: 0.50)")
    parser.add_argument("--dry-run", action="store_true",
                        help="If set, reports empty patches without deleting files")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "clean":
        print(f"=== Cleaning Patches in {args.images_dir} & {args.masks_dir} ===")
        clean_dataset_patches(
            images_dir=args.images_dir,
            masks_dir=args.masks_dir,
            empty_threshold=args.empty_threshold,
            dry_run=args.dry_run
        )
    elif args.mode == "split":
        print(f"=== Splitting Dataset {args.dataset_dir} into {args.output_dir} ===")
        split_dataset(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15
        )


if __name__ == "__main__":
    main()
