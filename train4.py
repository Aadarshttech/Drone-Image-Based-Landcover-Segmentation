#!/usr/bin/env python3
"""
Legacy training runner (train4.py).
Maintained for backward compatibility. Uses modern pipeline from `train.py` and `src`.
"""

import sys
from train import main as train_main
from src.metrics import compute_iou
from src.dataset import NepalDataset, NepalDataGenerator
from src.model import prepare_model


def train_model_with_deep():
    """Legacy entrypoint function."""
    print("Executing training pipeline via train_model_with_deep()...")
    train_main()


if __name__ == "__main__":
    train_main()
