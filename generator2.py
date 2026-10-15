"""
Legacy wrapper for NepalDataset and NepalDataGenerator.
Maintained for backward compatibility. Direct imports from `src.dataset` are recommended.
"""

from src.dataset import NepalDataset, NepalDataGenerator, LandcoverDataset, get_dataloader

__all__ = ["NepalDataset", "NepalDataGenerator", "LandcoverDataset", "get_dataloader"]
