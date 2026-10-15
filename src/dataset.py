"""
Dataset and DataLoader utilities for Drone Landcover Segmentation.
Handles multi-format aerial imagery (.tif, .png, .jpg) and segmentation masks.
"""

import os
from typing import Callable, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

try:
    import tifffile as tiff
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


class LandcoverDataset(Dataset):
    """
    Robust PyTorch Dataset for high-resolution drone imagery and landcover masks.
    Supports directory layouts:
      root_dir/images/[split]/... and root_dir/masks/[split]/...
      or root_dir/images/... and root_dir/masks/...
    """
    SUPPORTED_EXTS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')

    def __init__(
        self,
        root_dir: str,
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        num_classes: int = 3,
    ):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform
        self.num_classes = num_classes

        # Resolve images and masks directories
        self.images_dir, self.masks_dir = self._resolve_directories()

        # Build paired list of image and mask paths
        self.samples = self._find_pairs()
        if len(self.samples) == 0:
            print(f"[Warning] No matching image-mask pairs found in {self.images_dir}")

        self.last_valid_sample = None

    def _resolve_directories(self) -> Tuple[str, str]:
        """Locates images and masks directories based on optional split parameter."""
        # Check if split subdirectory exists
        if self.split:
            candidate_img = os.path.join(self.root_dir, "images", self.split)
            candidate_mask = os.path.join(self.root_dir, "masks", self.split)
            if os.path.isdir(candidate_img) and os.path.isdir(candidate_mask):
                return candidate_img, candidate_mask

        # Direct images and masks subdirectories
        direct_img = os.path.join(self.root_dir, "images")
        direct_mask = os.path.join(self.root_dir, "masks")
        if os.path.isdir(direct_img) and os.path.isdir(direct_mask):
            return direct_img, direct_mask

        # Fallback to root_dir directly
        return self.root_dir, self.root_dir

    def _find_pairs(self) -> List[Tuple[str, str]]:
        """Matches image files with corresponding mask files."""
        pairs = []
        if not os.path.isdir(self.images_dir):
            return pairs

        image_files = sorted([
            f for f in os.listdir(self.images_dir)
            if f.lower().endswith(self.SUPPORTED_EXTS) and not f.lower().endswith(('_mask.jpg', '_mask.png', '_mask.tif', '_mask.tiff'))
        ])

        for img_name in image_files:
            img_path = os.path.join(self.images_dir, img_name)
            base_name, ext = os.path.splitext(img_name)

            # Potential mask filenames
            candidate_names = [
                f"{base_name}_mask{ext}",
                f"{base_name}_mask.jpg",
                f"{base_name}_mask.png",
                f"{base_name}_mask.tif",
                f"{base_name}_mask.tiff",
                f"{base_name}{ext}",
            ]

            mask_path = None
            for c_name in candidate_names:
                candidate_path = os.path.join(self.masks_dir, c_name)
                if os.path.isfile(candidate_path):
                    mask_path = candidate_path
                    break

            if mask_path is not None:
                pairs.append((img_path, mask_path))

        return pairs

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str) -> np.ndarray:
        """Loads image file as uint8 RGB numpy array."""
        if path.lower().endswith(('.tif', '.tiff')) and HAS_TIFFFILE:
            img = tiff.imread(path)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[0] in (3, 4) and img.ndim == 3:  # CHW format
                img = np.transpose(img[:3], (1, 2, 0))
            elif img.shape[-1] > 3:
                img = img[..., :3]
            if img.dtype != np.uint8:
                img = (np.clip(img, 0, 1 if img.max() <= 1.0 else 255) * (255 if img.max() <= 1.0 else 1)).astype(np.uint8)
            return img
        else:
            with Image.open(path) as pil_img:
                return np.array(pil_img.convert('RGB'))

    def _load_mask(self, path: str) -> np.ndarray:
        """Loads mask file as integer numpy array."""
        if path.lower().endswith(('.tif', '.tiff')) and HAS_TIFFFILE:
            mask = tiff.imread(path)
            if mask.ndim == 3 and mask.shape[0] in (1, 2, 3, 4):  # CHW
                if mask.shape[0] == 1:
                    mask = mask[0]
                else:
                    # multi-channel or one-hot
                    return mask.astype(np.float32)
            elif mask.ndim == 3 and mask.shape[-1] in (3, 4):
                # RGB mask - map to integer class or channel
                mask = mask[..., 0]
            return mask.astype(np.int64)
        else:
            with Image.open(path) as pil_mask:
                mask = np.array(pil_mask)
                if mask.ndim == 3:
                    mask = mask[..., 0]
                return mask.astype(np.int64)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.samples[idx]
        try:
            image_np = self._load_image(img_path)
            mask_np = self._load_mask(mask_path)

            # Apply transforms
            if self.transform is not None:
                image = self.transform(image_np)
            else:
                image = transforms.ToTensor()(image_np)

            if self.mask_transform is not None:
                mask = self.mask_transform(mask_np)
            else:
                if isinstance(mask_np, np.ndarray):
                    if mask_np.ndim == 3:  # Already multi-channel [C, H, W]
                        mask = torch.from_numpy(mask_np).float()
                    else:
                        mask = torch.from_numpy(mask_np).long()
                else:
                    mask = torch.tensor(mask_np, dtype=torch.long)

            self.last_valid_sample = (image, mask)
            return image, mask

        except Exception as e:
            if self.last_valid_sample is not None:
                return self.last_valid_sample
            raise e


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    """Standard DataLoader generator with optimal defaults for training/evaluation."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
    )


# ==============================================================================
# Backward Compatibility Classes (for existing scripts: generator2.py, trygen.py)
# ==============================================================================

class NepalDataset(LandcoverDataset):
    """
    Backward-compatible subclass matching original NepalDataset initialization.
    """
    def __init__(self, data_path: str, transform: Optional[Callable] = None, training: bool = True):
        split = "train" if training else "val"
        super().__init__(root_dir=data_path, split=split, transform=transform)


class NepalDataGenerator:
    """
    Backward-compatible batch iterator mimicking original NepalDataGenerator interface.
    """
    def __init__(self, dataset: Dataset, batch_size: int = 16, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        self.iterator = iter(self.loader)

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        self.iterator = iter(self.loader)
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            raise StopIteration
