#!/usr/bin/env python3
"""
Training script for Drone Image-Based Landcover Segmentation.
Trains DeepLabV3 with early stopping, multi-class IoU tracking, and model checkpointing.
"""

import os
import sys
import copy
import argparse
import json
from typing import Dict, List
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm

from src.model import build_deeplabv3
from src.dataset import LandcoverDataset, get_dataloader
from src.metrics import compute_iou, compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepLabV3 for Landcover Segmentation")
    parser.add_argument("--data-dir", type=str, default="./data/sample", help="Path to dataset root directory")
    parser.add_argument("--num-classes", type=int, default=3, help="Number of segmentation classes (default: 3)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Adam optimizer")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--output-dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--save-name", type=str, default="best_model.pth", help="Filename for best model weights")
    parser.add_argument("--device", type=str, default="auto", help="Device to use: 'cuda', 'cpu', or 'auto'")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze ResNet backbone weights")
    return parser.parse_args()


def train_epoch(model, dataloader, criterion, optimizer, device, num_classes):
    model.train()
    running_loss = 0.0
    running_iou = 0.0

    for images, masks in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        logits = outputs['out'] if isinstance(outputs, dict) else outputs

        # Support both multi-channel one-hot targets and integer class maps
        if masks.ndim == 4 and masks.shape[1] == num_classes:
            target = torch.argmax(masks, dim=1).long()
        elif masks.ndim == 4 and masks.shape[1] == 1:
            target = masks.squeeze(1).long()
        else:
            target = masks.long()

        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        mean_iou, _ = compute_iou(logits, target, num_classes=num_classes)
        running_iou += mean_iou.item() if isinstance(mean_iou, torch.Tensor) else mean_iou

    epoch_loss = running_loss / max(len(dataloader), 1)
    epoch_iou = running_iou / max(len(dataloader), 1)
    return epoch_loss, epoch_iou


def validate(model, dataloader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            logits = outputs['out'] if isinstance(outputs, dict) else outputs

            if masks.ndim == 4 and masks.shape[1] == num_classes:
                target = torch.argmax(masks, dim=1).long()
            elif masks.ndim == 4 and masks.shape[1] == 1:
                target = masks.squeeze(1).long()
            else:
                target = masks.long()

            loss = criterion(logits, target)
            running_loss += loss.item()

            mean_iou, _ = compute_iou(logits, target, num_classes=num_classes)
            running_iou += mean_iou.item() if isinstance(mean_iou, torch.Tensor) else mean_iou

    val_loss = running_loss / max(len(dataloader), 1)
    val_iou = running_iou / max(len(dataloader), 1)
    return val_loss, val_iou


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Select device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"=== Drone Landcover Segmentation Training ===")
    print(f"Device: {device}")
    print(f"Dataset: {args.data_dir}")
    print(f"Classes: {args.num_classes}")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")

    # Standard transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Datasets and Loaders
    train_dataset = LandcoverDataset(root_dir=args.data_dir, split="train", transform=transform, num_classes=args.num_classes)
    val_dataset = LandcoverDataset(root_dir=args.data_dir, split="val", transform=transform, num_classes=args.num_classes)

    # If split subfolders do not exist or are empty, fallback to the root directory
    if len(train_dataset) == 0:
        print("[Notice] No 'train' subsplit found, loading all available pairs from root.")
        full_dataset = LandcoverDataset(root_dir=args.data_dir, split=None, transform=transform, num_classes=args.num_classes)
        if len(full_dataset) == 0:
            print(f"[Error] No dataset found in {args.data_dir}. Run 'python data/generate_sample_data.py' to generate sample data.")
            sys.exit(1)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = get_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # Model, Loss, Optimizer
    model = build_deeplabv3(num_classes=args.num_classes, pretrained=True, freeze_backbone=args.freeze_backbone)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float('inf')
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history = {"train_loss": [], "train_iou": [], "val_loss": [], "val_iou": []}

    save_path = os.path.join(args.output_dir, args.save_name)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_iou = train_epoch(model, train_loader, criterion, optimizer, device, args.num_classes)
        val_loss, val_iou = validate(model, val_loader, criterion, device, args.num_classes)

        history["train_loss"].append(train_loss)
        history["train_iou"].append(train_iou)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)

        print(f"Epoch {epoch:03d}/{args.epochs:03d} | Train Loss: {train_loss:.4f} - mIoU: {train_iou:.4f} | Val Loss: {val_loss:.4f} - mIoU: {val_iou:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, save_path)
            # Also save legacy named checkpoint for backward compatibility
            torch.save(best_weights, "bestmodeloverall22.pth")
            patience_counter = 0
            print(f"  --> Saved new best checkpoint to {save_path} (Val Loss: {best_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    # Save training history
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining completed! Best model saved at: {save_path}")


if __name__ == "__main__":
    main()
