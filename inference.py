#!/usr/bin/env python3
"""
Inference script for Drone Image-Based Landcover Segmentation.
Runs prediction on single images or entire directories and generates color overlays.
"""

import os
import argparse
from typing import Optional
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from src.model import build_deeplabv3
from src.utils import overlay_mask_on_image, create_color_mask, visualize_prediction, COLORMAP_DEFAULT


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference using trained DeepLabV3 model")
    parser.add_argument("--image", type=str, default=None, help="Path to single input drone image")
    parser.add_argument("--image-dir", type=str, default=None, help="Directory containing drone images to process")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to trained model weights (.pth)")
    parser.add_argument("--output-dir", type=str, default="inference_results", help="Directory to save output masks & overlays")
    parser.add_argument("--num-classes", type=int, default=3, help="Number of classes (default: 3)")
    parser.add_argument("--alpha", type=float, default=0.45, help="Alpha blending transparency for overlay")
    parser.add_argument("--device", type=str, default="auto", help="Execution device ('cuda', 'cpu', or 'auto')")
    return parser.parse_args()


def predict_image(
    model: torch.nn.Module,
    image_path: str,
    device: torch.device,
    transform: transforms.Compose,
    num_classes: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs model inference on a single image and returns (original_image_np, predicted_mask_2d).
    """
    with Image.open(image_path) as pil_img:
        orig_img = pil_img.convert("RGB")
        orig_np = np.array(orig_img)

    input_tensor = transform(orig_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        logits = output['out'] if isinstance(output, dict) else output
        pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

    return orig_np, pred_mask


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"=== Drone Landcover Segmentation Inference ===")
    print(f"Device: {device}")

    # Checkpoint resolution (support fallback to legacy name)
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path) and os.path.exists("bestmodeloverall22.pth"):
        ckpt_path = "bestmodeloverall22.pth"

    # Build model
    model = build_deeplabv3(num_classes=args.num_classes, pretrained=False)
    if os.path.exists(ckpt_path):
        print(f"Loading weights from: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"[Warning] Checkpoint '{ckpt_path}' not found. Using initialized weights for demonstration.")

    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Gather images to process
    image_paths = []
    if args.image and os.path.isfile(args.image):
        image_paths.append(args.image)
    elif args.image_dir and os.path.isdir(args.image_dir):
        image_paths = [
            os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))
        ]
    else:
        # Default test lookups
        sample_candidates = [
            "data/sample/images/test",
            "data/sample/images/train",
            "data"
        ]
        for c in sample_candidates:
            if os.path.isdir(c):
                found = [os.path.join(c, f) for f in os.listdir(c) if f.lower().endswith(('.jpg', '.png'))]
                if found:
                    image_paths = found[:3]
                    break

    if not image_paths:
        print("[Error] No input image specified and no default sample images found. Use --image <path> or --image-dir <path>.")
        return

    print(f"Processing {len(image_paths)} images...")
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        base, _ = os.path.splitext(filename)

        orig_np, pred_mask = predict_image(model, img_path, device, transform, args.num_classes)

        # 1. Save raw integer mask
        raw_mask_path = os.path.join(args.output_dir, f"{base}_pred_mask.png")
        Image.fromarray(pred_mask.astype(np.uint8)).save(raw_mask_path)

        # 2. Save colored mask
        color_mask = create_color_mask(pred_mask)
        color_mask_path = os.path.join(args.output_dir, f"{base}_color_mask.png")
        Image.fromarray(color_mask).save(color_mask_path)

        # 3. Save blended overlay
        overlay = overlay_mask_on_image(orig_np, pred_mask, alpha=args.alpha)
        overlay_path = os.path.join(args.output_dir, f"{base}_overlay.png")
        Image.fromarray(overlay).save(overlay_path)

        # 4. Save side-by-side plot
        vis_path = os.path.join(args.output_dir, f"{base}_comparison.png")
        visualize_prediction(orig_np, mask_pred=pred_mask, save_path=vis_path, title=f"Prediction: {filename}")

        print(f"Generated results for {filename} -> {args.output_dir}/")


if __name__ == "__main__":
    main()
