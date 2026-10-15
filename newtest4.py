#!/usr/bin/env python3
"""
Legacy testing and inference runner (newtest4.py).
Maintained for backward compatibility. Uses modern pipeline from `inference.py` and `src`.
"""

import os
from inference import main as inference_main, predict_image
from src.model import prepare_model
from src.utils import overlay_mask_on_image, create_color_mask, visualize_prediction


def infer_single_image_with_deep():
    """Legacy entrypoint function."""
    print("Executing inference pipeline via infer_single_image_with_deep()...")
    inference_main()


def test_model(model, image_path, transform=None):
    """Legacy wrapper function."""
    import torch
    from torchvision import transforms
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    orig_np, pred_mask = predict_image(model, image_path, device, transform)
    visualize_prediction(orig_np, mask_pred=pred_mask, title=f"Test Result: {os.path.basename(image_path)}")


if __name__ == "__main__":
    inference_main()
