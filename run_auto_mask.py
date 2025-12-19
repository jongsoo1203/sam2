import os
from typing import List, Dict, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import cv2

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
IMAGE_PATH = "anis.jpeg"
SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"


# -------------------------------------------------------------------
# Device setup
# -------------------------------------------------------------------
def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    if device.type == "cuda":
        # Turn on bfloat16 autocast for CUDA
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

        # Enable TF32 on Ampere+ GPUs
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    elif device.type == "mps":
        print(
            "\n[Warning] MPS support is preliminary. SAM 2 is trained on CUDA, so "
            "outputs may differ and performance might be degraded.\n"
            "See: https://github.com/pytorch/pytorch/issues/84936"
        )

    return device


# -------------------------------------------------------------------
# Visualization helper
# -------------------------------------------------------------------
def show_anns(anns: List[Dict[str, Any]], borders: bool = True) -> None:
    """Visualize SAM2 masks on the current matplotlib axes."""
    if not anns:
        return

    # Sort by area (largest first) so big segments don't get hidden
    sorted_anns = sorted(anns, key=lambda x: x["area"], reverse=True)

    ax = plt.gca()
    ax.set_autoscale_on(False)

    h, w = sorted_anns[0]["segmentation"].shape
    overlay = np.ones((h, w, 4), dtype=np.float32)
    overlay[:, :, 3] = 0.0  # fully transparent to start

    for ann in sorted_anns:
        mask = ann["segmentation"]

        # Random RGB + fixed alpha
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        overlay[mask] = color_mask

        if borders:
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )

            # Optionally smooth contours
            contours = [
                cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                for contour in contours
            ]

            cv2.drawContours(
                overlay,
                contours,
                contourIdx=-1,
                color=(0, 0, 1, 0.4),  # blue-ish border with alpha
                thickness=1,
            )

    ax.imshow(overlay)


# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------
def main() -> None:
    np.random.seed(3)
    device = get_device()

    # Load and show the original image
    image_pil = Image.open(IMAGE_PATH).convert("RGB")
    image_pil = ImageOps.exif_transpose(image_pil)
    image = np.array(image_pil)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis("off")
    plt.title("Input Image")
    plt.show()

    # Build SAM2 model
    sam2 = build_sam2(
        MODEL_CFG,
        SAM2_CHECKPOINT,
        device=device,
        apply_postprocessing=False,
    )

    # Automatic mask generation
    mask_generator = SAM2AutomaticMaskGenerator(sam2)
    masks = mask_generator.generate(image)

    print(f"Number of masks: {len(masks)}")
    if masks:
        print(f"Mask keys: {list(masks[0].keys())}")

    # Visualize masks
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_anns(masks)
    plt.axis("off")
    plt.title("SAM2 Masks")
    plt.show()


if __name__ == "__main__":
    main()
