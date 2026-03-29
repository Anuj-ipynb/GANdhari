# src/inference/infer.py

import torch
from PIL import Image
import torchvision.transforms as transforms
import os
import cv2
import numpy as np

from src.models.generator import UNetGenerator
from src.utils.metrics import compute_metrics


def canny_preprocess(image_path: str):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img = cv2.equalizeHist(img)  # 🔥 better edges
    edges = cv2.Canny(img, 50, 150)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


def run_inference(
    sketch_path: str,
    green_intensity: float = 0.65,
    density: float = 0.75,
    checkpoint_path: str = "outputs/checkpoints/G_final.pth",
    use_canny: bool = True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MUST match training
    G = UNetGenerator(ngf=32).to(device)

    # Load checkpoint
    if os.path.exists(checkpoint_path):
        G.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"✅ Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"❌ CHECKPOINT NOT FOUND: {checkpoint_path}")
        return None, {}

    G.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    # Load input
    if use_canny:
        sketch_np = canny_preprocess(sketch_path)
        sketch = Image.fromarray(sketch_np)
    else:
        sketch = Image.open(sketch_path).convert("RGB")

    input_tensor = transform(sketch).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            output = G(input_tensor)

    output = (output * 0.5 + 0.5).clamp(0, 1).cpu()
    result_img = transforms.ToPILImage()(output[0])

    # Save
    os.makedirs("outputs/results", exist_ok=True)
    save_path = "outputs/results/generated_sustainable_layout.png"
    result_img.save(save_path)

    print(f"✅ Generated image saved to: {save_path}")

    # 🔥 METRICS
    metrics = compute_metrics(sketch, result_img)

    return result_img, metrics