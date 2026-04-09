# src/inference/infer.py

import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
import os

from src.models.generator import UNetGenerator
from src.utils.metrics import compute_enhanced_metrics


# ---------------------------
# Tensor → Image
# ---------------------------
def tensor_to_image(tensor):
    img = tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))

    img = (img + 1) / 2
    img = np.clip(img, 0, 1)

    return (img * 255).astype(np.uint8)


# ---------------------------
# 🔥 FINAL POST-PROCESSING PIPELINE
# ---------------------------
def post_process(img):
    # --- 1. Gamma correction ---
    gamma = 1.5
    invGamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** invGamma) * 255 for i in np.arange(256)
    ]).astype("uint8")
    img = cv2.LUT(img, table)

    # --- 2. Contrast boost ---
    img = cv2.convertScaleAbs(img, alpha=1.3, beta=15)

    # --- 3. CLAHE (balanced) ---
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

    # --- 4. Color enhancement ---
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * 1.2, 0, 255).astype(np.uint8)
    v = np.clip(v * 1.1, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2RGB)

    # --- 5. Anti-checkerboard smoothing ---
    img = cv2.medianBlur(img, 3)

    # --- 6. Sharpen (controlled) ---
    kernel = np.array([
        [0, -1, 0],
        [-1, 4.2, -1],
        [0, -1, 0]
    ])
    sharp = cv2.filter2D(img, -1, kernel)

    # Blend sharpened + original (important)
    img = cv2.addWeighted(img, 0.7, sharp, 0.3, 0)

    # --- 7. Final slight blur (artifact suppression) ---
    img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


# ---------------------------
# 🔥 MAIN INFERENCE
# ---------------------------
def run_inference(
    sketch_path,
    checkpoint_path="outputs/checkpoints/checkpoint_final.pth"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = UNetGenerator(ngf=32).to(device)

    if not os.path.exists(checkpoint_path):
        return None, {"error": "Checkpoint not found"}

    ckpt = torch.load(checkpoint_path, map_location=device)
    G.load_state_dict(ckpt["G"] if "G" in ckpt else ckpt)
    G.eval()

    try:
        sketch = Image.open(sketch_path).convert("RGB")

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ])

        input_tensor = transform(sketch).unsqueeze(0).to(device)

        with torch.no_grad():
            output = G(input_tensor)[0]

        # Convert
        img = tensor_to_image(output)

        # 🔥 Apply FINAL post-processing
        img = post_process(img)

        result_img = Image.fromarray(img)

        # Save
        os.makedirs("outputs/results", exist_ok=True)
        save_path = "outputs/results/generated.png"
        result_img.save(save_path)

        # ✅ FIXED METRICS (correct inputs)
        metrics = compute_enhanced_metrics(sketch, result_img)

        # JSON-safe conversion
        metrics = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                   for k, v in metrics.items()}

        return result_img, metrics

    except Exception as e:
        return None, {"error": str(e)}