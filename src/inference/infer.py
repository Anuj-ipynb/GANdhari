# src/inference/infer.py
import torch
from PIL import Image
import torchvision.transforms as transforms
import os
import cv2
import numpy as np

from src.models.generator import UNetGenerator
from src.utils.metrics import compute_enhanced_metrics


def canny_preprocess(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray, 30, 150)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


def run_inference(
    sketch_path: str,
    green_intensity: float = 0.65,
    density: float = 0.75,
    checkpoint_path: str = "outputs/checkpoints/checkpoint_final.pth",
    use_canny: bool = True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    G = UNetGenerator(ngf=32).to(device)

    # Load checkpoint
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict) and "G" in ckpt:
            G.load_state_dict(ckpt["G"])
            print(f"✅ Loaded NEW checkpoint: {checkpoint_path}")
        else:
            G.load_state_dict(ckpt)
            print(f"✅ Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None, {"error": "Checkpoint not found"}

    G.eval()

    try:
        # === ROBUST INPUT PREPROCESSING (handles many sketch types) ===
        if use_canny:
            sketch_np = canny_preprocess(sketch_path)
            sketch = Image.fromarray(sketch_np)
        else:
            sketch = Image.open(sketch_path).convert("RGB")

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        input_tensor = transform(sketch).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad(), torch.amp.autocast("cuda"):
            output = G(input_tensor)

        output = (output * 0.5 + 0.5).clamp(0, 1).cpu()
        result_img = transforms.ToPILImage()(output[0])

        # === STRONG POST-PROCESSING (makes outputs usable) ===
        result_cv = np.array(result_img)
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8,8))
        gray = cv2.cvtColor(result_cv, cv2.COLOR_RGB2GRAY)
        enhanced = clahe.apply(gray)
        result_cv = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

        # Sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        result_cv = cv2.filter2D(result_cv, -1, kernel)

        result_cv = cv2.morphologyEx(result_cv, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
        result_img = Image.fromarray(result_cv)

        # Save
        os.makedirs("outputs/results", exist_ok=True)
        save_path = "outputs/results/generated_sustainable_layout.png"
        result_img.save(save_path)
        print(f"✅ Image saved: {save_path}")

        # Metrics
        raw_metrics = compute_enhanced_metrics(sketch, result_img)
        clean_metrics = {
            "building_density": round(raw_metrics.get("building_density", 0) * 100, 2),
            "road_coverage": round(raw_metrics.get("road_coverage", 0) * 100, 2),
            "green_coverage": round(raw_metrics.get("green_coverage", 0) * 100, 2),
            "osr_proxy": round(raw_metrics.get("osr_proxy", 0), 3),
            "road_connectivity": round(raw_metrics.get("road_connectivity", 0), 3),
            "building_compactness": round(raw_metrics.get("building_compactness", 1.0), 3),
            "sustainability_score": round(raw_metrics.get("sustainability_score", 50.0), 2),

            "Green Coverage": round(raw_metrics.get("green_coverage", 0), 4),
            "Building Density": round(raw_metrics.get("building_density", 0), 4),
            "Road Coverage": round(raw_metrics.get("road_coverage", 0), 4),
            "Sustainability Score": round(raw_metrics.get("sustainability_score", 50.0), 2),

            "SSIM": raw_metrics.get("SSIM"),
            "L1": raw_metrics.get("L1"),
            "Edge Consistency": raw_metrics.get("Edge Consistency")
        }

        print(f"✅ Metrics → Score: {clean_metrics['sustainability_score']:.1f} | Green: {clean_metrics['green_coverage']}%")

        return result_img, clean_metrics

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, {"error": str(e)}