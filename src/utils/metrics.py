# src/utils/metrics.py
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.measure import label, regionprops

# =============================================
# Basic Image Quality Metrics
# =============================================
def compute_metrics(real_img, fake_img):
    """SSIM, L1, and Edge Consistency"""
    real = np.array(real_img)
    fake = np.array(fake_img)

    if fake.shape[:2] != real.shape[:2]:
        fake = cv2.resize(fake, (real.shape[1], real.shape[0]))

    ssim_score = ssim(real, fake, channel_axis=2, data_range=255.0)
    l1 = np.mean(np.abs(real.astype(np.float32) - fake.astype(np.float32))) / 255.0

    real_edges = cv2.Canny(cv2.cvtColor(real, cv2.COLOR_RGB2GRAY), 80, 180)
    fake_edges = cv2.Canny(cv2.cvtColor(fake, cv2.COLOR_RGB2GRAY), 80, 180)
    edge_score = np.mean(real_edges == fake_edges)

    return {
        "SSIM": round(ssim_score, 4),
        "L1": round(l1, 4),
        "Edge Consistency": round(edge_score, 4)
    }


# =============================================
# Urban Structure Extraction (Final Version)
# =============================================
def extract_structures(img):
    """Robust extraction for Cityscapes-style outputs"""
    if isinstance(img, np.ndarray):
        gen_np = img
    else:
        gen_np = np.array(img.convert("RGB"))

    # Strong contrast enhancement - crucial for your dark outputs
    gray = cv2.cvtColor(gen_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=5.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Buildings: bright regions using adaptive Otsu
    _, buildings = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    buildings = cv2.morphologyEx(buildings, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Roads: thin edges only (prevent large dark areas being counted as roads)
    edges = cv2.Canny(gray, 35, 105)
    roads = cv2.dilate(edges, np.ones((3, 3), np.uint8))

    # Green: wide tolerant range for vegetation (Bengaluru-friendly)
    hsv = cv2.cvtColor(gen_np, cv2.COLOR_RGB2HSV)
    green = cv2.inRange(hsv, (25, 20, 25), (100, 255, 255))

    return buildings, roads, green


# =============================================
# Urban Planning Metrics
# =============================================
def compute_urban_metrics(img):
    buildings, roads, green = extract_structures(img)
    total = buildings.size * 1.0

    building_density = float(np.sum(buildings) / (255 * total))
    road_coverage = float(np.sum(roads) / (255 * total))
    green_coverage = float(np.sum(green) / (255 * total))

    osr_proxy = float(np.sum(green) / (np.sum(buildings) + np.sum(roads) + 1e-6))

    # Road connectivity (lower = better connected network)
    _, num_components = cv2.connectedComponents(roads)
    road_connectivity = float(num_components / (np.sum(roads) / 255 + 1e-6))

    return {
        "building_density": building_density,
        "road_coverage": road_coverage,
        "green_coverage": green_coverage,
        "osr_proxy": osr_proxy,
        "road_connectivity": road_connectivity,
        "building_compactness": 1.0,
    }


# =============================================
# Sustainability Score
# =============================================
def sustainability_score(metrics):
    """Bengaluru-aware sustainability scoring"""
    g = metrics["green_coverage"] * 100
    d = metrics["building_density"] * 100
    r = metrics["road_coverage"] * 100
    conn = metrics["road_connectivity"]

    score = (
        0.40 * g +                    # Green is most important
        0.25 * (100 - d) +            # Low density is good
        0.20 * r +                    # Good road coverage
        0.10 * (100 - conn * 8) +     # Penalize fragmented roads
        0.05 * 60                     # Base bonus
    )
    return round(max(0, min(100, score)), 2)


# =============================================
# Combined Enhanced Metrics (Main Function)
# =============================================
def compute_enhanced_metrics(real_img, fake_img):
    basic = compute_metrics(real_img, fake_img)
    urban = compute_urban_metrics(fake_img)
    score = sustainability_score(urban)

    combined = {**basic, **urban}
    combined["sustainability_score"] = score
    combined["Sustainability Score"] = score

    # Backward compatibility for app.py
    combined["Green Coverage"] = urban["green_coverage"]
    combined["Building Density"] = urban["building_density"]
    combined["Road Coverage"] = urban["road_coverage"]

    return combined