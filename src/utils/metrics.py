# src/utils/metrics.py

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim


# ---------------------------
# SAFE UTILITIES
# ---------------------------

def _to_uint8(img):
    if isinstance(img, np.ndarray):
        arr = img
    else:
        arr = np.array(img.convert("RGB"))

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


def _safe_float(x):
    try:
        return float(np.asarray(x).item())
    except Exception:
        return float(np.mean(x))


# ---------------------------
# BASIC IMAGE METRICS
# ---------------------------

def compute_metrics(real_img, fake_img):
    try:
        real = _to_uint8(real_img)
        fake = _to_uint8(fake_img)

        if fake.shape[:2] != real.shape[:2]:
            fake = cv2.resize(fake, (real.shape[1], real.shape[0]))

        ssim_score = ssim(real, fake, channel_axis=2, data_range=255)

        l1 = np.mean(np.abs(real.astype(np.float32) - fake.astype(np.float32))) / 255.0

        real_edges = cv2.Canny(cv2.cvtColor(real, cv2.COLOR_RGB2GRAY), 80, 180)
        fake_edges = cv2.Canny(cv2.cvtColor(fake, cv2.COLOR_RGB2GRAY), 80, 180)

        edge_score = np.sum((real_edges > 0) & (fake_edges > 0)) / (
            np.sum(real_edges > 0) + 1e-6
        )

        return {
            "SSIM": round(_safe_float(ssim_score), 4),
            "L1": round(_safe_float(l1), 4),
            "Edge Consistency": round(_safe_float(edge_score), 4),
        }

    except Exception:
        return {
            "SSIM": 0.0,
            "L1": 1.0,
            "Edge Consistency": 0.0,
        }


# ---------------------------
# STRUCTURE EXTRACTION (FINAL FIXED)
# ---------------------------

def extract_structures(img):
    img = _to_uint8(img)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # ---------------------------
    # GREEN
    # ---------------------------
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    green = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel)
    green = cv2.dilate(green, kernel, iterations=1)

    # ---------------------------
    # ROAD (STRICT FIX)
    # ---------------------------
    h, s, v = cv2.split(hsv)

    road_mask = (
        (s < 60) &
        (v > 60) &
        (v < 140)
    )

    roads = (road_mask.astype(np.uint8)) * 255
    roads = cv2.morphologyEx(roads, cv2.MORPH_OPEN, kernel)
    roads = cv2.morphologyEx(roads, cv2.MORPH_CLOSE, kernel)

    # ---------------------------
    # BUILDINGS (ADDED BACK ✅)
    # ---------------------------
    edges = cv2.Canny(gray, 50, 150)
    edges_dense = cv2.dilate(edges, np.ones((5, 5), np.uint8))

    buildings = (edges_dense > 0).astype(np.uint8) * 255
    buildings = cv2.bitwise_and(buildings, cv2.bitwise_not(green))
    buildings = cv2.bitwise_and(buildings, cv2.bitwise_not(roads))

    buildings = cv2.morphologyEx(buildings, cv2.MORPH_CLOSE, kernel)

    return buildings, roads, green


# ---------------------------
# URBAN METRICS
# ---------------------------

def compute_urban_metrics(img):
    try:
        buildings, roads, green = extract_structures(img)

        total = float(buildings.size)

        building_density = np.sum(buildings > 0) / total
        road_coverage = np.sum(roads > 0) / total
        green_coverage = np.sum(green > 0) / total

        osr_proxy = np.sum(green > 0) / (
            np.sum(buildings > 0) + np.sum(roads > 0) + 1e-6
        )

        num_labels, _ = cv2.connectedComponents((roads > 0).astype(np.uint8))
        road_connectivity = min(num_labels / 50.0, 1.0)

        contours, _ = cv2.findContours(buildings, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        compactness = 0.0
        if len(contours) > 0:
            vals = []
            for c in contours:
                area = cv2.contourArea(c)
                peri = cv2.arcLength(c, True)
                if peri > 0:
                    vals.append((4 * np.pi * area) / (peri * peri + 1e-6))
            if vals:
                compactness = np.mean(vals)

        return {
            "building_density": _safe_float(building_density),
            "road_coverage": _safe_float(road_coverage),
            "green_coverage": _safe_float(green_coverage),
            "osr_proxy": _safe_float(osr_proxy),
            "road_connectivity": _safe_float(road_connectivity),
            "building_compactness": _safe_float(compactness),
        }

    except Exception:
        return {
            "building_density": 0.0,
            "road_coverage": 0.0,
            "green_coverage": 0.0,
            "osr_proxy": 0.0,
            "road_connectivity": 0.0,
            "building_compactness": 0.0,
        }


# ---------------------------
# SUSTAINABILITY SCORE
# ---------------------------

def sustainability_score(metrics):
    try:
        g = metrics["green_coverage"] * 100
        d = metrics["building_density"] * 100
        r = metrics["road_coverage"] * 100
        conn = metrics["road_connectivity"] * 100

        score = (
            0.45 * g +
            0.20 * (100 - d) +
            0.20 * (100 - r) +
            0.10 * conn +
            0.05 * 50
        )

        return round(max(0, min(100, score)), 2)

    except Exception:
        return 0.0


# ---------------------------
# FINAL METRICS
# ---------------------------

def compute_enhanced_metrics(real_img, fake_img):
    try:
        basic = compute_metrics(real_img, fake_img)
        urban = compute_urban_metrics(fake_img)

        score = sustainability_score(urban)

        combined = {**basic, **urban}
        combined = {k: _safe_float(v) for k, v in combined.items()}

        combined["sustainability_score"] = score
        combined["Sustainability Score"] = score

        combined["Green Coverage"] = round(urban["green_coverage"], 4)
        combined["Building Density"] = round(urban["building_density"], 4)
        combined["Road Coverage"] = round(urban["road_coverage"], 4)

        return combined

    except Exception:
        return {
            "SSIM": 0.0,
            "L1": 1.0,
            "Edge Consistency": 0.0,
            "building_density": 0.0,
            "road_coverage": 0.0,
            "green_coverage": 0.0,
            "osr_proxy": 0.0,
            "road_connectivity": 0.0,
            "building_compactness": 0.0,
            "sustainability_score": 0.0,
        }