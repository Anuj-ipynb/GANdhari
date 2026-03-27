# src/utils/preprocessing.py
import cv2

def canny_preprocess(image_path):
    """Clean raw sketches using Canny edge detection"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read {image_path}")
    edges = cv2.Canny(img, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)