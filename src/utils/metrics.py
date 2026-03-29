# src/utils/metrics.py

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim


def compute_metrics(real_img, fake_img):
    """
    real_img, fake_img: PIL Images
    """

    real = np.array(real_img)
    fake = np.array(fake_img)

    # Resize safety
    fake = cv2.resize(fake, (real.shape[1], real.shape[0]))

    # -------------------
    # SSIM
    # -------------------
    ssim_score = ssim(real, fake, channel_axis=2, data_range=255)

    # -------------------
    # L1
    # -------------------
    l1 = np.mean(np.abs(real.astype(np.float32) - fake.astype(np.float32))) / 255.0

    # -------------------
    # Edge Consistency
    # -------------------
    real_edges = cv2.Canny(cv2.cvtColor(real, cv2.COLOR_RGB2GRAY), 100, 200)
    fake_edges = cv2.Canny(cv2.cvtColor(fake, cv2.COLOR_RGB2GRAY), 100, 200)

    edge_score = np.mean(real_edges == fake_edges)

    return {
        "SSIM": round(ssim_score, 4),
        "L1": round(l1, 4),
        "Edge Consistency": round(edge_score, 4)
    }