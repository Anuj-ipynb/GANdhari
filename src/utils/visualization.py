# src/utils/visualization.py
import os
from torchvision.utils import save_image

def save_sample(epoch, real, fake, path="outputs/samples"):
    """Save sample images during training"""
    os.makedirs(path, exist_ok=True)
    combined = torch.cat([real * 0.5 + 0.5, fake * 0.5 + 0.5], dim=0)
    save_image(combined, f"{path}/epoch_{epoch:04d}.png", nrow=2, normalize=True)