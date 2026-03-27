# filename: src/utils/metrics.py
from skimage.metrics import structural_similarity as ssim

def compute_ssim(real, fake):
    real_np = real[0].cpu().numpy().transpose(1,2,0)
    fake_np = fake[0].cpu().numpy().transpose(1,2,0)
    return ssim(real_np, fake_np, multichannel=True, data_range=2.0, channel_axis=2)