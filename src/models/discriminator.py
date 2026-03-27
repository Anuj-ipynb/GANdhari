# src/models/discriminator.py
import torch.nn as nn
import torch

class PatchGAN(nn.Module):
    """PatchGAN Discriminator - Standard for Pix2Pix"""
    def __init__(self, ndf=64):
        super().__init__()
        self.model = nn.Sequential(
            # 256 -> 128
            nn.Conv2d(6, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128 -> 64
            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64 -> 32
            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32 -> 31 (Patch output)
            nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf*8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x, y):
        # Concatenate input and target along channel dimension
        return self.model(torch.cat([x, y], dim=1))