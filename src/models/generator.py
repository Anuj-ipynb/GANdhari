# src/models/generator.py

import torch
import torch.nn as nn


class UNetGenerator(nn.Module):
    """
    FINAL Stable Pix2Pix Generator for GTX 1050 Ti

    Fixes:
    - No 1x1 InstanceNorm crash
    - Correct 256x256 output
    - Balanced depth for low VRAM
    """

    def __init__(self, ngf=64):
        super().__init__()

        # -------------------
        # Encoder (downsampling)
        # -------------------
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, ngf, 4, 2, 1),  # 256 -> 128
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1),  # 128 -> 64
            nn.InstanceNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1),  # 64 -> 32
            nn.InstanceNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.enc4 = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1),  # 32 -> 16
            nn.InstanceNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.enc5 = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1),  # 16 -> 8
            nn.InstanceNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.enc6 = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1),  # 8 -> 4
            nn.InstanceNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.enc7 = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1),  # 4 -> 2
            nn.LeakyReLU(0.2, inplace=True)  # NO norm (critical)
        )

        # -------------------
        # Decoder (upsampling)
        # -------------------
        self.dec6 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1),  # 2 -> 4
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(ngf * 8)
        )

        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1),  # 4 -> 8
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(ngf * 8)
        )

        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1),  # 8 -> 16
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(ngf * 8)
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 16, ngf * 4, 4, 2, 1),  # 16 -> 32
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(ngf * 4)
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 2, 4, 2, 1),  # 32 -> 64
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(ngf * 2)
        )

        # 🔥 FIX: split final into TWO stages
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf, 4, 2, 1),  # 64 -> 128
            nn.ReLU(inplace=True)
        )

        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1),  # 128 -> 256 ✅
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)   # 128
        e2 = self.enc2(e1)  # 64
        e3 = self.enc3(e2)  # 32
        e4 = self.enc4(e3)  # 16
        e5 = self.enc5(e4)  # 8
        e6 = self.enc6(e5)  # 4
        e7 = self.enc7(e6)  # 2

        # Decoder
        d6 = self.dec6(e7)                      # 4
        d5 = self.dec5(torch.cat([d6, e6], 1))  # 8
        d4 = self.dec4(torch.cat([d5, e5], 1))  # 16
        d3 = self.dec3(torch.cat([d4, e4], 1))  # 32
        d2 = self.dec2(torch.cat([d3, e3], 1))  # 64
        d1 = self.dec1(torch.cat([d2, e2], 1))  # 128
        out = self.final(d1)                    # 256 ✅

        return out