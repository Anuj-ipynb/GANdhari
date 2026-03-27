# src/training/trainer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from src.models.generator import UNetGenerator
from src.models.discriminator import PatchGAN
from src.data.dataset import CityscapesDataset


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -------------------------
        # Models
        # -------------------------
        self.G = UNetGenerator(ngf=config.get('ngf', 64)).to(self.device)
        self.D = PatchGAN(ndf=config.get('ndf', 64)).to(self.device)

        print("⚠️ torch.compile disabled (Pascal GPU detected)")

        # -------------------------
        # Optimizers
        # -------------------------
        self.opt_G = torch.optim.Adam(
            self.G.parameters(),
            lr=config['lr'],
            betas=(0.5, 0.999)
        )

        self.opt_D = torch.optim.Adam(
            self.D.parameters(),
            lr=config['lr'],
            betas=(0.5, 0.999)
        )

        # -------------------------
        # Losses
        # -------------------------
        self.criterion_GAN = nn.BCEWithLogitsLoss()
        self.criterion_L1 = nn.L1Loss()

        # -------------------------
        # AMP (NEW API - FIXED)
        # -------------------------
        self.scaler = torch.amp.GradScaler("cuda")

        # -------------------------
        # Dataset
        # -------------------------
        self.dataset = CityscapesDataset()
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )

        # -------------------------
        # Paths
        # -------------------------
        self.checkpoint_dir = config.get('checkpoint_dir', 'outputs/checkpoints')
        self.sample_dir = config.get('sample_dir', 'outputs/samples')

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)

        # -------------------------
        # Gradient Accumulation
        # -------------------------
        self.accum_steps = config.get("accum_steps", 4)

    def train(self):
        epochs = self.config.get('epochs', 50)

        print(f"Starting training | Epochs: {epochs}")

        for epoch in range(epochs):
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{epochs}")

            self.opt_G.zero_grad(set_to_none=True)
            self.opt_D.zero_grad(set_to_none=True)

            for i, (real_input, real_target) in enumerate(pbar):
                real_input = real_input.to(self.device, non_blocking=True)
                real_target = real_target.to(self.device, non_blocking=True)

                # =========================
                # 🔥 Train Discriminator
                # =========================
                with torch.amp.autocast("cuda"):
                    fake = self.G(real_input)

                    pred_real = self.D(real_input, real_target)
                    pred_fake = self.D(real_input, fake.detach())

                    loss_D_real = self.criterion_GAN(
                        pred_real,
                        torch.ones_like(pred_real)
                    )

                    loss_D_fake = self.criterion_GAN(
                        pred_fake,
                        torch.zeros_like(pred_fake)
                    )

                    loss_D = (loss_D_real + loss_D_fake) * 0.5
                    loss_D = loss_D / self.accum_steps

                self.scaler.scale(loss_D).backward()

                if (i + 1) % self.accum_steps == 0:
                    self.scaler.step(self.opt_D)
                    self.scaler.update()
                    self.opt_D.zero_grad(set_to_none=True)

                # =========================
                # 🚀 Train Generator
                # =========================
                with torch.amp.autocast("cuda"):
                    fake = self.G(real_input)
                    pred_fake_for_G = self.D(real_input, fake)

                    loss_G_GAN = self.criterion_GAN(
                        pred_fake_for_G,
                        torch.ones_like(pred_fake_for_G)
                    )

                    loss_G_L1 = self.criterion_L1(fake, real_target)

                    loss_G = loss_G_GAN + self.config.get('lambda_l1', 100) * loss_G_L1
                    loss_G = loss_G / self.accum_steps

                self.scaler.scale(loss_G).backward()

                if (i + 1) % self.accum_steps == 0:
                    self.scaler.step(self.opt_G)
                    self.scaler.update()
                    self.opt_G.zero_grad(set_to_none=True)

                # =========================
                # 🧹 VRAM Cleanup (CRITICAL)
                # =========================
                del fake, pred_real, pred_fake, pred_fake_for_G
                torch.cuda.empty_cache()

                # =========================
                # 📊 Logging
                # =========================
                pbar.set_postfix({
                    "G": f"{loss_G.item():.3f}",
                    "D": f"{loss_D.item():.3f}"
                })

            print(f"✅ Epoch {epoch+1} completed")

            # =========================
            # 💾 Checkpointing
            # =========================
            if (epoch + 1) % 10 == 0:
                torch.save(
                    self.G.state_dict(),
                    os.path.join(self.checkpoint_dir, f"G_epoch_{epoch+1}.pth")
                )

        # Final model
        torch.save(
            self.G.state_dict(),
            os.path.join(self.checkpoint_dir, "G_final.pth")
        )

        print("✅ Training finished successfully!")