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

        self.checkpoint_dir = config.get('checkpoint_dir', 'outputs/checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # -----------------------------
        # Models
        # -----------------------------
        self.G = UNetGenerator(ngf=config.get('ngf', 32)).to(self.device)
        self.D = PatchGAN(ndf=config.get('ndf', 64)).to(self.device)

        # -----------------------------
        # ✅ RESUME LOADING (CRITICAL FIX)
        # -----------------------------
        self.start_epoch = 0
        if config.get("resume", False) and config.get("resume_path"):
            if os.path.exists(config["resume_path"]):
                checkpoint = torch.load(config["resume_path"], map_location=self.device)

                self.G.load_state_dict(checkpoint["G"])
                self.D.load_state_dict(checkpoint["D"])
                self.start_epoch = checkpoint.get("epoch", 0) + 1

                print(f"✅ Loaded checkpoint from epoch {self.start_epoch}")
            else:
                print("⚠️ Resume checkpoint not found, starting fresh")

        # -----------------------------
        # Losses
        # -----------------------------
        self.criterion_GAN = nn.BCEWithLogitsLoss()
        self.criterion_L1 = nn.L1Loss()

        # -----------------------------
        # Optimizers
        # -----------------------------
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=config['lr'], betas=(0.5, 0.999))

        # -----------------------------
        # Dataset
        # -----------------------------
        self.dataset = CityscapesDataset()
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0
        )

        print(f"📊 Dataset size: {len(self.dataset)} images")

    def train(self):
        epochs = self.config.get('epochs', 10)
        lambda_l1 = self.config.get('lambda_l1', 70)

        print(f"🚀 Training for {epochs} epochs")
        print(f"λ_L1 = {lambda_l1}")

        for epoch in range(self.start_epoch, self.start_epoch + epochs):
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}")

            for real_input, real_target in pbar:
                real_input = real_input.to(self.device)
                real_target = real_target.to(self.device)

                # -----------------------------
                # Train Discriminator
                # -----------------------------
                fake = self.G(real_input).detach()

                pred_real = self.D(real_input, real_target)
                pred_fake = self.D(real_input, fake)

                loss_D = (
                    self.criterion_GAN(pred_real, torch.ones_like(pred_real)) +
                    self.criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
                ) * 0.5

                self.opt_D.zero_grad()
                loss_D.backward()
                self.opt_D.step()

                # -----------------------------
                # Train Generator
                # -----------------------------
                fake = self.G(real_input)
                pred_fake = self.D(real_input, fake)

                loss_G_GAN = self.criterion_GAN(pred_fake, torch.ones_like(pred_fake))
                loss_G_L1 = self.criterion_L1(fake, real_target)

                loss_G = loss_G_GAN + lambda_l1 * loss_G_L1

                self.opt_G.zero_grad()
                loss_G.backward()
                self.opt_G.step()

                pbar.set_postfix(
                    G=f"{loss_G.item():.4f}",
                    L1=f"{loss_G_L1.item():.4f}"
                )

            # -----------------------------
            # Save checkpoint
            # -----------------------------
            torch.save({
                "epoch": epoch,
                "G": self.G.state_dict(),
                "D": self.D.state_dict(),
            }, os.path.join(self.checkpoint_dir, "checkpoint_final.pth"))

            print(f"✅ Epoch {epoch+1} saved")

        print("🎉 Training complete!")