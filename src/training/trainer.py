# src/training/trainer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from src.models.generator import UNetGenerator
from src.models.discriminator import PatchGAN
from src.data.dataset import CityscapesDataset
from src.losses.perceptual_loss import VGGPerceptualLoss


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.checkpoint_dir = config.get('checkpoint_dir', 'outputs/checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Models
        self.G = UNetGenerator(ngf=config.get('ngf', 32)).to(self.device)
        self.D = PatchGAN(ndf=config.get('ndf', 64)).to(self.device)

        # Losses
        self.criterion_GAN = nn.BCEWithLogitsLoss()
        self.criterion_L1 = nn.L1Loss()
        self.perceptual_loss = VGGPerceptualLoss(self.device)

        # Optimizers
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=config['lr'], betas=(0.5, 0.999))

        # AMP + Accumulation
        self.scaler = torch.amp.GradScaler("cuda")
        self.accum_steps = config.get("accumulation_steps", 8)

        # Dataset
        self.dataset = CityscapesDataset()
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )

        # Resume Logic
        self.start_epoch = 0
        checkpoint_path = os.path.join(self.checkpoint_dir, "checkpoint_final.pth")

        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.G.load_state_dict(checkpoint["G"])
                self.D.load_state_dict(checkpoint["D"])
                if "opt_G" in checkpoint:
                    self.opt_G.load_state_dict(checkpoint["opt_G"])
                if "opt_D" in checkpoint:
                    self.opt_D.load_state_dict(checkpoint["opt_D"])
                self.start_epoch = checkpoint.get("epoch", 99) + 1
                print(f"✅ Resuming from epoch {self.start_epoch}")
            except Exception as e:
                print(f"⚠️ Failed to load checkpoint: {e}")
        else:
            print("⚠️ No checkpoint found. Starting from scratch.")

    def train(self):
        total_epochs = self.config.get('epochs', 50)
        lambda_l1 = self.config.get('lambda_l1', 120)
        lambda_perc = self.config.get('lambda_perc', 15)

        print(f"🚀 Starting fine-tuning from epoch {self.start_epoch} to {total_epochs}")
        print(f"λ_L1 = {lambda_l1} | λ_Perc = {lambda_perc} | LR = {self.config['lr']}")

        for epoch in range(self.start_epoch, total_epochs):
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{total_epochs}")

            for i, (real_input, real_target) in enumerate(pbar):
                real_input = real_input.to(self.device)
                real_target = real_target.to(self.device)

                # Train Discriminator
                with torch.amp.autocast("cuda"):
                    fake = self.G(real_input)
                    pred_real = self.D(real_input, real_target)
                    pred_fake = self.D(real_input, fake.detach())

                    loss_D = (
                        self.criterion_GAN(pred_real, torch.ones_like(pred_real)) +
                        self.criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
                    ) * 0.5 / self.accum_steps

                self.scaler.scale(loss_D).backward()

                if (i + 1) % self.accum_steps == 0:
                    self.scaler.step(self.opt_D)
                    self.scaler.update()
                    self.opt_D.zero_grad(set_to_none=True)

                # Train Generator with Perceptual Loss
                with torch.amp.autocast("cuda"):
                    fake = self.G(real_input)
                    pred_fake = self.D(real_input, fake)

                    loss_G_GAN = self.criterion_GAN(pred_fake, torch.ones_like(pred_fake))
                    loss_G_L1 = self.criterion_L1(fake, real_target)
                    loss_G_perc = self.perceptual_loss(fake, real_target)

                    loss_G = loss_G_GAN + lambda_l1 * loss_G_L1 + lambda_perc * loss_G_perc

                self.scaler.scale(loss_G).backward()

                if (i + 1) % self.accum_steps == 0:
                    self.scaler.step(self.opt_G)
                    self.scaler.update()
                    self.opt_G.zero_grad(set_to_none=True)

                del fake, pred_real, pred_fake
                torch.cuda.empty_cache()

                pbar.set_postfix(
                    G=f"{loss_G.item():.4f}",
                    Perc=f"{loss_G_perc.item():.4f}"
                )

            # Save checkpoint after every epoch
            torch.save({
                "epoch": epoch,
                "G": self.G.state_dict(),
                "D": self.D.state_dict(),
                "opt_G": self.opt_G.state_dict(),
                "opt_D": self.opt_D.state_dict()
            }, os.path.join(self.checkpoint_dir, "checkpoint_final.pth"))

            print(f"✅ Epoch {epoch+1} completed and saved")

        print("🎉 Fine-tuning completed successfully!")