# train.py

import yaml
import argparse
import os
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Pix2Pix Training (Resume Enabled)")

    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--tiny", action="store_true")

    args = parser.parse_args()

    # -----------------------------
    # Load config
    # -----------------------------
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # -----------------------------
    # Overrides
    # -----------------------------
    config["epochs"] = args.epochs
    config["lr"] = args.lr

    if args.tiny:
        config["ngf"] = 32
        config["ndf"] = 32
        print("⚡ Tiny mode enabled")

    # -----------------------------
    # ✅ RESUME LOGIC (FIXED)
    # -----------------------------
    ckpt_path = os.path.join(
        config["checkpoint_dir"],
        config["resume_checkpoint"]
    )

    if os.path.exists(ckpt_path):
        config["resume"] = True
        config["resume_path"] = ckpt_path
        mode = "RESUME"
    else:
        config["resume"] = False
        config["resume_path"] = None
        mode = "FROM SCRATCH"
        print("⚠️ Checkpoint not found, starting fresh")

    # -----------------------------
    # Debug info
    # -----------------------------
    print("=" * 60)
    print("🚀 TRAINING START")
    print(f"Epochs        : {config['epochs']}")
    print(f"Learning Rate : {config['lr']}")
    print(f"Batch Size    : {config['batch_size']}")
    print(f"ngf           : {config['ngf']}")
    print(f"Mode          : {mode}")
    print("=" * 60)

    # -----------------------------
    # Start training
    # -----------------------------
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()