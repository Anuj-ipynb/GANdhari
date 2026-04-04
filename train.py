# train.py

import yaml
import argparse
import os
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="SustainableUrbanPix2Pix Fine-Tuning")
    
    parser.add_argument("--additional_epochs", type=int, default=30,
                        help="Number of additional epochs to fine-tune (recommended)")
    parser.add_argument("--lr", type=float, default=0.00005,
                        help="Learning rate for fine-tuning")
    parser.add_argument('--tiny', action='store_true', 
                        help='Use tiny mode (ngf=32)')
    parser.add_argument('--config', default='config.yaml', help='Config file path')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Apply overrides
    config['lr'] = args.lr
    if args.tiny:
        config['ngf'] = 32
        config['ndf'] = 32
        print("⚡ Tiny Mode Enabled (ngf=32, ndf=32)")

    # Calculate total epochs = current checkpoint epoch + additional_epochs 
    # We assume checkpoint is at epoch 100
    current_epoch = 100   # Change this if your checkpoint is at different epoch
    total_epochs = current_epoch + args.additional_epochs

    config['epochs'] = total_epochs

    print("=" * 70)
    print("🚀 SustainableUrbanPix2Pix Fine-Tuning")
    print(f"Resuming from epoch {current_epoch}")
    print(f"Additional epochs : {args.additional_epochs}")
    print(f"Total epochs      : {total_epochs}")
    print(f"Learning Rate     : {config['lr']}")
    print(f"λ_L1              : {config.get('lambda_l1', 120)}")
    print(f"λ_Perc            : {config.get('lambda_perc', 15)}")
    print("=" * 70)

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()